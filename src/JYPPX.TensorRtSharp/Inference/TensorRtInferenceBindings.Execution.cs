using System;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtInferenceBindings
{
    /// <summary>
    /// Gets the current execution-context readiness snapshot.
    /// 获取当前 execution context 的就绪状态快照。
    /// </summary>
    /// <param name="runShapeInference">Whether to run TensorRT shape inference first. 是否先执行 TensorRT shape inference。</param>
    /// <returns>The readiness snapshot. 就绪状态快照。</returns>
    public TensorRtExecutionContextReadiness GetReadiness(bool runShapeInference = true)
    {
        ThrowIfDisposed();
        return _context.GetReadiness(_engine, runShapeInference);
    }

    /// <summary>
    /// Enqueues inference asynchronously on a CUDA stream, optionally synchronizing before return.
    /// 在 CUDA stream 上异步提交推理，并可选择返回前同步。
    /// </summary>
    /// <param name="stream">The CUDA stream used for enqueue. 用于 enqueue 的 CUDA stream。</param>
    /// <param name="synchronize">Whether to synchronize the stream before returning. 是否在返回前同步 stream。</param>
    /// <param name="runShapeInference">Whether to run TensorRT shape inference for readiness validation. 是否为就绪校验执行 TensorRT shape inference。</param>
    /// <returns>An execution summary. 执行摘要。</returns>
    public TensorRtInferenceExecutionSummary EnqueueAsync(CudaStream stream, bool synchronize = false, bool runShapeInference = true)
    {
        ThrowIfDisposed();
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        TensorRtExecutionContextReadiness readiness = PrepareForExecution(runShapeInference);

        _context.EnqueueAsync(stream);
        if (synchronize)
        {
            stream.Synchronize();
        }

        return new TensorRtInferenceExecutionSummary(ProfileIndex, _buffers.Count, synchronize, readiness);
    }

    /// <summary>
    /// Executes an explicit-batch network synchronously through TensorRT <c>executeV2</c>.
    /// 通过 TensorRT <c>executeV2</c> 同步执行 explicit-batch 网络。
    /// </summary>
    /// <param name="runShapeInference">Whether to run shape inference during readiness validation. 是否在就绪校验时执行 shape inference。</param>
    /// <returns>An execution summary. 执行摘要。</returns>
    public TensorRtInferenceExecutionSummary ExecuteV2(bool runShapeInference = true)
    {
        ThrowIfDisposed();
        if (_engine.Line == TensorRtApiLine.TensorRt8 && _engine.HasImplicitBatchDimensionCompatibility)
        {
            throw new InvalidOperationException("TensorRT 8 implicit-batch engines must use ExecuteLegacy.");
        }

        TensorRtExecutionContextReadiness readiness = PrepareForExecution(runShapeInference);
        _context.ExecuteV2();
        return new TensorRtInferenceExecutionSummary(ProfileIndex, _buffers.Count, synchronized: true, readiness: readiness);
    }

    /// <summary>
    /// Executes a TensorRT 8 implicit-batch network synchronously through legacy <c>execute</c>.
    /// 通过 legacy <c>execute</c> 同步执行 TensorRT 8 implicit-batch 网络。
    /// </summary>
    /// <param name="batchSize">The positive legacy batch size. 正数 legacy batch size。</param>
    /// <param name="runShapeInference">Whether to run shape inference during readiness validation. 是否在就绪校验时执行 shape inference。</param>
    /// <returns>An execution summary. 执行摘要。</returns>
    public TensorRtInferenceExecutionSummary ExecuteLegacy(int batchSize, bool runShapeInference = true)
    {
        ThrowIfDisposed();
        if (_engine.Line != TensorRtApiLine.TensorRt8)
        {
            throw new NotSupportedException("Legacy implicit-batch execution is available only for TensorRT 8.");
        }
        if (!_engine.HasImplicitBatchDimensionCompatibility)
        {
            throw new InvalidOperationException("Legacy Execute requires a TensorRT 8 implicit-batch engine.");
        }
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Legacy execution batch size must be positive.");
        }

        TensorRtExecutionContextReadiness readiness = PrepareForExecution(runShapeInference);
        _context.ExecuteLegacy(batchSize);
        return new TensorRtInferenceExecutionSummary(ProfileIndex, _buffers.Count, synchronized: true, readiness: readiness);
    }

    /// <summary>
    /// Enqueues a TensorRT 8 explicit-batch network through <c>enqueueV2</c> and synchronizes before returning.
    /// 通过 <c>enqueueV2</c> 提交 TensorRT 8 explicit-batch 网络，并在返回前完成同步。
    /// </summary>
    /// <param name="stream">The caller-owned CUDA stream. 调用方拥有的 CUDA stream。</param>
    /// <param name="runShapeInference">Whether to run shape inference during readiness validation. 是否在就绪校验时执行 shape inference。</param>
    /// <returns>An execution summary whose synchronized flag is always true. synchronized 恒为 true 的执行摘要。</returns>
    public TensorRtInferenceExecutionSummary EnqueueV2AndSynchronize(CudaStream stream, bool runShapeInference = true)
    {
        ThrowIfDisposed();
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }
        if (_engine.Line != TensorRtApiLine.TensorRt8)
        {
            throw new NotSupportedException("enqueueV2 compatibility execution is available only for TensorRT 8.");
        }
        if (_engine.HasImplicitBatchDimensionCompatibility)
        {
            throw new InvalidOperationException("enqueueV2 compatibility execution requires an explicit-batch engine.");
        }

        TensorRtExecutionContextReadiness readiness = PrepareForExecution(runShapeInference);
        _context.EnqueueV2(stream);
        stream.Synchronize();
        return new TensorRtInferenceExecutionSummary(ProfileIndex, _buffers.Count, synchronized: true, readiness: readiness);
    }

    private TensorRtExecutionContextReadiness PrepareForExecution(bool runShapeInference)
    {
        BindAll();
        TensorRtExecutionContextReadiness readiness = GetReadiness(runShapeInference);
        if (!readiness.IsReadyForEnqueue)
        {
            throw new InvalidOperationException("TensorRT execution context is not ready for inference: " + readiness);
        }

        return readiness;
    }
}
