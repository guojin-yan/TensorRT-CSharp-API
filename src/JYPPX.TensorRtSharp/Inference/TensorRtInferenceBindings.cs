using System;
using System.Collections.Generic;
using System.Text;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// High-level TensorRT inference binding set for deployment-oriented enqueue flows.
/// 面向模型部署 enqueue 流程的高层 TensorRT 推理绑定集。
/// </summary>
/// <remarks>
/// This class owns only the CUDA buffers that it allocates itself. Buffers supplied through
/// <see cref="UseDeviceBuffer(string, CudaMemory, TensorRtDims?)"/> remain owned by the caller.
/// 该类只拥有自己分配的 CUDA 缓冲区；通过
/// <see cref="UseDeviceBuffer(string, CudaMemory, TensorRtDims?)"/> 传入的缓冲区仍由调用方拥有。
/// </remarks>
public sealed partial class TensorRtInferenceBindings : IDisposable
{
    private readonly TensorRtEngine _engine;
    private readonly TensorRtExecutionContext _context;
    private readonly Dictionary<string, TensorRtInferenceBuffer> _buffers;
    private bool _disposed;

    /// <summary>
    /// Creates a binding set for one engine, execution context, and optimization profile.
    /// 为一个 engine、execution context 和 optimization profile 创建绑定集。
    /// </summary>
    /// <param name="engine">The TensorRT engine. TensorRT 引擎。</param>
    /// <param name="context">The execution context used for enqueue. 用于 enqueue 的 execution context。</param>
    /// <param name="profileIndex">The optimization profile index. Optimization profile 索引。</param>
    public TensorRtInferenceBindings(TensorRtEngine engine, TensorRtExecutionContext context, int profileIndex = 0)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _context = context ?? throw new ArgumentNullException(nameof(context));
        if (_engine.Line != _context.Line)
        {
            throw new ArgumentException("Engine and execution context must belong to the same TensorRT API line.", nameof(context));
        }

        if (profileIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(profileIndex), "Profile index must be greater than or equal to zero.");
        }

        ProfileIndex = profileIndex;
        Report = engine.GetBindingReport(context, profileIndex, runShapeInference: false);
        _buffers = new Dictionary<string, TensorRtInferenceBuffer>(StringComparer.Ordinal);
    }

    /// <summary>
    /// Gets the optimization profile index used by this binding set.
    /// 获取该绑定集使用的 optimization profile 索引。
    /// </summary>
    public int ProfileIndex { get; }

    /// <summary>
    /// Gets the latest engine binding report.
    /// 获取最近一次 engine 绑定报告。
    /// </summary>
    public TensorRtEngineBindingReport Report { get; private set; }

    /// <summary>
    /// Gets the current managed CUDA buffers keyed by TensorRT tensor name.
    /// 获取按 TensorRT tensor 名称索引的当前托管 CUDA 缓冲区。
    /// </summary>
    public IReadOnlyDictionary<string, TensorRtInferenceBuffer> Buffers => _buffers;

    /// <summary>
    /// Releases CUDA buffers owned by this binding set.
    /// 释放该绑定集拥有的 CUDA 缓冲区。
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        foreach (TensorRtInferenceBuffer buffer in _buffers.Values)
        {
            if (buffer.OwnsMemory)
            {
                buffer.Memory.Dispose();
            }
        }

        _buffers.Clear();
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private TensorRtEngineTensorBinding GetTensor(string tensorName)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
        }

        foreach (TensorRtEngineTensorBinding tensor in Report.Tensors)
        {
            if (StringComparer.Ordinal.Equals(tensor.Name, tensorName))
            {
                return tensor;
            }
        }

        throw new ArgumentException($"Tensor '{tensorName}' was not found in the engine binding report.", nameof(tensorName));
    }

    private void RefreshReport(bool runShapeInference)
    {
        Report = _engine.GetBindingReport(_context, ProfileIndex, runShapeInference);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(TensorRtInferenceBindings));
        }
    }
}
