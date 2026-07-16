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
public sealed class TensorRtInferenceBindings : IDisposable
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
    /// Sets a dynamic input shape on the execution context and refreshes the binding report.
    /// 在 execution context 上设置动态输入 shape，并刷新绑定报告。
    /// </summary>
    /// <param name="tensorName">The input tensor name. 输入 tensor 名称。</param>
    /// <param name="shape">The runtime input shape. 运行时输入 shape。</param>
    /// <returns>The current binding set for chaining. 当前绑定集，便于链式调用。</returns>
    public TensorRtInferenceBindings SetInputShape(string tensorName, TensorRtDims shape)
    {
        ThrowIfDisposed();
        if (shape == null)
        {
            throw new ArgumentNullException(nameof(shape));
        }

        TensorRtEngineTensorBinding tensor = GetTensor(tensorName);
        if (tensor.IOMode != TensorRtIOMode.Input)
        {
            throw new ArgumentException("Only input tensors can receive an input shape.", nameof(tensorName));
        }

        _context.SetInputShape(tensorName, shape);
        if (_buffers.TryGetValue(tensorName, out TensorRtInferenceBuffer? buffer))
        {
            buffer.RuntimeShape = shape;
            buffer.SizeInBytes = EstimateTensorByteSize(tensor, shape);
        }

        RefreshReport(runShapeInference: false);
        return this;
    }

    /// <summary>
    /// Allocates a CUDA buffer for the named tensor.
    /// 为指定 tensor 分配 CUDA 缓冲区。
    /// </summary>
    /// <param name="tensorName">The TensorRT tensor name. TensorRT tensor 名称。</param>
    /// <param name="runtimeShape">Optional runtime shape used for size estimation. 用于估算大小的可选运行时 shape。</param>
    /// <param name="sizeInBytes">Optional explicit allocation size. 可选的显式分配字节数。</param>
    /// <returns>The created buffer descriptor. 创建的缓冲区描述。</returns>
    public TensorRtInferenceBuffer AllocateDeviceBuffer(string tensorName, TensorRtDims? runtimeShape = null, int? sizeInBytes = null)
    {
        ThrowIfDisposed();
        TensorRtEngineTensorBinding tensor = GetTensor(tensorName);
        TensorRtDims shape = runtimeShape ?? ResolveRuntimeShape(tensor);
        int requiredBytes = sizeInBytes ?? EstimateTensorByteSize(tensor, shape);
        if (requiredBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes), "Tensor buffer size must be greater than zero.");
        }

        RemoveOwnedBuffer(tensor.Name);
        CudaMemory memory = new CudaMemory(requiredBytes);
        TensorRtInferenceBuffer buffer = new TensorRtInferenceBuffer(tensor, memory, shape, requiredBytes, ownsMemory: true);
        _buffers[tensor.Name] = buffer;
        return buffer;
    }

    /// <summary>
    /// Attaches an externally owned CUDA buffer to the named tensor.
    /// 将外部拥有的 CUDA 缓冲区附加到指定 tensor。
    /// </summary>
    /// <param name="tensorName">The TensorRT tensor name. TensorRT tensor 名称。</param>
    /// <param name="memory">The externally owned CUDA allocation. 外部拥有的 CUDA 设备内存。</param>
    /// <param name="runtimeShape">Optional runtime shape used for validation and diagnostics. 用于验证和诊断的可选运行时 shape。</param>
    /// <returns>The buffer descriptor. 缓冲区描述。</returns>
    public TensorRtInferenceBuffer UseDeviceBuffer(string tensorName, CudaMemory memory, TensorRtDims? runtimeShape = null)
    {
        ThrowIfDisposed();
        if (memory == null)
        {
            throw new ArgumentNullException(nameof(memory));
        }

        TensorRtEngineTensorBinding tensor = GetTensor(tensorName);
        TensorRtDims shape = runtimeShape ?? ResolveRuntimeShape(tensor);
        int requiredBytes = EstimateTensorByteSize(tensor, shape);
        if (memory.SizeInBytes < requiredBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(memory), "CUDA allocation is smaller than the estimated tensor binding size.");
        }

        RemoveOwnedBuffer(tensor.Name);
        TensorRtInferenceBuffer buffer = new TensorRtInferenceBuffer(tensor, memory, shape, requiredBytes, ownsMemory: false);
        _buffers[tensor.Name] = buffer;
        return buffer;
    }

    /// <summary>
    /// Copies single-precision input data into a named input buffer, allocating the buffer when needed.
    /// 将单精度输入数据复制到指定输入缓冲区；必要时自动分配缓冲区。
    /// </summary>
    /// <param name="tensorName">The input tensor name. 输入 tensor 名称。</param>
    /// <param name="values">The input values. 输入数据。</param>
    /// <param name="runtimeShape">Optional runtime shape used for allocation. 用于分配的可选运行时 shape。</param>
    /// <returns>The current binding set for chaining. 当前绑定集，便于链式调用。</returns>
    public TensorRtInferenceBindings CopyInputFromHost(string tensorName, float[] values, TensorRtDims? runtimeShape = null)
    {
        ThrowIfDisposed();
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        TensorRtInferenceBuffer buffer = EnsureBuffer(tensorName, runtimeShape, checked(values.Length * sizeof(float)));
        if (buffer.Tensor.IOMode != TensorRtIOMode.Input)
        {
            throw new ArgumentException("Only input tensors can be copied from host.", nameof(tensorName));
        }

        buffer.Memory.CopyFrom(values);
        return this;
    }

    /// <summary>
    /// Copies byte input data into a named input buffer, allocating the buffer when needed.
    /// 将字节输入数据复制到指定输入缓冲区；必要时自动分配缓冲区。
    /// </summary>
    /// <param name="tensorName">The input tensor name. 输入 tensor 名称。</param>
    /// <param name="bytes">The input bytes. 输入字节。</param>
    /// <param name="runtimeShape">Optional runtime shape used for allocation. 用于分配的可选运行时 shape。</param>
    /// <returns>The current binding set for chaining. 当前绑定集，便于链式调用。</returns>
    public TensorRtInferenceBindings CopyInputFromHost(string tensorName, byte[] bytes, TensorRtDims? runtimeShape = null)
    {
        ThrowIfDisposed();
        if (bytes == null)
        {
            throw new ArgumentNullException(nameof(bytes));
        }

        TensorRtInferenceBuffer buffer = EnsureBuffer(tensorName, runtimeShape, bytes.Length);
        if (buffer.Tensor.IOMode != TensorRtIOMode.Input)
        {
            throw new ArgumentException("Only input tensors can be copied from host.", nameof(tensorName));
        }

        buffer.Memory.CopyFrom(bytes);
        return this;
    }

    /// <summary>
    /// Binds one tensor buffer to the execution context.
    /// 将一个 tensor 缓冲区绑定到 execution context。
    /// </summary>
    /// <param name="tensorName">The TensorRT tensor name. TensorRT tensor 名称。</param>
    /// <returns>The current binding set for chaining. 当前绑定集，便于链式调用。</returns>
    public TensorRtInferenceBindings BindTensor(string tensorName)
    {
        ThrowIfDisposed();
        if (!_buffers.TryGetValue(tensorName, out TensorRtInferenceBuffer? buffer))
        {
            throw new InvalidOperationException($"Tensor '{tensorName}' does not have an attached CUDA buffer.");
        }

        if (buffer.Tensor.Location != TensorRtTensorLocation.Device)
        {
            throw new NotSupportedException("High-level inference bindings currently support device tensors only.");
        }

        if (buffer.Tensor.IOMode == TensorRtIOMode.Input)
        {
            _context.SetInputTensorAddress(buffer.Tensor.Name, buffer.Memory);
        }
        else if (buffer.Tensor.IOMode == TensorRtIOMode.Output)
        {
            _context.SetOutputTensorAddress(buffer.Tensor.Name, buffer.Memory);
        }
        else
        {
            _context.SetTensorAddress(buffer.Tensor.Name, buffer.Memory);
        }

        buffer.IsBound = true;
        return this;
    }

    /// <summary>
    /// Binds every attached CUDA buffer to the execution context.
    /// 将所有已附加 CUDA 缓冲区绑定到 execution context。
    /// </summary>
    /// <returns>The current binding set for chaining. 当前绑定集，便于链式调用。</returns>
    public TensorRtInferenceBindings BindAll()
    {
        ThrowIfDisposed();
        foreach (string tensorName in new List<string>(_buffers.Keys))
        {
            BindTensor(tensorName);
        }

        RefreshReport(runShapeInference: false);
        return this;
    }

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

    /// <summary>
    /// Copies a named output tensor to a single-precision managed array.
    /// 将指定输出 tensor 复制到单精度托管数组。
    /// </summary>
    /// <param name="tensorName">The output tensor name. 输出 tensor 名称。</param>
    /// <param name="elementCount">The number of float elements to read. 要读取的 float 元素数量。</param>
    /// <returns>The copied output values. 复制出的输出数据。</returns>
    public float[] ReadOutputSingles(string tensorName, int elementCount)
    {
        ThrowIfDisposed();
        if (!_buffers.TryGetValue(tensorName, out TensorRtInferenceBuffer? buffer))
        {
            throw new InvalidOperationException($"Tensor '{tensorName}' does not have an attached CUDA buffer.");
        }

        if (buffer.Tensor.IOMode != TensorRtIOMode.Output)
        {
            throw new ArgumentException("Only output tensors can be read as outputs.", nameof(tensorName));
        }

        return buffer.Memory.ToSingleArray(elementCount);
    }

    /// <summary>
    /// Creates a readable multi-line summary for deployment diagnostics.
    /// 创建用于部署诊断的多行可读摘要。
    /// </summary>
    /// <returns>A summary string. 摘要字符串。</returns>
    public string Describe()
    {
        ThrowIfDisposed();
        StringBuilder builder = new StringBuilder();
        builder.Append("TensorRtInferenceBindings profile=").Append(ProfileIndex).Append(" engine=").AppendLine(Report.EngineName);
        foreach (TensorRtInferenceBuffer buffer in _buffers.Values)
        {
            builder.Append("  ").AppendLine(buffer.ToString());
        }

        return builder.ToString();
    }

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

    private TensorRtInferenceBuffer EnsureBuffer(string tensorName, TensorRtDims? runtimeShape, int minimumBytes)
    {
        TensorRtEngineTensorBinding tensor = GetTensor(tensorName);
        if (_buffers.TryGetValue(tensor.Name, out TensorRtInferenceBuffer? existing))
        {
            if (existing.Memory.SizeInBytes < minimumBytes)
            {
                throw new InvalidOperationException("Existing CUDA buffer is smaller than the requested host payload.");
            }

            return existing;
        }

        TensorRtDims shape = runtimeShape ?? ResolveRuntimeShape(tensor);
        int estimatedBytes = EstimateTensorByteSize(tensor, shape);
        return AllocateDeviceBuffer(tensor.Name, shape, Math.Max(estimatedBytes, minimumBytes));
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

    private TensorRtDims ResolveRuntimeShape(TensorRtEngineTensorBinding tensor)
    {
        try
        {
            TensorRtDims shape = _context.GetTensorShape(tensor.Name);
            if (CanEstimate(shape))
            {
                return shape;
            }
        }
        catch (Exception)
        {
        }

        if (tensor.EngineShape != null && CanEstimate(tensor.EngineShape))
        {
            return tensor.EngineShape;
        }

        if (tensor.ProfileOptShape != null && CanEstimate(tensor.ProfileOptShape))
        {
            return tensor.ProfileOptShape;
        }

        throw new InvalidOperationException($"Tensor '{tensor.Name}' does not have a concrete runtime shape. Set input shapes or provide a runtime shape explicitly.");
    }

    private static bool CanEstimate(TensorRtDims shape)
    {
        if (shape == null || shape.Values.Length == 0)
        {
            return false;
        }

        foreach (int value in shape.Values)
        {
            if (value <= 0)
            {
                return false;
            }
        }

        return true;
    }

    private static int EstimateTensorByteSize(TensorRtEngineTensorBinding tensor, TensorRtDims shape)
    {
        if (!CanEstimate(shape))
        {
            throw new InvalidOperationException($"Tensor '{tensor.Name}' shape is not concrete enough for byte-size estimation.");
        }

        long elementCount = 1;
        foreach (int value in shape.Values)
        {
            elementCount = checked(elementCount * value);
        }

        int bytesPerComponent = tensor.EffectiveBytesPerComponent;
        int componentsPerElement = tensor.EffectiveComponentsPerElement;
        if (bytesPerComponent <= 0)
        {
            throw new NotSupportedException(
                $"Tensor '{tensor.Name}' data type {tensor.DataType} does not have an integral byte-size fallback.");
        }

        long bytes = checked(elementCount * bytesPerComponent * componentsPerElement);
        if (bytes <= 0 || bytes > int.MaxValue)
        {
            throw new InvalidOperationException($"Tensor '{tensor.Name}' estimated byte size is outside the managed allocation range.");
        }

        return (int)bytes;
    }

    private void RefreshReport(bool runShapeInference)
    {
        Report = _engine.GetBindingReport(_context, ProfileIndex, runShapeInference);
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

    private void RemoveOwnedBuffer(string tensorName)
    {
        if (_buffers.TryGetValue(tensorName, out TensorRtInferenceBuffer? existing) && existing.OwnsMemory)
        {
            existing.Memory.Dispose();
        }

        _buffers.Remove(tensorName);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(TensorRtInferenceBindings));
        }
    }
}
