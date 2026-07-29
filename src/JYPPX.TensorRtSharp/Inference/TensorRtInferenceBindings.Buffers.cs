using System;
using JYPPX.CudaSharp;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtInferenceBindings
{
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

    private void RemoveOwnedBuffer(string tensorName)
    {
        if (_buffers.TryGetValue(tensorName, out TensorRtInferenceBuffer? existing) && existing.OwnsMemory)
        {
            existing.Memory.Dispose();
        }

        _buffers.Remove(tensorName);
    }
}
