using System;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtExecutionContext : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtExecutionContext(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    public TensorRtApiLine Line { get; }

    internal SafeTensorRtObjectHandle Handle => _handle;

    public string Name
    {
        get => NativeBridgeApi.GetExecutionContextName(Line, _handle);
        set => NativeBridgeApi.SetExecutionContextName(Line, _handle, value);
    }

    public bool DebugSync
    {
        get => NativeBridgeApi.GetExecutionContextDebugSync(Line, _handle);
        set => NativeBridgeApi.SetExecutionContextDebugSync(Line, _handle, value);
    }

    public bool AllInputDimensionsSpecified => NativeBridgeApi.AllInputDimensionsSpecified(Line, _handle);

    public bool AllInputShapesSpecified => NativeBridgeApi.AllInputShapesSpecified(Line, _handle);

    public ulong DeviceMemorySizeInBytes => NativeBridgeApi.GetExecutionContextDeviceMemorySize(Line, _handle);

    public ulong PersistentCacheLimitInBytes
    {
        get => NativeBridgeApi.GetExecutionContextPersistentCacheLimit(Line, _handle);
        set => NativeBridgeApi.SetExecutionContextPersistentCacheLimit(Line, _handle, value);
    }

    public void SetInputShape(string tensorName, TensorRtDims dims)
    {
        NativeBridgeApi.SetInputShape(Line, _handle, tensorName, dims);
    }

    public void SetBindingDimensions(int bindingIndex, TensorRtDims dims)
    {
        NativeBridgeApi.SetBindingDimensions(Line, _handle, bindingIndex, dims);
    }

    public void SetTensorAddress(string tensorName, CudaMemory memory)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(nameof(memory));
        }

        NativeBridgeApi.SetTensorAddress(Line, _handle, tensorName, memory.Handle);
    }

    /// <summary>
    /// Binds a CUDA allocation to a named input tensor.
    /// 将 CUDA 设备内存分配绑定到指定输入 tensor。
    /// </summary>
    /// <param name="tensorName">The input tensor name. 输入 tensor 名称。</param>
    /// <param name="memory">The CUDA allocation used as the input buffer. 作为输入缓冲区的 CUDA 设备内存。</param>
    public void SetInputTensorAddress(string tensorName, CudaMemory memory)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(nameof(memory));
        }

        NativeBridgeApi.SetInputTensorAddress(Line, _handle, tensorName, memory.Handle);
    }

    /// <summary>
    /// Binds a CUDA allocation to a named output tensor.
    /// 将 CUDA 设备内存分配绑定到指定输出 tensor。
    /// </summary>
    /// <param name="tensorName">The output tensor name. 输出 tensor 名称。</param>
    /// <param name="memory">The CUDA allocation used as the output buffer. 作为输出缓冲区的 CUDA 设备内存。</param>
    public void SetOutputTensorAddress(string tensorName, CudaMemory memory)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(nameof(memory));
        }

        NativeBridgeApi.SetOutputTensorAddress(Line, _handle, tensorName, memory.Handle);
    }

    public void SetDeviceMemory(CudaMemory memory)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(nameof(memory));
        }

        NativeBridgeApi.SetExecutionContextDeviceMemory(Line, _handle, memory.Handle);
    }

    public ulong UpdateDeviceMemorySizeForShapes()
    {
        return NativeBridgeApi.UpdateExecutionContextDeviceMemorySizeForShapes(Line, _handle);
    }

    /// <summary>
    /// Runs TensorRT shape inference for the current execution context.
    /// 对当前 execution context 执行 TensorRT shape inference。
    /// </summary>
    /// <returns>
    /// The number of tensors whose shapes are still insufficiently specified.
    /// 仍未完全指定 shape 的 tensor 数量。
    /// </returns>
    public int InferShapes()
    {
        return NativeBridgeApi.InferExecutionContextShapes(Line, _handle);
    }

    public void SetInputConsumedEvent(CudaEvent cudaEvent)
    {
        if (cudaEvent == null)
        {
            throw new ArgumentNullException(nameof(cudaEvent));
        }

        NativeBridgeApi.SetExecutionContextInputConsumedEvent(Line, _handle, cudaEvent.Handle);
    }

    public TensorRtDims GetTensorShape(string tensorName)
    {
        return NativeBridgeApi.GetExecutionContextTensorShape(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets a TensorRT 11 execution-context tensor shape with 64-bit dimension extents.
    /// 获取 TensorRT 11 execution context 中张量的形状，并保留 64 位维度 extent。
    /// </summary>
    /// <param name="tensorName">The context tensor name. Context 张量名称。</param>
    /// <returns>The runtime tensor shape reported by TensorRT. TensorRT 报告的运行时张量形状。</returns>
    public TensorRtDims64 GetTensorShape64(string tensorName)
    {
        return NativeBridgeApi.GetExecutionContextTensorShape64(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets one TensorRT 11 runtime tensor shape extent as a 64-bit value.
    /// 以 64 位整数获取 TensorRT 11 运行时张量形状的单个 extent。
    /// </summary>
    /// <param name="tensorName">The context tensor name. Context 张量名称。</param>
    /// <param name="dimensionIndex">The zero-based dimension index. 从零开始的维度索引。</param>
    /// <returns>The runtime shape extent reported by TensorRT. TensorRT 报告的运行时 shape extent。</returns>
    public long GetTensorShapeDimensionExtent64(string tensorName, int dimensionIndex)
    {
        return NativeBridgeApi.GetExecutionContextTensorShapeDimensionExtent64(Line, _handle, tensorName, dimensionIndex);
    }

    public TensorRtDims GetTensorStrides(string tensorName)
    {
        return NativeBridgeApi.GetExecutionContextTensorStrides(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets TensorRT 11 runtime tensor strides with 64-bit values.
    /// 获取 TensorRT 11 运行时张量 strides，并保留 64 位值。
    /// </summary>
    /// <param name="tensorName">The context tensor name. Context 张量名称。</param>
    /// <returns>The runtime tensor strides reported by TensorRT. TensorRT 报告的运行时 tensor strides。</returns>
    public TensorRtDims64 GetTensorStrides64(string tensorName)
    {
        return NativeBridgeApi.GetExecutionContextTensorStrides64(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets one TensorRT 11 runtime stride extent as a 64-bit value.
    /// 以 64 位整数获取 TensorRT 11 运行时张量 stride 的单个 extent。
    /// </summary>
    /// <param name="tensorName">The context tensor name. Context 张量名称。</param>
    /// <param name="dimensionIndex">The zero-based dimension index. 从零开始的维度索引。</param>
    /// <returns>The runtime stride extent reported by TensorRT. TensorRT 报告的运行时 stride extent。</returns>
    public long GetTensorStrideDimensionExtent64(string tensorName, int dimensionIndex)
    {
        return NativeBridgeApi.GetExecutionContextTensorStrideDimensionExtent64(Line, _handle, tensorName, dimensionIndex);
    }

    public bool IsTensorAddressBound(string tensorName)
    {
        return NativeBridgeApi.IsExecutionContextTensorAddressBound(Line, _handle, tensorName);
    }

    public void EnqueueAsync(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeBridgeApi.EnqueueAsync(Line, _handle, stream.Handle);
    }

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
