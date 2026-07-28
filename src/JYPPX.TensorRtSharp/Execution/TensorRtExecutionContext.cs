using System;
using System.Collections.Generic;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Execution Context wrapper.
/// 表示托管 TensorRT Tensor Rt Execution Context 包装器。
/// </summary>
public sealed partial class TensorRtExecutionContext : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;
    private TensorRtProfiler? _profilerKeepAlive;
    private readonly object _auxiliaryStreamLeaseLock = new object();
    private TensorRtAuxiliaryStreamHandleLease? _auxiliaryStreamLease;
    private int _auxiliaryStreamAssignedCount;
    private bool _auxiliaryStreamsCleared = true;
    private bool _auxiliaryStreamContextDisposed;
    private string _auxiliaryStreamDiagnostic = "No caller-provided auxiliary CUDA streams are assigned.";

    internal TensorRtExecutionContext(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    /// <summary>
    /// Gets or sets the Line value.
    /// 获取或设置 Line 值。
    /// </summary>
    public TensorRtApiLine Line { get; }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets or sets the Name value.
    /// 获取或设置 Name 值。
    /// </summary>
    public string Name
    {
        get => NativeBridgeApi.GetExecutionContextName(Line, _handle);
        set => NativeBridgeApi.SetExecutionContextName(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets the Debug Sync value.
    /// 获取或设置 Debug Sync 值。
    /// </summary>
    public bool DebugSync
    {
        get => NativeBridgeApi.GetExecutionContextDebugSync(Line, _handle);
        set => NativeBridgeApi.SetExecutionContextDebugSync(Line, _handle, value);
    }

    /// <summary>
    /// Gets the All Input Dimensions Specified value.
    /// 获取 All Input Dimensions Specified 值。
    /// </summary>
    public bool AllInputDimensionsSpecified => NativeBridgeApi.AllInputDimensionsSpecified(Line, _handle);

    /// <summary>
    /// Gets or sets the All Input Shapes Specified value.
    /// 获取或设置 All Input Shapes Specified 值。
    /// </summary>
    public bool AllInputShapesSpecified => NativeBridgeApi.AllInputShapesSpecified(Line, _handle);

    /// <summary>
    /// Gets or sets the Device Memory Size In Bytes value.
    /// 获取或设置 Device Memory Size In Bytes 值。
    /// </summary>
    public ulong DeviceMemorySizeInBytes => NativeBridgeApi.GetExecutionContextDeviceMemorySize(Line, _handle);

    /// <summary>
    /// Gets or sets the Persistent Cache Limit In Bytes value.
    /// 获取或设置 Persistent Cache Limit In Bytes 值。
    /// </summary>
    public ulong PersistentCacheLimitInBytes
    {
        get => NativeBridgeApi.GetExecutionContextPersistentCacheLimit(Line, _handle);
        set => NativeBridgeApi.SetExecutionContextPersistentCacheLimit(Line, _handle, value);
    }

    /// <summary>
    /// Sets the Input Shape value.
    /// 设置 Input Shape 值。
    /// </summary>
    public void SetInputShape(string tensorName, TensorRtDims dims)
    {
        NativeBridgeApi.SetInputShape(Line, _handle, tensorName, dims);
    }

    /// <summary>
    /// Sets the Binding Dimensions value.
    /// 设置 Binding Dimensions 值。
    /// </summary>
    public void SetBindingDimensions(int bindingIndex, TensorRtDims dims)
    {
        NativeBridgeApi.SetBindingDimensions(Line, _handle, bindingIndex, dims);
    }

    /// <summary>
    /// Sets the Tensor Address value.
    /// 设置 Tensor Address 值。
    /// </summary>
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

    /// <summary>
    /// Sets the Device Memory value.
    /// 设置 Device Memory 值。
    /// </summary>
    public void SetDeviceMemory(CudaMemory memory)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(nameof(memory));
        }

        NativeBridgeApi.SetExecutionContextDeviceMemory(Line, _handle, memory.Handle);
    }

    /// <summary>
    /// Updates the Device Memory Size For Shapes value.
    /// 更新 Device Memory Size For Shapes 值。
    /// </summary>
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

    /// <summary>
    /// Sets the Input Consumed Event value.
    /// 设置 Input Consumed Event 值。
    /// </summary>
    public void SetInputConsumedEvent(CudaEvent cudaEvent)
    {
        if (cudaEvent == null)
        {
            throw new ArgumentNullException(nameof(cudaEvent));
        }

        NativeBridgeApi.SetExecutionContextInputConsumedEvent(Line, _handle, cudaEvent.Handle);
    }

    /// <summary>
    /// Gets the Tensor Shape value.
    /// 获取 Tensor Shape 值。
    /// </summary>
    public TensorRtDims GetTensorShape(string tensorName)
    {
        return NativeBridgeApi.GetExecutionContextTensorShape(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets TensorRT 8 legacy shape-binding values for a binding index.
    /// 获取 TensorRT 8 legacy shape binding 的运行时取值。
    /// </summary>
    /// <param name="bindingIndex">The legacy binding index. legacy binding 索引。</param>
    /// <returns>Caller-owned copied shape-binding values. 调用方拥有的 shape-binding 值副本。</returns>
    public int[] GetShapeBinding(int bindingIndex)
    {
        return NativeBridgeApi.GetExecutionContextShapeBinding(Line, _handle, bindingIndex);
    }

    /// <summary>
    /// Sets copied values for a TensorRT 8 legacy input shape binding.
    /// 为 TensorRT 8 legacy input shape binding 设置复制后的值。
    /// </summary>
    /// <param name="bindingIndex">The legacy input shape-binding index. legacy input shape-binding 索引。</param>
    /// <param name="values">Caller-owned shape values copied for the native call. 调用方拥有且会被复制用于原生调用的 shape 值。</param>
    /// <returns><see langword="true"/> when TensorRT accepts the values. TensorRT 接受这些值时返回 <see langword="true"/>。</returns>
    public bool SetInputShapeBinding(int bindingIndex, IReadOnlyList<int> values)
    {
        return NativeBridgeApi.SetExecutionContextInputShapeBinding(Line, _handle, bindingIndex, values);
    }

    /// <summary>Sets copied TensorRT 8 legacy input shape-binding values. 设置复制后的 TensorRT 8 legacy input shape-binding 值。</summary>
    /// <param name="bindingIndex">The legacy input shape-binding index. legacy input shape-binding 索引。</param>
    /// <param name="values">Caller-owned shape values. 调用方拥有的 shape 值。</param>
    /// <returns><see langword="true"/> when TensorRT accepts the values. TensorRT 接受这些值时返回 <see langword="true"/>。</returns>
    public bool SetInputShapeBinding(int bindingIndex, params int[] values)
    {
        if (values == null) { throw new ArgumentNullException(nameof(values)); }
        return SetInputShapeBinding(bindingIndex, (IReadOnlyList<int>)values);
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

    /// <summary>
    /// Gets the Tensor Strides value.
    /// 获取 Tensor Strides 值。
    /// </summary>
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

    /// <summary>
    /// Checks whether Tensor Address Bound is true.
    /// 检查 Tensor Address Bound 是否为 true。
    /// </summary>
    public bool IsTensorAddressBound(string tensorName)
    {
        return NativeBridgeApi.IsExecutionContextTensorAddressBound(Line, _handle, tensorName);
    }

    /// <summary>
    /// Enqueues the TensorRT execution work.
    /// 将 TensorRT execution work 加入队列。
    /// </summary>
    public void EnqueueAsync(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeBridgeApi.EnqueueAsync(Line, _handle, stream.Handle);
    }

    /// <summary>
    /// Releases the native TensorRT resources held by this object.
    /// 释放此对象持有的 native TensorRT 资源。
    /// </summary>
    public void Dispose()
    {
        TensorRtAuxiliaryStreamHandleLease? auxiliaryStreamLease;
        lock (_auxiliaryStreamLeaseLock)
        {
            if (_auxiliaryStreamContextDisposed)
            {
                return;
            }

            _auxiliaryStreamContextDisposed = true;
            auxiliaryStreamLease = _auxiliaryStreamLease;
            TryClearAuxStreamsForDispose();
            _auxiliaryStreamLease = null;
            _auxiliaryStreamAssignedCount = 0;
            _auxiliaryStreamsCleared = true;
            _auxiliaryStreamDiagnostic = "Execution context disposed; auxiliary stream leases were released after native context teardown.";
        }

        TensorRtProfiler? profiler = _profilerKeepAlive;
        try
        {
            if (profiler != null)
            {
                TryClearProfilerForDispose();
            }
        }
        finally
        {
            try
            {
                _handle.Dispose();
                GC.KeepAlive(profiler);
            }
            finally
            {
                auxiliaryStreamLease?.Dispose();
                DetachProfiler();
                GC.SuppressFinalize(this);
            }
        }
    }

    private void TryClearAuxStreamsForDispose()
    {
        try
        {
            NativeBridgeApi.ClearExecutionContextAuxStreams(Line, _handle);
        }
        catch (Exception exception) when (
            exception is BridgeProbeException ||
            exception is TensorRtException ||
            exception is EntryPointNotFoundException ||
            exception is DllNotFoundException ||
            exception is BadImageFormatException ||
            exception is ObjectDisposedException)
        {
            // Keep any stream lease until after context teardown even when native clear is unavailable.
        }
    }

    private void TryClearProfilerForDispose()
    {
        try
        {
            NativeBridgeApi.ClearExecutionContextProfiler(Line, _handle);
        }
        catch (BridgeProbeException)
        {
            // Dispose must still release the context handle. Keep the profiler alive until after
            // the context handle is released so TensorRT never observes a freed borrowed profiler.
        }
    }

    private TensorRtProfiler? DetachProfiler()
    {
        TensorRtProfiler? profiler = _profilerKeepAlive;
        if (profiler != null)
        {
            _profilerKeepAlive = null;
            profiler.DetachBorrower();
        }

        return profiler;
    }
}
