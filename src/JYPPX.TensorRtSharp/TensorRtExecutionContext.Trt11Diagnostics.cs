using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Gets whether an input-consumed CUDA event is currently set on this TensorRT 11 execution context.
    /// 获取此 TensorRT 11 execution context 当前是否设置了 input-consumed CUDA event。
    /// </summary>
    public bool IsInputConsumedEventSet => NativeBridgeApi.IsExecutionContextInputConsumedEventSet(Line, _handle);

    /// <summary>
    /// Gets whether a named output tensor has an explicit device address bound.
    /// 获取指定输出 tensor 是否已绑定显式设备地址。
    /// </summary>
    /// <param name="tensorName">The output tensor name. / 输出 tensor 名称。</param>
    /// <returns><c>true</c> if TensorRT reports an output address. / TensorRT 报告已有输出地址时返回 <c>true</c>。</returns>
    public bool IsOutputTensorAddressSet(string tensorName)
    {
        return NativeBridgeApi.IsExecutionContextOutputTensorAddressSet(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets whether a named output tensor uses a TensorRT output allocator.
    /// 获取指定输出 tensor 是否使用 TensorRT output allocator；适用于 TensorRT 8/10/11，不会暴露 allocator 指针或接管其生命周期。
    /// </summary>
    /// <param name="tensorName">The output tensor name. / 输出 tensor 名称。</param>
    /// <returns><c>true</c> if TensorRT reports an output allocator. / TensorRT 报告存在 output allocator 时返回 <c>true</c>。</returns>
    public bool HasOutputAllocator(string tensorName)
    {
        return NativeBridgeApi.HasExecutionContextOutputAllocator(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets whether this execution context has a temporary-storage allocator attached.
    /// 获取此 execution context 是否绑定了 temporary-storage allocator；适用于 TensorRT 8/10/11，不会暴露 allocator 指针或接管其生命周期。
    /// </summary>
    public bool HasTemporaryStorageAllocator => NativeBridgeApi.HasExecutionContextTemporaryStorageAllocator(Line, _handle);
}
