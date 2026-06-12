using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>
    /// Marks a tensor as a TensorRT debug tensor.
    /// 将指定 tensor 标记为 TensorRT debug tensor。
    /// </summary>
    /// <param name="tensor">The network tensor to mark. / 要标记的 network tensor。</param>
    /// <returns><see langword="true"/> when TensorRT accepts or already has the mark. / TensorRT 接受标记或该 tensor 已被标记时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// Debug tensors can be emitted by runtime debug listeners. Tensor names should be stable before enabling this.
    /// debug tensor 可由运行时 debug listener 输出；启用前应保证 tensor 名称稳定。
    /// </remarks>
    public bool MarkDebugTensor(TensorRtTensor tensor)
    {
        ValidateInputTensor(tensor, nameof(tensor));
        return NativeBridgeApi.MarkNetworkDebugTensor(Line, _handle, tensor.Handle);
    }

    /// <summary>
    /// Removes the TensorRT debug mark from a tensor.
    /// 移除指定 tensor 的 TensorRT debug 标记。
    /// </summary>
    /// <param name="tensor">The network tensor to unmark. / 要取消标记的 network tensor。</param>
    /// <returns><see langword="true"/> when TensorRT accepts or already has no mark. / TensorRT 接受取消或该 tensor 本就未标记时返回 <see langword="true"/>。</returns>
    public bool UnmarkDebugTensor(TensorRtTensor tensor)
    {
        ValidateInputTensor(tensor, nameof(tensor));
        return NativeBridgeApi.UnmarkNetworkDebugTensor(Line, _handle, tensor.Handle);
    }

    /// <summary>
    /// Gets whether a tensor is currently marked as a TensorRT debug tensor.
    /// 获取指定 tensor 当前是否已被标记为 TensorRT debug tensor。
    /// </summary>
    /// <param name="tensor">The network tensor to query. / 要查询的 network tensor。</param>
    public bool IsDebugTensor(TensorRtTensor tensor)
    {
        ValidateInputTensor(tensor, nameof(tensor));
        return NativeBridgeApi.IsNetworkDebugTensor(Line, _handle, tensor.Handle);
    }

    /// <summary>
    /// Asks TensorRT 11 to mark unfused tensors as debug tensors.
    /// 请求 TensorRT 11 将未融合 tensor 标记为 debug tensor。
    /// </summary>
    /// <returns><see langword="true"/> when TensorRT accepts the request. / TensorRT 接受请求时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// This preserves optimizer fusion behavior better than marking individual tensors, but TensorRT may use internal tensor names.
    /// 该方式比逐个标记 tensor 更能保留优化器融合行为，但 TensorRT 可能使用内部 tensor 名称。
    /// </remarks>
    public bool MarkUnfusedTensorsAsDebugTensors()
    {
        return NativeBridgeApi.MarkNetworkUnfusedTensorsAsDebugTensors(Line, _handle);
    }

    /// <summary>
    /// Removes the TensorRT 11 unfused-debug-tensor network mark.
    /// 移除 TensorRT 11 未融合 tensor debug 标记。
    /// </summary>
    /// <returns><see langword="true"/> when TensorRT accepts the request. / TensorRT 接受请求时返回 <see langword="true"/>。</returns>
    public bool UnmarkUnfusedTensorsAsDebugTensors()
    {
        return NativeBridgeApi.UnmarkNetworkUnfusedTensorsAsDebugTensors(Line, _handle);
    }

    /// <summary>
    /// Marks an INT32 tensor as a TensorRT shape output.
    /// 将 INT32 tensor 标记为 TensorRT shape output。
    /// </summary>
    /// <param name="tensor">The tensor whose value should be exposed through shape output APIs. / 需要通过 shape output API 暴露值的 tensor。</param>
    /// <returns><see langword="true"/> when TensorRT accepts the shape-output mark. / TensorRT 接受 shape-output 标记时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// TensorRT requires shape outputs to have build-time constant dimensions and no more than one dimension.
    /// TensorRT 要求 shape output 的维度可在构建期确定，且维度数量不超过一维。
    /// </remarks>
    public bool MarkOutputForShapes(TensorRtTensor tensor)
    {
        ValidateInputTensor(tensor, nameof(tensor));
        return NativeBridgeApi.MarkNetworkOutputForShapes(Line, _handle, tensor.Handle);
    }

    /// <summary>
    /// Removes a TensorRT shape-output mark from a tensor.
    /// 从指定 tensor 移除 TensorRT shape-output 标记。
    /// </summary>
    /// <param name="tensor">The tensor to unmark. / 要取消标记的 tensor。</param>
    /// <returns><see langword="true"/> when TensorRT accepts the request. / TensorRT 接受请求时返回 <see langword="true"/>。</returns>
    public bool UnmarkOutputForShapes(TensorRtTensor tensor)
    {
        ValidateInputTensor(tensor, nameof(tensor));
        return NativeBridgeApi.UnmarkNetworkOutputForShapes(Line, _handle, tensor.Handle);
    }
}
