using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Layer wrapper.
/// 表示托管 TensorRT Tensor Rt Layer 包装器。
/// </summary>
public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets the Element Wise Operation value.
    /// 获取 Element Wise Operation 值。
    /// </summary>
    public TensorRtElementWiseOperation GetElementWiseOperation()
    {
        return NativeBridgeApi.GetElementWiseOperation(Line, _handle);
    }

    /// <summary>
    /// Sets the Element Wise Operation value.
    /// 设置 Element Wise Operation 值。
    /// </summary>
    public void SetElementWiseOperation(TensorRtElementWiseOperation operation)
    {
        NativeBridgeApi.SetElementWiseOperation(Line, _handle, operation);
    }

}
