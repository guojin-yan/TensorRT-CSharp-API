using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Network Definition wrapper.
/// 表示托管 TensorRT Tensor Rt Network Definition 包装器。
/// </summary>
public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>
    /// Adds a Fill layer or object.
    /// 添加 Fill 层或对象。
    /// </summary>
    public TensorRtLayer AddFill(TensorRtDims dimensions, TensorRtFillOperation operation)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddFillLayer(Line, _handle, dimensions, operation));
    }

}
