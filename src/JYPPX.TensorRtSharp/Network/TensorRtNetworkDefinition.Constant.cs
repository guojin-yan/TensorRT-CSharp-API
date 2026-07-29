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
    /// Adds a Constant layer or object.
    /// 添加 Constant 层或对象。
    /// </summary>
    public TensorRtLayer AddConstant(TensorRtDims shape, TensorRtWeights weights)
    {
        if (shape == null)
        {
            throw new ArgumentNullException(nameof(shape));
        }

        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddConstantLayer(Line, _handle, shape, weights));
    }

}
