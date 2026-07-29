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
    /// Adds a Padding layer or object.
    /// 添加 Padding 层或对象。
    /// </summary>
    public TensorRtLayer AddPadding(TensorRtTensor input, TensorRtDims prePadding, TensorRtDims postPadding)
    {
        ValidateInputTensor(input, nameof(input));
        if (prePadding == null)
        {
            throw new ArgumentNullException(nameof(prePadding));
        }

        if (postPadding == null)
        {
            throw new ArgumentNullException(nameof(postPadding));
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddPaddingLayer(Line, _handle, input.Handle, prePadding, postPadding));
    }

}
