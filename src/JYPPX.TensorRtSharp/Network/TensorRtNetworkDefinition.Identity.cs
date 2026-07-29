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
    /// Adds a Identity layer or object.
    /// 添加 Identity 层或对象。
    /// </summary>
    public TensorRtLayer AddIdentity(TensorRtTensor input)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddIdentityLayer(Line, _handle, input.Handle));
    }

}
