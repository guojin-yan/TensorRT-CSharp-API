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
    /// Adds a Select layer or object.
    /// 添加 Select 层或对象。
    /// </summary>
    public TensorRtLayer AddSelect(TensorRtTensor condition, TensorRtTensor thenInput, TensorRtTensor elseInput)
    {
        if (condition == null)
        {
            throw new ArgumentNullException(nameof(condition));
        }

        if (thenInput == null)
        {
            throw new ArgumentNullException(nameof(thenInput));
        }

        if (elseInput == null)
        {
            throw new ArgumentNullException(nameof(elseInput));
        }

        if (condition.Line != Line || thenInput.Line != Line || elseInput.Line != Line)
        {
            throw new ArgumentException("Input tensors must belong to the same TensorRT API line as the network.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddSelectLayer(Line, _handle, condition.Handle, thenInput.Handle, elseInput.Handle));
    }

}
