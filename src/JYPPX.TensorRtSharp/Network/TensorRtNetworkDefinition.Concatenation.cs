using System;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Network Definition wrapper.
/// 表示托管 TensorRT Tensor Rt Network Definition 包装器。
/// </summary>
public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>
    /// Adds a Concatenation layer or object.
    /// 添加 Concatenation 层或对象。
    /// </summary>
    public TensorRtLayer AddConcatenation(params TensorRtTensor[] inputs)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (inputs.Length < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(inputs), "Concatenation requires at least two input tensors.");
        }

        SafeTensorRtObjectHandle[] inputHandles = new SafeTensorRtObjectHandle[inputs.Length];
        for (int index = 0; index < inputs.Length; index++)
        {
            if (inputs[index] == null)
            {
                throw new ArgumentNullException(nameof(inputs), "Input tensors must not contain null entries.");
            }

            if (inputs[index].Line != Line)
            {
                throw new ArgumentException("Input tensors must belong to the same TensorRT API line as the network.");
            }

            inputHandles[index] = inputs[index].Handle;
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddConcatenationLayer(Line, _handle, inputHandles));
    }

}
