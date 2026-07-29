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
    /// Adds a Slice layer or object.
    /// 添加 Slice 层或对象。
    /// </summary>
    public TensorRtLayer AddSlice(TensorRtTensor input, TensorRtDims start, TensorRtDims size, TensorRtDims stride)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (start == null)
        {
            throw new ArgumentNullException(nameof(start));
        }

        if (size == null)
        {
            throw new ArgumentNullException(nameof(size));
        }

        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddSliceLayer(Line, _handle, input.Handle, start, size, stride));
    }

}
