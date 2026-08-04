using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddConcatenationLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle[] inputs)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (inputs.Length < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(inputs), "Concatenation requires at least two input tensors.");
        }

        IntPtr[] inputHandles = new IntPtr[inputs.Length];
        for (int index = 0; index < inputs.Length; index++)
        {
            if (inputs[index] == null || inputs[index].IsInvalid)
            {
                throw new ArgumentException("Input tensor handles must not be null or invalid.", nameof(inputs));
            }

            inputHandles[index] = inputs[index].DangerousGetHandle();
        }

        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_concatenation(network, inputHandles, inputHandles.Length, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_concatenation(network, inputHandles, inputHandles.Length, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_concatenation(network, inputHandles, inputHandles.Length, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetConcatenationAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int axis)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_concatenation_layer_set_axis(layer, axis),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_concatenation_layer_set_axis(layer, axis),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_concatenation_layer_set_axis(layer, axis),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetConcatenationAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int axis;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_concatenation_layer_get_axis(layer, out axis),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_concatenation_layer_get_axis(layer, out axis),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_concatenation_layer_get_axis(layer, out axis),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return axis;
    }

}
