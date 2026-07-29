using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddScaleLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        TensorRtScaleMode mode,
        TensorRtWeights? shift,
        TensorRtWeights? scale,
        TensorRtWeights? power,
        int channelAxis)
    {
        TensorRtDataType dataType = GetScaleWeightsDataType(shift, scale, power);
        ValidateOptionalWeightsDataType(dataType, shift, nameof(shift));
        ValidateOptionalWeightsDataType(dataType, scale, nameof(scale));
        ValidateOptionalWeightsDataType(dataType, power, nameof(power));

        TensorRtWeights.PinnedScope? shiftPinned = PinOptionalWeights(shift);
        TensorRtWeights.PinnedScope? scalePinned = PinOptionalWeights(scale);
        TensorRtWeights.PinnedScope? powerPinned = PinOptionalWeights(power);
        try
        {
            SafeTensorRtObjectHandle layer;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_scale_nd(network, input, (int)mode, (int)dataType, shiftPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(shift?.ElementCount ?? 0), scalePinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(scale?.ElementCount ?? 0), powerPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(power?.ElementCount ?? 0), channelAxis, out layer),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_scale_nd(network, input, (int)mode, (int)dataType, shiftPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(shift?.ElementCount ?? 0), scalePinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(scale?.ElementCount ?? 0), powerPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(power?.ElementCount ?? 0), channelAxis, out layer),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_scale_nd(network, input, (int)mode, (int)dataType, shiftPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(shift?.ElementCount ?? 0), scalePinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(scale?.ElementCount ?? 0), powerPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(power?.ElementCount ?? 0), channelAxis, out layer),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
            return layer;
        }
        finally
        {
            powerPinned?.Dispose();
            scalePinned?.Dispose();
            shiftPinned?.Dispose();
        }
    }

    private static TensorRtDataType GetScaleWeightsDataType(TensorRtWeights? shift, TensorRtWeights? scale, TensorRtWeights? power)
    {
        if (shift != null && !shift.IsEmpty)
        {
            return shift.DataType;
        }

        if (scale != null && !scale.IsEmpty)
        {
            return scale.DataType;
        }

        if (power != null && !power.IsEmpty)
        {
            return power.DataType;
        }

        return TensorRtDataType.Float;
    }

}
