using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static int GetErrorCodeExclusiveUpperBound(TensorRtApiLine line)
    {
        int upperBound;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_error_code_get_exclusive_upper_bound(out upperBound),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_error_code_get_exclusive_upper_bound(out upperBound),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_error_code_get_exclusive_upper_bound(out upperBound),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        if (upperBound <= 0 || upperBound > 1_000_000)
        {
            throw new BridgeProbeException(
                BridgeStatusCode.InvalidState,
                BridgeErrorCategory.TensorRt,
                "TensorRT returned an invalid ErrorCode exclusive upper bound.");
        }
        return upperBound;
    }
}
