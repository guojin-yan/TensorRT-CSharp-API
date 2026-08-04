using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static void SetFillAlphaInt64(TensorRtApiLine line, SafeTensorRtObjectHandle layer, long value)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetFillAlphaInt64));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_fill_layer_set_alpha_int64(layer, value);
        NativeStatus.ThrowIfFailed(status);
    }

    public static long GetFillAlphaInt64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetFillAlphaInt64));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_fill_layer_get_alpha_int64(layer, out long value);
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static void SetFillBetaInt64(TensorRtApiLine line, SafeTensorRtObjectHandle layer, long value)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetFillBetaInt64));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_fill_layer_set_beta_int64(layer, value);
        NativeStatus.ThrowIfFailed(status);
    }

    public static long GetFillBetaInt64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetFillBetaInt64));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_fill_layer_get_beta_int64(layer, out long value);
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static bool IsFillAlphaBetaInt64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(IsFillAlphaBetaInt64));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_fill_layer_is_alpha_beta_int64(layer, out int value);
        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }
}
