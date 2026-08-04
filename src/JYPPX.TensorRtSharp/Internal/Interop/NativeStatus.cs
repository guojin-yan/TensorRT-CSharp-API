using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static class NativeStatus
{
    public static void ThrowIfFailed(BridgeStatusCode statusCode)
    {
        if (statusCode == BridgeStatusCode.Ok)
        {
            return;
        }

        string message = NativeBridgeApi.GetLastErrorMessageOrFallback($"Bridge call failed with status '{statusCode}'.");
        BridgeErrorCategory category = NativeBridgeApi.GetLastErrorCategory();
        throw new BridgeProbeException(statusCode, category, message);
    }
}

