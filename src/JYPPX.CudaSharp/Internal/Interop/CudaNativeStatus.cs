using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static class CudaNativeStatus
{
    public static void ThrowIfFailed(BridgeStatusCode statusCode)
    {
        if (statusCode == BridgeStatusCode.Ok)
        {
            return;
        }

        string message = NativeBridgeApi.GetLastErrorMessageOrFallback($"CUDA bridge call failed with status '{statusCode}'.");
        BridgeErrorCategory category = NativeBridgeApi.GetLastErrorCategory();
        throw new CudaException(statusCode, category, message);
    }
}

