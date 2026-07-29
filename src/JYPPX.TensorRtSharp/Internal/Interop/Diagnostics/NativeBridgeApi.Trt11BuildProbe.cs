namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool TryRunTrt11MinimalBuildChain(out string message)
    {
        return TryRunTrtMinimalBuildChain(Trt11Bindings, out message);
    }

    public static bool TryBuildTrt11SerializedNetworkOnly(out string message)
    {
        return TryBuildSerializedNetworkOnly(Trt11Bindings, out message);
    }
}
