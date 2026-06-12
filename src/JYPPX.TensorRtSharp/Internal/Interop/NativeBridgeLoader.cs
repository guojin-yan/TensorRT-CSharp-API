using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static class NativeBridgeLoader
{
    public static void EnsureInitialized()
    {
        NativeBridgeLibraryLoader.EnsureInitialized(typeof(NativeMethodsCommon).Assembly);
    }
}
