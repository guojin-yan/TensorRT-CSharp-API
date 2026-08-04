using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static class NativeBridgeLoader
{
    public static void EnsureInitialized()
    {
        NativeBridgeLibraryLoader.EnsureInitialized(typeof(NativeMethodsCommon).Assembly);
    }
}
