using Microsoft.Win32.SafeHandles;

namespace JYPPX.Shared.Interop;

/// <summary>
/// Common base type for native bridge SafeHandle implementations.
/// </summary>
public abstract class SafeBridgeHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    protected SafeBridgeHandle()
        : base(true)
    {
    }
}

