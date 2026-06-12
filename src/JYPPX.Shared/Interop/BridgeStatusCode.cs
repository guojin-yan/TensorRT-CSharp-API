namespace JYPPX.Shared.Interop;

/// <summary>
/// Status codes returned by the native bridge.
/// </summary>
public enum BridgeStatusCode
{
    Ok = 0,
    InvalidArgument = 1,
    BufferTooSmall = 2,
    NotFound = 3,
    NotSupported = 4,
    DependencyMissing = 5,
    NotReady = 6,
    RuntimeError = 7,
    InvalidState = 8,
    OutOfMemory = 9,
    NotImplemented = 10
}

