using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Thrown when the CUDA bridge reports a non-success status.
/// </summary>
public sealed class CudaException : NativeBridgeException
{
    public CudaException(BridgeStatusCode statusCode, BridgeErrorCategory errorCategory, string message)
        : base(statusCode, errorCategory, message)
    {
    }
}

