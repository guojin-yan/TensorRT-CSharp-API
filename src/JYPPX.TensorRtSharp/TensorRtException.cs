using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Base exception for TensorRT bridge operations.
/// </summary>
public class TensorRtException : NativeBridgeException
{
    public TensorRtException(BridgeStatusCode statusCode, BridgeErrorCategory errorCategory, string message)
        : base(statusCode, errorCategory, message)
    {
    }
}

