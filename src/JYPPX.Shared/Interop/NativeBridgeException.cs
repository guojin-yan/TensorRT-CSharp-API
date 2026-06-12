using System;

namespace JYPPX.Shared.Interop;

/// <summary>
/// Base exception for managed wrappers around the native bridge.
/// </summary>
public class NativeBridgeException : InvalidOperationException
{
    public NativeBridgeException(BridgeStatusCode statusCode, BridgeErrorCategory errorCategory, string message)
        : base(message)
    {
        StatusCode = statusCode;
        ErrorCategory = errorCategory;
    }

    public BridgeStatusCode StatusCode { get; }
    public BridgeErrorCategory ErrorCategory { get; }
}

