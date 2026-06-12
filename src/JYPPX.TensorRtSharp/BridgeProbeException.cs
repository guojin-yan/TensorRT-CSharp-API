using System;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Thrown when the bridge reports a non-success status for a probe operation.
/// </summary>
public sealed class BridgeProbeException : TensorRtException
{
    public BridgeProbeException(BridgeStatusCode statusCode, BridgeErrorCategory errorCategory, string message)
        : base(statusCode, errorCategory, message)
    {
    }
}
