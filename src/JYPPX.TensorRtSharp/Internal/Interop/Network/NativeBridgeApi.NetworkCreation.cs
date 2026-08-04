using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateNetwork(TensorRtApiLine line, SafeTensorRtObjectHandle builder, uint creationFlags)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.NetworkCreate(builder, creationFlags, out SafeTensorRtObjectHandle network);
        NativeStatus.ThrowIfFailed(status);
        return network;
    }

}
