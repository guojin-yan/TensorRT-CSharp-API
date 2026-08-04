using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle BuildSerializedNetwork(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle config)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.SerializedBuild(builder, network, config, out SafeTensorRtObjectHandle hostMemory);
        NativeStatus.ThrowIfFailed(status);
        return hostMemory;
    }

}
