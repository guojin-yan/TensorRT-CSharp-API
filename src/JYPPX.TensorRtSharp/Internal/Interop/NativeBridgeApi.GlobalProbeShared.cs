using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    private static BridgeProbeException UnsupportedGlobalRuntimeProbeLine()
    {
        return new BridgeProbeException(
            BridgeStatusCode.NotSupported,
            BridgeErrorCategory.TensorRt,
            "Global TensorRT runtime and plugin registry probes are exposed by this bridge for TensorRT 8, 10, and 11.");
    }

}
