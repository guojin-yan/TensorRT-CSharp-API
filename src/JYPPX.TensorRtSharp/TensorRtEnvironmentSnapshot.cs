using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// High-level environment snapshot for the native TensorRT bridge.
/// </summary>
public sealed class TensorRtEnvironmentSnapshot
{
    public TensorRtEnvironmentSnapshot(
        BridgeBuildInfo buildInfo,
        BridgeRuntimeInfo runtimeInfo,
        BridgeCapabilityInfo capabilityInfo,
        TensorRtAdapterInfo tensorRt8,
        TensorRtAdapterInfo tensorRt10,
        TensorRtAdapterInfo tensorRt11)
    {
        BuildInfo = buildInfo;
        RuntimeInfo = runtimeInfo;
        CapabilityInfo = capabilityInfo;
        TensorRt8 = tensorRt8;
        TensorRt10 = tensorRt10;
        TensorRt11 = tensorRt11;
    }

    public BridgeBuildInfo BuildInfo { get; }
    public BridgeRuntimeInfo RuntimeInfo { get; }
    public BridgeCapabilityInfo CapabilityInfo { get; }
    public TensorRtAdapterInfo TensorRt8 { get; }
    public TensorRtAdapterInfo TensorRt10 { get; }
    public TensorRtAdapterInfo TensorRt11 { get; }
}
