namespace JYPPX.Shared.Interop;

/// <summary>
/// Runtime-facing bridge state reported by the native layer.
/// </summary>
public sealed class BridgeRuntimeInfo
{
    public BridgeRuntimeInfo(
        int abiVersion,
        string bridgeName,
        string bridgeBanner,
        string lastErrorMessage,
        BridgeErrorCategory lastErrorCategory,
        bool cudaToolkitAvailable,
        bool tensorRtAvailable)
    {
        AbiVersion = abiVersion;
        BridgeName = bridgeName;
        BridgeBanner = bridgeBanner;
        LastErrorMessage = lastErrorMessage;
        LastErrorCategory = lastErrorCategory;
        CudaToolkitAvailable = cudaToolkitAvailable;
        TensorRtAvailable = tensorRtAvailable;
    }

    public int AbiVersion { get; }
    public string BridgeName { get; }
    public string BridgeBanner { get; }
    public string LastErrorMessage { get; }
    public BridgeErrorCategory LastErrorCategory { get; }
    public bool CudaToolkitAvailable { get; }
    public bool TensorRtAvailable { get; }
}

