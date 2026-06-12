namespace JYPPX.Shared.Interop;

/// <summary>
/// Build-time metadata reported by the native bridge.
/// </summary>
public sealed class BridgeBuildInfo
{
    public BridgeBuildInfo(
        int abiVersion,
        int bridgeVersionMajor,
        int bridgeVersionMinor,
        int bridgeVersionPatch,
        string bridgeName,
        string bridgeBanner,
        string compilerId,
        string compilerVersion,
        string systemName,
        string systemProcessor,
        string buildConfiguration,
        string cudaToolkitVersion,
        string tensorRtVersion,
        bool hasCudaToolkit,
        bool hasTensorRt,
        bool cudaBindingsEnabled,
        bool tensorRtBindingsEnabled)
    {
        AbiVersion = abiVersion;
        BridgeVersionMajor = bridgeVersionMajor;
        BridgeVersionMinor = bridgeVersionMinor;
        BridgeVersionPatch = bridgeVersionPatch;
        BridgeName = bridgeName;
        BridgeBanner = bridgeBanner;
        CompilerId = compilerId;
        CompilerVersion = compilerVersion;
        SystemName = systemName;
        SystemProcessor = systemProcessor;
        BuildConfiguration = buildConfiguration;
        CudaToolkitVersion = cudaToolkitVersion;
        TensorRtVersion = tensorRtVersion;
        HasCudaToolkit = hasCudaToolkit;
        HasTensorRt = hasTensorRt;
        CudaBindingsEnabled = cudaBindingsEnabled;
        TensorRtBindingsEnabled = tensorRtBindingsEnabled;
    }

    public int AbiVersion { get; }
    public int BridgeVersionMajor { get; }
    public int BridgeVersionMinor { get; }
    public int BridgeVersionPatch { get; }
    public string BridgeName { get; }
    public string BridgeBanner { get; }
    public string CompilerId { get; }
    public string CompilerVersion { get; }
    public string SystemName { get; }
    public string SystemProcessor { get; }
    public string BuildConfiguration { get; }
    public string CudaToolkitVersion { get; }
    public string TensorRtVersion { get; }
    public bool HasCudaToolkit { get; }
    public bool HasTensorRt { get; }
    public bool CudaBindingsEnabled { get; }
    public bool TensorRtBindingsEnabled { get; }
}

