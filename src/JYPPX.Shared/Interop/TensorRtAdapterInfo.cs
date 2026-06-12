namespace JYPPX.Shared.Interop;

/// <summary>
/// Adapter availability snapshot for a specific TensorRT major line.
/// </summary>
public sealed class TensorRtAdapterInfo
{
    public TensorRtAdapterInfo(
        TensorRtApiLine line,
        bool vendorDependencyAvailable,
        bool runtimeCreationSupported,
        bool builderCreationSupported,
        bool networkCreationSupported,
        bool engineDeserializationSupported,
        string detectedVersion,
        string statusMessage)
    {
        Line = line;
        VendorDependencyAvailable = vendorDependencyAvailable;
        RuntimeCreationSupported = runtimeCreationSupported;
        BuilderCreationSupported = builderCreationSupported;
        NetworkCreationSupported = networkCreationSupported;
        EngineDeserializationSupported = engineDeserializationSupported;
        DetectedVersion = detectedVersion;
        StatusMessage = statusMessage;
    }

    public TensorRtApiLine Line { get; }
    public bool VendorDependencyAvailable { get; }
    public bool RuntimeCreationSupported { get; }
    public bool BuilderCreationSupported { get; }
    public bool NetworkCreationSupported { get; }
    public bool EngineDeserializationSupported { get; }
    public string DetectedVersion { get; }
    public string StatusMessage { get; }
}

