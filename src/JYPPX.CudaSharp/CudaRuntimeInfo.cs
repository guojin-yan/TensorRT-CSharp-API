namespace JYPPX.CudaSharp;

/// <summary>
/// Runtime-level CUDA bridge information.
/// </summary>
public sealed class CudaRuntimeInfo
{
    public CudaRuntimeInfo(
        bool vendorDependencyAvailable,
        bool supportsStreams,
        bool supportsEvents,
        bool supportsMemory,
        int runtimeVersion,
        int driverVersion,
        int deviceCount,
        string statusMessage)
    {
        VendorDependencyAvailable = vendorDependencyAvailable;
        SupportsStreams = supportsStreams;
        SupportsEvents = supportsEvents;
        SupportsMemory = supportsMemory;
        RuntimeVersion = runtimeVersion;
        DriverVersion = driverVersion;
        DeviceCount = deviceCount;
        StatusMessage = statusMessage;
    }

    public bool VendorDependencyAvailable { get; }
    public bool SupportsStreams { get; }
    public bool SupportsEvents { get; }
    public bool SupportsMemory { get; }
    public int RuntimeVersion { get; }
    public int DriverVersion { get; }
    public int DeviceCount { get; }
    public string StatusMessage { get; }
}

