namespace JYPPX.CudaSharp;

/// <summary>
/// Runtime-level CUDA bridge information.
/// CUDA bridge 的 runtime 层信息。
/// </summary>
public sealed class CudaRuntimeInfo
{
    /// <summary>
    /// Initializes a snapshot of CUDA runtime capability information.
    /// 初始化一份 CUDA runtime 能力信息快照。
    /// </summary>
    /// <param name="vendorDependencyAvailable">Whether vendor CUDA dependencies are available. 是否可用厂商 CUDA 依赖。</param>
    /// <param name="supportsStreams">Whether stream APIs are supported. 是否支持 stream API。</param>
    /// <param name="supportsEvents">Whether event APIs are supported. 是否支持 event API。</param>
    /// <param name="supportsMemory">Whether memory APIs are supported. 是否支持 memory API。</param>
    /// <param name="runtimeVersion">The CUDA runtime version. CUDA runtime 版本。</param>
    /// <param name="driverVersion">The CUDA driver version. CUDA driver 版本。</param>
    /// <param name="deviceCount">The visible CUDA device count. 可见 CUDA 设备数量。</param>
    /// <param name="statusMessage">The status or diagnostic message. 状态或诊断消息。</param>
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

    /// <summary>
    /// Gets whether vendor CUDA dependencies are currently available.
    /// 获取当前是否可用厂商 CUDA 依赖。
    /// </summary>
    public bool VendorDependencyAvailable { get; }
    /// <summary>
    /// Gets whether stream APIs are supported by the loaded runtime.
    /// 获取当前加载的 runtime 是否支持 stream API。
    /// </summary>
    public bool SupportsStreams { get; }
    /// <summary>
    /// Gets whether event APIs are supported by the loaded runtime.
    /// 获取当前加载的 runtime 是否支持 event API。
    /// </summary>
    public bool SupportsEvents { get; }
    /// <summary>
    /// Gets whether memory APIs are supported by the loaded runtime.
    /// 获取当前加载的 runtime 是否支持 memory API。
    /// </summary>
    public bool SupportsMemory { get; }
    /// <summary>
    /// Gets the CUDA runtime version.
    /// 获取 CUDA runtime 版本。
    /// </summary>
    public int RuntimeVersion { get; }
    /// <summary>
    /// Gets the CUDA driver version.
    /// 获取 CUDA driver 版本。
    /// </summary>
    public int DriverVersion { get; }
    /// <summary>
    /// Gets the visible CUDA device count.
    /// 获取可见 CUDA 设备数量。
    /// </summary>
    public int DeviceCount { get; }
    /// <summary>
    /// Gets the status or diagnostic message.
    /// 获取状态或诊断消息。
    /// </summary>
    public string StatusMessage { get; }
}
