namespace JYPPX.Shared.Interop;

/// <summary>
/// Adapter availability snapshot for a specific TensorRT major line.
/// 指定 TensorRT 主版本线的适配器可用性快照。
/// </summary>
public sealed class TensorRtAdapterInfo
{
    /// <summary>
    /// Creates a TensorRT adapter availability snapshot.
    /// 创建 TensorRT 适配器可用性快照。
    /// </summary>
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

    /// <summary>
    /// Gets the TensorRT API line represented by this snapshot.
    /// 获取该快照对应的 TensorRT API 主版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }
    /// <summary>
    /// Gets whether vendor-side dependencies were found.
    /// 获取是否找到了厂商侧依赖。
    /// </summary>
    public bool VendorDependencyAvailable { get; }
    /// <summary>
    /// Gets whether TensorRT runtime creation is supported.
    /// 获取是否支持创建 TensorRT runtime。
    /// </summary>
    public bool RuntimeCreationSupported { get; }
    /// <summary>
    /// Gets whether TensorRT builder creation is supported.
    /// 获取是否支持创建 TensorRT builder。
    /// </summary>
    public bool BuilderCreationSupported { get; }
    /// <summary>
    /// Gets whether TensorRT network creation is supported.
    /// 获取是否支持创建 TensorRT network。
    /// </summary>
    public bool NetworkCreationSupported { get; }
    /// <summary>
    /// Gets whether TensorRT engine deserialization is supported.
    /// 获取是否支持 TensorRT engine 反序列化。
    /// </summary>
    public bool EngineDeserializationSupported { get; }
    /// <summary>
    /// Gets the detected vendor version string when available.
    /// 获取可用时检测到的厂商版本字符串。
    /// </summary>
    public string DetectedVersion { get; }
    /// <summary>
    /// Gets the status or diagnostic message for this adapter line.
    /// 获取该适配线的状态或诊断消息。
    /// </summary>
    public string StatusMessage { get; }
}
