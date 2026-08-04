namespace JYPPX.TensorRtSharp.Shared.Interop;

/// <summary>
/// Runtime-facing bridge state reported by the native layer.
/// 原生层报告的面向运行时的 bridge 状态。
/// </summary>
public sealed class BridgeRuntimeInfo
{
    /// <summary>
    /// Creates a bridge runtime-information snapshot.
    /// 创建 bridge 运行时信息快照。
    /// </summary>
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

    /// <summary>
    /// Gets the native bridge ABI version.
    /// 获取原生 bridge ABI 版本。
    /// </summary>
    public int AbiVersion { get; }
    /// <summary>
    /// Gets the logical bridge name.
    /// 获取逻辑 bridge 名称。
    /// </summary>
    public string BridgeName { get; }
    /// <summary>
    /// Gets the bridge banner string.
    /// 获取 bridge banner 字符串。
    /// </summary>
    public string BridgeBanner { get; }
    /// <summary>
    /// Gets the last bridge error message.
    /// 获取最近一次 bridge 错误消息。
    /// </summary>
    public string LastErrorMessage { get; }
    /// <summary>
    /// Gets the category of the last bridge error.
    /// 获取最近一次 bridge 错误所属类别。
    /// </summary>
    public BridgeErrorCategory LastErrorCategory { get; }
    /// <summary>
    /// Gets whether CUDA toolkit dependencies are currently available.
    /// 获取当前 CUDA toolkit 依赖是否可用。
    /// </summary>
    public bool CudaToolkitAvailable { get; }
    /// <summary>
    /// Gets whether TensorRT dependencies are currently available.
    /// 获取当前 TensorRT 依赖是否可用。
    /// </summary>
    public bool TensorRtAvailable { get; }
}
