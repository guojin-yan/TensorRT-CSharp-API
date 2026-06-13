using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// High-level environment snapshot for the native TensorRT bridge.
/// 原生 TensorRT bridge 的高层环境快照。
/// </summary>
public sealed class TensorRtEnvironmentSnapshot
{
    /// <summary>
    /// Initializes a high-level environment snapshot for the native TensorRT bridge.
    /// 初始化原生 TensorRT bridge 的高层环境快照。
    /// </summary>
    /// <param name="buildInfo">The bridge build information. bridge 构建信息。</param>
    /// <param name="runtimeInfo">The bridge runtime information. bridge 运行时信息。</param>
    /// <param name="capabilityInfo">The bridge capability information. bridge 能力信息。</param>
    /// <param name="tensorRt8">The TensorRT 8 adapter information. TensorRT 8 适配器信息。</param>
    /// <param name="tensorRt10">The TensorRT 10 adapter information. TensorRT 10 适配器信息。</param>
    /// <param name="tensorRt11">The TensorRT 11 adapter information. TensorRT 11 适配器信息。</param>
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

    /// <summary>
    /// Gets the bridge build information.
    /// 获取 bridge 构建信息。
    /// </summary>
    public BridgeBuildInfo BuildInfo { get; }
    /// <summary>
    /// Gets the bridge runtime information.
    /// 获取 bridge 运行时信息。
    /// </summary>
    public BridgeRuntimeInfo RuntimeInfo { get; }
    /// <summary>
    /// Gets the bridge capability information.
    /// 获取 bridge 能力信息。
    /// </summary>
    public BridgeCapabilityInfo CapabilityInfo { get; }
    /// <summary>
    /// Gets the TensorRT 8 adapter information.
    /// 获取 TensorRT 8 适配器信息。
    /// </summary>
    public TensorRtAdapterInfo TensorRt8 { get; }
    /// <summary>
    /// Gets the TensorRT 10 adapter information.
    /// 获取 TensorRT 10 适配器信息。
    /// </summary>
    public TensorRtAdapterInfo TensorRt10 { get; }
    /// <summary>
    /// Gets the TensorRT 11 adapter information.
    /// 获取 TensorRT 11 适配器信息。
    /// </summary>
    public TensorRtAdapterInfo TensorRt11 { get; }
}
