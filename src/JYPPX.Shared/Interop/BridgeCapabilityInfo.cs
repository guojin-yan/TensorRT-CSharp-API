namespace JYPPX.Shared.Interop;

/// <summary>
/// Capability flags surfaced by the native bridge.
/// 原生 bridge 暴露出的能力标志集合。
/// </summary>
public sealed class BridgeCapabilityInfo
{
    /// <summary>
    /// Creates a bridge capability snapshot.
    /// 创建 bridge 能力快照。
    /// </summary>
    public BridgeCapabilityInfo(
        bool supportsTrt8Adapter,
        bool supportsTrt10Adapter,
        bool supportsTrt11Adapter,
        bool supportsTrt8RuntimeCreation,
        bool supportsTrt10RuntimeCreation,
        bool supportsTrt11RuntimeCreation,
        bool supportsTrt8BuilderCreation,
        bool supportsTrt10BuilderCreation,
        bool supportsTrt11BuilderCreation,
        bool supportsLastErrorQuery,
        bool supportsBuildInfoQuery,
        bool supportsRuntimeInfoQuery)
    {
        SupportsTrt8Adapter = supportsTrt8Adapter;
        SupportsTrt10Adapter = supportsTrt10Adapter;
        SupportsTrt11Adapter = supportsTrt11Adapter;
        SupportsTrt8RuntimeCreation = supportsTrt8RuntimeCreation;
        SupportsTrt10RuntimeCreation = supportsTrt10RuntimeCreation;
        SupportsTrt11RuntimeCreation = supportsTrt11RuntimeCreation;
        SupportsTrt8BuilderCreation = supportsTrt8BuilderCreation;
        SupportsTrt10BuilderCreation = supportsTrt10BuilderCreation;
        SupportsTrt11BuilderCreation = supportsTrt11BuilderCreation;
        SupportsLastErrorQuery = supportsLastErrorQuery;
        SupportsBuildInfoQuery = supportsBuildInfoQuery;
        SupportsRuntimeInfoQuery = supportsRuntimeInfoQuery;
    }

    /// <summary>
    /// Gets whether the bridge exposes a TensorRT 8 adapter line.
    /// 获取 bridge 是否暴露 TensorRT 8 适配线。
    /// </summary>
    public bool SupportsTrt8Adapter { get; }
    /// <summary>
    /// Gets whether the bridge exposes a TensorRT 10 adapter line.
    /// 获取 bridge 是否暴露 TensorRT 10 适配线。
    /// </summary>
    public bool SupportsTrt10Adapter { get; }
    /// <summary>
    /// Gets whether the bridge exposes a TensorRT 11 adapter line.
    /// 获取 bridge 是否暴露 TensorRT 11 适配线。
    /// </summary>
    public bool SupportsTrt11Adapter { get; }
    /// <summary>
    /// Gets whether TensorRT 8 runtime creation is supported.
    /// 获取是否支持创建 TensorRT 8 runtime。
    /// </summary>
    public bool SupportsTrt8RuntimeCreation { get; }
    /// <summary>
    /// Gets whether TensorRT 10 runtime creation is supported.
    /// 获取是否支持创建 TensorRT 10 runtime。
    /// </summary>
    public bool SupportsTrt10RuntimeCreation { get; }
    /// <summary>
    /// Gets whether TensorRT 11 runtime creation is supported.
    /// 获取是否支持创建 TensorRT 11 runtime。
    /// </summary>
    public bool SupportsTrt11RuntimeCreation { get; }
    /// <summary>
    /// Gets whether TensorRT 8 builder creation is supported.
    /// 获取是否支持创建 TensorRT 8 builder。
    /// </summary>
    public bool SupportsTrt8BuilderCreation { get; }
    /// <summary>
    /// Gets whether TensorRT 10 builder creation is supported.
    /// 获取是否支持创建 TensorRT 10 builder。
    /// </summary>
    public bool SupportsTrt10BuilderCreation { get; }
    /// <summary>
    /// Gets whether TensorRT 11 builder creation is supported.
    /// 获取是否支持创建 TensorRT 11 builder。
    /// </summary>
    public bool SupportsTrt11BuilderCreation { get; }
    /// <summary>
    /// Gets whether bridge last-error querying is supported.
    /// 获取是否支持 bridge 最近错误查询。
    /// </summary>
    public bool SupportsLastErrorQuery { get; }
    /// <summary>
    /// Gets whether bridge build-info querying is supported.
    /// 获取是否支持 bridge 构建信息查询。
    /// </summary>
    public bool SupportsBuildInfoQuery { get; }
    /// <summary>
    /// Gets whether bridge runtime-info querying is supported.
    /// 获取是否支持 bridge 运行时信息查询。
    /// </summary>
    public bool SupportsRuntimeInfoQuery { get; }
}
