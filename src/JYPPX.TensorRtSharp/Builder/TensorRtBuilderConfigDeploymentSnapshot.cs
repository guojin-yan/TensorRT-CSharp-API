using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Captures TensorRT builder configuration values that materially affect engine deployment.
/// 捕获会实质影响 engine 部署结果的 TensorRT builder 配置值。
/// </summary>
public sealed class TensorRtBuilderConfigDeploymentSnapshot
{
    internal TensorRtBuilderConfigDeploymentSnapshot(
        int optimizationProfileCount,
        bool isProfileStreamSet,
        bool hasCalibrationProfile,
        TensorRtBuilderFlags flags,
        TensorRtEngineCapability engineCapability,
        TensorRtHardwareCompatibilityLevel hardwareCompatibilityLevel,
        TensorRtRuntimePlatform runtimePlatform,
        ulong workspaceMemoryPoolLimitInBytes,
        int optimizationLevel,
        TensorRtProfilingVerbosity profilingVerbosity,
        int maxAuxStreams,
        int averageTimingIterations,
        TensorRtTacticSources tacticSources,
        TensorRtDeviceType defaultDeviceType,
        int dlaCore,
        TensorRtTilingOptimizationLevel tilingOptimizationLevel,
        long l2LimitForTilingInBytes,
        int maxTactics,
        bool hasTimingCache,
        int pluginToSerializeCount,
        bool hasProgressMonitor,
        string remoteAutoTuningConfig,
        TensorRtBuilderConfigSerializedPluginSnapshot serializedPluginSnapshot,
        IReadOnlyList<string> diagnostics)
    {
        OptimizationProfileCount = optimizationProfileCount;
        IsProfileStreamSet = isProfileStreamSet;
        HasCalibrationProfile = hasCalibrationProfile;
        Flags = flags;
        EngineCapability = engineCapability;
        HardwareCompatibilityLevel = hardwareCompatibilityLevel;
        RuntimePlatform = runtimePlatform;
        WorkspaceMemoryPoolLimitInBytes = workspaceMemoryPoolLimitInBytes;
        OptimizationLevel = optimizationLevel;
        ProfilingVerbosity = profilingVerbosity;
        MaxAuxStreams = maxAuxStreams;
        AverageTimingIterations = averageTimingIterations;
        TacticSources = tacticSources;
        DefaultDeviceType = defaultDeviceType;
        DlaCore = dlaCore;
        TilingOptimizationLevel = tilingOptimizationLevel;
        L2LimitForTilingInBytes = l2LimitForTilingInBytes;
        MaxTactics = maxTactics;
        HasTimingCache = hasTimingCache;
        PluginToSerializeCount = pluginToSerializeCount;
        HasProgressMonitor = hasProgressMonitor;
        RemoteAutoTuningConfig = remoteAutoTuningConfig;
        SerializedPluginSnapshot = serializedPluginSnapshot;
        Diagnostics = diagnostics;
    }

    /// <summary>
    /// Gets the number of optimization profiles attached to the config.
    /// 获取已附加到配置的 optimization profile 数量。
    /// </summary>
    public int OptimizationProfileCount { get; }

    /// <summary>
    /// Gets whether a profiling CUDA stream is set.
    /// 获取是否已设置 profiling CUDA stream。
    /// </summary>
    public bool IsProfileStreamSet { get; }

    /// <summary>
    /// Gets whether a calibration profile is attached.
    /// 获取是否已附加 calibration profile。
    /// </summary>
    public bool HasCalibrationProfile { get; }

    /// <summary>
    /// Gets the TensorRT 11 builder flag bitmask.
    /// 获取 TensorRT 11 builder flag 位掩码。
    /// </summary>
    public TensorRtBuilderFlags Flags { get; }

    /// <summary>
    /// Gets configured engine capability.
    /// 获取已配置的 engine capability。
    /// </summary>
    public TensorRtEngineCapability EngineCapability { get; }

    /// <summary>
    /// Gets configured hardware compatibility level.
    /// 获取已配置的硬件兼容级别。
    /// </summary>
    public TensorRtHardwareCompatibilityLevel HardwareCompatibilityLevel { get; }

    /// <summary>
    /// Gets configured runtime platform.
    /// 获取已配置的 runtime platform。
    /// </summary>
    public TensorRtRuntimePlatform RuntimePlatform { get; }

    /// <summary>
    /// Gets workspace memory-pool limit.
    /// 获取 workspace memory pool 限制。
    /// </summary>
    public ulong WorkspaceMemoryPoolLimitInBytes { get; }

    /// <summary>
    /// Gets builder optimization level.
    /// 获取 builder optimization level。
    /// </summary>
    public int OptimizationLevel { get; }

    /// <summary>
    /// Gets profiling verbosity.
    /// 获取 profiling verbosity。
    /// </summary>
    public TensorRtProfilingVerbosity ProfilingVerbosity { get; }

    /// <summary>
    /// Gets maximum auxiliary stream count requested during build.
    /// 获取构建时请求的最大辅助 stream 数量。
    /// </summary>
    public int MaxAuxStreams { get; }

    /// <summary>
    /// Gets average timing iterations.
    /// 获取 average timing iterations。
    /// </summary>
    public int AverageTimingIterations { get; }

    /// <summary>
    /// Gets tactic-source bit flags.
    /// 获取 tactic source 位标志。
    /// </summary>
    public TensorRtTacticSources TacticSources { get; }

    /// <summary>
    /// Gets default device type for layers.
    /// 获取 layer 默认设备类型。
    /// </summary>
    public TensorRtDeviceType DefaultDeviceType { get; }

    /// <summary>
    /// Gets selected DLA core index.
    /// 获取选择的 DLA core 索引。
    /// </summary>
    public int DlaCore { get; }

    /// <summary>
    /// Gets TensorRT 11 tiling optimization level.
    /// 获取 TensorRT 11 tiling 优化级别。
    /// </summary>
    public TensorRtTilingOptimizationLevel TilingOptimizationLevel { get; }

    /// <summary>
    /// Gets L2 byte limit for tiling optimization.
    /// 获取 tiling 优化使用的 L2 字节上限。
    /// </summary>
    public long L2LimitForTilingInBytes { get; }

    /// <summary>
    /// Gets maximum tactic count considered during build.
    /// 获取构建时最多考虑的 tactic 数量。
    /// </summary>
    public int MaxTactics { get; }

    /// <summary>
    /// Gets whether a timing cache is attached.
    /// 获取是否已绑定 timing cache。
    /// </summary>
    public bool HasTimingCache { get; }

    /// <summary>
    /// Gets number of plugin libraries configured for serialization.
    /// 获取配置为随 engine 序列化的 plugin library 数量。
    /// </summary>
    public int PluginToSerializeCount { get; }

    /// <summary>
    /// Gets whether a progress monitor is attached.
    /// 获取是否已绑定 progress monitor。
    /// </summary>
    public bool HasProgressMonitor { get; }

    /// <summary>
    /// Gets remote auto-tuning configuration text.
    /// 获取 remote auto-tuning 配置文本。
    /// </summary>
    public string RemoteAutoTuningConfig { get; }

    /// <summary>
    /// Gets copied serialized plugin-path inventory details when available.
    /// 获取可用时复制出的 serialized plugin path inventory 详情。
    /// </summary>
    public TensorRtBuilderConfigSerializedPluginSnapshot SerializedPluginSnapshot { get; }

    /// <summary>
    /// Gets non-fatal diagnostics collected while building the snapshot.
    /// 获取构建快照时收集的非致命诊断信息。
    /// </summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>
    /// Converts this copied builder-config deployment snapshot into a compact pointer-free summary.
    /// 将当前已复制 builder-config deployment 快照转换为紧凑、无指针逃逸的摘要。
    /// </summary>
    /// <remarks>
    /// This method only reads managed snapshot values. It does not call TensorRT, expose native config pointers,
    /// or promote local deployment diagnostics to runtime proof.
    /// 该方法只读取托管快照值；不会调用 TensorRT、暴露原生 config 指针，也不会将本地部署诊断晋级为 runtime proof。
    /// </remarks>
    public TensorRtBuilderConfigDeploymentSummary ToSummary()
    {
        return new TensorRtBuilderConfigDeploymentSummary(
            OptimizationProfileCount,
            IsProfileStreamSet,
            HasCalibrationProfile,
            Flags,
            EngineCapability,
            HardwareCompatibilityLevel,
            RuntimePlatform,
            WorkspaceMemoryPoolLimitInBytes,
            OptimizationLevel,
            ProfilingVerbosity,
            MaxAuxStreams,
            AverageTimingIterations,
            TacticSources,
            DefaultDeviceType,
            DlaCore,
            TilingOptimizationLevel,
            L2LimitForTilingInBytes,
            MaxTactics,
            HasTimingCache,
            PluginToSerializeCount,
            SerializedPluginSnapshot.Count,
            SerializedPluginSnapshot.PluginLibraryPaths.Count,
            HasProgressMonitor,
            !string.IsNullOrWhiteSpace(RemoteAutoTuningConfig),
            Diagnostics.Count);
    }

    /// <summary>
    /// Creates a compact one-line diagnostic summary.
    /// 创建紧凑的单行诊断摘要。
    /// </summary>
    public override string ToString()
    {
        return $"profiles={OptimizationProfileCount} flags={Flags} capability={EngineCapability} workspace={WorkspaceMemoryPoolLimitInBytes} opt={OptimizationLevel} aux={MaxAuxStreams} plugins={SerializedPluginSnapshot.Count}/{SerializedPluginSnapshot.PluginLibraryPaths.Count} diagnostics={Diagnostics.Count}";
    }
}
