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

/// <summary>
/// Compact pointer-free summary of TensorRT builder configuration deployment settings.
/// TensorRT builder config deployment 设置的紧凑无指针摘要。
/// </summary>
public sealed class TensorRtBuilderConfigDeploymentSummary
{
    internal TensorRtBuilderConfigDeploymentSummary(
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
        int copiedSerializedPluginCount,
        int copiedSerializedPluginPathCount,
        bool hasProgressMonitor,
        bool hasRemoteAutoTuningConfig,
        int diagnosticCount)
    {
        OptimizationProfileCount = optimizationProfileCount < 0 ? 0 : optimizationProfileCount;
        IsProfileStreamSet = isProfileStreamSet;
        HasCalibrationProfile = hasCalibrationProfile;
        Flags = flags;
        EngineCapability = engineCapability;
        HardwareCompatibilityLevel = hardwareCompatibilityLevel;
        RuntimePlatform = runtimePlatform;
        WorkspaceMemoryPoolLimitInBytes = workspaceMemoryPoolLimitInBytes;
        OptimizationLevel = optimizationLevel;
        ProfilingVerbosity = profilingVerbosity;
        MaxAuxStreams = maxAuxStreams < 0 ? 0 : maxAuxStreams;
        AverageTimingIterations = averageTimingIterations < 0 ? 0 : averageTimingIterations;
        TacticSources = tacticSources;
        DefaultDeviceType = defaultDeviceType;
        DlaCore = dlaCore;
        TilingOptimizationLevel = tilingOptimizationLevel;
        L2LimitForTilingInBytes = l2LimitForTilingInBytes < 0 ? 0 : l2LimitForTilingInBytes;
        MaxTactics = maxTactics < 0 ? 0 : maxTactics;
        HasTimingCache = hasTimingCache;
        PluginToSerializeCount = pluginToSerializeCount < 0 ? 0 : pluginToSerializeCount;
        CopiedSerializedPluginCount = copiedSerializedPluginCount < 0 ? 0 : copiedSerializedPluginCount;
        CopiedSerializedPluginPathCount = copiedSerializedPluginPathCount < 0 ? 0 : copiedSerializedPluginPathCount;
        HasProgressMonitor = hasProgressMonitor;
        HasRemoteAutoTuningConfig = hasRemoteAutoTuningConfig;
        DiagnosticCount = diagnosticCount < 0 ? 0 : diagnosticCount;
    }

    /// <summary>Gets copied optimization profile count. 获取已复制 optimization profile 数量。</summary>
    public int OptimizationProfileCount { get; }

    /// <summary>Gets whether a profile stream is set. 获取是否已设置 profile stream。</summary>
    public bool IsProfileStreamSet { get; }

    /// <summary>Gets whether a calibration profile is attached. 获取是否已附加 calibration profile。</summary>
    public bool HasCalibrationProfile { get; }

    /// <summary>Gets copied builder flags. 获取已复制 builder flags。</summary>
    public TensorRtBuilderFlags Flags { get; }

    /// <summary>Gets copied engine capability. 获取已复制 engine capability。</summary>
    public TensorRtEngineCapability EngineCapability { get; }

    /// <summary>Gets copied hardware compatibility level. 获取已复制硬件兼容级别。</summary>
    public TensorRtHardwareCompatibilityLevel HardwareCompatibilityLevel { get; }

    /// <summary>Gets copied runtime platform. 获取已复制 runtime platform。</summary>
    public TensorRtRuntimePlatform RuntimePlatform { get; }

    /// <summary>Gets copied workspace memory-pool limit. 获取已复制 workspace memory pool 限制。</summary>
    public ulong WorkspaceMemoryPoolLimitInBytes { get; }

    /// <summary>Gets copied optimization level. 获取已复制 optimization level。</summary>
    public int OptimizationLevel { get; }

    /// <summary>Gets copied profiling verbosity. 获取已复制 profiling verbosity。</summary>
    public TensorRtProfilingVerbosity ProfilingVerbosity { get; }

    /// <summary>Gets copied max auxiliary stream count. 获取已复制最大 auxiliary stream 数量。</summary>
    public int MaxAuxStreams { get; }

    /// <summary>Gets copied average timing iteration count. 获取已复制 average timing iteration 数量。</summary>
    public int AverageTimingIterations { get; }

    /// <summary>Gets copied tactic sources. 获取已复制 tactic sources。</summary>
    public TensorRtTacticSources TacticSources { get; }

    /// <summary>Gets copied default device type. 获取已复制默认 device type。</summary>
    public TensorRtDeviceType DefaultDeviceType { get; }

    /// <summary>Gets copied DLA core. 获取已复制 DLA core。</summary>
    public int DlaCore { get; }

    /// <summary>Gets copied tiling optimization level. 获取已复制 tiling optimization level。</summary>
    public TensorRtTilingOptimizationLevel TilingOptimizationLevel { get; }

    /// <summary>Gets copied L2 byte limit for tiling. 获取已复制 tiling L2 字节限制。</summary>
    public long L2LimitForTilingInBytes { get; }

    /// <summary>Gets copied max tactic count. 获取已复制 max tactic 数量。</summary>
    public int MaxTactics { get; }

    /// <summary>Gets whether a timing cache is attached. 获取是否已绑定 timing cache。</summary>
    public bool HasTimingCache { get; }

    /// <summary>Gets TensorRT-reported plugin-to-serialize count. 获取 TensorRT 报告的待序列化 plugin 数量。</summary>
    public int PluginToSerializeCount { get; }

    /// <summary>Gets copied serialized plugin record count. 获取已复制 serialized plugin 记录数量。</summary>
    public int CopiedSerializedPluginCount { get; }

    /// <summary>Gets copied serialized plugin path count. 获取已复制 serialized plugin path 数量。</summary>
    public int CopiedSerializedPluginPathCount { get; }

    /// <summary>Gets whether a progress monitor is attached. 获取是否已绑定 progress monitor。</summary>
    public bool HasProgressMonitor { get; }

    /// <summary>Gets whether remote auto-tuning config text is present. 获取是否存在 remote auto-tuning config 文本。</summary>
    public bool HasRemoteAutoTuningConfig { get; }

    /// <summary>Gets diagnostic count collected while building the snapshot. 获取构建快照时收集的诊断数量。</summary>
    public int DiagnosticCount { get; }

    /// <summary>Gets whether copied plugin records cover the TensorRT-reported plugin count. 获取已复制 plugin 记录是否覆盖 TensorRT 报告数量。</summary>
    public bool CopiedPluginCountMatchesReportedCount => CopiedSerializedPluginCount == PluginToSerializeCount;

    /// <summary>Gets whether this summary is copied and pointer-free. 获取该摘要是否为复制型且无指针逃逸。</summary>
    public bool PointerFreeCopiedSummary => true;

    /// <summary>Gets whether this summary can be promoted as runtime proof. 获取该摘要是否可晋级为 runtime proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether deferred records can be deleted because of this summary. 获取是否可因该摘要删除 deferred 记录。</summary>
    public bool CanDeleteDeferredRecord => false;

    /// <summary>Formats this summary for smoke output and logs. 将该摘要格式化为 smoke 输出和日志。</summary>
    public override string ToString()
    {
        return $"profiles={OptimizationProfileCount} flags={Flags} capability={EngineCapability} workspace={WorkspaceMemoryPoolLimitInBytes} opt={OptimizationLevel} aux={MaxAuxStreams} plugins={PluginToSerializeCount}/{CopiedSerializedPluginCount}/{CopiedSerializedPluginPathCount} diagnostics={DiagnosticCount} RuntimeProof={CanPromoteRuntimeProof}";
    }
}
