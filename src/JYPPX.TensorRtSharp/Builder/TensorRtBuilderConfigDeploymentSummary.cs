using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

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
