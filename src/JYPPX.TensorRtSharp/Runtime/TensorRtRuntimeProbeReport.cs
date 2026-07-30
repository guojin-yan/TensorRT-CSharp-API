using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// TensorRT runtime staged probe report.
/// TensorRT runtime 分阶段探针报告。
/// </summary>
public sealed class TensorRtRuntimeProbeReport
{
    internal TensorRtRuntimeProbeReport(
        TensorRtApiLine line,
        TensorRtGlobalRuntimeVersion? globalVersion,
        TensorRtPluginRegistryInventory? globalPluginRegistry,
        IReadOnlyList<TensorRtRuntimeProbeStage> stages)
    {
        Line = line;
        GlobalVersion = globalVersion;
        GlobalPluginRegistry = globalPluginRegistry;
        Stages = stages ?? Array.Empty<TensorRtRuntimeProbeStage>();
    }

    /// <summary>
    /// Gets the TensorRT API line used for the probe.
    /// 获取探针使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the global runtime version snapshot when it was collected.
    /// 获取已采集的全局 runtime 版本快照。
    /// </summary>
    public TensorRtGlobalRuntimeVersion? GlobalVersion { get; }

    /// <summary>
    /// Gets the global plugin registry inventory when it was collected.
    /// 获取已采集的全局 plugin registry inventory。
    /// </summary>
    public TensorRtPluginRegistryInventory? GlobalPluginRegistry { get; }

    /// <summary>
    /// Gets each probe stage in execution order.
    /// 获取按执行顺序排列的探针阶段。
    /// </summary>
    public IReadOnlyList<TensorRtRuntimeProbeStage> Stages { get; }

    /// <summary>
    /// Gets whether runtime creation succeeded.
    /// 获取 runtime creation 是否成功。
    /// </summary>
    public bool RuntimeCreationSucceeded => Stages.Any(stage => stage.Name == "RuntimeCreate" && stage.Succeeded);

    /// <summary>
    /// Gets the first failed stage, or <see langword="null"/> when all stages succeeded.
    /// 获取第一个失败阶段；全部成功时为 <see langword="null"/>。
    /// </summary>
    public TensorRtRuntimeProbeStage? FirstFailure => Stages.FirstOrDefault(stage => !stage.Succeeded);

    /// <summary>
    /// Formats the runtime probe report for diagnostics.
    /// 将 runtime 探针报告格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString()
    {
        string version = GlobalVersion?.ToString() ?? "version=n/a";
        string registry = GlobalPluginRegistry != null ? $"globalCreators={GlobalPluginRegistry.CreatorCount}/{GlobalPluginRegistry.RecursiveCreatorCount?.ToString() ?? "n/a"}" : "globalCreators=n/a";
        string firstFailure = FirstFailure?.Name ?? "None";
        return $"{Line}:{version}:{registry}:runtimeCreate={RuntimeCreationSucceeded}:firstFailure={firstFailure}";
    }
}
