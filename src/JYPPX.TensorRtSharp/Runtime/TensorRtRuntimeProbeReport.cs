using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Read-only TensorRT global runtime version details collected without creating a runtime.
/// 无需创建 runtime 即可采集的 TensorRT 全局 runtime 版本信息。
/// </summary>
public sealed class TensorRtGlobalRuntimeVersion
{
    internal TensorRtGlobalRuntimeVersion(
        TensorRtApiLine line,
        int inferLibVersion,
        int major,
        int minor,
        int patch,
        int build,
        int onnxParserVersion,
        bool hasGlobalLogger)
    {
        Line = line;
        InferLibVersion = inferLibVersion;
        Major = major;
        Minor = minor;
        Patch = patch;
        Build = build;
        OnnxParserVersion = onnxParserVersion;
        HasGlobalLogger = hasGlobalLogger;
    }

    /// <summary>
    /// Gets the TensorRT API line used for the query.
    /// 获取查询使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the packed TensorRT infer library version reported by TensorRT.
    /// 获取 TensorRT 报告的 packed infer library version。
    /// </summary>
    public int InferLibVersion { get; }

    /// <summary>
    /// Gets the TensorRT infer library major version.
    /// 获取 TensorRT infer library major version。
    /// </summary>
    public int Major { get; }

    /// <summary>
    /// Gets the TensorRT infer library minor version.
    /// 获取 TensorRT infer library minor version。
    /// </summary>
    public int Minor { get; }

    /// <summary>
    /// Gets the TensorRT infer library patch version.
    /// 获取 TensorRT infer library patch version。
    /// </summary>
    public int Patch { get; }

    /// <summary>
    /// Gets the TensorRT infer library build version.
    /// 获取 TensorRT infer library build version。
    /// </summary>
    public int Build { get; }

    /// <summary>
    /// Gets the ONNX parser version reported by TensorRT's parser library.
    /// 获取 TensorRT ONNX parser library 报告的版本。
    /// </summary>
    public int OnnxParserVersion { get; }

    /// <summary>
    /// Gets whether TensorRT exposes a non-null global logger pointer.
    /// 获取 TensorRT 是否暴露非空全局 logger 指针。
    /// </summary>
    public bool HasGlobalLogger { get; }

    /// <summary>
    /// Formats the global runtime version snapshot for diagnostics.
    /// 将全局 runtime 版本快照格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString() => $"{Line}:{Major}.{Minor}.{Patch}.{Build}:packed={InferLibVersion}:onnx={OnnxParserVersion}:globalLogger={HasGlobalLogger}";
}

/// <summary>
/// Result for one TensorRT runtime probe stage.
/// TensorRT runtime 分阶段探针的单阶段结果。
/// </summary>
public sealed class TensorRtRuntimeProbeStage
{
    internal TensorRtRuntimeProbeStage(string name, bool succeeded, string message)
    {
        Name = name ?? string.Empty;
        Succeeded = succeeded;
        Message = message ?? string.Empty;
    }

    /// <summary>
    /// Gets the stage name.
    /// 获取阶段名称。
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets whether the stage succeeded.
    /// 获取该阶段是否成功。
    /// </summary>
    public bool Succeeded { get; }

    /// <summary>
    /// Gets the diagnostic message for this stage.
    /// 获取该阶段的诊断消息。
    /// </summary>
    public string Message { get; }

    /// <summary>
    /// Formats the probe stage for diagnostics.
    /// 将探针阶段格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString() => $"{Name}={Succeeded}:{Message}";
}

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
