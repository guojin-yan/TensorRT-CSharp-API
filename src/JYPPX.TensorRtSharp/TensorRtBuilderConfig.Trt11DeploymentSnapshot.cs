using System;
using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtBuilderConfig
{
    /// <summary>
    /// Builds a TensorRT 11 deployment snapshot for this builder configuration.
    /// 为当前 builder configuration 构建 TensorRT 11 部署快照。
    /// </summary>
    /// <returns>A deployment snapshot with build-time knobs that affect engine portability and runtime behavior. 包含影响 engine 可移植性和运行时行为的构建参数快照。</returns>
    /// <remarks>
    /// The snapshot is read-only and does not mutate the builder config.
    /// 该快照只读取当前配置，不会修改 builder config。
    /// </remarks>
    public TensorRtBuilderConfigDeploymentSnapshot GetDeploymentSnapshot()
    {
        List<string> diagnostics = new List<string>();

        return new TensorRtBuilderConfigDeploymentSnapshot(
            OptimizationProfileCount,
            IsProfileStreamSet,
            TryCollect("HasCalibrationProfile", diagnostics, () => HasCalibrationProfile, false),
            TryCollect("Flags", diagnostics, GetFlags, TensorRtBuilderFlags.None),
            GetEngineCapability(),
            GetHardwareCompatibilityLevel(),
            TryCollect("RuntimePlatform", diagnostics, GetRuntimePlatform, TensorRtRuntimePlatform.SameAsBuild),
            GetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace),
            GetOptimizationLevel(),
            GetProfilingVerbosity(),
            GetMaxAuxStreams(),
            GetAverageTimingIterations(),
            GetTacticSources(),
            TryCollect("DefaultDeviceType", diagnostics, GetDefaultDeviceType, TensorRtDeviceType.Gpu),
            TryCollect("DlaCore", diagnostics, GetDlaCore, -1),
            TryCollect("TilingOptimizationLevel", diagnostics, GetTilingOptimizationLevel, TensorRtTilingOptimizationLevel.None),
            TryCollect("L2LimitForTiling", diagnostics, GetL2LimitForTiling, 0L),
            TryCollect("MaxTactics", diagnostics, GetMaxTactics, 0),
            TryCollect("HasTimingCache", diagnostics, () => HasTimingCache, false),
            TryCollect("PluginToSerializeCount", diagnostics, () => PluginToSerializeCount, 0),
            TryCollect("HasProgressMonitor", diagnostics, () => HasProgressMonitor, false),
            TryCollect("RemoteAutoTuningConfig", diagnostics, GetRemoteAutoTuningConfig, string.Empty),
            TryCollect(
                "SerializedPluginSnapshot",
                diagnostics,
                GetSerializedPluginSnapshot,
                new TensorRtBuilderConfigSerializedPluginSnapshot(Line, 0, Array.Empty<string>(), false, "Unavailable")),
            diagnostics);
    }

    private static T TryCollect<T>(string fieldName, List<string> diagnostics, Func<T> getter, T fallback)
    {
        try
        {
            return getter();
        }
        catch (Exception ex)
        {
            diagnostics.Add($"{fieldName}: {ex.Message}");
            return fallback;
        }
    }
}
