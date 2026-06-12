using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtEngine
{
    /// <summary>
    /// Gets TensorRT 11 profile tensor values V2 for a named shape tensor.
    /// 获取命名 shape tensor 的 TensorRT 11 profile tensor values V2。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. Engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. Optimization profile 索引。</param>
    /// <param name="selector">The min/opt/max selector. Min/Opt/Max 选择器。</param>
    /// <param name="valueCount">
    /// Number of values to copy. TensorRT does not expose a count query on <c>ICudaEngine</c>, so callers should pass the expected shape-value count.
    /// 要复制的值数量。TensorRT 在 <c>ICudaEngine</c> 上不提供数量查询，因此调用者应传入预期的 shape-value 数量。
    /// </param>
    /// <returns>
    /// Copied profile tensor values, or an empty array when TensorRT reports no values for the tensor.
    /// 复制出的 profile tensor values；当 TensorRT 未报告该 tensor 的值时返回空数组。
    /// </returns>
    public long[] GetProfileTensorValuesV2(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector, int valueCount)
    {
        return NativeBridgeApi.GetEngineProfileTensorValuesV2(Line, _handle, tensorName, profileIndex, selector, valueCount);
    }

    /// <summary>
    /// Builds a TensorRT 11 deployment snapshot for this engine.
    /// 为当前 engine 构建 TensorRT 11 部署快照。
    /// </summary>
    /// <param name="profileIndex">The optimization profile used for profile-specific fields. 用于 profile 相关字段的 optimization profile 索引。</param>
    /// <returns>A deployment snapshot with engine metadata and tensor bindings. 包含 engine 元数据和 tensor binding 的部署快照。</returns>
    /// <remarks>
    /// This method intentionally aggregates many deployment-critical getters into one managed object so applications can log or validate an engine before binding buffers.
    /// 该方法会有意把大量部署关键 getter 聚合到一个托管对象中，便于应用在绑定缓冲区前记录或验证 engine。
    /// </remarks>
    public TensorRtEngineDeploymentSnapshot GetDeploymentSnapshot(int profileIndex = 0)
    {
        if (profileIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(profileIndex), "Profile index must be greater than or equal to zero.");
        }

        List<string> diagnostics = new List<string>();
        List<TensorRtEngineTensorBinding> tensors = new List<TensorRtEngineTensorBinding>();
        int tensorCount = IOTensorCount;
        for (int index = 0; index < tensorCount; index++)
        {
            try
            {
                tensors.Add(GetTensorBinding(index, profileIndex));
            }
            catch (Exception ex)
            {
                diagnostics.Add($"TensorBinding[{index}]: {ex.Message}");
            }
        }

        ulong profileMemory = TryCollect("ProfileDeviceMemorySize", diagnostics, () => GetDeviceMemorySizeForProfile(profileIndex), 0UL);
        ulong profileMemoryV2 = TryCollect("ProfileDeviceMemorySizeV2", diagnostics, () => GetDeviceMemorySizeForProfileV2(profileIndex), 0UL);
        long totalWeights = TryCollect("EngineStat.TotalWeightsSize", diagnostics, () => GetEngineStat(TensorRtEngineStat.TotalWeightsSize), 0L);
        long strippedWeights = TryCollect("EngineStat.StrippedWeightsSize", diagnostics, () => GetEngineStat(TensorRtEngineStat.StrippedWeightsSize), 0L);

        return new TensorRtEngineDeploymentSnapshot(
            Name,
            profileIndex,
            tensorCount,
            LayerCount,
            OptimizationProfileCount,
            DeviceMemorySizeInBytes,
            DeviceMemorySizeV2InBytes,
            profileMemory,
            profileMemoryV2,
            AuxiliaryStreamCount,
            Capability,
            TacticSources,
            ProfilingVerbosity,
            EngineHardwareCompatibilityLevel,
            IsRefittable,
            StreamableWeightsSizeInBytes,
            WeightStreamingBudgetV2InBytes,
            WeightStreamingAutomaticBudgetInBytes,
            WeightStreamingScratchMemorySizeInBytes,
            totalWeights,
            strippedWeights,
            tensors,
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
