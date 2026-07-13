using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtEngine
{
    /// <summary>
    /// Gets TensorRT 10 legacy profile tensor values for a named shape tensor, using the engine tensor shape to infer the value count.
    /// 使用 engine tensor shape 自动推导值数量，并获取命名 shape tensor 的 TensorRT 10 legacy profile tensor values。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. Engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. Optimization profile 索引。</param>
    /// <param name="selector">The min/opt/max selector. Min/Opt/Max 选择器。</param>
    /// <returns>
    /// Copied profile tensor values, or an empty array when TensorRT reports no values for the tensor.
    /// 复制出的 profile tensor values；当 TensorRT 未报告该 tensor 的值时返回空数组。
    /// </returns>
    /// <remarks>
    /// This convenience overload can infer the value count only when <see cref="GetTensorShape"/> reports a fully static shape.
    /// 当 <see cref="GetTensorShape"/> 返回完全静态 shape 时，该便利 overload 才能自动推导值数量。
    /// </remarks>
    public int[] GetProfileTensorValues(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector)
    {
        return GetProfileTensorValues(tensorName, profileIndex, selector, InferProfileTensorValueCount(tensorName));
    }

    /// <summary>
    /// Gets TensorRT 10 legacy profile tensor values for a named shape tensor.
    /// 获取命名 shape tensor 的 TensorRT 10 legacy profile tensor values。
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
    /// <remarks>
    /// TensorRT 10 deprecates this legacy int32 path in favor of <c>GetProfileTensorValuesV2</c>.
    /// TensorRT 10 已将该 int32 legacy 路径标记为 deprecated；新代码请优先使用 <c>GetProfileTensorValuesV2</c>。
    /// </remarks>
    public int[] GetProfileTensorValues(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector, int valueCount)
    {
        return NativeBridgeApi.GetEngineProfileTensorValues(Line, _handle, tensorName, profileIndex, selector, valueCount);
    }

    /// <summary>
    /// Gets TensorRT 10/11 profile tensor values V2 for a named shape tensor, using the engine tensor shape to infer the value count.
    /// 使用 engine tensor shape 自动推导值数量，并获取命名 shape tensor 的 TensorRT 10/11 profile tensor values V2。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. Engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. Optimization profile 索引。</param>
    /// <param name="selector">The min/opt/max selector. Min/Opt/Max 选择器。</param>
    /// <returns>
    /// Copied profile tensor values, or an empty array when TensorRT reports no values for the tensor.
    /// 复制出的 profile tensor values；当 TensorRT 未报告该 tensor 的值时返回空数组。
    /// </returns>
    /// <remarks>
    /// This convenience overload can infer the value count only when <see cref="GetTensorShape"/> reports a fully static shape.
    /// 当 <see cref="GetTensorShape"/> 返回完全静态 shape 时，该便利 overload 才能自动推导值数量。
    /// </remarks>
    public long[] GetProfileTensorValuesV2(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector)
    {
        return GetProfileTensorValuesV2(tensorName, profileIndex, selector, InferProfileTensorValueCount(tensorName));
    }

    /// <summary>
    /// Gets TensorRT 10/11 profile tensor values V2 for a named shape tensor.
    /// 获取命名 shape tensor 的 TensorRT 10/11 profile tensor values V2。
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
    /// Gets a copied profile tensor values snapshot using the engine tensor shape to infer the value count.
    /// 使用 engine tensor shape 自动推导值数量，并获取 copied profile tensor values 快照。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. Engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. Optimization profile 索引。</param>
    /// <param name="selector">The min/opt/max selector. Min/Opt/Max 选择器。</param>
    /// <returns>A copied profile tensor values snapshot. 复制出的 profile tensor values 快照。</returns>
    public TensorRtEngineProfileTensorValuesSnapshot GetProfileTensorValuesSnapshot(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector)
    {
        return GetProfileTensorValuesSnapshot(tensorName, profileIndex, selector, InferProfileTensorValueCount(tensorName));
    }

    /// <summary>
    /// Gets a copied profile tensor values snapshot for one tensor/profile/selector query.
    /// 获取单个 tensor/profile/selector 查询对应的 copied profile tensor values 快照。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. Engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. Optimization profile 索引。</param>
    /// <param name="selector">The min/opt/max selector. Min/Opt/Max 选择器。</param>
    /// <param name="valueCount">The expected number of profile tensor values to copy. 要复制的 profile tensor values 数量。</param>
    /// <returns>A copied profile tensor values snapshot. 复制出的 profile tensor values 快照。</returns>
    /// <remarks>
    /// The snapshot uses caller-owned arrays and never exposes TensorRT-owned pointers. TensorRT 10 may provide both the
    /// legacy int32 path and the V2 int64 path; TensorRT 11 provides the V2 path.
    /// 该快照使用调用方拥有的数组，不暴露 TensorRT-owned pointer。TensorRT 10 可能同时提供 legacy int32 与 V2 int64 路径；
    /// TensorRT 11 提供 V2 路径。
    /// </remarks>
    public TensorRtEngineProfileTensorValuesSnapshot GetProfileTensorValuesSnapshot(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector, int valueCount)
    {
        ValidateProfileTensorSnapshotInput(tensorName, profileIndex, valueCount);

        List<string> diagnostics = new List<string>();
        int[] legacyValues = Array.Empty<int>();
        long[] valuesV2 = Array.Empty<long>();
        bool hasLegacyValues = false;
        bool hasValuesV2 = false;

        if (Line == JYPPX.Shared.Interop.TensorRtApiLine.TensorRt10)
        {
            try
            {
                legacyValues = GetProfileTensorValues(tensorName, profileIndex, selector, valueCount);
                hasLegacyValues = true;
            }
            catch (Exception exception) when (IsProfileTensorSnapshotProbeException(exception))
            {
                diagnostics.Add($"LegacyInt32Values: {exception.Message}");
            }
        }

        try
        {
            valuesV2 = GetProfileTensorValuesV2(tensorName, profileIndex, selector, valueCount);
            hasValuesV2 = true;
        }
        catch (Exception exception) when (IsProfileTensorSnapshotProbeException(exception))
        {
            diagnostics.Add($"ValuesV2: {exception.Message}");
        }

        return new TensorRtEngineProfileTensorValuesSnapshot(
            Line,
            tensorName,
            profileIndex,
            selector,
            valueCount,
            hasLegacyValues,
            legacyValues,
            hasValuesV2,
            valuesV2,
            diagnostics);
    }

    /// <summary>
    /// Tries to get a copied profile tensor values snapshot using the engine tensor shape to infer the value count.
    /// 尝试使用 engine tensor shape 自动推导值数量，并获取 copied profile tensor values 快照。
    /// </summary>
    public bool TryGetProfileTensorValuesSnapshot(
        string tensorName,
        int profileIndex,
        TensorRtOptimizationProfileSelector selector,
        out TensorRtEngineProfileTensorValuesSnapshot snapshot,
        out string diagnostic)
    {
        try
        {
            return TryGetProfileTensorValuesSnapshot(tensorName, profileIndex, selector, InferProfileTensorValueCount(tensorName), out snapshot, out diagnostic);
        }
        catch (Exception exception) when (IsProfileTensorSnapshotProbeException(exception))
        {
            snapshot = CreateUnavailableProfileTensorValuesSnapshot(tensorName ?? string.Empty, profileIndex, selector, 0, exception.Message);
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to get a copied profile tensor values snapshot for one tensor/profile/selector query.
    /// 尝试获取单个 tensor/profile/selector 查询对应的 copied profile tensor values 快照。
    /// </summary>
    public bool TryGetProfileTensorValuesSnapshot(
        string tensorName,
        int profileIndex,
        TensorRtOptimizationProfileSelector selector,
        int valueCount,
        out TensorRtEngineProfileTensorValuesSnapshot snapshot,
        out string diagnostic)
    {
        try
        {
            snapshot = GetProfileTensorValuesSnapshot(tensorName, profileIndex, selector, valueCount);
            diagnostic = snapshot.Diagnostics.Count == 0 ? "OK" : string.Join("; ", snapshot.Diagnostics);
            return snapshot.HasAnyValues;
        }
        catch (Exception exception) when (IsProfileTensorSnapshotProbeException(exception))
        {
            snapshot = CreateUnavailableProfileTensorValuesSnapshot(tensorName ?? string.Empty, profileIndex, selector, valueCount, exception.Message);
            diagnostic = exception.Message;
            return false;
        }
    }

    private int InferProfileTensorValueCount(string tensorName)
    {
        TensorRtDims shape = GetTensorShape(tensorName);
        if (shape.Rank == 0)
        {
            return 1;
        }

        long count = 1;
        foreach (int extent in shape.Values)
        {
            if (extent <= 0)
            {
                throw new InvalidOperationException($"Tensor '{tensorName}' has dynamic or invalid shape {shape}; call the overload that accepts valueCount.");
            }

            count *= extent;
            if (count > int.MaxValue)
            {
                throw new InvalidOperationException($"Tensor '{tensorName}' shape {shape} is too large to copy with the managed profile tensor values helper.");
            }
        }

        return (int)count;
    }

    private static void ValidateProfileTensorSnapshotInput(string tensorName, int profileIndex, int valueCount)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name is required.", nameof(tensorName));
        }

        if (profileIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(profileIndex), "Profile index must be greater than or equal to zero.");
        }

        if (valueCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(valueCount), "Value count must be greater than zero.");
        }
    }

    private TensorRtEngineProfileTensorValuesSnapshot CreateUnavailableProfileTensorValuesSnapshot(
        string tensorName,
        int profileIndex,
        TensorRtOptimizationProfileSelector selector,
        int valueCount,
        string diagnostic)
    {
        return new TensorRtEngineProfileTensorValuesSnapshot(
            Line,
            tensorName,
            profileIndex,
            selector,
            valueCount,
            hasLegacyInt32Values: false,
            legacyInt32Values: Array.Empty<int>(),
            hasValuesV2: false,
            valuesV2: Array.Empty<long>(),
            diagnostics: new[] { diagnostic });
    }

    private static bool IsProfileTensorSnapshotProbeException(Exception exception)
    {
        return exception is BridgeProbeException ||
            exception is NotSupportedException ||
            exception is InvalidOperationException ||
            exception is ArgumentException;
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
        List<TensorRtEngineProfileTensorValuesSnapshot> profileTensorValues = new List<TensorRtEngineProfileTensorValuesSnapshot>();
        int tensorCount = IOTensorCount;
        for (int index = 0; index < tensorCount; index++)
        {
            try
            {
                TensorRtEngineTensorBinding binding = GetTensorBinding(index, profileIndex);
                tensors.Add(binding);
                int valueCount = InferProfileTensorSnapshotValueCount(binding);
                profileTensorValues.Add(GetProfileTensorValuesSnapshot(binding.Name, profileIndex, TensorRtOptimizationProfileSelector.Opt, valueCount));
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
            profileTensorValues,
            diagnostics);
    }

    private static int InferProfileTensorSnapshotValueCount(TensorRtEngineTensorBinding binding)
    {
        if (binding == null)
        {
            throw new ArgumentNullException(nameof(binding));
        }

        if (binding.EngineShape.Rank == 0)
        {
            return 1;
        }

        long count = 1;
        foreach (int extent in binding.EngineShape.Values)
        {
            if (extent <= 0)
            {
                return Math.Max(1, binding.EngineShape.Rank);
            }

            count *= extent;
            if (count > int.MaxValue)
            {
                return binding.EngineShape.Rank;
            }
        }

        return Math.Max(1, (int)count);
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
