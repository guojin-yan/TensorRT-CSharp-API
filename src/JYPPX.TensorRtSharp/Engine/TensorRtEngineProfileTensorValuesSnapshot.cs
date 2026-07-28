using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Captures copied ICudaEngine profile tensor values for one tensor/profile/selector query.
/// 捕获单个 tensor/profile/selector 查询对应的 ICudaEngine profile tensor values 复制快照。
/// </summary>
public sealed class TensorRtEngineProfileTensorValuesSnapshot
{
    internal TensorRtEngineProfileTensorValuesSnapshot(
        TensorRtApiLine line,
        string tensorName,
        int profileIndex,
        TensorRtOptimizationProfileSelector selector,
        int requestedValueCount,
        bool hasLegacyInt32Values,
        IReadOnlyList<int> legacyInt32Values,
        bool hasValuesV2,
        IReadOnlyList<long> valuesV2,
        IReadOnlyList<string> diagnostics)
    {
        Line = line;
        TensorName = tensorName;
        ProfileIndex = profileIndex;
        Selector = selector;
        RequestedValueCount = requestedValueCount;
        HasLegacyInt32Values = hasLegacyInt32Values;
        LegacyInt32Values = Array.AsReadOnly(ToArray(legacyInt32Values));
        HasValuesV2 = hasValuesV2;
        ValuesV2 = Array.AsReadOnly(ToArray(valuesV2));
        Diagnostics = Array.AsReadOnly(ToArray(diagnostics));
    }

    /// <summary>
    /// Gets the TensorRT API line used for this query.
    /// 获取本次查询使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the engine tensor name.
    /// 获取 engine tensor 名称。
    /// </summary>
    public string TensorName { get; }

    /// <summary>
    /// Gets the optimization profile index.
    /// 获取 optimization profile 索引。
    /// </summary>
    public int ProfileIndex { get; }

    /// <summary>
    /// Gets the min/opt/max selector.
    /// 获取 min/opt/max 选择器。
    /// </summary>
    public TensorRtOptimizationProfileSelector Selector { get; }

    /// <summary>
    /// Gets the caller-requested value count used for the copied-buffer query.
    /// 获取 copied-buffer 查询使用的调用方请求值数量。
    /// </summary>
    public int RequestedValueCount { get; }

    /// <summary>
    /// Gets whether the TensorRT 10 legacy int32 values path produced copied values.
    /// 获取 TensorRT 10 legacy int32 values 路径是否产出了复制值。
    /// </summary>
    public bool HasLegacyInt32Values { get; }

    /// <summary>
    /// Gets copied TensorRT 10 legacy int32 profile tensor values.
    /// 获取复制出的 TensorRT 10 legacy int32 profile tensor values。
    /// </summary>
    public IReadOnlyList<int> LegacyInt32Values { get; }

    /// <summary>
    /// Gets whether the TensorRT 10/11 V2 int64 values path produced copied values.
    /// 获取 TensorRT 10/11 V2 int64 values 路径是否产出了复制值。
    /// </summary>
    public bool HasValuesV2 { get; }

    /// <summary>
    /// Gets copied TensorRT 10/11 V2 int64 profile tensor values.
    /// 获取复制出的 TensorRT 10/11 V2 int64 profile tensor values。
    /// </summary>
    public IReadOnlyList<long> ValuesV2 { get; }

    /// <summary>
    /// Gets whether either copied values path succeeded.
    /// 获取是否至少有一个 copied values 路径成功。
    /// </summary>
    public bool HasAnyValues => HasLegacyInt32Values || HasValuesV2;

    /// <summary>
    /// Gets non-fatal diagnostics collected while building the snapshot.
    /// 获取构建快照时收集到的非致命诊断。
    /// </summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>
    /// Creates a compact one-line diagnostic summary.
    /// 创建紧凑的单行诊断摘要。
    /// </summary>
    public override string ToString()
    {
        return $"{Line}:{TensorName}:profile={ProfileIndex}:selector={Selector}:requested={RequestedValueCount}:legacy={LegacyInt32Values.Count}:v2={ValuesV2.Count}:diagnostics={Diagnostics.Count}";
    }

    private static T[] ToArray<T>(IReadOnlyList<T> values)
    {
        if (values == null)
        {
            return Array.Empty<T>();
        }

        T[] copy = new T[values.Count];
        for (int i = 0; i < values.Count; i++)
        {
            copy[i] = values[i];
        }

        return copy;
    }
}
