using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents the min/opt/max 64-bit value range of a TensorRT shape tensor input.
/// 表示 TensorRT shape tensor 输入的 min/opt/max 64 位整数取值范围。
/// </summary>
public sealed class TensorRtOptimizationProfileShapeValueRangeV2
{
    /// <summary>
    /// Creates a TensorRT optimization profile 64-bit shape-value range.
    /// 创建 TensorRT optimization profile 的 64 位 shape-value 范围对象。
    /// </summary>
    /// <param name="min">Minimum values. 最小取值。</param>
    /// <param name="opt">Optimization target values. 优化目标取值。</param>
    /// <param name="max">Maximum values. 最大取值。</param>
    public TensorRtOptimizationProfileShapeValueRangeV2(IReadOnlyList<long> min, IReadOnlyList<long> opt, IReadOnlyList<long> max)
    {
        Min = min;
        Opt = opt;
        Max = max;
    }

    /// <summary>
    /// Gets the minimum supported 64-bit shape tensor values.
    /// 获取最小支持的 64 位 shape tensor 取值。
    /// </summary>
    public IReadOnlyList<long> Min { get; }

    /// <summary>
    /// Gets the optimization target 64-bit shape tensor values.
    /// 获取优化目标的 64 位 shape tensor 取值。
    /// </summary>
    public IReadOnlyList<long> Opt { get; }

    /// <summary>
    /// Gets the maximum supported 64-bit shape tensor values.
    /// 获取最大支持的 64 位 shape tensor 取值。
    /// </summary>
    public IReadOnlyList<long> Max { get; }
}
