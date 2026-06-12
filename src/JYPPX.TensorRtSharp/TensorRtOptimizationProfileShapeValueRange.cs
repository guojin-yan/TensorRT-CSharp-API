using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents the min/opt/max integer value range of a TensorRT shape tensor input.
/// 表示 TensorRT shape tensor 输入的 min/opt/max 整数取值范围。
/// </summary>
public sealed class TensorRtOptimizationProfileShapeValueRange
{
    /// <summary>
    /// Creates a TensorRT optimization profile shape-value range.
    /// 创建 TensorRT optimization profile shape-value 范围对象。
    /// </summary>
    /// <param name="min">Minimum values. 最小取值。</param>
    /// <param name="opt">Optimization target values. 优化目标取值。</param>
    /// <param name="max">Maximum values. 最大取值。</param>
    public TensorRtOptimizationProfileShapeValueRange(IReadOnlyList<int> min, IReadOnlyList<int> opt, IReadOnlyList<int> max)
    {
        Min = min;
        Opt = opt;
        Max = max;
    }

    /// <summary>
    /// Gets the minimum supported shape tensor values.
    /// 获取最小支持的 shape tensor 取值。
    /// </summary>
    public IReadOnlyList<int> Min { get; }

    /// <summary>
    /// Gets the optimization target shape tensor values.
    /// 获取优化目标 shape tensor 取值。
    /// </summary>
    public IReadOnlyList<int> Opt { get; }

    /// <summary>
    /// Gets the maximum supported shape tensor values.
    /// 获取最大支持的 shape tensor 取值。
    /// </summary>
    public IReadOnlyList<int> Max { get; }
}
