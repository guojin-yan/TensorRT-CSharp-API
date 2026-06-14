namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents the min/opt/max dimension range of a TensorRT optimization profile input.
/// 表示 TensorRT 优化 profile 输入张量的 min/opt/max 维度范围。
/// </summary>
public sealed class TensorRtOptimizationProfileShapeRange
{
    /// <summary>
    /// Creates a TensorRT optimization profile dimension range.
    /// 创建 TensorRT optimization profile 维度范围对象。
    /// </summary>
    /// <param name="min">Minimum supported shape. 最小支持形状。</param>
    /// <param name="opt">Optimization target shape. 优化目标形状。</param>
    /// <param name="max">Maximum supported shape. 最大支持形状。</param>
    public TensorRtOptimizationProfileShapeRange(TensorRtDims min, TensorRtDims opt, TensorRtDims max)
    {
        Min = min;
        Opt = opt;
        Max = max;
    }

    /// <summary>
    /// Gets the minimum supported shape.
    /// 获取最小支持形状。
    /// </summary>
    public TensorRtDims Min { get; }

    /// <summary>
    /// Gets the optimization target shape.
    /// 获取优化目标形状。
    /// </summary>
    public TensorRtDims Opt { get; }

    /// <summary>
    /// Gets the maximum supported shape.
    /// 获取最大支持形状。
    /// </summary>
    public TensorRtDims Max { get; }

    /// <summary>
    /// Formats the min/opt/max shape range for diagnostics.
    /// 将 min/opt/max shape 范围格式化为便于诊断的字符串。
    /// </summary>
    /// <returns>A display string containing minimum, optimal, and maximum shapes. 包含最小、最优和最大形状的显示字符串。</returns>
    public override string ToString()
    {
        return $"Min={Min}, Opt={Opt}, Max={Max}";
    }
}
