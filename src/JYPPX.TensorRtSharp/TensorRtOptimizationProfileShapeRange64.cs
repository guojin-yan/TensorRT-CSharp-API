namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes a TensorRT 11 min/opt/max shape range using 64-bit dimension extents.
/// 使用 64 位维度 extent 描述 TensorRT 11 的 min/opt/max shape 范围。
/// </summary>
public sealed class TensorRtOptimizationProfileShapeRange64
{
    /// <summary>
    /// Creates a new 64-bit optimization profile shape range.
    /// 创建新的 64 位 optimization profile shape 范围。
    /// </summary>
    /// <param name="min">The minimum shape. 最小 shape。</param>
    /// <param name="opt">The optimization target shape. 优化目标 shape。</param>
    /// <param name="max">The maximum shape. 最大 shape。</param>
    public TensorRtOptimizationProfileShapeRange64(TensorRtDims64 min, TensorRtDims64 opt, TensorRtDims64 max)
    {
        Min = min;
        Opt = opt;
        Max = max;
    }

    /// <summary>
    /// Gets the minimum supported shape.
    /// 获取最小支持 shape。
    /// </summary>
    public TensorRtDims64 Min { get; }

    /// <summary>
    /// Gets the optimization target shape.
    /// 获取优化目标 shape。
    /// </summary>
    public TensorRtDims64 Opt { get; }

    /// <summary>
    /// Gets the maximum supported shape.
    /// 获取最大支持 shape。
    /// </summary>
    public TensorRtDims64 Max { get; }

    /// <summary>
    /// Returns a compact diagnostic string.
    /// 返回紧凑的诊断字符串。
    /// </summary>
    public override string ToString()
    {
        return $"min={Min}, opt={Opt}, max={Max}";
    }
}
