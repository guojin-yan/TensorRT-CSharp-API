using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents TensorRT TensorRtResizeMode values.
/// 表示 TensorRT TensorRtResizeMode 枚举值。
/// </summary>
public enum TensorRtResizeMode
{
    /// <summary>
    /// Represents the Nearest value of TensorRtResizeMode.
    /// 表示 TensorRtResizeMode 的 Nearest 取值。
    /// </summary>
    Nearest = 0,
    /// <summary>
    /// Represents the Linear value of TensorRtResizeMode.
    /// 表示 TensorRtResizeMode 的 Linear 取值。
    /// </summary>
    Linear = 1,
    /// <summary>
    /// Represents the Cubic value of TensorRtResizeMode.
    /// 表示 TensorRtResizeMode 的 Cubic 取值。
    /// </summary>
    Cubic = 2
}
/// <summary>
/// Selects the TensorRT interpolation algorithm for resize and grid-sample style layers.
/// 选择 TensorRT resize / grid-sample 等层使用的插值算法。
/// </summary>
public enum TensorRtInterpolationMode
{
    /// <summary>
    /// Nearest-neighbor interpolation.
    /// 最近邻插值。
    /// </summary>
    Nearest = 0,

    /// <summary>
    /// Linear interpolation; TensorRT maps this to linear, bilinear, or trilinear by rank.
    /// 线性插值；TensorRT 会按张量维度映射为 linear、bilinear 或 trilinear。
    /// </summary>
    Linear = 1,

    /// <summary>
    /// Cubic interpolation.
    /// 三次插值。
    /// </summary>
    Cubic = 2
}

/// <summary>
/// Represents TensorRT TensorRtResizeCoordinateTransformation values.
/// 表示 TensorRT TensorRtResizeCoordinateTransformation 枚举值。
/// </summary>
public enum TensorRtResizeCoordinateTransformation
{
    /// <summary>
    /// Represents the AlignCorners value of TensorRtResizeCoordinateTransformation.
    /// 表示 TensorRtResizeCoordinateTransformation 的 AlignCorners 取值。
    /// </summary>
    AlignCorners = 0,
    /// <summary>
    /// Represents the Asymmetric value of TensorRtResizeCoordinateTransformation.
    /// 表示 TensorRtResizeCoordinateTransformation 的 Asymmetric 取值。
    /// </summary>
    Asymmetric = 1,
    /// <summary>
    /// Represents the HalfPixel value of TensorRtResizeCoordinateTransformation.
    /// 表示 TensorRtResizeCoordinateTransformation 的 HalfPixel 取值。
    /// </summary>
    HalfPixel = 2
}

/// <summary>
/// Represents TensorRT TensorRtResizeSelector values.
/// 表示 TensorRT TensorRtResizeSelector 枚举值。
/// </summary>
public enum TensorRtResizeSelector
{
    /// <summary>
    /// Represents the Formula value of TensorRtResizeSelector.
    /// 表示 TensorRtResizeSelector 的 Formula 取值。
    /// </summary>
    Formula = 0,
    /// <summary>
    /// Represents the Upper value of TensorRtResizeSelector.
    /// 表示 TensorRtResizeSelector 的 Upper 取值。
    /// </summary>
    Upper = 1
}

/// <summary>
/// Represents TensorRT TensorRtResizeRoundMode values.
/// 表示 TensorRT TensorRtResizeRoundMode 枚举值。
/// </summary>
public enum TensorRtResizeRoundMode
{
    /// <summary>
    /// Represents the HalfUp value of TensorRtResizeRoundMode.
    /// 表示 TensorRtResizeRoundMode 的 HalfUp 取值。
    /// </summary>
    HalfUp = 0,
    /// <summary>
    /// Represents the HalfDown value of TensorRtResizeRoundMode.
    /// 表示 TensorRtResizeRoundMode 的 HalfDown 取值。
    /// </summary>
    HalfDown = 1,
    /// <summary>
    /// Represents the Floor value of TensorRtResizeRoundMode.
    /// 表示 TensorRtResizeRoundMode 的 Floor 取值。
    /// </summary>
    Floor = 2,
    /// <summary>
    /// Represents the Ceil value of TensorRtResizeRoundMode.
    /// 表示 TensorRtResizeRoundMode 的 Ceil 取值。
    /// </summary>
    Ceil = 3
}
