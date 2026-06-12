namespace JYPPX.TensorRtSharp;

/// <summary>
/// Identifies how TensorRT uses a weight tensor during refitting.
/// 标识 TensorRT 在 refit 过程中如何使用某个权重张量。
/// </summary>
public enum TensorRtWeightsRole
{
    /// <summary>
    /// Kernel weights for convolution-like layers.
    /// 卷积类层使用的 kernel 权重。
    /// </summary>
    Kernel = 0,

    /// <summary>
    /// Bias weights for convolution-like layers.
    /// 卷积类层使用的 bias 权重。
    /// </summary>
    Bias = 1,

    /// <summary>
    /// Shift weights for scale layers.
    /// scale 层使用的 shift 权重。
    /// </summary>
    Shift = 2,

    /// <summary>
    /// Scale weights for scale layers.
    /// scale 层使用的 scale 权重。
    /// </summary>
    Scale = 3,

    /// <summary>
    /// Constant layer weights.
    /// constant 层权重。
    /// </summary>
    Constant = 4,

    /// <summary>
    /// Any other TensorRT weight role.
    /// 其他 TensorRT 权重角色。
    /// </summary>
    Any = 5
}
