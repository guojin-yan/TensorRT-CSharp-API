namespace JYPPX.TensorRtSharp;

/// <summary>
/// Severity values used by TensorRT logger callbacks.
/// TensorRT logger 回调使用的严重级别。
/// </summary>
public enum TensorRtLogSeverity
{
    /// <summary>
    /// An internal TensorRT error. TensorRT 内部错误。
    /// </summary>
    InternalError = 0,

    /// <summary>
    /// A TensorRT error. TensorRT 错误。
    /// </summary>
    Error = 1,

    /// <summary>
    /// A TensorRT warning. TensorRT 警告。
    /// </summary>
    Warning = 2,

    /// <summary>
    /// Informational TensorRT output. TensorRT 信息输出。
    /// </summary>
    Info = 3,

    /// <summary>
    /// Verbose TensorRT output. TensorRT 详细输出。
    /// </summary>
    Verbose = 4
}
