namespace JYPPX.TensorRtSharp.Shared.Interop;

/// <summary>
/// Broad source categories for bridge error messages.
/// bridge 错误消息的宽泛来源类别。
/// </summary>
public enum BridgeErrorCategory
{
    /// <summary>
    /// No specific error category was reported. 未报告具体错误类别。
    /// </summary>
    None = 0,
    /// <summary>
    /// The error originated from common bridge code. 错误来自通用 bridge 代码。
    /// </summary>
    Common = 1,
    /// <summary>
    /// The error originated from CUDA-related code. 错误来自 CUDA 相关代码。
    /// </summary>
    Cuda = 2,
    /// <summary>
    /// The error originated from TensorRT-related code. 错误来自 TensorRT 相关代码。
    /// </summary>
    TensorRt = 3,
    /// <summary>
    /// The error originated from file or I/O handling. 错误来自文件或 I/O 处理。
    /// </summary>
    Io = 4
}
