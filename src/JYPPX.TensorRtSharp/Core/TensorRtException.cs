using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Base exception for TensorRT bridge operations.
/// TensorRT bridge 操作的基础异常类型。
/// </summary>
public class TensorRtException : NativeBridgeException
{
    /// <summary>
    /// Initializes a TensorRT bridge exception from bridge status information.
    /// 使用 bridge 状态信息初始化 TensorRT 异常。
    /// </summary>
    /// <param name="statusCode">The bridge status code. Bridge 状态码。</param>
    /// <param name="errorCategory">The bridge error category. Bridge 错误分类。</param>
    /// <param name="message">The human-readable error message. 可读错误消息。</param>
    public TensorRtException(BridgeStatusCode statusCode, BridgeErrorCategory errorCategory, string message)
        : base(statusCode, errorCategory, message)
    {
    }
}
