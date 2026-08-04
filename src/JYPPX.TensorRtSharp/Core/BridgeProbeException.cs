using System;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Thrown when the bridge reports a non-success status for a probe operation.
/// 当 bridge 在探测操作中返回非成功状态时引发。
/// </summary>
public sealed class BridgeProbeException : TensorRtException
{
    /// <summary>
    /// Initializes a probe exception from bridge status information.
    /// 使用 bridge 状态信息初始化探测异常。
    /// </summary>
    /// <param name="statusCode">The bridge status code. Bridge 状态码。</param>
    /// <param name="errorCategory">The bridge error category. Bridge 错误分类。</param>
    /// <param name="message">The human-readable error message. 可读错误消息。</param>
    public BridgeProbeException(BridgeStatusCode statusCode, BridgeErrorCategory errorCategory, string message)
        : base(statusCode, errorCategory, message)
    {
    }
}
