using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Thrown when the CUDA bridge reports a non-success status.
/// 当 CUDA bridge 返回非成功状态时引发。
/// </summary>
public sealed class CudaException : NativeBridgeException
{
    /// <summary>
    /// Initializes a CUDA bridge exception from the reported bridge status.
    /// 使用 bridge 返回的状态信息初始化 CUDA 异常。
    /// </summary>
    /// <param name="statusCode">The bridge status code. Bridge 状态码。</param>
    /// <param name="errorCategory">The bridge error category. Bridge 错误分类。</param>
    /// <param name="message">The human-readable error message. 可读错误消息。</param>
    public CudaException(BridgeStatusCode statusCode, BridgeErrorCategory errorCategory, string message)
        : base(statusCode, errorCategory, message)
    {
    }
}
