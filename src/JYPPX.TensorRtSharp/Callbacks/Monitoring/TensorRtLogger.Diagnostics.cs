using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLogger
{
    /// <summary>
    /// Synchronously emits a diagnostic message through the native logger object.
    /// 通过 native logger 对象同步发送一条诊断消息。
    /// </summary>
    /// <param name="severity">The severity to use. 要使用的严重级别。</param>
    /// <param name="message">The message to copy to native UTF-8 memory for the call. 调用时复制到 native UTF-8 内存的消息。</param>
    /// <returns><c>true</c> when the message was accepted without a managed callback exception; otherwise <c>false</c>. 若消息未触发托管回调异常则返回 <c>true</c>，否则返回 <c>false</c>。</returns>
    /// <remarks>
    /// This method is intended for diagnostics and smoke tests. It does not expose the native logger pointer and does not change logger ownership.
    /// 该方法用于诊断和 smoke 测试；它不会暴露 native logger 指针，也不会改变 logger 所有权。
    /// </remarks>
    public bool EmitDiagnostic(TensorRtLogSeverity severity, string message)
    {
        if (message == null)
        {
            throw new ArgumentNullException(nameof(message));
        }

        ThrowIfDisposed();
        return NativeBridgeApi.EmitLoggerDiagnostic(Line, _handle, severity, message);
    }
}
