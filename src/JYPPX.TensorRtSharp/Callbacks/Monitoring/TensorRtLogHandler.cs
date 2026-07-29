namespace JYPPX.TensorRtSharp;

/// <summary>
/// Receives TensorRT logger messages copied across the native boundary.
/// 接收从 native 边界复制过来的 TensorRT logger 消息。
/// </summary>
/// <param name="severity">The TensorRT log severity. TensorRT 日志严重级别。</param>
/// <param name="message">The copied UTF-8 log message. 复制后的 UTF-8 日志消息。</param>
public delegate void TensorRtLogHandler(TensorRtLogSeverity severity, string message);
