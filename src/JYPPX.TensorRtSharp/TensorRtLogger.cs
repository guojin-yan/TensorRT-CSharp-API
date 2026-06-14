using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT logger.
/// TensorRT logger 的托管封装。
/// </summary>
public sealed class TensorRtLogger : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    /// <summary>
    /// Creates a TensorRT logger for one TensorRT API line.
    /// 为一个 TensorRT API line 创建 TensorRT logger。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    public TensorRtLogger(TensorRtApiLine line)
    {
        NativeBridgeLoader.EnsureInitialized();
        Line = line;
        _handle = NativeBridgeApi.CreateLogger(line);
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this logger.
    /// 获取当前 logger 使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Releases the TensorRT logger handle.
    /// 释放 TensorRT logger 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
