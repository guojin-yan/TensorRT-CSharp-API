using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Provides a safe managed wrapper over TensorRT ONNX parser-refitter diagnostics.
/// 提供 TensorRT ONNX parser-refitter 诊断信息的安全托管封装。
/// </summary>
public sealed partial class TensorRtOnnxParserRefitter : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;
    private readonly TensorRtRefitter _refitterKeepAlive;
    private readonly TensorRtLogger _loggerKeepAlive;
    private readonly TensorRtPinnedInitializerSet _initializerPins = new TensorRtPinnedInitializerSet();
    private bool _disposed;

    internal TensorRtOnnxParserRefitter(TensorRtRefitter refitter, TensorRtLogger logger)
    {
        if (refitter == null)
        {
            throw new ArgumentNullException(nameof(refitter));
        }

        if (logger == null)
        {
            throw new ArgumentNullException(nameof(logger));
        }

        if (refitter.Line != logger.Line)
        {
            throw new ArgumentException("Refitter and logger must belong to the same TensorRT API line.");
        }

        Line = refitter.Line;
        _refitterKeepAlive = refitter;
        _loggerKeepAlive = logger;

        _refitterKeepAlive.AttachBorrower(Line);
        _loggerKeepAlive.AttachBorrower(Line);
        try
        {
            _handle = NativeBridgeApi.CreateOnnxParserRefitter(Line, refitter.Handle, logger.Handle);
        }
        catch
        {
            _loggerKeepAlive.DetachBorrower();
            _refitterKeepAlive.DetachBorrower();
            throw;
        }
    }

    /// <summary>
    /// Gets the TensorRT API line used by this parser-refitter.
    /// 获取当前 parser-refitter 使用的 TensorRT API 版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Releases the native ONNX parser-refitter handle.
    /// 释放原生 ONNX parser-refitter 句柄。
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _handle.Dispose();
        _initializerPins.Dispose();
        GC.KeepAlive(_loggerKeepAlive);
        GC.KeepAlive(_refitterKeepAlive);
        _loggerKeepAlive.DetachBorrower();
        _refitterKeepAlive.DetachBorrower();
        GC.SuppressFinalize(this);
    }

    private static byte[] CopyModelSegment(ArraySegment<byte> modelData, string argumentName)
    {
        if (modelData.Array == null)
        {
            throw new ArgumentException("ONNX model segment must reference a byte array.", argumentName);
        }

        byte[] buffer = new byte[modelData.Count];
        Buffer.BlockCopy(modelData.Array, modelData.Offset, buffer, 0, modelData.Count);
        return buffer;
    }

    private static byte[] CopyModelStream(Stream modelStream, string argumentName)
    {
        if (modelStream == null)
        {
            throw new ArgumentNullException(argumentName);
        }

        if (!modelStream.CanRead)
        {
            throw new ArgumentException("ONNX model stream must be readable.", argumentName);
        }

        using MemoryStream copy = new MemoryStream();
        modelStream.CopyTo(copy);
        return copy.ToArray();
    }
}
