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
/// Parses ONNX models into a TensorRT network definition.
/// 将 ONNX 模型解析到 TensorRT network definition。
/// </summary>
public sealed partial class TensorRtOnnxParser : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;
    private readonly TensorRtLogger? _loggerKeepAlive;
    private readonly TensorRtPinnedInitializerSet _initializerPins = new TensorRtPinnedInitializerSet();
    private bool _disposed;

    /// <summary>
    /// Creates an ONNX parser for the specified TensorRT logger and network.
    /// 使用指定的 TensorRT logger 和 network 创建 ONNX parser。
    /// </summary>
    /// <param name="logger">The TensorRT logger used by the parser. Parser 使用的 TensorRT logger。</param>
    /// <param name="network">The target TensorRT network definition. 目标 TensorRT network definition。</param>
    /// <remarks>
    /// TensorRT borrows the logger pointer. This parser keeps the managed logger attached until the parser is disposed.
    /// TensorRT 只借用 logger 指针；当前 parser 会保持托管 logger 借用关系直到 parser 释放。
    /// </remarks>
    public TensorRtOnnxParser(TensorRtLogger logger, TensorRtNetworkDefinition network)
    {
        if (logger == null)
        {
            throw new ArgumentNullException(nameof(logger));
        }

        if (network == null)
        {
            throw new ArgumentNullException(nameof(network));
        }

        if (logger.Line != network.Line)
        {
            throw new ArgumentException("Logger and network must belong to the same TensorRT API line.");
        }

        Line = logger.Line;
        _loggerKeepAlive = logger;
        _loggerKeepAlive.AttachBorrower(Line);
        try
        {
            _handle = NativeBridgeApi.CreateOnnxParser(Line, logger.Handle, network.Handle);
        }
        catch
        {
            _loggerKeepAlive.DetachBorrower();
            throw;
        }
    }

    internal TensorRtOnnxParser(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this parser.
    /// 获取当前 parser 使用的 TensorRT API 版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the number of parser errors currently reported by TensorRT.
    /// 获取 TensorRT 当前报告的 parser 错误数量。
    /// </summary>
    public int ErrorCount => NativeBridgeApi.GetOnnxParserErrorCount(Line, _handle);

    /// <summary>
    /// Gets or sets the ONNX parser flag bitmask.
    /// 获取或设置 ONNX parser 标志位掩码。
    /// </summary>
    public TensorRtOnnxParserFlags Flags
    {
        get => (TensorRtOnnxParserFlags)NativeBridgeApi.GetOnnxParserFlags(Line, _handle);
        set
        {
            ValidateParserFlags(value);
            NativeBridgeApi.SetOnnxParserFlags(Line, _handle, (uint)value);
        }
    }

    /// <summary>
    /// Releases the native ONNX parser handle.
    /// 释放原生 ONNX parser 句柄。
    /// </summary>
    public void Dispose()
    {
        SafeTensorRtObjectHandleLease? builderConfigLease;
        lock (_builderConfigAttachmentLock)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _handle.Dispose();
            builderConfigLease = _builderConfigLease;
            _builderConfigLease = null;
        }

        builderConfigLease?.Dispose();
        _initializerPins.Dispose();
        GC.KeepAlive(_loggerKeepAlive);
        _loggerKeepAlive?.DetachBorrower();
        GC.SuppressFinalize(this);
    }

    private void ValidateParserFlags(TensorRtOnnxParserFlags flags)
    {
        TensorRtOnnxParserFlags trt11OnlyFlags =
            TensorRtOnnxParserFlags.ReportCapabilityDla |
            TensorRtOnnxParserFlags.EnablePluginOverride |
            TensorRtOnnxParserFlags.AdjustForDla;
        TensorRtOnnxParserFlags knownFlags =
            TensorRtOnnxParserFlags.NativeInstanceNormalization |
            TensorRtOnnxParserFlags.EnableUInt8AndAsymmetricQuantizationDla |
            trt11OnlyFlags;
        if ((flags & ~knownFlags) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(flags), flags, "Unknown ONNX parser flag bits were specified.");
        }

        if (Line == TensorRtApiLine.TensorRt8 &&
            (flags & TensorRtOnnxParserFlags.EnableUInt8AndAsymmetricQuantizationDla) != 0)
        {
            throw new NotSupportedException("TensorRT 8 ONNX parser does not expose EnableUInt8AndAsymmetricQuantizationDla.");
        }

        if (Line != TensorRtApiLine.TensorRt11 && (flags & trt11OnlyFlags) != 0)
        {
            throw new NotSupportedException("ReportCapabilityDla, EnablePluginOverride, and AdjustForDla require TensorRT 11.");
        }
    }

    private void ValidateParserFlag(TensorRtOnnxParserFlag flag)
    {
        if (flag < TensorRtOnnxParserFlag.NativeInstanceNormalization || flag > TensorRtOnnxParserFlag.AdjustForDla)
        {
            throw new ArgumentOutOfRangeException(nameof(flag), flag, "Unknown ONNX parser flag was specified.");
        }

        if (Line == TensorRtApiLine.TensorRt8 && flag == TensorRtOnnxParserFlag.EnableUInt8AndAsymmetricQuantizationDla)
        {
            throw new NotSupportedException("TensorRT 8 ONNX parser does not expose EnableUInt8AndAsymmetricQuantizationDla.");
        }

        if (Line != TensorRtApiLine.TensorRt11 && flag >= TensorRtOnnxParserFlag.ReportCapabilityDla)
        {
            throw new NotSupportedException("ReportCapabilityDla, EnablePluginOverride, and AdjustForDla require TensorRT 11.");
        }
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
