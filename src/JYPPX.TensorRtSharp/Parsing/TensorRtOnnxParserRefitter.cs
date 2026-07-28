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
public sealed class TensorRtOnnxParserRefitter : IDisposable
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
    /// Gets the number of parser-refitter errors currently reported by TensorRT.
    /// 获取 TensorRT 当前报告的 parser-refitter 错误数量。
    /// </summary>
    public int ErrorCount => NativeBridgeApi.GetOnnxParserRefitterErrorCount(Line, _handle);

    /// <summary>
    /// Refits the target engine from serialized ONNX model bytes.
    /// 使用已序列化 ONNX 模型字节重整目标 engine。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化 ONNX 模型字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT reports the refit succeeded. TensorRT 报告重整成功时返回 <c>true</c>。</returns>
    /// <remarks>
    /// The model buffer is pinned only for the native call; TensorRT parser-refitter diagnostics remain available through copied diagnostic APIs.
    /// 模型缓冲区仅在 native 调用期间短期 pin；TensorRT parser-refitter 诊断仍通过复制型诊断 API 获取。
    /// </remarks>
    public bool RefitFromBytes(byte[] modelData, string? modelPath = null)
    {
        return NativeBridgeApi.RefitOnnxParserRefitterFromBytes(Line, _handle, modelData, modelPath);
    }

    /// <summary>
    /// Refits the target engine from a serialized ONNX model byte-array segment.
    /// 使用托管字节数组片段中的 ONNX 模型重整目标 engine。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model byte segment. 已序列化 ONNX 模型字节片段。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT reports the refit succeeded. TensorRT 报告重整成功时返回 <c>true</c>。</returns>
    public bool RefitFromBytes(ArraySegment<byte> modelData, string? modelPath = null)
    {
        return RefitFromBytes(CopyModelSegment(modelData, nameof(modelData)), modelPath);
    }

#if NETCOREAPP3_1_OR_GREATER || NET5_0_OR_GREATER || NET6_0_OR_GREATER || NET7_0_OR_GREATER || NET8_0_OR_GREATER || NET9_0_OR_GREATER || NET10_0_OR_GREATER
    /// <summary>
    /// Refits the target engine from serialized ONNX model bytes in a read-only span.
    /// 使用只读 span 中的 ONNX 模型字节重整目标 engine。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化 ONNX 模型字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT reports the refit succeeded. TensorRT 报告重整成功时返回 <c>true</c>。</returns>
    public bool RefitFromBytes(ReadOnlySpan<byte> modelData, string? modelPath = null)
    {
        return RefitFromBytes(modelData.ToArray(), modelPath);
    }

#endif
    /// <summary>
    /// Refits the target engine from serialized ONNX model bytes read from a stream.
    /// 使用从 stream 读取的 ONNX 模型字节重整目标 engine。
    /// </summary>
    /// <param name="modelStream">Readable stream containing serialized ONNX model bytes. 包含已序列化 ONNX 模型字节的可读 stream。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT reports the refit succeeded. TensorRT 报告重整成功时返回 <c>true</c>。</returns>
    public bool RefitFromBytes(Stream modelStream, string? modelPath = null)
    {
        return RefitFromBytes(CopyModelStream(modelStream, nameof(modelStream)), modelPath);
    }

    /// <summary>
    /// Refits the target engine from an ONNX model file.
    /// 使用 ONNX 模型文件重整目标 engine。
    /// </summary>
    /// <param name="filePath">The ONNX model path. ONNX 模型路径。</param>
    /// <returns><c>true</c> when TensorRT reports the refit succeeded. TensorRT 报告重整成功时返回 <c>true</c>。</returns>
    public bool RefitFromFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("ONNX model file was not found.", filePath);
        }

        return NativeBridgeApi.RefitOnnxParserRefitterFromFile(Line, _handle, filePath);
    }

    /// <summary>
    /// Loads serialized ONNX model-proto bytes into the TensorRT 11 parser-refitter without refitting immediately.
    /// 将已序列化 ONNX model proto 字节加载到 TensorRT 11 parser-refitter，但暂不立即重整。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model-proto bytes. 已序列化 ONNX model proto 字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT accepts the model proto. TensorRT 接受 model proto 时返回 <c>true</c>。</returns>
    public bool LoadModelProto(byte[] modelData, string? modelPath = null)
    {
        return NativeBridgeApi.LoadOnnxParserRefitterModelProto(Line, _handle, modelData, modelPath);
    }

    /// <summary>
    /// Loads serialized ONNX model-proto bytes from a byte-array segment into the TensorRT 11 parser-refitter.
    /// 从托管字节数组片段加载 ONNX model proto 到 TensorRT 11 parser-refitter。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model-proto byte segment. 已序列化 ONNX model proto 字节片段。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT accepts the model proto. TensorRT 接受 model proto 时返回 <c>true</c>。</returns>
    public bool LoadModelProto(ArraySegment<byte> modelData, string? modelPath = null)
    {
        return LoadModelProto(CopyModelSegment(modelData, nameof(modelData)), modelPath);
    }

#if NETCOREAPP3_1_OR_GREATER || NET5_0_OR_GREATER || NET6_0_OR_GREATER || NET7_0_OR_GREATER || NET8_0_OR_GREATER || NET9_0_OR_GREATER || NET10_0_OR_GREATER
    /// <summary>
    /// Loads serialized ONNX model-proto bytes from a read-only span into the TensorRT 11 parser-refitter.
    /// 从只读 span 加载 ONNX model proto 到 TensorRT 11 parser-refitter。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model-proto bytes. 已序列化 ONNX model proto 字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT accepts the model proto. TensorRT 接受 model proto 时返回 <c>true</c>。</returns>
    public bool LoadModelProto(ReadOnlySpan<byte> modelData, string? modelPath = null)
    {
        return LoadModelProto(modelData.ToArray(), modelPath);
    }

#endif
    /// <summary>
    /// Loads serialized ONNX model-proto bytes from a stream into the TensorRT 11 parser-refitter.
    /// 从 stream 加载 ONNX model proto 到 TensorRT 11 parser-refitter。
    /// </summary>
    /// <param name="modelStream">Readable stream containing serialized ONNX model-proto bytes. 包含已序列化 ONNX model proto 字节的可读 stream。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT accepts the model proto. TensorRT 接受 model proto 时返回 <c>true</c>。</returns>
    public bool LoadModelProto(Stream modelStream, string? modelPath = null)
    {
        return LoadModelProto(CopyModelStream(modelStream, nameof(modelStream)), modelPath);
    }

    /// <summary>
    /// Loads an external ONNX initializer into the TensorRT 11 parser-refitter.
    /// 将外部 ONNX initializer 加载到 TensorRT 11 parser-refitter。
    /// </summary>
    /// <param name="name">Initializer name. initializer 名称。</param>
    /// <param name="data">Initializer data. initializer 数据。</param>
    /// <returns><c>true</c> when TensorRT accepts the initializer. TensorRT 接受 initializer 时返回 <c>true</c>。</returns>
    /// <remarks>
    /// The initializer data is copied into an owned managed array and pinned until this parser-refitter is disposed.
    /// initializer 数据会复制到当前 parser-refitter 拥有的托管数组，并 pin 到 parser-refitter 释放为止。
    /// </remarks>
    public bool LoadInitializer(string name, byte[] data)
    {
        return _initializerPins.LoadOrReplace(
            name,
            data,
            (pointer, size) => NativeBridgeApi.LoadOnnxParserRefitterInitializer(Line, _handle, name, pointer, size));
    }

    /// <summary>
    /// Loads an external ONNX initializer from a byte-array segment into the TensorRT 11 parser-refitter.
    /// 从托管字节数组片段加载外部 ONNX initializer 到 TensorRT 11 parser-refitter。
    /// </summary>
    /// <param name="name">Initializer name. initializer 名称。</param>
    /// <param name="data">Initializer data segment. initializer 数据片段。</param>
    /// <returns><c>true</c> when TensorRT accepts the initializer. TensorRT 接受 initializer 时返回 <c>true</c>。</returns>
    public bool LoadInitializer(string name, ArraySegment<byte> data)
    {
        return LoadInitializer(name, CopyModelSegment(data, nameof(data)));
    }

#if NETCOREAPP3_1_OR_GREATER || NET5_0_OR_GREATER || NET6_0_OR_GREATER || NET7_0_OR_GREATER || NET8_0_OR_GREATER || NET9_0_OR_GREATER || NET10_0_OR_GREATER
    /// <summary>
    /// Loads an external ONNX initializer from a read-only span into the TensorRT 11 parser-refitter.
    /// 从只读 span 加载外部 ONNX initializer 到 TensorRT 11 parser-refitter。
    /// </summary>
    /// <param name="name">Initializer name. initializer 名称。</param>
    /// <param name="data">Initializer data. initializer 数据。</param>
    /// <returns><c>true</c> when TensorRT accepts the initializer. TensorRT 接受 initializer 时返回 <c>true</c>。</returns>
    public bool LoadInitializer(string name, ReadOnlySpan<byte> data)
    {
        return LoadInitializer(name, data.ToArray());
    }

#endif
    /// <summary>
    /// Loads an external ONNX initializer from a stream into the TensorRT 11 parser-refitter.
    /// 从 stream 加载外部 ONNX initializer 到 TensorRT 11 parser-refitter。
    /// </summary>
    /// <param name="name">Initializer name. initializer 名称。</param>
    /// <param name="dataStream">Readable stream containing initializer data. 包含 initializer 数据的可读 stream。</param>
    /// <returns><c>true</c> when TensorRT accepts the initializer. TensorRT 接受 initializer 时返回 <c>true</c>。</returns>
    public bool LoadInitializer(string name, Stream dataStream)
    {
        return LoadInitializer(name, CopyModelStream(dataStream, nameof(dataStream)));
    }

    /// <summary>
    /// Refits the model proto previously loaded through <see cref="LoadModelProto(byte[], string?)"/>.
    /// 重整先前通过 <see cref="LoadModelProto(byte[], string?)"/> 加载的 model proto。
    /// </summary>
    /// <returns><c>true</c> when TensorRT reports the refit succeeded. TensorRT 报告重整成功时返回 <c>true</c>。</returns>
    public bool RefitLoadedModel()
    {
        return NativeBridgeApi.RefitOnnxParserRefitterLoadedModel(Line, _handle);
    }

    /// <summary>
    /// Gets one parser-refitter error by index.
    /// 按索引获取一个 parser-refitter 错误。
    /// </summary>
    /// <param name="index">Zero-based parser-refitter error index. 从零开始的 parser-refitter 错误索引。</param>
    /// <returns>The copied parser-refitter error information. 已复制的 parser-refitter 错误信息。</returns>
    public TensorRtParserErrorInfo GetError(int index)
    {
        return NativeBridgeApi.GetOnnxParserRefitterError(Line, _handle, index);
    }

    /// <summary>
    /// Gets one detailed parser-refitter diagnostic by index.
    /// 按索引获取一条详细 parser-refitter 诊断信息。
    /// </summary>
    /// <param name="index">Zero-based parser-refitter error index. 从零开始的 parser-refitter 错误索引。</param>
    /// <returns>The copied detailed parser-refitter diagnostic. 已复制的详细 parser-refitter 诊断信息。</returns>
    /// <remarks>
    /// Variable-length strings are copied through caller-owned buffers; no borrowed native parser-error pointer is exposed.
    /// 可变长度字符串通过调用方缓冲区复制；不会暴露 borrowed native parser-error 指针。
    /// </remarks>
    public TensorRtOnnxParserDiagnostic GetDiagnostic(int index)
    {
        return NativeBridgeApi.GetOnnxParserRefitterDiagnostic(Line, _handle, index);
    }

    /// <summary>
    /// Gets all parser-refitter errors currently reported by TensorRT.
    /// 获取 TensorRT 当前报告的全部 parser-refitter 错误。
    /// </summary>
    /// <returns>The copied parser-refitter errors. 已复制的 parser-refitter 错误列表。</returns>
    public IReadOnlyList<TensorRtParserErrorInfo> GetErrors()
    {
        int count = ErrorCount;
        List<TensorRtParserErrorInfo> errors = new List<TensorRtParserErrorInfo>(count);
        for (int index = 0; index < count; index++)
        {
            errors.Add(GetError(index));
        }

        return errors;
    }

    /// <summary>
    /// Gets all detailed parser-refitter diagnostics currently reported by TensorRT.
    /// 获取 TensorRT 当前报告的全部详细 parser-refitter 诊断信息。
    /// </summary>
    /// <returns>The copied detailed parser-refitter diagnostics. 已复制的详细 parser-refitter 诊断信息列表。</returns>
    public IReadOnlyList<TensorRtOnnxParserDiagnostic> GetDiagnostics()
    {
        int count = ErrorCount;
        List<TensorRtOnnxParserDiagnostic> diagnostics = new List<TensorRtOnnxParserDiagnostic>(count);
        for (int index = 0; index < count; index++)
        {
            diagnostics.Add(GetDiagnostic(index));
        }

        return diagnostics;
    }

    /// <summary>
    /// Captures copied parser-refitter diagnostics without exposing TensorRT-owned parser-error pointers.
    /// 捕获已复制的 parser-refitter 诊断信息，不暴露 TensorRT 拥有的 parser-error 指针。
    /// </summary>
    /// <returns>A pointer-free parser-refitter diagnostic snapshot. 无指针逃逸的 parser-refitter 诊断快照。</returns>
    public TensorRtOnnxParserRefitterDiagnosticSnapshot GetDiagnosticSnapshot()
    {
        IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics = GetDiagnostics();
        return new TensorRtOnnxParserRefitterDiagnosticSnapshot(
            Line,
            ErrorCount,
            diagnostics,
            BuildDiagnosticSummary(diagnostics));
    }

    /// <summary>
    /// Clears parser-refitter errors stored by TensorRT.
    /// 清理 TensorRT 保存的 parser-refitter 错误。
    /// </summary>
    public void ClearErrors()
    {
        NativeBridgeApi.ClearOnnxParserRefitterErrors(Line, _handle);
    }

    /// <summary>
    /// Builds a readable parser-refitter diagnostic summary for logs and deployment troubleshooting.
    /// 为日志和部署排障生成可读的 parser-refitter 诊断摘要。
    /// </summary>
    /// <returns>A parser-refitter diagnostic summary string. parser-refitter 诊断摘要字符串。</returns>
    public string GetDiagnosticSummary()
    {
        return BuildDiagnosticSummary(GetDiagnostics());
    }

    private static string BuildDiagnosticSummary(IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)
    {
        if (diagnostics.Count == 0)
        {
            return "ONNX parser refitter reported no errors.";
        }

        StringBuilder builder = new StringBuilder();
        builder.Append("ONNX parser refitter errors: ");
        builder.Append(diagnostics.Count);

        for (int index = 0; index < diagnostics.Count; index++)
        {
            TensorRtOnnxParserDiagnostic error = diagnostics[index];
            builder.AppendLine();
            builder.Append('#');
            builder.Append(error.Index);
            builder.Append(" code=");
            builder.Append(error.Code);
            if (error.Line >= 0)
            {
                builder.Append(" line=");
                builder.Append(error.Line);
            }

            if (!string.IsNullOrWhiteSpace(error.File))
            {
                builder.Append(" file=");
                builder.Append(error.File);
            }

            if (!string.IsNullOrWhiteSpace(error.NodeName))
            {
                builder.Append(" node=");
                builder.Append(error.NodeName);
            }

            if (!string.IsNullOrWhiteSpace(error.NodeOperator))
            {
                builder.Append(" op=");
                builder.Append(error.NodeOperator);
            }

            if (error.LocalFunctionStack.Count > 0)
            {
                builder.Append(" stack=");
                builder.Append(string.Join(">", error.LocalFunctionStack));
            }

            if (!string.IsNullOrWhiteSpace(error.Description))
            {
                builder.Append(": ");
                builder.Append(error.Description);
            }
        }

        return builder.ToString();
    }

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
