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
    /// Parses an ONNX model from a file path into the target network.
    /// 从文件路径解析 ONNX 模型到目标 network。
    /// </summary>
    /// <param name="filePath">The ONNX model path. ONNX 模型路径。</param>
    /// <param name="verbosity">TensorRT parser verbosity level. TensorRT parser 日志详细级别。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool ParseFromFile(string filePath, int verbosity = 1)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("ONNX model file was not found.", filePath);
        }

        return NativeBridgeApi.ParseOnnxFromFile(Line, _handle, filePath, verbosity);
    }

    /// <summary>
    /// Parses ONNX model bytes into the target network.
    /// 将 ONNX 模型字节解析到目标 network。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化的 ONNX 模型字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool Parse(byte[] modelData, string? modelPath = null)
    {
        return NativeBridgeApi.ParseOnnxFromMemory(Line, _handle, modelData, modelPath);
    }

    /// <summary>
    /// Parses ONNX model bytes from a managed byte-array segment into the target network.
    /// 将托管字节数组片段中的 ONNX 模型解析到目标 network。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model byte segment. 已序列化 ONNX 模型字节片段。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    /// <remarks>
    /// The segment is copied into an exact managed byte array before native interop.
    /// 调用 native interop 前会将片段复制为精确长度的托管字节数组。
    /// </remarks>
    public bool Parse(ArraySegment<byte> modelData, string? modelPath = null)
    {
        return Parse(CopyModelSegment(modelData, nameof(modelData)), modelPath);
    }

#if NETCOREAPP3_1_OR_GREATER || NET5_0_OR_GREATER || NET6_0_OR_GREATER || NET7_0_OR_GREATER || NET8_0_OR_GREATER || NET9_0_OR_GREATER || NET10_0_OR_GREATER
    /// <summary>
    /// Parses ONNX model bytes from a managed read-only span into the target network.
    /// 将托管只读 span 中的 ONNX 模型字节解析到目标 network。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化 ONNX 模型字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    /// <remarks>
    /// The span is copied into a managed byte array before native interop, and TensorRT does not retain caller-owned memory.
    /// 调用 native interop 前会将 span 复制到托管字节数组，TensorRT 不会保留调用方拥有的内存。
    /// </remarks>
    public bool Parse(ReadOnlySpan<byte> modelData, string? modelPath = null)
    {
        return Parse(modelData.ToArray(), modelPath);
    }

#endif
    /// <summary>
    /// Parses ONNX model bytes from a managed stream into the target network.
    /// 从托管 stream 读取 ONNX 模型字节并解析到目标 network。
    /// </summary>
    /// <param name="modelStream">The readable stream containing serialized ONNX model bytes. 包含已序列化 ONNX 模型字节的可读 stream。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    /// <remarks>
    /// The stream is copied into managed memory before native interop; this is not a TensorRT model-proto or external-initializer lifetime bridge.
    /// 调用 native interop 前会将 stream 内容复制到托管内存；该入口不是 TensorRT model proto 或 external initializer 生命周期桥。
    /// </remarks>
    public bool Parse(Stream modelStream, string? modelPath = null)
    {
        return Parse(CopyModelStream(modelStream, nameof(modelStream)), modelPath);
    }

    /// <summary>
    /// Loads serialized ONNX model-proto bytes into the TensorRT 11 parser without parsing immediately.
    /// 将已序列化 ONNX model proto 字节加载到 TensorRT 11 parser，但暂不立即解析。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model-proto bytes. 已序列化的 ONNX model proto 字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT accepts the model proto. TensorRT 接受 model proto 时返回 <c>true</c>。</returns>
    /// <remarks>
    /// TensorRT 11 does not retain the model-proto byte buffer after this call returns; the managed buffer is pinned only for the native call.
    /// TensorRT 11 在该调用返回后不会继续持有 model-proto 字节缓冲区；托管缓冲区仅在 native 调用期间短期 pin。
    /// </remarks>
    public bool LoadModelProto(byte[] modelData, string? modelPath = null)
    {
        return NativeBridgeApi.LoadOnnxParserModelProto(Line, _handle, modelData, modelPath);
    }

    /// <summary>
    /// Loads serialized ONNX model-proto bytes from a byte-array segment into the TensorRT 11 parser.
    /// 从托管字节数组片段加载 ONNX model proto 到 TensorRT 11 parser。
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
    /// Loads serialized ONNX model-proto bytes from a read-only span into the TensorRT 11 parser.
    /// 从只读 span 加载 ONNX model proto 到 TensorRT 11 parser。
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
    /// Loads serialized ONNX model-proto bytes from a stream into the TensorRT 11 parser.
    /// 从 stream 加载 ONNX model proto 到 TensorRT 11 parser。
    /// </summary>
    /// <param name="modelStream">Readable stream containing serialized ONNX model-proto bytes. 包含已序列化 ONNX model proto 字节的可读 stream。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT accepts the model proto. TensorRT 接受 model proto 时返回 <c>true</c>。</returns>
    public bool LoadModelProto(Stream modelStream, string? modelPath = null)
    {
        return LoadModelProto(CopyModelStream(modelStream, nameof(modelStream)), modelPath);
    }

    /// <summary>
    /// Loads an external ONNX initializer into the TensorRT 11 parser.
    /// 将外部 ONNX initializer 加载到 TensorRT 11 parser。
    /// </summary>
    /// <param name="name">Initializer name. initializer 名称。</param>
    /// <param name="data">Initializer data. initializer 数据。</param>
    /// <returns><c>true</c> when TensorRT accepts the initializer. TensorRT 接受 initializer 时返回 <c>true</c>。</returns>
    /// <remarks>
    /// The initializer data is copied into an owned managed array and pinned until this parser is disposed, matching TensorRT's lifetime requirement.
    /// initializer 数据会复制到当前 parser 拥有的托管数组，并 pin 到 parser 释放为止，以满足 TensorRT 生命周期要求。
    /// </remarks>
    public bool LoadInitializer(string name, byte[] data)
    {
        return _initializerPins.LoadOrReplace(
            name,
            data,
            (pointer, size) => NativeBridgeApi.LoadOnnxParserInitializer(Line, _handle, name, pointer, size));
    }

    /// <summary>
    /// Loads an external ONNX initializer from a byte-array segment into the TensorRT 11 parser.
    /// 从托管字节数组片段加载外部 ONNX initializer 到 TensorRT 11 parser。
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
    /// Loads an external ONNX initializer from a read-only span into the TensorRT 11 parser.
    /// 从只读 span 加载外部 ONNX initializer 到 TensorRT 11 parser。
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
    /// Loads an external ONNX initializer from a stream into the TensorRT 11 parser.
    /// 从 stream 加载外部 ONNX initializer 到 TensorRT 11 parser。
    /// </summary>
    /// <param name="name">Initializer name. initializer 名称。</param>
    /// <param name="dataStream">Readable stream containing initializer data. 包含 initializer 数据的可读 stream。</param>
    /// <returns><c>true</c> when TensorRT accepts the initializer. TensorRT 接受 initializer 时返回 <c>true</c>。</returns>
    public bool LoadInitializer(string name, Stream dataStream)
    {
        return LoadInitializer(name, CopyModelStream(dataStream, nameof(dataStream)));
    }

    /// <summary>
    /// Parses the model proto previously loaded through <see cref="LoadModelProto(byte[], string?)"/>.
    /// 解析先前通过 <see cref="LoadModelProto(byte[], string?)"/> 加载的 model proto。
    /// </summary>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool ParseLoadedModel()
    {
        return NativeBridgeApi.ParseOnnxLoadedModelProto(Line, _handle);
    }

    /// <summary>
    /// Attempts to parse ONNX model bytes and returns copied TensorRT parser diagnostics.
    /// 尝试解析 ONNX 模型字节并返回已复制的 TensorRT parser 诊断信息。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化 ONNX 模型字节。</param>
    /// <param name="diagnostics">Copied parser diagnostics collected after the parse attempt. 解析尝试后收集到的 parser 诊断信息副本。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool TryParse(byte[] modelData, out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)
    {
        return TryParse(modelData, null, out diagnostics);
    }

    /// <summary>
    /// Attempts to parse ONNX model bytes and returns copied TensorRT parser diagnostics.
    /// 尝试解析 ONNX 模型字节并返回已复制的 TensorRT parser 诊断信息。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化 ONNX 模型字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <param name="diagnostics">Copied parser diagnostics collected after the parse attempt. 解析尝试后收集到的 parser 诊断信息副本。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool TryParse(byte[] modelData, string? modelPath, out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)
    {
        bool parsed = Parse(modelData, modelPath);
        diagnostics = GetDiagnostics();
        return parsed;
    }

    /// <summary>
    /// Attempts to parse ONNX model bytes from a managed stream and returns copied TensorRT parser diagnostics.
    /// 尝试从托管 stream 解析 ONNX 模型字节并返回已复制的 TensorRT parser 诊断信息。
    /// </summary>
    /// <param name="modelStream">The readable stream containing serialized ONNX model bytes. 包含已序列化 ONNX 模型字节的可读 stream。</param>
    /// <param name="diagnostics">Copied parser diagnostics collected after the parse attempt. 解析尝试后收集到的 parser 诊断信息副本。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool TryParse(Stream modelStream, out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)
    {
        return TryParse(modelStream, null, out diagnostics);
    }

    /// <summary>
    /// Attempts to parse ONNX model bytes from a managed stream and returns copied TensorRT parser diagnostics.
    /// 尝试从托管 stream 解析 ONNX 模型字节并返回已复制的 TensorRT parser 诊断信息。
    /// </summary>
    /// <param name="modelStream">The readable stream containing serialized ONNX model bytes. 包含已序列化 ONNX 模型字节的可读 stream。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <param name="diagnostics">Copied parser diagnostics collected after the parse attempt. 解析尝试后收集到的 parser 诊断信息副本。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool TryParse(Stream modelStream, string? modelPath, out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)
    {
        bool parsed = Parse(modelStream, modelPath);
        diagnostics = GetDiagnostics();
        return parsed;
    }

    /// <summary>
    /// Gets one parser error by index.
    /// 按索引获取一个 parser 错误。
    /// </summary>
    /// <param name="index">Zero-based parser error index. 从零开始的 parser 错误索引。</param>
    /// <returns>The parser error information. Parser 错误信息。</returns>
    public TensorRtParserErrorInfo GetError(int index)
    {
        return NativeBridgeApi.GetOnnxParserError(Line, _handle, index);
    }

    /// <summary>
    /// Gets one detailed parser diagnostic by index.
    /// 按索引获取一条详细解析器诊断信息。
    /// </summary>
    /// <param name="index">Zero-based parser error index. 从零开始的解析器错误索引。</param>
    /// <returns>The detailed parser diagnostic. 详细解析器诊断信息。</returns>
    /// <remarks>
    /// This path reads variable-length strings from the native bridge and avoids truncating long parser messages.
    /// 该路径从原生桥接读取可变长度字符串，可避免长解析错误消息被固定结构体截断。
    /// </remarks>
    public TensorRtOnnxParserDiagnostic GetDiagnostic(int index)
    {
        return NativeBridgeApi.GetOnnxParserDiagnostic(Line, _handle, index);
    }

    /// <summary>
    /// Clears parser errors stored by TensorRT.
    /// 清理 TensorRT 保存的 parser 错误。
    /// </summary>
    public void ClearErrors()
    {
        NativeBridgeApi.ClearOnnxParserErrors(Line, _handle);
    }

    /// <summary>
    /// Returns whether the selected TensorRT parser reports support for an ONNX operator.
    /// 返回当前 TensorRT parser 是否报告支持指定 ONNX operator。
    /// </summary>
    /// <param name="operatorName">The ONNX operator name, for example <c>Identity</c>. ONNX operator 名称，例如 <c>Identity</c>。</param>
    /// <returns><c>true</c> when TensorRT reports support for the operator. TensorRT 报告支持该 operator 时返回 <c>true</c>。</returns>
    public bool SupportsOperator(string operatorName)
    {
        return NativeBridgeApi.OnnxParserSupportsOperator(Line, _handle, operatorName);
    }

    /// <summary>
    /// Returns whether TensorRT reports the ONNX subgraph at the specified index as supported.
    /// 返回 TensorRT 是否报告指定索引处的 ONNX subgraph 受支持。
    /// </summary>
    /// <param name="index">Zero-based subgraph index. 从零开始的 subgraph 索引。</param>
    /// <returns><c>true</c> when TensorRT reports the subgraph as supported. TensorRT 报告该 subgraph 受支持时返回 <c>true</c>。</returns>
    /// <remarks>
    /// This is a copied scalar query over the native parser state. The native bridge keeps TensorRT version guards in place and does not expose a borrowed parser pointer.
    /// 这是对 native parser 状态的标量只读查询；native bridge 保留 TensorRT 版本保护，不暴露 borrowed parser 指针。
    /// </remarks>
    public bool IsSubgraphSupported(long index)
    {
        return NativeBridgeApi.IsOnnxParserSubgraphSupported(Line, _handle, index);
    }

    /// <summary>
    /// Gets a single ONNX parser flag.
    /// 获取单个 ONNX parser 标志。
    /// </summary>
    /// <param name="flag">The parser flag to query. 要查询的 parser 标志。</param>
    /// <returns><c>true</c> when the flag is enabled. 标志启用时返回 <c>true</c>。</returns>
    public bool GetFlag(TensorRtOnnxParserFlag flag)
    {
        ValidateParserFlag(flag);
        return NativeBridgeApi.GetOnnxParserFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Enables a single ONNX parser flag.
    /// 启用单个 ONNX parser 标志。
    /// </summary>
    /// <param name="flag">The parser flag to enable. 要启用的 parser 标志。</param>
    public void SetFlag(TensorRtOnnxParserFlag flag)
    {
        ValidateParserFlag(flag);
        NativeBridgeApi.SetOnnxParserFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Clears a single ONNX parser flag.
    /// 清除单个 ONNX parser 标志。
    /// </summary>
    /// <param name="flag">The parser flag to clear. 要清除的 parser 标志。</param>
    public void ClearFlag(TensorRtOnnxParserFlag flag)
    {
        ValidateParserFlag(flag);
        NativeBridgeApi.ClearOnnxParserFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Gets all parser errors currently reported by TensorRT.
    /// 获取 TensorRT 当前报告的全部 parser 错误。
    /// </summary>
    /// <returns>The parser error list. Parser 错误列表。</returns>
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
    /// Gets all detailed parser diagnostics currently reported by TensorRT.
    /// 获取 TensorRT 当前报告的全部详细解析器诊断信息。
    /// </summary>
    /// <returns>The detailed parser diagnostics. 详细解析器诊断信息列表。</returns>
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
    /// Builds a readable parser error summary for logs and diagnostics.
    /// 为日志和诊断生成可读的 parser 错误摘要。
    /// </summary>
    /// <returns>A parser error summary string. Parser 错误摘要字符串。</returns>
    public string GetErrorSummary()
    {
        return GetDiagnosticSummary();
    }

    /// <summary>
    /// Builds a readable parser diagnostic summary for logs and deployment troubleshooting.
    /// 为日志和部署排障生成可读的解析器诊断摘要。
    /// </summary>
    /// <returns>A parser diagnostic summary string. 解析器诊断摘要字符串。</returns>
    public string GetDiagnosticSummary()
    {
        return BuildDiagnosticSummary(GetDiagnostics());
    }

    private static string BuildDiagnosticSummary(IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)
    {
        if (diagnostics.Count == 0)
        {
            return "ONNX parser reported no errors.";
        }

        StringBuilder builder = new StringBuilder();
        builder.Append("ONNX parser errors: ");
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
