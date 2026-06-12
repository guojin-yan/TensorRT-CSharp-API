namespace JYPPX.TensorRtSharp;

/// <summary>
/// Provides fixed-buffer ONNX parser error information returned by TensorRT.
/// 表示 TensorRT 返回的固定缓冲区 ONNX parser 错误信息。
/// </summary>
/// <remarks>
/// Use <see cref="TensorRtOnnxParserDiagnostic"/> when long diagnostic strings must not be truncated.
/// 当需要避免长诊断字符串被截断时，请使用 <see cref="TensorRtOnnxParserDiagnostic"/>。
/// </remarks>
public sealed class TensorRtParserErrorInfo
{
    /// <summary>
    /// Creates parser error information.
    /// 创建 parser 错误信息。
    /// </summary>
    public TensorRtParserErrorInfo(
        int index,
        int code,
        int line,
        int node,
        string description,
        string file,
        string functionName,
        string nodeName,
        string nodeOperator)
    {
        Index = index;
        Code = code;
        Line = line;
        Node = node;
        Description = description;
        File = file;
        FunctionName = functionName;
        NodeName = nodeName;
        NodeOperator = nodeOperator;
    }

    /// <summary>
    /// Gets the zero-based parser error index.
    /// 获取从零开始的 parser 错误索引。
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Gets the TensorRT parser error code.
    /// 获取 TensorRT parser 错误代码。
    /// </summary>
    public int Code { get; }

    /// <summary>
    /// Gets the source line reported by the parser.
    /// 获取 parser 报告的源码行号。
    /// </summary>
    public int Line { get; }

    /// <summary>
    /// Gets the ONNX node index reported by the parser.
    /// 获取 parser 报告的 ONNX 节点索引。
    /// </summary>
    public int Node { get; }

    /// <summary>
    /// Gets the parser error message stored in the fixed native buffer.
    /// 获取固定原生缓冲区中的 parser 错误消息。
    /// </summary>
    public string Description { get; }

    /// <summary>
    /// Gets the source file reported by the parser.
    /// 获取 parser 报告的源码文件。
    /// </summary>
    public string File { get; }

    /// <summary>
    /// Gets the source function reported by the parser.
    /// 获取 parser 报告的源码函数。
    /// </summary>
    public string FunctionName { get; }

    /// <summary>
    /// Gets the ONNX node name when available.
    /// 获取可用的 ONNX 节点名称。
    /// </summary>
    public string NodeName { get; }

    /// <summary>
    /// Gets the ONNX node operator when available.
    /// 获取可用的 ONNX 节点算子类型。
    /// </summary>
    public string NodeOperator { get; }

    /// <summary>
    /// Converts the error to a compact log-friendly string.
    /// 将错误信息转换为适合日志输出的简短字符串。
    /// </summary>
    /// <returns>A readable parser error string. 可读的 parser 错误字符串。</returns>
    public override string ToString()
    {
        return string.IsNullOrWhiteSpace(Description)
            ? $"ONNX parser error #{Index} code={Code}"
            : $"ONNX parser error #{Index} code={Code}: {Description}";
    }
}
