using System;
using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Provides detailed diagnostics for one ONNX parser error reported by TensorRT.
/// 表示 TensorRT ONNX 解析器报告的一条详细错误诊断信息。
/// </summary>
public sealed class TensorRtOnnxParserDiagnostic
{
    /// <summary>
    /// Creates a parser diagnostic value.
    /// 创建一个解析器诊断值。
    /// </summary>
    /// <param name="index">Zero-based error index in the parser. 解析器中的从零开始错误索引。</param>
    /// <param name="code">TensorRT parser error code. TensorRT 解析器错误代码。</param>
    /// <param name="line">Source line reported by the parser, or a negative value when unavailable. 解析器报告的源码行号；不可用时通常为负值。</param>
    /// <param name="node">ONNX node index reported by the parser, or a negative value when unavailable. 解析器报告的 ONNX 节点索引；不可用时通常为负值。</param>
    /// <param name="description">Full parser error message. 完整的解析器错误消息。</param>
    /// <param name="file">Source file reported by the parser. 解析器报告的源码文件。</param>
    /// <param name="functionName">Source function reported by the parser. 解析器报告的源码函数。</param>
    /// <param name="nodeName">ONNX node name when exposed by the TensorRT major line. TensorRT 大版本支持时返回 ONNX 节点名称。</param>
    /// <param name="nodeOperator">ONNX node operator when exposed by the TensorRT major line. TensorRT 大版本支持时返回 ONNX 节点算子类型。</param>
    /// <param name="localFunctionStack">TensorRT 10 local function stack entries, or an empty list on TensorRT 8. TensorRT 10 的本地函数栈；TensorRT 8 通常为空。</param>
    public TensorRtOnnxParserDiagnostic(
        int index,
        int code,
        int line,
        int node,
        string description,
        string file,
        string functionName,
        string nodeName,
        string nodeOperator,
        IReadOnlyList<string> localFunctionStack)
    {
        Index = index;
        Code = code;
        Line = line;
        Node = node;
        Description = description ?? string.Empty;
        File = file ?? string.Empty;
        FunctionName = functionName ?? string.Empty;
        NodeName = nodeName ?? string.Empty;
        NodeOperator = nodeOperator ?? string.Empty;
        LocalFunctionStack = localFunctionStack ?? Array.Empty<string>();
    }

    /// <summary>
    /// Gets the zero-based error index in the parser.
    /// 获取解析器中的从零开始错误索引。
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Gets the TensorRT parser error code.
    /// 获取 TensorRT 解析器错误代码。
    /// </summary>
    public int Code { get; }

    /// <summary>
    /// Gets the source line reported by the parser.
    /// 获取解析器报告的源码行号。
    /// </summary>
    public int Line { get; }

    /// <summary>
    /// Gets the ONNX node index reported by the parser.
    /// 获取解析器报告的 ONNX 节点索引。
    /// </summary>
    public int Node { get; }

    /// <summary>
    /// Gets the full parser error message without the fixed-size native struct truncation limit.
    /// 获取完整解析器错误消息，不受固定长度原生结构体缓冲区截断限制。
    /// </summary>
    public string Description { get; }

    /// <summary>
    /// Gets the source file reported by the parser.
    /// 获取解析器报告的源码文件。
    /// </summary>
    public string File { get; }

    /// <summary>
    /// Gets the source function reported by the parser.
    /// 获取解析器报告的源码函数。
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
    /// Gets the TensorRT 10 local function stack entries.
    /// 获取 TensorRT 10 本地函数栈条目。
    /// </summary>
    public IReadOnlyList<string> LocalFunctionStack { get; }

    /// <summary>
    /// Converts the diagnostic to a compact log-friendly string.
    /// 将诊断信息转换为适合日志输出的简短字符串。
    /// </summary>
    /// <returns>A readable parser diagnostic string. 可读的解析器诊断字符串。</returns>
    public override string ToString()
    {
        string location = string.IsNullOrWhiteSpace(File) ? string.Empty : $" file={File}";
        string node = string.IsNullOrWhiteSpace(NodeName) ? string.Empty : $" node={NodeName}";
        string op = string.IsNullOrWhiteSpace(NodeOperator) ? string.Empty : $" op={NodeOperator}";
        return string.IsNullOrWhiteSpace(Description)
            ? $"ONNX parser diagnostic #{Index} code={Code}{location}{node}{op}"
            : $"ONNX parser diagnostic #{Index} code={Code}{location}{node}{op}: {Description}";
    }
}
