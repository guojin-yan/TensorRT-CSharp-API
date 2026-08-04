using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Parses ONNX models into a TensorRT network definition.
/// 将 ONNX 模型解析到 TensorRT network definition。
/// </summary>
public sealed partial class TensorRtOnnxParser
{
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

}
