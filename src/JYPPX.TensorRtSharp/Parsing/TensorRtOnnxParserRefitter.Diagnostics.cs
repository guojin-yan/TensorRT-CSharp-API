using System.Collections.Generic;
using System.Text;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOnnxParserRefitter
{
    /// <summary>
    /// Gets the number of parser-refitter errors currently reported by TensorRT.
    /// 获取 TensorRT 当前报告的 parser-refitter 错误数量。
    /// </summary>
    public int ErrorCount => NativeBridgeApi.GetOnnxParserRefitterErrorCount(Line, _handle);

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
}
