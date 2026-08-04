using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Captures copied, pointer-free ONNX parser-refitter diagnostics.
/// 捕获已复制、无指针逃逸的 ONNX parser-refitter 诊断信息。
/// </summary>
public sealed class TensorRtOnnxParserRefitterDiagnosticSnapshot
{
    /// <summary>
    /// Creates an ONNX parser-refitter diagnostic snapshot.
    /// 创建 ONNX parser-refitter 诊断快照。
    /// </summary>
    /// <param name="line">TensorRT API line used by the parser-refitter. Parser-refitter 所属 TensorRT API 版本线。</param>
    /// <param name="errorCount">Current parser-refitter error count reported by TensorRT. TensorRT 当前报告的 parser-refitter 错误数量。</param>
    /// <param name="diagnostics">Copied parser-refitter diagnostics. 已复制的 parser-refitter 诊断列表。</param>
    /// <param name="diagnosticSummary">Readable parser-refitter diagnostic summary. 可读 parser-refitter 诊断摘要。</param>
    public TensorRtOnnxParserRefitterDiagnosticSnapshot(
        TensorRtApiLine line,
        int errorCount,
        IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics,
        string diagnosticSummary)
    {
        Line = line;
        ErrorCount = errorCount;
        Diagnostics = diagnostics ?? Array.Empty<TensorRtOnnxParserDiagnostic>();
        DiagnosticSummary = diagnosticSummary ?? string.Empty;
    }

    /// <summary>
    /// Gets the TensorRT API line used by the parser-refitter.
    /// 获取 parser-refitter 使用的 TensorRT API 版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the parser-refitter error count reported when the snapshot was captured.
    /// 获取捕获快照时 TensorRT 报告的 parser-refitter 错误数量。
    /// </summary>
    public int ErrorCount { get; }

    /// <summary>
    /// Gets copied detailed parser-refitter diagnostics.
    /// 获取已复制的详细 parser-refitter 诊断列表。
    /// </summary>
    public IReadOnlyList<TensorRtOnnxParserDiagnostic> Diagnostics { get; }

    /// <summary>
    /// Gets a readable parser-refitter diagnostic summary.
    /// 获取可读 parser-refitter 诊断摘要。
    /// </summary>
    public string DiagnosticSummary { get; }

    /// <summary>
    /// Creates a compact pointer-free summary from the copied managed snapshot values.
    /// 从已复制的托管快照值创建紧凑、无指针逃逸的摘要。
    /// </summary>
    /// <remarks>
    /// This method only reads values already held by this managed snapshot. It does not call TensorRT.
    /// It does not expose native pointers, and it does not promote the snapshot to runtime or external-model proof.
    /// 此方法只读取当前托管快照中已有的值，不调用 TensorRT、不暴露原生指针，也不将快照提升为运行时或外部模型 proof。
    /// </remarks>
    /// <returns>A compact parser-refitter diagnostic summary. 紧凑 parser-refitter 诊断摘要。</returns>
    public TensorRtOnnxParserRefitterDiagnosticSummary ToSummary()
    {
        return new TensorRtOnnxParserRefitterDiagnosticSummary(
            Line,
            ErrorCount,
            Diagnostics.Count,
            DiagnosticSummary.Length);
    }

    /// <summary>
    /// Converts the snapshot to a compact log-friendly string.
    /// 将快照转换为适合日志输出的紧凑字符串。
    /// </summary>
    /// <returns>A compact parser-refitter diagnostic snapshot string. 紧凑 parser-refitter 诊断快照字符串。</returns>
    public override string ToString()
    {
        return $"Line={(int)Line} Errors={ErrorCount} Diagnostics={Diagnostics.Count}";
    }
}
