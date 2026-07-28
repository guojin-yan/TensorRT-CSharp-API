using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

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

/// <summary>
/// Summarizes copied, pointer-free ONNX parser-refitter diagnostics.
/// 汇总已复制、无指针逃逸的 ONNX parser-refitter 诊断信息。
/// </summary>
public sealed class TensorRtOnnxParserRefitterDiagnosticSummary
{
    /// <summary>
    /// Creates a compact ONNX parser-refitter diagnostic summary.
    /// 创建紧凑 ONNX parser-refitter 诊断摘要。
    /// </summary>
    /// <param name="line">TensorRT API line used by the parser-refitter. Parser-refitter 所属 TensorRT API 版本线。</param>
    /// <param name="errorCount">Parser-refitter error count copied into the source snapshot. 源快照中复制的 parser-refitter 错误数量。</param>
    /// <param name="copiedDiagnosticCount">Number of copied parser-refitter diagnostics. 已复制 parser-refitter 诊断数量。</param>
    /// <param name="diagnosticSummaryLength">Length of the copied readable diagnostic summary. 已复制可读诊断摘要长度。</param>
    public TensorRtOnnxParserRefitterDiagnosticSummary(
        TensorRtApiLine line,
        int errorCount,
        int copiedDiagnosticCount,
        int diagnosticSummaryLength)
    {
        Line = line;
        ErrorCount = errorCount;
        CopiedDiagnosticCount = copiedDiagnosticCount;
        DiagnosticSummaryLength = diagnosticSummaryLength;
    }

    /// <summary>
    /// Gets the TensorRT API line used by the parser-refitter.
    /// 获取 parser-refitter 使用的 TensorRT API 版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the parser-refitter error count copied into the source snapshot.
    /// 获取源快照中复制的 parser-refitter 错误数量。
    /// </summary>
    public int ErrorCount { get; }

    /// <summary>
    /// Gets the number of copied parser-refitter diagnostics.
    /// 获取已复制 parser-refitter 诊断数量。
    /// </summary>
    public int CopiedDiagnosticCount { get; }

    /// <summary>
    /// Gets the length of the copied readable diagnostic summary.
    /// 获取已复制可读诊断摘要长度。
    /// </summary>
    public int DiagnosticSummaryLength { get; }

    /// <summary>
    /// Gets the runtime evidence kind represented by this copied summary.
    /// 获取该 copied summary 表示的 runtime evidence 类型。
    /// </summary>
    public string RuntimeEvidenceKind => "copied-readonly-summary";

    /// <summary>
    /// Gets whether this summary is runtime execution evidence.
    /// 获取该摘要是否为 runtime execution evidence。
    /// </summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>
    /// Gets whether this summary is runtime execution proof.
    /// 获取该摘要是否为 runtime execution proof。
    /// </summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>
    /// Gets whether this summary is copied and pointer-free.
    /// 获取该摘要是否为复制型且不暴露指针。
    /// </summary>
    public bool PointerFreeCopiedSummary => true;

    /// <summary>
    /// Gets whether this summary can be promoted as runtime proof.
    /// 获取该摘要是否可晋级为 runtime proof。
    /// </summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>
    /// Gets whether this summary can promote public release proof.
    /// 获取该摘要是否可晋级为 public release proof。
    /// </summary>
    public bool CanPromoteReleaseProof => false;

    /// <summary>
    /// Gets whether this summary allows deleting deferred records.
    /// 获取该摘要是否允许删除 deferred 记录。
    /// </summary>
    public bool CanDeleteDeferredRecord => false;

    /// <summary>
    /// Converts the summary to a compact log-friendly string.
    /// 将摘要转换为适合日志输出的紧凑字符串。
    /// </summary>
    /// <returns>A compact parser-refitter diagnostic summary string. 紧凑 parser-refitter 诊断摘要字符串。</returns>
    public override string ToString()
    {
        return $"Line={(int)Line} Errors={ErrorCount} CopiedDiagnostics={CopiedDiagnosticCount} DiagnosticSummaryLength={DiagnosticSummaryLength} RuntimeProof={CanPromoteRuntimeProof}";
    }
}
