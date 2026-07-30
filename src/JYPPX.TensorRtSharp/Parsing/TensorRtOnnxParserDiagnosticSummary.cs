using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Summarizes copied, pointer-free ONNX parser diagnostic inventory state.
/// 汇总已复制、无指针逃逸的 ONNX parser 诊断 inventory 状态。
/// </summary>
public sealed class TensorRtOnnxParserDiagnosticSummary
{
    /// <summary>
    /// Creates a compact ONNX parser diagnostic summary.
    /// 创建紧凑 ONNX parser 诊断摘要。
    /// </summary>
    /// <param name="line">TensorRT API line used by the parser. Parser 所属 TensorRT API 版本线。</param>
    /// <param name="errorCount">Parser error count copied into the source snapshot. 源快照中复制的 parser 错误数量。</param>
    /// <param name="copiedDiagnosticCount">Number of copied parser diagnostics. 已复制 parser 诊断数量。</param>
    /// <param name="diagnosticSummaryLength">Length of the copied readable diagnostic summary. 已复制可读诊断摘要长度。</param>
    /// <param name="usedVCPluginLibraryCount">Number of copied VC plugin library paths. 已复制 VC plugin library 路径数量。</param>
    /// <param name="identityOperatorSupported">Whether TensorRT reported ONNX Identity operator support when the snapshot was captured. 捕获快照时 TensorRT 是否报告支持 ONNX Identity 算子。</param>
    public TensorRtOnnxParserDiagnosticSummary(
        TensorRtApiLine line,
        int errorCount,
        int copiedDiagnosticCount,
        int diagnosticSummaryLength,
        int usedVCPluginLibraryCount,
        bool identityOperatorSupported)
    {
        Line = line;
        ErrorCount = errorCount;
        CopiedDiagnosticCount = copiedDiagnosticCount;
        DiagnosticSummaryLength = diagnosticSummaryLength;
        UsedVCPluginLibraryCount = usedVCPluginLibraryCount;
        IdentityOperatorSupported = identityOperatorSupported;
    }

    /// <summary>
    /// Gets the TensorRT API line used by the parser.
    /// 获取 parser 使用的 TensorRT API 版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the parser error count copied into the source snapshot.
    /// 获取源快照中复制的 parser 错误数量。
    /// </summary>
    public int ErrorCount { get; }

    /// <summary>
    /// Gets the number of copied parser diagnostics.
    /// 获取已复制 parser 诊断数量。
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
    /// Gets the number of copied VC plugin library paths.
    /// 获取已复制 VC plugin library 路径数量。
    /// </summary>
    public int UsedVCPluginLibraryCount { get; }

    /// <summary>
    /// Gets whether TensorRT reported support for the ONNX Identity operator when the snapshot was captured.
    /// 获取捕获快照时 TensorRT 是否报告支持 ONNX Identity 算子。
    /// </summary>
    public bool IdentityOperatorSupported { get; }

    /// <summary>
    /// Converts the summary to a compact log-friendly string.
    /// 将摘要转换为适合日志输出的紧凑字符串。
    /// </summary>
    /// <returns>A compact parser diagnostic summary string. 紧凑 parser 诊断摘要字符串。</returns>
    public override string ToString()
    {
        return $"Line={(int)Line} Errors={ErrorCount} CopiedDiagnostics={CopiedDiagnosticCount} DiagnosticSummaryLength={DiagnosticSummaryLength} UsedVCPluginLibraries={UsedVCPluginLibraryCount} SupportsIdentity={IdentityOperatorSupported} RuntimeProof={CanPromoteRuntimeProof}";
    }
}
