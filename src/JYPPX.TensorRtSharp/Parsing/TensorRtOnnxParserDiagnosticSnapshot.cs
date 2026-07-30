using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Captures copied, pointer-free ONNX parser diagnostics and inventory state.
/// 捕获已复制、无指针逃逸的 ONNX parser 诊断与 inventory 状态。
/// </summary>
public sealed class TensorRtOnnxParserDiagnosticSnapshot
{
    /// <summary>
    /// Creates an ONNX parser diagnostic snapshot.
    /// 创建 ONNX parser 诊断快照。
    /// </summary>
    /// <param name="line">TensorRT API line used by the parser. Parser 所属 TensorRT API 版本线。</param>
    /// <param name="errorCount">Current parser error count reported by TensorRT. TensorRT 当前报告的 parser 错误数量。</param>
    /// <param name="diagnostics">Copied parser diagnostics. 已复制的 parser 诊断列表。</param>
    /// <param name="diagnosticSummary">Readable parser diagnostic summary. 可读 parser 诊断摘要。</param>
    /// <param name="usedVCPluginLibraries">Copied VC plugin library paths reported by the parser. 已复制的 parser VC plugin library 路径。</param>
    /// <param name="identityOperatorSupported">Whether TensorRT reports support for the ONNX Identity operator. TensorRT 是否报告支持 ONNX Identity 算子。</param>
    public TensorRtOnnxParserDiagnosticSnapshot(
        TensorRtApiLine line,
        int errorCount,
        IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics,
        string diagnosticSummary,
        IReadOnlyList<string> usedVCPluginLibraries,
        bool identityOperatorSupported)
    {
        Line = line;
        ErrorCount = errorCount;
        Diagnostics = diagnostics ?? Array.Empty<TensorRtOnnxParserDiagnostic>();
        DiagnosticSummary = diagnosticSummary ?? string.Empty;
        UsedVCPluginLibraries = usedVCPluginLibraries ?? Array.Empty<string>();
        IdentityOperatorSupported = identityOperatorSupported;
    }

    /// <summary>
    /// Gets the TensorRT API line used by the parser.
    /// 获取 parser 使用的 TensorRT API 版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the parser error count reported when the snapshot was captured.
    /// 获取捕获快照时 TensorRT 报告的 parser 错误数量。
    /// </summary>
    public int ErrorCount { get; }

    /// <summary>
    /// Gets copied detailed parser diagnostics.
    /// 获取已复制的详细 parser 诊断列表。
    /// </summary>
    public IReadOnlyList<TensorRtOnnxParserDiagnostic> Diagnostics { get; }

    /// <summary>
    /// Gets a readable parser diagnostic summary.
    /// 获取可读 parser 诊断摘要。
    /// </summary>
    public string DiagnosticSummary { get; }

    /// <summary>
    /// Gets copied VC plugin library paths reported by the parser.
    /// 获取已复制的 parser VC plugin library 路径。
    /// </summary>
    public IReadOnlyList<string> UsedVCPluginLibraries { get; }

    /// <summary>
    /// Gets whether TensorRT reports support for the ONNX Identity operator.
    /// 获取 TensorRT 是否报告支持 ONNX Identity 算子。
    /// </summary>
    public bool IdentityOperatorSupported { get; }

    /// <summary>
    /// Creates a compact pointer-free summary from the copied managed snapshot values.
    /// 从已复制的托管快照值创建紧凑、无指针逃逸的摘要。
    /// </summary>
    /// <remarks>
    /// This method only reads values already held by this managed snapshot. It does not call TensorRT.
    /// It does not expose native pointers, and it does not promote the snapshot to runtime or external-model proof.
    /// 此方法只读取当前托管快照中已有的值，不调用 TensorRT、不暴露原生指针，也不将快照提升为运行时或外部模型 proof。
    /// </remarks>
    /// <returns>A compact parser diagnostic summary. 紧凑 parser 诊断摘要。</returns>
    public TensorRtOnnxParserDiagnosticSummary ToSummary()
    {
        return new TensorRtOnnxParserDiagnosticSummary(
            Line,
            ErrorCount,
            Diagnostics.Count,
            DiagnosticSummary.Length,
            UsedVCPluginLibraries.Count,
            IdentityOperatorSupported);
    }

    /// <summary>
    /// Converts the snapshot to a compact log-friendly string.
    /// 将快照转换为适合日志输出的紧凑字符串。
    /// </summary>
    /// <returns>A compact parser diagnostic snapshot string. 紧凑 parser 诊断快照字符串。</returns>
    public override string ToString()
    {
        return $"Line={(int)Line} Errors={ErrorCount} Diagnostics={Diagnostics.Count} UsedVCPluginLibraries={UsedVCPluginLibraries.Count} SupportsIdentity={IdentityOperatorSupported}";
    }
}
