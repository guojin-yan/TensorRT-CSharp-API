namespace JYPPX.TensorRtSharp;

/// <summary>
/// Summarizes copied ONNX parser model-support diagnostics without exposing native pointers.
/// 汇总已复制的 ONNX parser 模型支持性诊断，不暴露原生指针。
/// </summary>
public sealed class TensorRtOnnxModelSupportSummary
{
    internal TensorRtOnnxModelSupportSummary(
        bool isSupported,
        long reportedSupportedSubgraphCount,
        long reportedUnsupportedSubgraphCount,
        int copiedSubgraphCount,
        long copiedSupportedSubgraphCount,
        long copiedUnsupportedSubgraphCount,
        long copiedNodeCount)
    {
        IsSupported = isSupported;
        ReportedSupportedSubgraphCount = reportedSupportedSubgraphCount < 0 ? 0 : reportedSupportedSubgraphCount;
        ReportedUnsupportedSubgraphCount = reportedUnsupportedSubgraphCount < 0 ? 0 : reportedUnsupportedSubgraphCount;
        CopiedSubgraphCount = copiedSubgraphCount < 0 ? 0 : copiedSubgraphCount;
        CopiedSupportedSubgraphCount = copiedSupportedSubgraphCount < 0 ? 0 : copiedSupportedSubgraphCount;
        CopiedUnsupportedSubgraphCount = copiedUnsupportedSubgraphCount < 0 ? 0 : copiedUnsupportedSubgraphCount;
        CopiedNodeCount = copiedNodeCount < 0 ? 0 : copiedNodeCount;
    }

    /// <summary>Gets whether TensorRT reported full-model support. 获取 TensorRT 是否报告完整模型受支持。</summary>
    public bool IsSupported { get; }

    /// <summary>Gets the supported subgraph count reported by TensorRT. 获取 TensorRT 报告的受支持子图数量。</summary>
    public long ReportedSupportedSubgraphCount { get; }

    /// <summary>Gets the unsupported subgraph count reported by TensorRT. 获取 TensorRT 报告的不受支持子图数量。</summary>
    public long ReportedUnsupportedSubgraphCount { get; }

    /// <summary>Gets the number of copied managed subgraph records. 获取已复制到托管侧的子图记录数量。</summary>
    public int CopiedSubgraphCount { get; }

    /// <summary>Gets the number of copied subgraphs marked supported. 获取已复制且标记为支持的子图数量。</summary>
    public long CopiedSupportedSubgraphCount { get; }

    /// <summary>Gets the number of copied subgraphs marked unsupported. 获取已复制且标记为不支持的子图数量。</summary>
    public long CopiedUnsupportedSubgraphCount { get; }

    /// <summary>Gets the total number of copied ONNX node indexes. 获取已复制 ONNX node index 总数。</summary>
    public long CopiedNodeCount { get; }

    /// <summary>Gets whether copied subgraph support counts match the TensorRT-reported counts. 获取已复制子图计数是否与 TensorRT 报告计数一致。</summary>
    public bool CopiedSubgraphCountsMatchReportedCounts =>
        ReportedSupportedSubgraphCount == CopiedSupportedSubgraphCount &&
        ReportedUnsupportedSubgraphCount == CopiedUnsupportedSubgraphCount;

    /// <summary>Gets the runtime evidence kind represented by this copied summary. 获取该 copied summary 表示的 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "copied-readonly-summary";

    /// <summary>Gets whether this summary is runtime execution evidence. 获取该摘要是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this summary is runtime execution proof. 获取该摘要是否为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets whether the summary is copied and pointer-free. 获取该摘要是否为复制型且无指针逃逸。</summary>
    public bool PointerFreeCopiedSummary => true;

    /// <summary>Gets whether this summary can be promoted as runtime execution proof. 获取该摘要是否可晋级为 runtime execution proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether this summary can promote public release proof. 获取该摘要是否可晋级为 public release proof。</summary>
    public bool CanPromoteReleaseProof => false;

    /// <summary>Gets whether this summary allows deleting deferred records. 获取该摘要是否允许删除 deferred 记录。</summary>
    public bool CanDeleteDeferredRecord => false;

    /// <summary>
    /// Returns a compact log-friendly support summary string.
    /// 返回适合日志输出的紧凑支持性摘要。
    /// </summary>
    public override string ToString()
    {
        return $"Supported={IsSupported} CopiedSubgraphs={CopiedSubgraphCount} CopiedNodes={CopiedNodeCount} ReportedSupported={ReportedSupportedSubgraphCount} ReportedUnsupported={ReportedUnsupportedSubgraphCount} CountMatch={CopiedSubgraphCountsMatchReportedCounts} RuntimeProof={CanPromoteRuntimeProof}";
    }
}
