using System;
using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes whether TensorRT reports support for an ONNX model and its parser subgraphs.
/// 描述 TensorRT 是否支持指定 ONNX 模型以及 parser 子图支持情况。
/// </summary>
public sealed class TensorRtOnnxModelSupportReport
{
    /// <summary>
    /// Creates an ONNX model support report.
    /// 创建 ONNX 模型支持性报告。
    /// </summary>
    /// <param name="isSupported">Whether TensorRT reports the full model as supported. TensorRT 是否报告完整模型受支持。</param>
    /// <param name="supportedSubgraphCount">Number of supported subgraphs. 受支持子图数量。</param>
    /// <param name="unsupportedSubgraphCount">Number of unsupported subgraphs. 不受支持子图数量。</param>
    /// <param name="subgraphs">Per-subgraph support details. 每个子图的支持性详情。</param>
    public TensorRtOnnxModelSupportReport(
        bool isSupported,
        long supportedSubgraphCount,
        long unsupportedSubgraphCount,
        IReadOnlyList<TensorRtOnnxSubgraphSupportInfo> subgraphs)
    {
        IsSupported = isSupported;
        SupportedSubgraphCount = supportedSubgraphCount;
        UnsupportedSubgraphCount = unsupportedSubgraphCount;
        Subgraphs = subgraphs ?? Array.Empty<TensorRtOnnxSubgraphSupportInfo>();
    }

    /// <summary>
    /// Gets whether TensorRT reports the whole ONNX model as supported.
    /// 获取 TensorRT 是否报告整个 ONNX 模型受支持。
    /// </summary>
    public bool IsSupported { get; }

    /// <summary>
    /// Gets the number of supported subgraphs reported by TensorRT.
    /// 获取 TensorRT 报告的受支持子图数量。
    /// </summary>
    public long SupportedSubgraphCount { get; }

    /// <summary>
    /// Gets the number of unsupported subgraphs reported by TensorRT.
    /// 获取 TensorRT 报告的不受支持子图数量。
    /// </summary>
    public long UnsupportedSubgraphCount { get; }

    /// <summary>
    /// Gets the per-subgraph support details.
    /// 获取每个子图的支持性详情。
    /// </summary>
    public IReadOnlyList<TensorRtOnnxSubgraphSupportInfo> Subgraphs { get; }

    /// <summary>
    /// Converts this report into a compact pointer-free summary.
    /// 将当前报告转换为紧凑、无指针逃逸的摘要。
    /// </summary>
    /// <remarks>
    /// This method only reads copied managed values that are already present in the report.
    /// It does not call TensorRT, does not expose parser-owned tensor pointers, and does not promote
    /// the report to runtime execution or package-consumer proof.
    /// 此方法只读取报告中已经复制到托管侧的值；不会调用 TensorRT、不会暴露 parser 拥有的 tensor 指针，
    /// 也不会将报告晋级为 runtime execution 或 package-consumer proof。
    /// </remarks>
    /// <returns>A compact ONNX model support summary. 紧凑 ONNX 模型支持性摘要。</returns>
    public TensorRtOnnxModelSupportSummary ToSummary()
    {
        long copiedNodeCount = 0;
        long copiedSupportedSubgraphCount = 0;
        long copiedUnsupportedSubgraphCount = 0;

        for (int index = 0; index < Subgraphs.Count; index++)
        {
            TensorRtOnnxSubgraphSupportInfo subgraph = Subgraphs[index];
            copiedNodeCount += subgraph.Nodes.Count;
            if (subgraph.IsSupported)
            {
                copiedSupportedSubgraphCount++;
            }
            else
            {
                copiedUnsupportedSubgraphCount++;
            }
        }

        return new TensorRtOnnxModelSupportSummary(
            IsSupported,
            SupportedSubgraphCount,
            UnsupportedSubgraphCount,
            Subgraphs.Count,
            copiedSupportedSubgraphCount,
            copiedUnsupportedSubgraphCount,
            copiedNodeCount);
    }

    /// <summary>
    /// Returns a compact support summary.
    /// 返回紧凑的支持性摘要。
    /// </summary>
    public override string ToString()
    {
        return $"Supported={IsSupported} Subgraphs={Subgraphs.Count} SupportedSubgraphs={SupportedSubgraphCount} UnsupportedSubgraphs={UnsupportedSubgraphCount}";
    }
}

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

/// <summary>
/// Describes one ONNX parser subgraph support result.
/// 描述一条 ONNX parser 子图支持性结果。
/// </summary>
public sealed class TensorRtOnnxSubgraphSupportInfo
{
    /// <summary>
    /// Creates a subgraph support result.
    /// 创建子图支持性结果。
    /// </summary>
    /// <param name="index">Zero-based subgraph index. 从零开始的子图索引。</param>
    /// <param name="isSupported">Whether TensorRT reports the subgraph as supported. TensorRT 是否报告该子图受支持。</param>
    /// <param name="nodes">ONNX node indexes included in this subgraph. 当前子图包含的 ONNX 节点索引。</param>
    public TensorRtOnnxSubgraphSupportInfo(long index, bool isSupported, IReadOnlyList<long> nodes)
    {
        Index = index;
        IsSupported = isSupported;
        Nodes = nodes ?? Array.Empty<long>();
    }

    /// <summary>
    /// Gets the zero-based subgraph index.
    /// 获取从零开始的子图索引。
    /// </summary>
    public long Index { get; }

    /// <summary>
    /// Gets whether TensorRT reports this subgraph as supported.
    /// 获取 TensorRT 是否报告该子图受支持。
    /// </summary>
    public bool IsSupported { get; }

    /// <summary>
    /// Gets the ONNX node indexes included in this subgraph.
    /// 获取当前子图包含的 ONNX 节点索引。
    /// </summary>
    public IReadOnlyList<long> Nodes { get; }

    /// <summary>
    /// Returns a compact subgraph summary.
    /// 返回紧凑的子图摘要。
    /// </summary>
    public override string ToString()
    {
        return $"Index={Index} Supported={IsSupported} Nodes={Nodes.Count}";
    }
}
