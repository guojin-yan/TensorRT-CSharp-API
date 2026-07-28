using System.Collections.Generic;

namespace JYPPX.CudaSharp;

/// <summary>
/// Captures a bounded copied diagnostic snapshot for a CUDA graph.
/// 捕获 CUDA graph 的有界复制型诊断快照。
/// </summary>
public sealed class CudaGraphDiagnosticSnapshot
{
    /// <summary>
    /// Initializes a CUDA graph diagnostic snapshot.
    /// 初始化 CUDA graph 诊断快照。
    /// </summary>
    /// <param name="topology">The graph-level topology snapshot. graph 级拓扑快照。</param>
    /// <param name="nodes">The bounded node snapshots. 有界 node 快照。</param>
    /// <param name="rootNodes">The bounded root-node snapshots. 有界 root-node 快照。</param>
    /// <param name="edges">The bounded edge snapshots. 有界 edge 快照。</param>
    public CudaGraphDiagnosticSnapshot(
        CudaGraphTopologySnapshot topology,
        IReadOnlyList<CudaGraphNodeSnapshot> nodes,
        IReadOnlyList<CudaGraphNodeSnapshot> rootNodes,
        IReadOnlyList<CudaGraphEdgeSnapshot> edges)
    {
        Topology = topology;
        Nodes = nodes;
        RootNodes = rootNodes;
        Edges = edges;
    }

    /// <summary>
    /// Gets the graph-level topology snapshot.
    /// 获取 graph 级拓扑快照。
    /// </summary>
    public CudaGraphTopologySnapshot Topology { get; }

    /// <summary>
    /// Gets the bounded node snapshots.
    /// 获取有界 node 快照。
    /// </summary>
    public IReadOnlyList<CudaGraphNodeSnapshot> Nodes { get; }

    /// <summary>
    /// Gets the bounded root-node snapshots.
    /// 获取有界 root-node 快照。
    /// </summary>
    public IReadOnlyList<CudaGraphNodeSnapshot> RootNodes { get; }

    /// <summary>
    /// Gets the bounded edge snapshots.
    /// 获取有界 edge 快照。
    /// </summary>
    public IReadOnlyList<CudaGraphEdgeSnapshot> Edges { get; }

    /// <summary>
    /// Converts this bounded graph diagnostic snapshot into a compact pointer-free summary.
    /// 将当前有界 graph 诊断快照转换为紧凑、无指针逃逸的摘要。
    /// </summary>
    /// <remarks>
    /// This method only reads copied managed snapshot values. It does not call CUDA, does not expose
    /// native graph or node pointers, and does not promote local diagnostics to runtime proof.
    /// 此方法只读取已经复制到托管侧的快照值；不会调用 CUDA、不会暴露原生 graph 或 node 指针，
    /// 也不会将本地诊断晋级为 runtime proof。
    /// </remarks>
    public CudaGraphDiagnosticSummary ToSummary()
    {
        ulong listedDependencyCount = 0;
        ulong listedDependencyWithEdgeDataCount = 0;
        ulong listedDependentCount = 0;
        ulong listedDependentWithEdgeDataCount = 0;
        ulong listedEdgeDataCount = 0;

        for (int index = 0; index < Nodes.Count; index++)
        {
            CudaGraphNodeTopologySnapshot topology = Nodes[index].Topology;
            listedDependencyCount += topology.DependencyCount;
            listedDependencyWithEdgeDataCount += topology.DependencyWithEdgeDataCount;
            listedDependentCount += topology.DependentCount;
            listedDependentWithEdgeDataCount += topology.DependentWithEdgeDataCount;
        }

        for (int index = 0; index < Edges.Count; index++)
        {
            if (Edges[index].HasEdgeData)
            {
                listedEdgeDataCount++;
            }
        }

        return new CudaGraphDiagnosticSummary(
            Topology.NodeCount,
            Topology.RootNodeCount,
            Topology.EdgeCount,
            Topology.EdgeWithDataCount,
            Nodes.Count,
            RootNodes.Count,
            Edges.Count,
            listedEdgeDataCount,
            listedDependencyCount,
            listedDependencyWithEdgeDataCount,
            listedDependentCount,
            listedDependentWithEdgeDataCount);
    }

    /// <summary>
    /// Formats this snapshot for diagnostics.
    /// 将该快照格式化为诊断字符串。
    /// </summary>
    public override string ToString() =>
        $"{Topology}, ListedNodes={Nodes.Count}, ListedRoots={RootNodes.Count}, ListedEdges={Edges.Count}";
}

/// <summary>
/// Summarizes copied CUDA graph diagnostics without exposing native graph or node pointers.
/// 汇总已复制的 CUDA graph 诊断信息，不暴露原生 graph 或 node 指针。
/// </summary>
public sealed class CudaGraphDiagnosticSummary
{
    internal CudaGraphDiagnosticSummary(
        ulong reportedNodeCount,
        ulong reportedRootNodeCount,
        ulong reportedEdgeCount,
        ulong reportedEdgeWithDataCount,
        int copiedNodeSnapshotCount,
        int copiedRootNodeSnapshotCount,
        int copiedEdgeSnapshotCount,
        ulong copiedEdgeDataSnapshotCount,
        ulong copiedNodeDependencyCount,
        ulong copiedNodeDependencyWithEdgeDataCount,
        ulong copiedNodeDependentCount,
        ulong copiedNodeDependentWithEdgeDataCount)
    {
        ReportedNodeCount = reportedNodeCount;
        ReportedRootNodeCount = reportedRootNodeCount;
        ReportedEdgeCount = reportedEdgeCount;
        ReportedEdgeWithDataCount = reportedEdgeWithDataCount;
        CopiedNodeSnapshotCount = copiedNodeSnapshotCount < 0 ? 0 : copiedNodeSnapshotCount;
        CopiedRootNodeSnapshotCount = copiedRootNodeSnapshotCount < 0 ? 0 : copiedRootNodeSnapshotCount;
        CopiedEdgeSnapshotCount = copiedEdgeSnapshotCount < 0 ? 0 : copiedEdgeSnapshotCount;
        CopiedEdgeDataSnapshotCount = copiedEdgeDataSnapshotCount;
        CopiedNodeDependencyCount = copiedNodeDependencyCount;
        CopiedNodeDependencyWithEdgeDataCount = copiedNodeDependencyWithEdgeDataCount;
        CopiedNodeDependentCount = copiedNodeDependentCount;
        CopiedNodeDependentWithEdgeDataCount = copiedNodeDependentWithEdgeDataCount;
    }

    /// <summary>Gets the graph node count reported by CUDA. 获取 CUDA 报告的 graph node 数量。</summary>
    public ulong ReportedNodeCount { get; }

    /// <summary>Gets the graph root-node count reported by CUDA. 获取 CUDA 报告的 root node 数量。</summary>
    public ulong ReportedRootNodeCount { get; }

    /// <summary>Gets the graph edge count reported by CUDA. 获取 CUDA 报告的 edge 数量。</summary>
    public ulong ReportedEdgeCount { get; }

    /// <summary>Gets the graph edge-data count reported by CUDA. 获取 CUDA edge-data 查询报告的 edge 数量。</summary>
    public ulong ReportedEdgeWithDataCount { get; }

    /// <summary>Gets the bounded copied node snapshot count. 获取有界复制的 node snapshot 数量。</summary>
    public int CopiedNodeSnapshotCount { get; }

    /// <summary>Gets the bounded copied root-node snapshot count. 获取有界复制的 root-node snapshot 数量。</summary>
    public int CopiedRootNodeSnapshotCount { get; }

    /// <summary>Gets the bounded copied edge snapshot count. 获取有界复制的 edge snapshot 数量。</summary>
    public int CopiedEdgeSnapshotCount { get; }

    /// <summary>Gets the copied edge snapshots that include CUDA edge data. 获取包含 CUDA edge data 的已复制 edge snapshot 数量。</summary>
    public ulong CopiedEdgeDataSnapshotCount { get; }

    /// <summary>Gets copied dependency counts summarized from node snapshots. 获取从 node snapshot 汇总出的 dependency 数量。</summary>
    public ulong CopiedNodeDependencyCount { get; }

    /// <summary>Gets copied dependency-with-edge-data counts summarized from node snapshots. 获取从 node snapshot 汇总出的 dependency edge-data 数量。</summary>
    public ulong CopiedNodeDependencyWithEdgeDataCount { get; }

    /// <summary>Gets copied dependent counts summarized from node snapshots. 获取从 node snapshot 汇总出的 dependent 数量。</summary>
    public ulong CopiedNodeDependentCount { get; }

    /// <summary>Gets copied dependent-with-edge-data counts summarized from node snapshots. 获取从 node snapshot 汇总出的 dependent edge-data 数量。</summary>
    public ulong CopiedNodeDependentWithEdgeDataCount { get; }

    /// <summary>Gets whether copied node snapshots cover the reported node count. 获取已复制 node snapshot 是否覆盖报告的 node 数量。</summary>
    public bool CopiedNodeSnapshotsCoverReportedCount => (ulong)CopiedNodeSnapshotCount == ReportedNodeCount;

    /// <summary>Gets whether copied root snapshots cover the reported root count. 获取已复制 root snapshot 是否覆盖报告的 root 数量。</summary>
    public bool CopiedRootSnapshotsCoverReportedCount => (ulong)CopiedRootNodeSnapshotCount == ReportedRootNodeCount;

    /// <summary>Gets whether copied edge snapshots cover the reported edge-data count. 获取已复制 edge snapshot 是否覆盖报告的 edge-data 数量。</summary>
    public bool CopiedEdgeDataSnapshotsCoverReportedCount => CopiedEdgeDataSnapshotCount == ReportedEdgeWithDataCount;

    /// <summary>Gets the runtime evidence kind represented by this copied summary. 获取该 copied summary 表示的 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "copied-readonly-summary";

    /// <summary>Gets whether this summary is runtime execution evidence. 获取该摘要是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this summary is runtime execution proof. 获取该摘要是否为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets whether this summary is copied and pointer-free. 获取该摘要是否为复制型且无指针逃逸。</summary>
    public bool PointerFreeCopiedSummary => true;

    /// <summary>Gets whether this summary can be promoted as runtime proof. 获取该摘要是否可晋级为 runtime proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether this summary can promote public release proof. 获取该摘要是否可晋级为 public release proof。</summary>
    public bool CanPromoteReleaseProof => false;

    /// <summary>Gets whether deferred history can be deleted because of this summary. 获取是否可因该摘要删除 deferred history。</summary>
    public bool CanDeleteDeferredRecord => false;

    /// <summary>Formats this summary for logs and smoke output. 将该摘要格式化为日志和 smoke 输出。</summary>
    public override string ToString()
    {
        return $"ReportedNodes={ReportedNodeCount} CopiedNodes={CopiedNodeSnapshotCount} ReportedRoots={ReportedRootNodeCount} CopiedRoots={CopiedRootNodeSnapshotCount} ReportedEdges={ReportedEdgeCount} CopiedEdges={CopiedEdgeSnapshotCount} EdgeData={CopiedEdgeDataSnapshotCount}/{ReportedEdgeWithDataCount} RuntimeProof={CanPromoteRuntimeProof}";
    }
}
