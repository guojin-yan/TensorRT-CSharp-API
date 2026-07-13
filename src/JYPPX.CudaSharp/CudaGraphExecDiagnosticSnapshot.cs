using System.Collections.Generic;

namespace JYPPX.CudaSharp;

/// <summary>
/// Captures copied executable-state diagnostics for multiple CUDA graph nodes.
/// 捕获多个 CUDA graph node 的复制型 executable 状态诊断。
/// </summary>
public sealed class CudaGraphExecDiagnosticSnapshot
{
    /// <summary>
    /// Initializes a CUDA graph executable diagnostic snapshot.
    /// 初始化 CUDA graph executable 诊断快照。
    /// </summary>
    /// <param name="nodeStates">The copied node-state snapshots. 复制出的 node 状态快照。</param>
    public CudaGraphExecDiagnosticSnapshot(IReadOnlyList<CudaGraphExecNodeStateSnapshot> nodeStates)
    {
        NodeStates = nodeStates;
    }

    /// <summary>
    /// Gets the copied node-state snapshots.
    /// 获取复制出的 node 状态快照。
    /// </summary>
    public IReadOnlyList<CudaGraphExecNodeStateSnapshot> NodeStates { get; }

    /// <summary>
    /// Converts this executable graph diagnostic snapshot into a compact pointer-free summary.
    /// 将当前 executable graph 诊断快照转换为紧凑、无指针逃逸的摘要。
    /// </summary>
    /// <remarks>
    /// This method reads copied managed node-state snapshots only. It does not call CUDA and does not
    /// expose native graph executable or node pointers.
    /// 此方法只读取已复制的托管 node-state snapshot，不调用 CUDA，也不暴露原生 graph executable 或 node 指针。
    /// </remarks>
    public CudaGraphExecDiagnosticSummary ToSummary()
    {
        int enabledCount = 0;
        int disabledCount = 0;
        int snapshotsWithNodeToken = 0;
        ulong flagsOr = 0;

        for (int index = 0; index < NodeStates.Count; index++)
        {
            CudaGraphExecNodeStateSnapshot state = NodeStates[index];
            if (state.Enabled)
            {
                enabledCount++;
            }
            else
            {
                disabledCount++;
            }

            if (state.HasNode)
            {
                snapshotsWithNodeToken++;
            }

            flagsOr |= state.Flags;
        }

        return new CudaGraphExecDiagnosticSummary(
            NodeStates.Count,
            enabledCount,
            disabledCount,
            snapshotsWithNodeToken,
            flagsOr);
    }

    /// <summary>
    /// Formats this snapshot for diagnostics.
    /// 将该快照格式化为诊断字符串。
    /// </summary>
    public override string ToString() => $"NodeStates={NodeStates.Count}";
}

/// <summary>
/// Summarizes copied CUDA graph executable diagnostics without exposing native pointers.
/// 汇总已复制的 CUDA graph executable 诊断信息，不暴露原生指针。
/// </summary>
public sealed class CudaGraphExecDiagnosticSummary
{
    internal CudaGraphExecDiagnosticSummary(
        int copiedNodeStateCount,
        int enabledNodeStateCount,
        int disabledNodeStateCount,
        int snapshotsWithNodeTokenCount,
        ulong flagsOr)
    {
        CopiedNodeStateCount = copiedNodeStateCount < 0 ? 0 : copiedNodeStateCount;
        EnabledNodeStateCount = enabledNodeStateCount < 0 ? 0 : enabledNodeStateCount;
        DisabledNodeStateCount = disabledNodeStateCount < 0 ? 0 : disabledNodeStateCount;
        SnapshotsWithNodeTokenCount = snapshotsWithNodeTokenCount < 0 ? 0 : snapshotsWithNodeTokenCount;
        FlagsOr = flagsOr;
    }

    /// <summary>Gets the copied node-state snapshot count. 获取已复制 node-state snapshot 数量。</summary>
    public int CopiedNodeStateCount { get; }

    /// <summary>Gets the copied enabled node-state count. 获取已复制 enabled node-state 数量。</summary>
    public int EnabledNodeStateCount { get; }

    /// <summary>Gets the copied disabled node-state count. 获取已复制 disabled node-state 数量。</summary>
    public int DisabledNodeStateCount { get; }

    /// <summary>Gets the copied snapshots that include graph-owned node value tokens. 获取包含 graph-owned node 值 token 的已复制 snapshot 数量。</summary>
    public int SnapshotsWithNodeTokenCount { get; }

    /// <summary>Gets the bitwise OR of copied executable graph flags. 获取已复制 executable graph flags 的按位 OR。</summary>
    public ulong FlagsOr { get; }

    /// <summary>Gets whether copied enabled and disabled counts cover all node states. 获取 enabled/disabled 计数是否覆盖全部 node state。</summary>
    public bool CopiedStateCountsMatchNodeStateCount => EnabledNodeStateCount + DisabledNodeStateCount == CopiedNodeStateCount;

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
        return $"NodeStates={CopiedNodeStateCount} Enabled={EnabledNodeStateCount} Disabled={DisabledNodeStateCount} NodeTokens={SnapshotsWithNodeTokenCount} FlagsOr={FlagsOr} RuntimeProof={CanPromoteRuntimeProof}";
    }
}
