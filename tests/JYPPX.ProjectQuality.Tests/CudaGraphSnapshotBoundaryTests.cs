using JYPPX.CudaSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaGraphSnapshotBoundaryTests
{
    [Fact]
    public void ManagedCudaGraphSnapshotsExposeCopiedScalarsOnly()
    {
        string graph =
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.TopologyDiagnostics.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.NodeInspection.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.NodeRelations.cs");
        string graphExec = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphExec.cs");
        string device = ReadSource(
            "src", "JYPPX.CudaSharp", "Devices", "CudaDevice.GraphResources.cs");
        string graphSnapshot = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphTopologySnapshot.cs");
        string nodeSnapshot = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphNodeTopologySnapshot.cs");
        string execNodeSnapshot = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphExecNodeStateSnapshot.cs");
        string nodeListSnapshot = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphNodeSnapshot.cs");
        string adjacentNodeSnapshot = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphAdjacentNodeSnapshot.cs");
        string edgeSnapshot = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphEdgeSnapshot.cs");
        string diagnosticSnapshot = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphDiagnosticSnapshot.cs");
        string execDiagnosticSnapshot = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphExecDiagnosticSnapshot.cs");
        string deviceGraphMemoryInfo = ReadSource("src", "JYPPX.CudaSharp", "Devices", "CudaDeviceGraphMemoryInfo.cs");

        Assert.Contains("public CudaGraphTopologySnapshot GetTopologySnapshot()", graph);
        Assert.Contains("public static CudaGraphNodeTopologySnapshot GetNodeTopologySnapshot(CudaGraphNode node)", graph);
        Assert.Contains("public CudaGraphExecNodeStateSnapshot GetNodeStateSnapshot(CudaGraphNode node)", graphExec);
        Assert.Contains("public IReadOnlyList<CudaGraphNodeSnapshot> GetNodeSnapshotList", graph);
        Assert.Contains("public IReadOnlyList<CudaGraphNodeSnapshot> GetRootNodeSnapshotList", graph);
        Assert.Contains("public IReadOnlyList<CudaGraphEdgeSnapshot> GetEdgeSnapshotList", graph);
        Assert.Contains("public static IReadOnlyList<CudaGraphAdjacentNodeSnapshot> GetNodeDependencySnapshotList", graph);
        Assert.Contains("public static IReadOnlyList<CudaGraphAdjacentNodeSnapshot> GetNodeDependentSnapshotList", graph);
        Assert.Contains("public CudaGraphDiagnosticSnapshot GetDiagnosticSnapshot", graph);
        Assert.Contains("public CudaGraphExecDiagnosticSnapshot GetDiagnosticSnapshot(IReadOnlyList<CudaGraphNode> nodes)", graphExec);
        Assert.Contains("public sealed class CudaGraphTopologySnapshot", graphSnapshot);
        Assert.Contains("public sealed class CudaGraphNodeTopologySnapshot", nodeSnapshot);
        Assert.Contains("public sealed class CudaGraphExecNodeStateSnapshot", execNodeSnapshot);
        Assert.Contains("public sealed class CudaGraphNodeSnapshot", nodeListSnapshot);
        Assert.Contains("public sealed class CudaGraphAdjacentNodeSnapshot", adjacentNodeSnapshot);
        Assert.Contains("public sealed class CudaGraphEdgeSnapshot", edgeSnapshot);
        Assert.Contains("public sealed class CudaGraphDiagnosticSnapshot", diagnosticSnapshot);
        Assert.Contains("public sealed class CudaGraphExecDiagnosticSnapshot", execDiagnosticSnapshot);
        Assert.Contains("public CudaGraphDiagnosticSummary ToSummary()", diagnosticSnapshot);
        Assert.Contains("public sealed class CudaGraphDiagnosticSummary", diagnosticSnapshot);
        Assert.Contains("public CudaGraphExecDiagnosticSummary ToSummary()", execDiagnosticSnapshot);
        Assert.Contains("public sealed class CudaGraphExecDiagnosticSummary", execDiagnosticSnapshot);
        Assert.Contains("public static CudaDeviceGraphMemorySummary GetGraphMemorySummary(int ordinal)", device);
        Assert.Contains("public static CudaDeviceGraphMemorySummary CurrentGraphMemorySummary", device);
        Assert.Contains("public CudaDeviceGraphMemorySummary ToSummary()", deviceGraphMemoryInfo);
        Assert.Contains("public sealed class CudaDeviceGraphMemorySummary", deviceGraphMemoryInfo);
        Assert.Contains("public ulong NodeCount", graphSnapshot);
        Assert.Contains("public ulong RootNodeCount", graphSnapshot);
        Assert.Contains("public ulong EdgeCount", graphSnapshot);
        Assert.Contains("public ulong EdgeWithDataCount", graphSnapshot);
        Assert.Contains("public CudaGraphNodeType NodeType", nodeSnapshot);
        Assert.Contains("public ulong DependencyCount", nodeSnapshot);
        Assert.Contains("public ulong DependencyWithEdgeDataCount", nodeSnapshot);
        Assert.Contains("public ulong DependentCount", nodeSnapshot);
        Assert.Contains("public ulong DependentWithEdgeDataCount", nodeSnapshot);
        Assert.Contains("public bool Enabled", execNodeSnapshot);
        Assert.Contains("public ulong Flags", execNodeSnapshot);
        Assert.Contains("public string NodeText", nodeListSnapshot);
        Assert.Contains("public string EdgeDataState", adjacentNodeSnapshot);
        Assert.Contains("public string EdgeDataState", edgeSnapshot);
        Assert.Contains("public IReadOnlyList<CudaGraphNodeSnapshot> Nodes", diagnosticSnapshot);
        Assert.Contains("public IReadOnlyList<CudaGraphExecNodeStateSnapshot> NodeStates", execDiagnosticSnapshot);
        Assert.Contains("public ulong ReportedNodeCount", diagnosticSnapshot);
        Assert.Contains("public int CopiedNodeSnapshotCount", diagnosticSnapshot);
        Assert.Contains("public ulong CopiedEdgeDataSnapshotCount", diagnosticSnapshot);
        Assert.Contains("public bool CopiedEdgeDataSnapshotsCoverReportedCount", diagnosticSnapshot);
        Assert.Contains("public bool PointerFreeCopiedSummary", diagnosticSnapshot);
        Assert.Contains("public bool CanPromoteRuntimeProof", diagnosticSnapshot);
        Assert.Contains("public bool CanDeleteDeferredRecord", diagnosticSnapshot);
        Assert.Contains("public int CopiedNodeStateCount", execDiagnosticSnapshot);
        Assert.Contains("public int EnabledNodeStateCount", execDiagnosticSnapshot);
        Assert.Contains("public int DisabledNodeStateCount", execDiagnosticSnapshot);
        Assert.Contains("public bool CopiedStateCountsMatchNodeStateCount", execDiagnosticSnapshot);
        Assert.Contains("public bool PointerFreeCopiedSummary", execDiagnosticSnapshot);
        Assert.Contains("public bool CanPromoteRuntimeProof", execDiagnosticSnapshot);
        Assert.Contains("public bool CanDeleteDeferredRecord", execDiagnosticSnapshot);
        Assert.Contains("public int CopiedScalarCounterCount => 4", deviceGraphMemoryInfo);
        Assert.Contains("public bool CurrentCountersWithinHighWatermarks", deviceGraphMemoryInfo);
        Assert.Contains("public bool PointerFreeCopiedSummary", deviceGraphMemoryInfo);
        Assert.Contains("public bool CanPromoteRuntimeProof", deviceGraphMemoryInfo);
        Assert.Contains("public bool CanDeleteDeferredRecord", deviceGraphMemoryInfo);

        string combined = graph + graphExec + device + graphSnapshot + nodeSnapshot + execNodeSnapshot + nodeListSnapshot + adjacentNodeSnapshot + edgeSnapshot + diagnosticSnapshot + execDiagnosticSnapshot + deviceGraphMemoryInfo;
        Assert.DoesNotContain("public IntPtr", combined);
        Assert.DoesNotContain("public nint", combined);
        Assert.DoesNotContain("public UIntPtr", graphSnapshot + nodeSnapshot + execNodeSnapshot + nodeListSnapshot + adjacentNodeSnapshot + edgeSnapshot + diagnosticSnapshot + execDiagnosticSnapshot + deviceGraphMemoryInfo);
    }

    [Fact]
    public void DeviceGraphMemoryInfoSummaryKeepsCopiedCountersOutOfRuntimeProof()
    {
        CudaDeviceGraphMemoryInfo info = new CudaDeviceGraphMemoryInfo(
            deviceOrdinal: 0,
            usedMemoryCurrentBytes: 16,
            usedMemoryHighBytes: 32,
            reservedMemoryCurrentBytes: 64,
            reservedMemoryHighBytes: 128);

        CudaDeviceGraphMemorySummary summary = info.ToSummary();

        Assert.Equal(0, summary.DeviceOrdinal);
        Assert.Equal(4, summary.CopiedScalarCounterCount);
        Assert.True(summary.CurrentCountersWithinHighWatermarks);
        Assert.Equal("copied-readonly-summary", summary.RuntimeEvidenceKind);
        Assert.False(summary.IsRuntimeExecutionEvidence);
        Assert.False(summary.IsRuntimeExecutionProof);
        Assert.True(summary.PointerFreeCopiedSummary);
        Assert.False(summary.CanPromoteRuntimeProof);
        Assert.False(summary.CanPromoteReleaseProof);
        Assert.False(summary.CanDeleteDeferredRecord);
        Assert.Contains("Counters=4", summary.ToString());
    }

    [Fact]
    public void SnapshotHelpersReuseExistingSafeBridgeQueries()
    {
        string graph =
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.TopologyDiagnostics.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.NodeInspection.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.NodeRelations.cs");
        string graphExec = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphExec.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Graph", "NativeCudaApi.Graph.cs");

        Assert.Contains("GetEdgeWithEdgeDataCount()", graph);
        Assert.Contains("GetDependencyWithEdgeDataCount(node)", graph);
        Assert.Contains("GetDependentWithEdgeDataCount(node)", graph);
        Assert.Contains("GetNode(index)", graph);
        Assert.Contains("GetRootNode(index)", graph);
        Assert.Contains("GetEdgeWithEdgeData(index)", graph);
        Assert.Contains("GetEdge(index)", graph);
        Assert.Contains("GetDependencyWithEdgeData(node, index)", graph);
        Assert.Contains("GetDependency(node, index)", graph);
        Assert.Contains("GetDependentWithEdgeData(node, index)", graph);
        Assert.Contains("GetDependent(node, index)", graph);
        Assert.Contains("GetNodeEnabled(node)", graphExec);
        Assert.Contains("Flags", graphExec);
        Assert.Contains("new CudaGraphExecNodeStateSnapshot((ulong)index, node, GetNodeEnabled(node), Flags)", graphExec);
        Assert.Contains("GetGraphEdgeV2Count", interop);
        Assert.Contains("GetGraphEdgeV2", interop);
        Assert.Contains("GetGraphNodeDependencyV2Count", interop);
        Assert.Contains("GetGraphNodeDependencyV2", interop);
        Assert.Contains("GetGraphNodeDependentV2Count", interop);
        Assert.Contains("GetGraphNodeDependentV2", interop);
        Assert.Contains("GetGraphExecNodeEnabled", interop);
        Assert.Contains("GetGraphExecFlags", interop);
    }

    [Fact]
    public void CudaGraphSmokeCoversSnapshotApis()
    {
        string program = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");

        Assert.Contains("GetTopologySnapshot", program);
        Assert.Contains("GetNodeTopologySnapshot", program);
        Assert.Contains("GetNodeStateSnapshot", program);
        Assert.Contains("GetNodeSnapshotList", program);
        Assert.Contains("GetRootNodeSnapshotList", program);
        Assert.Contains("GetEdgeSnapshotList", program);
        Assert.Contains("GetNodeDependencySnapshotList", program);
        Assert.Contains("GetNodeDependentSnapshotList", program);
        Assert.Contains("GetDiagnosticSnapshot", program);
        Assert.Contains("CudaGraphSnapshots", program);
        Assert.Contains("CudaGraphSnapshotLists", program);
        Assert.Contains("GraphSnapshot=", program);
        Assert.Contains("GraphDiagnosticSummary=", program);
        Assert.Contains("GraphExecDiagnosticSummary=", program);
        Assert.Contains("CudaDeviceGraphMemorySummary", program);
        Assert.Contains("Summary=[", program);
        Assert.Contains("AfterSummary=[", program);
        Assert.Contains("RootNodeSnapshot=", program);
        Assert.Contains("ChildNodeSnapshot=", program);
        Assert.Contains("ExecNodeSnapshot=", program);
        Assert.Contains("EdgeDataState=", program);
        Assert.Contains("Skipped:", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
