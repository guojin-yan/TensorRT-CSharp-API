using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaGraphNodeIdentityBoundaryTests
{
    [Fact]
    public void CudaGraphNodeIdentityApisAreLiftedWithoutBorrowedGraphExposure()
    {
        string topologyManifest = ReadSource("native", "manifests", "cuda", "cuda-twenty-fifth-batch-graph-topology.manifest.json");
        string identityManifest = ReadSource("native", "manifests", "cuda", "cuda-twenty-ninth-batch-graph-memory-node-identity.manifest.json");
        string deferredManifest = ReadSource("native", "manifests", "cuda", "cuda-thirty-seventh-batch-graph-boundaries.manifest.json");
        string runtimeHeader = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string nativeSource = ReadSource("native", "src", "cuda", "modules", "graph", "device_graph_memory.inc");
        string topologySource = ReadSource("native", "src", "cuda", "modules", "graph", "node_topology.inc");
        string coverageScript = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("cuda-graph-get-id-safe", topologyManifest);
        Assert.Contains("cuda-graph-exec-get-id-safe", topologyManifest);
        Assert.Contains("cuda-graph-node-is-in-graph-safe", identityManifest);
        Assert.Contains("cuda-graph-node-get-local-id-safe", identityManifest);
        Assert.Contains("cuda-graph-node-get-tools-id-safe", identityManifest);
        Assert.Contains("\"type\": \"uintptr_t\", \"direction\": \"in\", \"managedType\": \"UIntPtr\"", identityManifest);
        Assert.Contains("\"type\": \"JYPPX_Boolean*\", \"direction\": \"out\", \"managedType\": \"out int\"", identityManifest);

        Assert.Contains("cuda-graph-node-get-containing-graph-deferred", deferredManifest);
        Assert.Contains("jyppx_cuda_graph_node_is_in_graph_safe", runtimeHeader);
        Assert.Contains("jyppx_cuda_graph_node_get_local_id_safe", runtimeHeader);
        Assert.Contains("jyppx_cuda_graph_node_get_tools_id_safe", runtimeHeader);
        Assert.Contains("jyppx_cuda_graph_get_id_safe", runtimeHeader);
        Assert.Contains("jyppx_cuda_graph_exec_get_id_safe", runtimeHeader);

        Assert.Contains("cudaGraphNodeGetContainingGraph", nativeSource);
        Assert.Contains("containing_graph == graph_object->handle", nativeSource);
        Assert.Contains("cudaGraphNodeGetLocalId", nativeSource);
        Assert.Contains("cudaGraphNodeGetToolsId", nativeSource);
        Assert.Contains("cudaGraphGetId", topologySource);
        Assert.Contains("cudaGraphExecGetId", topologySource);
        Assert.Contains("CUDART_VERSION >= 13000", nativeSource);
        Assert.Contains("CUDART_VERSION >= 13000", topologySource);

        Assert.Contains("\"cudaGraphNodeGetContainingGraph\" = @(\"graph-node-get-containing-graph-deferred\", \"graph-node-is-in-graph-safe\")", coverageScript);
    }

    [Fact]
    public void ManagedCudaGraphNodeIdentityApiUsesValueTokensAndDoesNotExposeNativePointers()
    {
        string graph =
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.ConditionalHandles.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.NodeInspection.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.NodeRelations.cs");
        string graphExec = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphExec.cs");
        string graphNode = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphNode.cs");
        string interopGraph = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Graph", "NativeCudaApi.Graph.cs");
        string interopIdentity = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Graph", "NativeCudaApi.GraphMemory.cs");

        Assert.Contains("public uint Id => NativeCudaApi.GetGraphId(_handle);", graph);
        Assert.Contains("public uint Id => NativeCudaApi.GetGraphExecId(_handle);", graphExec);
        Assert.Contains("public bool ContainsNode(CudaGraphNode node)", graph);
        Assert.Contains("public static uint GetNodeLocalId(CudaGraphNode node)", graph);
        Assert.Contains("public static ulong GetNodeToolsId(CudaGraphNode node)", graph);
        Assert.Contains("public readonly struct CudaGraphNode", graphNode);
        Assert.Contains("internal UIntPtr Token", graphNode);

        Assert.Contains("GetGraphId(SafeCudaGraphHandle graph)", interopGraph);
        Assert.Contains("GetGraphExecId(SafeCudaGraphExecHandle graphExec)", interopGraph);
        Assert.Contains("IsGraphNodeInGraph(SafeCudaGraphHandle graph, CudaGraphNode node)", interopIdentity);
        Assert.Contains("GetGraphNodeLocalId(CudaGraphNode node)", interopIdentity);
        Assert.Contains("GetGraphNodeToolsId(CudaGraphNode node)", interopIdentity);
        Assert.DoesNotContain("public IntPtr", graph + graphExec + graphNode);
        Assert.DoesNotContain("public nint", graph + graphExec + graphNode);
    }

    [Fact]
    public void CudaGraphSmokeCoversNodeIdentityProbe()
    {
        string program = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");

        Assert.Contains("ProbeGraphNodeIdentity", program);
        Assert.Contains("ContainsNode", program);
        Assert.Contains("GetNodeLocalId", program);
        Assert.Contains("GetNodeToolsId", program);
        Assert.Contains("GraphId=", program);
        Assert.Contains("ExecId=", program);
        Assert.Contains("NodeIdentity=", program);
        Assert.Contains("Skipped=True Reason=CudaException", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
