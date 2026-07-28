using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaGraphEdgeDataBoundaryTests
{
    [Fact]
    public void CudaGraphEdgeDataApisAreLiftedWithoutDeletingDeferredRecords()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-thirty-ninth-batch-graph-edge-data.manifest.json");
        string edgeListManifest = ReadSource("native", "manifests", "cuda", "cuda-fortieth-batch-graph-edge-list-edge-data.manifest.json");
        string deferredManifest = ReadSource("native", "manifests", "cuda", "cuda-thirty-seventh-batch-graph-boundaries.manifest.json");
        string legacyDeferredManifest = ReadSource("native", "manifests", "cuda", "cuda-twenty-third-batch-deferred-coverage.manifest.json");
        string nativeSource = ReadSource("native", "src", "cuda", "modules", "graph", "node_topology.inc");
        string typesHeader = ReadSource("native", "include", "jyppx", "cuda", "types.h");

        Assert.Contains("cuda-graph-add-dependency-v2-safe", manifest);
        Assert.Contains("cuda-graph-node-get-dependencies-v2-count-safe", manifest);
        Assert.Contains("cuda-graph-node-get-dependency-v2-safe", manifest);
        Assert.Contains("cuda-graph-node-get-dependent-nodes-v2-count-safe", manifest);
        Assert.Contains("cuda-graph-node-get-dependent-node-v2-safe", manifest);
        Assert.Contains("cuda-graph-remove-dependency-v2-safe", manifest);
        Assert.Contains("\"type\": \"JYPPX_CudaGraphEdgeData*\", \"direction\": \"out\", \"managedType\": \"out NativeCudaGraphEdgeData\"", manifest);
        Assert.Contains("\"type\": \"const JYPPX_CudaGraphEdgeData*\", \"direction\": \"in\", \"managedType\": \"in NativeCudaGraphEdgeData\"", manifest);
        Assert.Contains("cuda-graph-get-edges-v2-count-safe", edgeListManifest);
        Assert.Contains("cuda-graph-get-edge-v2-safe", edgeListManifest);
        Assert.Contains("\"type\": \"JYPPX_CudaGraphEdgeData*\", \"direction\": \"out\", \"managedType\": \"out NativeCudaGraphEdgeData\"", edgeListManifest);

        Assert.Contains("cuda-graph-node-get-dependencies-v2-deferred", deferredManifest);
        Assert.Contains("cuda-graph-node-get-dependent-nodes-v2-deferred", deferredManifest);
        Assert.Contains("cuda-graph-remove-dependencies-v2-deferred", deferredManifest);
        Assert.Contains("cuda-cuda-graph-get-edges-v2-deferred", legacyDeferredManifest);

        Assert.Contains("typedef struct JYPPX_CudaGraphEdgeData", typesHeader);
        Assert.Contains("cudaGraphAddDependencies_v2", nativeSource);
        Assert.Contains("cudaGraphGetEdges_v2", nativeSource);
        Assert.Contains("cudaGraphNodeGetDependencies_v2", nativeSource);
        Assert.Contains("cudaGraphNodeGetDependentNodes_v2", nativeSource);
        Assert.Contains("cudaGraphRemoveDependencies_v2", nativeSource);
        Assert.Contains("CUDART_VERSION >= 12090", nativeSource);
        Assert.Contains("CUDART_VERSION >= 13000", nativeSource);
        Assert.Contains("jyppx_cuda_graph_edge_data_not_supported", nativeSource);
    }

    [Fact]
    public void ManagedCudaGraphEdgeDataApiDoesNotExposeNativePointers()
    {
        string edgeData = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphEdgeData.cs");
        string graph = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.cs");
        string graphEdge = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphEdge.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Graph", "NativeCudaApi.Graph.cs");
        string structs = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeStructs.cs");

        Assert.Contains("public readonly struct CudaGraphEdgeData", edgeData);
        Assert.Contains("public enum CudaGraphDependencyType", edgeData);
        Assert.Contains("public readonly struct CudaGraphNodeDependency", edgeData);
        Assert.Contains("public readonly struct CudaGraphEdgeWithData", graphEdge);
        Assert.Contains("GetDependencyWithEdgeDataCount", graph);
        Assert.Contains("GetDependencyWithEdgeData", graph);
        Assert.Contains("GetDependentWithEdgeDataCount", graph);
        Assert.Contains("GetDependentWithEdgeData", graph);
        Assert.Contains("GetEdgeWithEdgeDataCount", graph);
        Assert.Contains("GetEdgeWithEdgeData", graph);
        Assert.Contains("GetGraphEdgeV2Count", interop);
        Assert.Contains("GetGraphEdgeV2", interop);
        Assert.Contains("AddDependency(CudaGraphNode fromNode, CudaGraphNode toNode, CudaGraphEdgeData edgeData)", graph);
        Assert.Contains("AddGraphDependencyV2", interop);
        Assert.Contains("RemoveDependency(CudaGraphNode fromNode, CudaGraphNode toNode, CudaGraphEdgeData edgeData)", graph);
        Assert.Contains("NativeCudaGraphEdgeData", structs);
        Assert.Contains("in nativeEdgeData", interop);
        Assert.DoesNotContain("public IntPtr", edgeData + graph + graphEdge);
        Assert.DoesNotContain("public nint", edgeData + graph + graphEdge);
    }

    [Fact]
    public void CudaGraphEdgeDataApisAreDeclaredInPublicNativeHeaderAndCoverageAliases()
    {
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("jyppx_cuda_graph_node_get_dependencies_v2_count_safe", header);
        Assert.Contains("jyppx_cuda_graph_node_get_dependency_v2_safe", header);
        Assert.Contains("jyppx_cuda_graph_node_get_dependent_nodes_v2_count_safe", header);
        Assert.Contains("jyppx_cuda_graph_node_get_dependent_node_v2_safe", header);
        Assert.Contains("jyppx_cuda_graph_add_dependency_v2_safe", header);
        Assert.Contains("jyppx_cuda_graph_remove_dependency_v2_safe", header);
        Assert.Contains("\"cudaGraphAddDependencies\" = @(\"cuda-graph-add-dependencies-deferred\", \"graph-add-dependency-safe\", \"graph-add-dependency-v2-safe\")", script);
        Assert.Contains("\"cudaGraphAddDependencies_v2\" = @(\"cuda-graph-add-dependencies-v2-deferred\", \"graph-add-dependency-v2-safe\")", script);
        Assert.Contains("jyppx_cuda_graph_get_edges_v2_count_safe", header);
        Assert.Contains("jyppx_cuda_graph_get_edge_v2_safe", header);
        Assert.Contains("\"cudaGraphGetEdges\" = @(\"graph-get-edges-v2-deferred\", \"graph-get-edges-v2-count-safe\", \"graph-get-edge-v2-safe\")", script);
        Assert.Contains("\"cudaGraphGetEdges_v2\" = @(\"graph-get-edges-v2-deferred\", \"graph-get-edges-v2-count-safe\", \"graph-get-edge-v2-safe\")", script);
        Assert.Contains("\"cudaGraphNodeGetDependencies_v2\" = @(\"graph-node-get-dependencies-v2-deferred\", \"graph-node-get-dependencies-v2-count-safe\", \"graph-node-get-dependency-v2-safe\")", script);
        Assert.Contains("\"cudaGraphNodeGetDependentNodes_v2\" = @(\"graph-node-get-dependent-nodes-v2-deferred\", \"graph-node-get-dependent-nodes-v2-count-safe\", \"graph-node-get-dependent-node-v2-safe\")", script);
        Assert.Contains("\"cudaGraphRemoveDependencies_v2\" = @(\"graph-remove-dependencies-v2-deferred\", \"graph-remove-dependency-v2-safe\")", script);
    }

    [Fact]
    public void CudaGraphSmokeCoversEdgeDataQueryAndRemove()
    {
        string program = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");

        Assert.Contains("ProbeGraphEdgeData", program);
        Assert.Contains("GetDependencyWithEdgeDataCount", program);
        Assert.Contains("GetDependencyWithEdgeData", program);
        Assert.Contains("GetDependentWithEdgeDataCount", program);
        Assert.Contains("GetDependentWithEdgeData", program);
        Assert.Contains("GetEdgeWithEdgeDataCount", program);
        Assert.Contains("GetEdgeWithEdgeData", program);
        Assert.Contains("AddDependency(root, child, dependency.EdgeData)", program);
        Assert.Contains("RemoveDependency(root, child, dependency.EdgeData)", program);
        Assert.Contains("GraphEdges=", program);
        Assert.Contains("EdgeData=", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
