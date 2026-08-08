using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaGraphBorrowedHandleSafetyGateTests
{
    [Fact]
    public void BorrowedHandleDeferredRecordsRemainUntilSafeAbiExists()
    {
        string twentyThirdDeferredManifest = ReadSource("native", "manifests", "cuda", "cuda-twenty-third-batch-deferred-coverage.manifest.json");
        string thirtySeventhDeferredManifest = ReadSource("native", "manifests", "cuda", "cuda-thirty-seventh-batch-graph-boundaries.manifest.json");
        string twentyThirdDeferredSource = ReadSource("native", "src", "cuda", "modules", "deferred", "twenty_third_batch_deferred.inc");
        string thirtySeventhDeferredSource = ReadSource("native", "src", "cuda", "modules", "deferred", "thirty_seventh_batch_graph_deferred.inc");
        string coverageScript = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("cuda-cuda-graph-child-graph-node-get-graph-deferred", twentyThirdDeferredManifest);
        Assert.Contains("cuda-cuda-graph-event-record-node-get-event-deferred", twentyThirdDeferredManifest);
        Assert.Contains("cuda-cuda-graph-event-wait-node-get-event-deferred", twentyThirdDeferredManifest);
        Assert.Contains("cuda-graph-node-get-containing-graph-deferred", thirtySeventhDeferredManifest);
        Assert.Contains("cuda-graph-node-get-params-deferred", thirtySeventhDeferredManifest);

        Assert.Contains("cudaGraphChildGraphNodeGetGraph deferred", twentyThirdDeferredSource);
        Assert.Contains("cudaGraphEventRecordNodeGetEvent deferred", twentyThirdDeferredSource);
        Assert.Contains("cudaGraphEventWaitNodeGetEvent deferred", twentyThirdDeferredSource);
        Assert.Contains("cudaGraphNodeGetContainingGraph deferred", thirtySeventhDeferredSource);
        Assert.Contains("cudaGraphNodeGetParams deferred", thirtySeventhDeferredSource);

        Assert.Contains("\"cudaGraphNodeGetContainingGraph\" = @(\"graph-node-get-containing-graph-deferred\", \"graph-node-is-in-graph-safe\")", coverageScript);
        Assert.Contains("\"cudaGraphNodeGetParams\" = @(\"graph-node-get-params-deferred\")", coverageScript);
        Assert.Contains("\"cudaGraphEventRecordNodeGetEvent\" = @(\"graph-event-record-node-get-event-deferred\", \"graph-event-record-node-has-event-safe\")", coverageScript);
        Assert.Contains("\"cudaGraphEventWaitNodeGetEvent\" = @(\"graph-event-wait-node-get-event-deferred\", \"graph-event-wait-node-has-event-safe\")", coverageScript);
    }

    [Fact]
    public void ManagedGraphSurfaceDoesNotExposeBorrowedNativeHandlesPublicly()
    {
        string graph =
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.ConditionalHandles.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.GraphComposition.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.NodeCreation.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.TopologyDiagnostics.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.NodeInspection.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.NodeMutation.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.NodeRelations.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.Instantiation.cs");
        string graphExec = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphExec.cs");
        string graphNode = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphNode.cs");
        string publicSurface = graph + graphExec + graphNode;

        Assert.DoesNotContain("public IntPtr", publicSurface);
        Assert.DoesNotContain("public nint", publicSurface);
        Assert.DoesNotContain("public CudaGraph GetChildGraph", publicSurface);
        Assert.DoesNotContain("GetContainingGraph", publicSurface);
        Assert.DoesNotContain("GetEvent(CudaGraphNode", publicSurface);
        Assert.DoesNotContain("GetParams(CudaGraphNode", publicSurface);
        Assert.Contains("ContainsNode(CudaGraphNode node)", graph);
        Assert.Contains("GetChildGraphSnapshot(CudaGraphNode node)", graph);
        Assert.Contains("internal UIntPtr Token", graphNode);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
