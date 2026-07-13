using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaGraphBorrowedHandleSafetyGateTests
{
    [Fact]
    public void BorrowedHandleSafetyGateDocumentsDeferredApisAndSafeAlternatives()
    {
        string article = ReadSource("docs", "articles", "zh-cn", "cuda-graph-borrowed-handle-safety-gate.md");
        string toc = ReadSource("docs", "toc.yml");

        Assert.Contains("cuda-graph-borrowed-handle-safety-gate", article);
        Assert.Contains("cudaGraphChildGraphNodeGetGraph", article);
        Assert.Contains("cudaGraphEventRecordNodeGetEvent", article);
        Assert.Contains("cudaGraphEventWaitNodeGetEvent", article);
        Assert.Contains("cudaGraphNodeGetContainingGraph", article);
        Assert.Contains("cudaGraphNodeGetParams", article);
        Assert.Contains("copied metadata/snapshot", article);
        Assert.Contains("typed copied descriptor", article);
        Assert.Contains("owner-referenced borrowed view", article);
        Assert.Contains("event node presence/status snapshot", article);
        Assert.Contains("EventRecordNodeHasEvent", article);
        Assert.Contains("EventWaitNodeHasEvent", article);
        Assert.Contains("not runtime proof", article, StringComparison.OrdinalIgnoreCase);

        Assert.Contains("CUDA Graph Borrowed Handle Safety Gate", toc);
        Assert.Contains("articles/zh-cn/cuda-graph-borrowed-handle-safety-gate.md", toc);
    }

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
        string graph = ReadSource("src", "JYPPX.CudaSharp", "CudaGraph.cs");
        string graphExec = ReadSource("src", "JYPPX.CudaSharp", "CudaGraphExec.cs");
        string graphNode = ReadSource("src", "JYPPX.CudaSharp", "CudaGraphNode.cs");
        string publicSurface = graph + graphExec + graphNode;

        Assert.DoesNotContain("public IntPtr", publicSurface);
        Assert.DoesNotContain("public nint", publicSurface);
        Assert.DoesNotContain("GetChildGraph", publicSurface);
        Assert.DoesNotContain("GetContainingGraph", publicSurface);
        Assert.DoesNotContain("GetEvent(CudaGraphNode", publicSurface);
        Assert.DoesNotContain("GetParams(CudaGraphNode", publicSurface);
        Assert.Contains("ContainsNode(CudaGraphNode node)", graph);
        Assert.Contains("internal UIntPtr Token", graphNode);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
