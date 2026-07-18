using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaOwnerScopedGraphDiagnosticsUpliftTests
{
    [Fact]
    public void ManifestAndNativeAbiDefineTwelveOwnerScopedEntries()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-forty-ninth-batch-owner-scoped-graph-diagnostics.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string types = ReadSource("native", "include", "jyppx", "cuda", "types.h");
        string native = ReadSource("native", "src", "cuda", "modules", "graph", "owner_scoped_diagnostics.inc");

        foreach (string id in new[]
        {
            "cuda-graph-add-memset-node-owner-safe",
            "cuda-graph-add-memset-node-after-owner-safe",
            "cuda-graph-exec-memset-node-set-params-owner-safe",
            "cuda-graph-destroy-node-owner-scoped-safe",
            "cuda-graph-kernel-node-get-params-copied-snapshot-safe",
            "cuda-graph-host-node-get-params-copied-snapshot-safe",
            "cuda-graph-mem-alloc-node-get-params-copied-snapshot-safe",
            "cuda-graph-mem-free-node-get-params-copied-snapshot-safe",
            "cuda-graph-external-semaphore-signal-node-get-params-copied-snapshot-safe",
            "cuda-graph-external-semaphore-wait-node-get-params-copied-snapshot-safe",
            "cuda-stream-get-capture-info-copied-summary-safe",
            "cuda-stream-update-capture-dependencies-owner-token-array-safe"
        })
        {
            Assert.Contains(id, manifest, StringComparison.Ordinal);
        }

        Assert.Contains("JYPPX_CudaGraphKernelNodeParamsSnapshot", types, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_graph_destroy_node_owner_scoped_safe", header, StringComparison.Ordinal);
        Assert.Contains("node-owner-mismatch", native, StringComparison.Ordinal);
        Assert.Contains("params.func != nullptr ? JYPPX_TRUE : JYPPX_FALSE", native, StringComparison.Ordinal);
        Assert.Contains("params.dptr != nullptr ? JYPPX_TRUE : JYPPX_FALSE", native, StringComparison.Ordinal);
        Assert.Contains("cudaStreamGetCaptureInfo_v2", native, StringComparison.Ordinal);
        Assert.Contains("cudaStreamGetCaptureInfo_v3", native, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 13000", native, StringComparison.Ordinal);
        Assert.Contains("catch (const std::bad_alloc&)", native, StringComparison.Ordinal);
        Assert.Contains("native-exception-caught", native, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfaceUsesOwnersCopiedSnapshotsAndTypedModes()
    {
        string graph = ReadSource("src", "JYPPX.CudaSharp", "CudaGraph.cs");
        string graphExec = ReadSource("src", "JYPPX.CudaSharp", "CudaGraphExec.cs");
        string stream = ReadSource("src", "JYPPX.CudaSharp", "CudaStream.cs");
        string captureInfo = ReadSource("src", "JYPPX.CudaSharp", "CudaStreamCaptureInfo.cs");
        string flags = ReadSource("src", "JYPPX.CudaSharp", "CudaFlags.cs");
        string publicSurface = graph + graphExec + stream + captureInfo + flags +
            ReadSource("src", "JYPPX.CudaSharp", "CudaGraphKernelNodeParametersSnapshot.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "CudaGraphHostNodeParametersSnapshot.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "CudaGraphMemoryAllocationNodeSnapshot.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "CudaGraphMemoryFreeNodeSnapshot.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "CudaGraphExternalSemaphoreNodeSnapshot.cs");

        Assert.Contains("public CudaGraphNode AddMemsetNode(CudaMemory destination", graph, StringComparison.Ordinal);
        Assert.Contains("public void RemoveNode(CudaGraphNode node)", graph, StringComparison.Ordinal);
        Assert.Contains("GetKernelNodeParametersSnapshot", graph, StringComparison.Ordinal);
        Assert.Contains("GetMemoryAllocationNodeSnapshot", graph, StringComparison.Ordinal);
        Assert.Contains("public void SetMemsetNodeParameters", graphExec, StringComparison.Ordinal);
        Assert.Contains("public void UpdateCaptureDependencies", stream, StringComparison.Ordinal);
        Assert.Contains("public enum CudaStreamCaptureDependencyMode", flags, StringComparison.Ordinal);
        Assert.Contains("public bool HasCapturedGraph", captureInfo, StringComparison.Ordinal);
        Assert.Contains("public ulong DependencyCount", captureInfo, StringComparison.Ordinal);

        Assert.DoesNotContain("public IntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", publicSurface, StringComparison.Ordinal);
    }

    [Fact]
    public void CoveragePrioritizesRealAliasesAndRetainsDeferredHistory()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string deferredGraph = ReadSource("native", "manifests", "cuda", "cuda-twenty-third-batch-deferred-coverage.manifest.json");
        string deferredStream = ReadSource("native", "manifests", "cuda", "cuda-thirty-fifth-batch-stream-device-boundaries.manifest.json");

        Assert.Contains("\"cudaGraphDestroyNode\" = @(\"id:cuda-graph-destroy-node-owner-scoped-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaGraphKernelNodeGetParams\" = @(\"id:cuda-graph-kernel-node-get-params-copied-snapshot-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaStreamGetCaptureInfo_v3\" = @(\"id:cuda-stream-get-capture-info-copied-summary-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaStreamUpdateCaptureDependencies\" = @(\"id:cuda-stream-update-capture-dependencies-owner-token-array-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("cuda-cuda-graph-destroy-node-deferred", deferredGraph, StringComparison.Ordinal);
        Assert.Contains("cuda-stream-get-capture-info-v3-deferred", deferredStream, StringComparison.Ordinal);
        Assert.Contains("cuda-stream-update-capture-dependencies-deferred", deferredStream, StringComparison.Ordinal);
    }

    [Fact]
    public void CandidateReviewKeepsUnsafeOwnershipApisDeferredAndRecordsResolvedCaptureOwnership()
    {
        string review = ReadSource("artifacts", "interface-coverage", "cuda-owner-scoped-candidate-review.md");

        Assert.Contains("`cudaStreamBeginCaptureToGraph` | implement | owner-scoped session retains stream/graph wrappers", review, StringComparison.Ordinal);
        Assert.Contains("`cudaGraphAddKernelNode` | requires kernel function and argument pointers", review, StringComparison.Ordinal);
        Assert.Contains("CUDA graph user-object APIs | require release callbacks", review, StringComparison.Ordinal);
        Assert.Contains("Old deferred manifests remain in place", review, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageRowsAreImplementedWithDeferredHistory()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "cuda-runtime-interface-comparison.csv");
        string[] rows = comparison.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string function in new[]
        {
            "cudaGraphAddMemsetNode",
            "cudaGraphDestroyNode",
            "cudaGraphExecMemsetNodeSetParams",
            "cudaGraphExternalSemaphoresSignalNodeGetParams",
            "cudaGraphExternalSemaphoresWaitNodeGetParams",
            "cudaGraphHostNodeGetParams",
            "cudaGraphKernelNodeGetParams",
            "cudaGraphMemAllocNodeGetParams",
            "cudaGraphMemFreeNodeGetParams",
            "cudaStreamGetCaptureInfo_v3",
            "cudaStreamUpdateCaptureDependencies"
        })
        {
            Assert.Contains(rows, row =>
                row.Contains($"\"{function}\"", StringComparison.Ordinal) &&
                row.Contains("\"implemented-with-deferred-history\"", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void SmokeAndPackageConsumerCompileTheNewSurface()
    {
        string smoke = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        Assert.Contains("ProbeOwnerScopedGraphDiagnostics", smoke, StringComparison.Ordinal);
        Assert.Contains("UpdateCaptureDependencies(Array.Empty<CudaGraphNode>()", smoke, StringComparison.Ordinal);
        Assert.Contains("GetMemoryAllocationNodeSnapshot", smoke, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaGraphKernelNodeParametersSnapshot)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaStream.UpdateCaptureDependencies)", consumer, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
