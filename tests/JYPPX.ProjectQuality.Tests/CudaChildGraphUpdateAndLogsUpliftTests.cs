using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaChildGraphUpdateAndLogsUpliftTests
{
    [Fact]
    public void ManifestAndNativeAbiUseCopiedOwnerBoundBoundaries()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-forty-eighth-batch-child-graph-update-logs.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string graphNative = ReadSource("native", "src", "cuda", "modules", "graph", "child_graph_update.inc");
        string logsNative = ReadSource("native", "src", "cuda", "modules", "cuda_logs.inc");

        foreach (string id in new[]
        {
            "cuda-graph-add-child-graph-node-safe",
            "cuda-graph-child-graph-node-has-embedded-graph-safe",
            "cuda-graph-child-graph-node-get-node-count-safe",
            "cuda-graph-exec-child-graph-node-set-params-safe",
            "cuda-graph-exec-update-copied-metadata-safe",
            "cuda-graph-instantiate-with-params-safe",
            "cuda-graph-kernel-node-copy-attributes-safe",
            "cuda-logs-current-cursor-safe",
            "cuda-logs-dump-to-memory-caller-buffer-safe",
            "cuda-logs-dump-to-file-safe"
        })
        {
            Assert.Contains(id, manifest, StringComparison.Ordinal);
        }

        Assert.Contains("jyppx_cuda_graph_child_graph_node_get_node_count_safe", header, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_graph_exec_update_copied_metadata_safe", header, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_logs_dump_to_memory_safe", header, StringComparison.Ordinal);
        Assert.Contains("#if JYPPX_HAS_CUDA_TOOLKIT\nJYPPX_StatusCode get_cuda_child_graph", graphNative.Replace("\r\n", "\n"), StringComparison.Ordinal);
        Assert.Contains("update_status != cudaSuccess && update_status != cudaErrorGraphExecUpdateFailure", graphNative, StringComparison.Ordinal);
        Assert.Contains("cudaGraphKernelNodeCopyAttributes(\n            jyppx_cuda_node_from_token(source_node),\n            jyppx_cuda_node_from_token(destination_node))", graphNative.Replace("\r\n", "\n"), StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 12000", graphNative, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 13020", logsNative, StringComparison.Ordinal);
        Assert.Contains("kMaximumCudaLogBufferSize = 25600", logsNative, StringComparison.Ordinal);
        Assert.Contains("output_buffer[written_size] = '\\0'", logsNative, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfaceUsesSnapshotsTypedCursorsAndManagedOwners()
    {
        string graph = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.cs");
        string graphExec = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphExec.cs");
        string childSnapshot = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphChildSnapshot.cs");
        string updateSnapshot = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphExecUpdateSnapshot.cs");
        string cursor = ReadSource("src", "JYPPX.CudaSharp", "Diagnostics", "CudaLogCursor.cs");
        string logSnapshot = ReadSource("src", "JYPPX.CudaSharp", "Diagnostics", "CudaLogSnapshot.cs");
        string runtimeLogs = ReadSource("src", "JYPPX.CudaSharp", "Diagnostics", "CudaRuntimeLogs.cs");
        string publicSurface = graph + graphExec + childSnapshot + updateSnapshot + cursor + logSnapshot + runtimeLogs;

        Assert.Contains("public CudaGraphNode AddChildGraphNode(CudaGraph childGraph)", graph, StringComparison.Ordinal);
        Assert.Contains("public CudaGraphChildSnapshot GetChildGraphSnapshot(CudaGraphNode node)", graph, StringComparison.Ordinal);
        Assert.Contains("public static void CopyKernelNodeAttributes", graph, StringComparison.Ordinal);
        Assert.Contains("public CudaGraphExec InstantiateWithParameters", graph, StringComparison.Ordinal);
        Assert.Contains("public void SetChildGraphNodeParameters", graphExec, StringComparison.Ordinal);
        Assert.Contains("public CudaGraphExecUpdateSnapshot Update(CudaGraph graph)", graphExec, StringComparison.Ordinal);
        Assert.Contains("public static class CudaRuntimeLogs", runtimeLogs, StringComparison.Ordinal);
        Assert.Contains("public readonly struct CudaLogCursor", cursor, StringComparison.Ordinal);
        Assert.Contains("public CudaLogCursor? NextCursor", logSnapshot, StringComparison.Ordinal);

        Assert.DoesNotContain("public IntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", publicSurface, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageMatcherPrioritizesExplicitRealAliasesAndKeepsDeferredHistory()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string deferredGraph = ReadSource("native", "manifests", "cuda", "cuda-twenty-third-batch-deferred-coverage.manifest.json");
        string deferredLogs = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");

        Assert.Contains("function Find-ExplicitCudaManifestApis", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaGraphNodeGetContainingGraph\" = @(\"id:*graph-node-is-in-graph-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaGraphExecUpdate\" = @(\"id:cuda-graph-exec-update-copied-metadata-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaLogsCurrent\" = @(\"id:cuda-logs-current-cursor-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("$explicitMatches = @(Find-ExplicitCudaManifestApis $ManifestApis $FunctionName)", coverage, StringComparison.Ordinal);
        Assert.Contains("cuda-cuda-graph-exec-update-deferred", deferredGraph, StringComparison.Ordinal);
        Assert.Contains("cuda-logs-current-deferred", deferredLogs, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageRowsAreImplementedWithDeferredHistory()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "cuda-runtime-interface-comparison.csv");
        string[] rows = comparison.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string function in new[]
        {
            "cudaGraphAddChildGraphNode",
            "cudaGraphChildGraphNodeGetGraph",
            "cudaGraphExecChildGraphNodeSetParams",
            "cudaGraphExecUpdate",
            "cudaGraphInstantiateWithParams",
            "cudaGraphKernelNodeCopyAttributes",
            "cudaGraphNodeGetContainingGraph",
            "cudaLogsCurrent",
            "cudaLogsDumpToMemory",
            "cudaLogsDumpToFile"
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
        string graphSmoke = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");
        string cudaSmoke = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");
        string packageConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        Assert.Contains("ProbeChildGraphUpdate", graphSmoke, StringComparison.Ordinal);
        Assert.Contains("GetChildGraphSnapshot", graphSmoke, StringComparison.Ordinal);
        Assert.Contains("SetChildGraphNodeParameters", graphSmoke, StringComparison.Ordinal);
        Assert.Contains("InstantiateWithParameters", graphSmoke, StringComparison.Ordinal);
        Assert.Contains("ProbeCudaRuntimeLogs", cudaSmoke, StringComparison.Ordinal);
        Assert.Contains("CudaRuntimeLogs.DumpToMemory", cudaSmoke, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaGraphExecUpdateSnapshot)", packageConsumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaRuntimeLogs.DumpToFile)", packageConsumer, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
