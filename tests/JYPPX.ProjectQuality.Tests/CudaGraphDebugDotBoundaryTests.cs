using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaGraphDebugDotBoundaryTests
{
    [Fact]
    public void CudaGraphDebugDotApiIsLiftedWithoutDeletingDeferredRecord()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-forty-first-batch-graph-debug-dot.manifest.json");
        string deferredManifest = ReadSource("native", "manifests", "cuda", "cuda-twenty-third-batch-deferred-coverage.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string nativeSource = ReadSource("native", "src", "cuda", "modules", "graph", "node_topology.inc");
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("cuda-graph-debug-dot-print-safe", manifest);
        Assert.Contains("jyppx_cuda_graph_debug_dot_print_safe", manifest);
        Assert.Contains("\"type\": \"const char*\", \"direction\": \"in\", \"managedType\": \"IntPtr\"", manifest);
        Assert.Contains("\"type\": \"uint32_t\", \"direction\": \"in\", \"managedType\": \"uint\"", manifest);
        Assert.Contains("cuda-cuda-graph-debug-dot-print-deferred", deferredManifest);
        Assert.Contains("jyppx_cuda_graph_debug_dot_print_safe", header);
        Assert.Contains("cudaGraphDebugDotPrint", nativeSource);
        Assert.Contains("CUDA graph debug DOT output path must not be null or empty", nativeSource);
        Assert.Contains("\"cudaGraphDebugDotPrint\" = @(\"graph-debug-dot-print-deferred\", \"graph-debug-dot-print-safe\")", script);
    }

    [Fact]
    public void ManagedCudaGraphDebugDotWrapperUsesStringPathAndValueFlags()
    {
        string graph = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.cs");
        string flags = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphDebugDotFlags.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Graph", "NativeCudaApi.Graph.cs");
        string utf8 = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Utf8Interop.cs");

        Assert.Contains("ExportDebugDot(string path, CudaGraphDebugDotFlags flags = CudaGraphDebugDotFlags.None)", graph);
        Assert.Contains("public enum CudaGraphDebugDotFlags : uint", flags);
        Assert.Contains("Verbose = 1U << 0", flags);
        Assert.Contains("ConditionalNodeParams = 1U << 15", flags);
        Assert.Contains("ExportGraphDebugDot", interop);
        Assert.Contains("Utf8Interop.ToNativeString(path)", interop);
        Assert.Contains("path.IndexOf('\\0')", interop);
        Assert.Contains("Utf8StringScope", utf8);
        Assert.DoesNotContain("public IntPtr", graph + flags);
        Assert.DoesNotContain("public nint", graph + flags);
    }

    [Fact]
    public void CudaGraphSmokeCoversDebugDotExport()
    {
        string program = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");

        Assert.Contains("ProbeGraphDebugDot", program);
        Assert.Contains("ExportDebugDot", program);
        Assert.Contains("ContainsDigraph", program);
        Assert.Contains("DebugDot=", program);
        Assert.Contains("Skipped=True Reason=CudaException", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
