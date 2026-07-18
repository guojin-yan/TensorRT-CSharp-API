using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaStreamCaptureToGraphUpliftTests
{
    [Fact]
    public void ManifestAndNativeBoundaryModelAnOwnerScopedCaptureSession()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-fifty-seventh-batch-stream-capture-to-graph.manifest.json");
        string native = ReadSource("native", "src", "cuda", "modules", "graph", "stream_capture_variants.inc");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");

        Assert.Contains("cuda-stream-begin-capture-to-graph-owner-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("cuda-stream-end-capture-into-existing-graph-owner-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 12030", manifest, StringComparison.Ordinal);
        Assert.Contains("cudaStreamBeginCaptureToGraph", native, StringComparison.Ordinal);
        Assert.Contains("cudaStreamEndCapture", native, StringComparison.Ordinal);
        Assert.Contains("capture-owner-mismatch", native, StringComparison.Ordinal);
        Assert.Contains("make_stream_capture_edge_data", native, StringComparison.Ordinal);
        Assert.Contains("std::bad_alloc", native, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_stream_begin_capture_to_graph_safe", header, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_stream_end_capture_into_graph_safe", header, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfacePinsOwnersAndDoesNotExposeNativePointers()
    {
        string stream = ReadSource("src", "JYPPX.CudaSharp", "CudaStream.cs");
        string graph = ReadSource("src", "JYPPX.CudaSharp", "CudaGraph.cs");
        string session = ReadSource("src", "JYPPX.CudaSharp", "CudaStreamCaptureToGraphSession.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Graph", "NativeCudaApi.StreamCaptureVariants.cs");
        string smoke = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");

        Assert.Contains("BeginCaptureToGraph", stream, StringComparison.Ordinal);
        Assert.Contains("EnterCaptureToGraphSession", stream + graph, StringComparison.Ordinal);
        Assert.Contains("cannot be disposed while a stream-to-graph capture session is active", stream + graph, StringComparison.Ordinal);
        Assert.Contains("public sealed class CudaStreamCaptureToGraphSession", session, StringComparison.Ordinal);
        Assert.Contains("public CudaGraph Graph", session, StringComparison.Ordinal);
        Assert.Contains("EndStreamCaptureIntoGraph", session + interop, StringComparison.Ordinal);
        Assert.Contains("ProbeStreamCaptureVariants(stream, device, ByteCount)", smoke, StringComparison.Ordinal);
        Assert.Contains("ToGraph=True", smoke, StringComparison.Ordinal);

        Assert.DoesNotContain("public IntPtr", session, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", session, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", session, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", session, StringComparison.Ordinal);
    }

    [Fact]
    public void CoveragePromotesBeginCaptureToGraphWhileKeepingDeferredHistory()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string deferred = ReadSource("native", "manifests", "cuda", "cuda-thirty-fifth-batch-stream-device-boundaries.manifest.json");
        string comparison = ReadSource("artifacts", "interface-coverage", "cuda-runtime-interface-comparison.csv");

        Assert.Contains("\"cudaStreamBeginCaptureToGraph\" = @(\"id:cuda-stream-begin-capture-to-graph-owner-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaStreamBeginCaptureToGraph\" = @(\"id:*stream-begin-capture-to-graph-deferred\")", coverage, StringComparison.Ordinal);
        Assert.Contains("cuda-stream-begin-capture-to-graph-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains(comparison.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries), row =>
            row.Contains("\"cudaStreamBeginCaptureToGraph\"", StringComparison.Ordinal) &&
            row.Contains("\"implemented-with-deferred-history\"", StringComparison.Ordinal));
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
