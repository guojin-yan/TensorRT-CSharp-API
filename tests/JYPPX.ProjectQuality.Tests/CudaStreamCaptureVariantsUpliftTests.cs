using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaStreamCaptureVariantsUpliftTests
{
    [Fact]
    public void VariantManifestAndNativeBoundaryContainTypedCopiedInputs()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-fifty-sixth-batch-stream-capture-variants.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string native = ReadSource("native", "src", "cuda", "modules", "graph", "stream_capture_variants.inc");

        foreach (string id in new[]
        {
            "cuda-stream-get-capture-info-ptsz-copied-scalars-safe",
            "cuda-stream-update-capture-dependencies-ptsz-owner-token-array-safe",
            "cuda-stream-update-capture-dependencies-v2-owner-token-edge-data-safe"
        })
        {
            Assert.Contains(id, manifest, StringComparison.Ordinal);
        }

        Assert.Contains("jyppx_cuda_stream_get_capture_info_ptsz_copied_scalars_safe", header, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_stream_update_capture_dependencies_ptsz_safe", header, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_stream_update_capture_dependencies_v2_safe", header, StringComparison.Ordinal);
        Assert.Contains("cudaStreamGetCaptureInfo_ptsz", native, StringComparison.Ordinal);
        Assert.Contains("cudaStreamUpdateCaptureDependencies_ptsz", native, StringComparison.Ordinal);
        Assert.Contains("cudaStreamUpdateCaptureDependencies_v2", native, StringComparison.Ordinal);
        Assert.Contains("make_stream_capture_edge_data", native, StringComparison.Ordinal);
        Assert.Contains("kMaximumCaptureDependencyCount", native, StringComparison.Ordinal);
        Assert.Contains("catch (const std::bad_alloc&)", native, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 12000 && CUDART_VERSION < 13000", native, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 11030 && CUDART_VERSION < 12030", native, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 12030 && CUDART_VERSION < 13000", native, StringComparison.Ordinal);
        Assert.Contains("cuda-version-not-supported", native, StringComparison.Ordinal);
        Assert.Contains("requires CUDA runtime 12.3 through 12.9", native, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfaceUsesScalarSnapshotAndCopiedEdgeDataWithoutNativeHandles()
    {
        string stream = ReadSource("src", "JYPPX.CudaSharp", "Streams", "CudaStream.cs");
        string scalar = ReadSource("src", "JYPPX.CudaSharp", "Streams", "CudaStreamCaptureScalarInfo.cs");
        string edgeData = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphEdgeData.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Graph", "NativeCudaApi.StreamCaptureVariants.cs");
        string smoke = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");
        string publicSurface = stream + scalar + edgeData;

        Assert.Contains("public CudaStreamCaptureScalarInfo GetCaptureInfoPtzs()", stream, StringComparison.Ordinal);
        Assert.Contains("public bool TryGetCaptureInfoPtzs", stream, StringComparison.Ordinal);
        Assert.Contains("public void UpdateCaptureDependenciesPtzs", stream, StringComparison.Ordinal);
        Assert.Contains("public void UpdateCaptureDependenciesV2", stream, StringComparison.Ordinal);
        Assert.Contains("public readonly struct CudaStreamCaptureScalarInfo", scalar, StringComparison.Ordinal);
        Assert.Contains("CudaGraphNodeDependency", stream + interop, StringComparison.Ordinal);
        Assert.Contains("GCHandle.Alloc(edgeData, GCHandleType.Pinned)", interop, StringComparison.Ordinal);
        Assert.Contains("ProbeStreamCaptureVariants", smoke, StringComparison.Ordinal);
        Assert.Contains("PtzsUpdate", smoke, StringComparison.Ordinal);
        Assert.Contains("V2Update", smoke, StringComparison.Ordinal);

        Assert.DoesNotContain("public IntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", publicSurface, StringComparison.Ordinal);
    }

    [Fact]
    public void CoveragePromotesVariantRowsWhileKeepingDeferredHistory()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string deferred = ReadSource("native", "manifests", "cuda", "cuda-thirty-fifth-batch-stream-device-boundaries.manifest.json");

        Assert.Contains("\"cudaStreamGetCaptureInfo_ptsz\" = @(\"id:cuda-stream-get-capture-info-ptsz-copied-scalars-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaStreamUpdateCaptureDependencies_ptsz\" = @(\"id:cuda-stream-update-capture-dependencies-ptsz-owner-token-array-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaStreamUpdateCaptureDependencies_v2\" = @(\"id:cuda-stream-update-capture-dependencies-v2-owner-token-edge-data-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("cuda-stream-get-capture-info-ptsz-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("cuda-stream-update-capture-dependencies-ptsz-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("cuda-stream-update-capture-dependencies-v2-deferred", deferred, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageRowsForVariantsAreImplementedWithDeferredHistory()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "cuda-runtime-interface-comparison.csv");
        string[] rows = comparison.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string function in new[]
        {
            "cudaStreamGetCaptureInfo_ptsz",
            "cudaStreamUpdateCaptureDependencies_ptsz",
            "cudaStreamUpdateCaptureDependencies_v2"
        })
        {
            Assert.Contains(rows, row =>
                row.Contains($"\"{function}\"", StringComparison.Ordinal) &&
                row.Contains("\"implemented-with-deferred-history\"", StringComparison.Ordinal));
        }
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
