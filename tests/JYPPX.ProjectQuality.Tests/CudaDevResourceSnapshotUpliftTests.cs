using System.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaDevResourceSnapshotUpliftTests
{
    private static readonly string[] OfficialFunctions =
    {
        "cudaDeviceGetDevResource",
        "cudaExecutionCtxGetDevResource",
        "cudaStreamGetDevResource"
    };

    [Fact]
    public void ManifestNativeAbiAndGeneratedBindingsDefineThreePointerFreeSnapshots()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-fifty-ninth-batch-device-resource-snapshots.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string types = ReadSource("native", "include", "jyppx", "cuda", "types.h");
        string native = ReadSource("native", "src", "cuda", "modules", "deployment", "dev_resource_snapshots.inc");
        string generated = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Generated", "NativeMethodsCuda.Generated.g.cs");

        foreach (string id in new[]
        {
            "cuda-device-get-dev-resource-copied-snapshot-safe",
            "cuda-execution-context-get-dev-resource-copied-snapshot-safe",
            "cuda-stream-get-dev-resource-copied-snapshot-safe"
        })
        {
            Assert.Contains(id, manifest, StringComparison.Ordinal);
        }

        foreach (string function in OfficialFunctions)
        {
            Assert.Contains(function, native, StringComparison.Ordinal);
        }

        Assert.Contains("JYPPX_CudaDevResourceSnapshot", types, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_device_get_dev_resource_snapshot_safe", header, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_execution_context_get_dev_resource_snapshot_safe", header, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_stream_get_dev_resource_snapshot_safe", header, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 13000", native, StringComparison.Ordinal);
        Assert.Contains("cuda-version-not-supported", native, StringComparison.Ordinal);
        Assert.Contains("native-exception-caught", native, StringComparison.Ordinal);
        Assert.Contains("nextResource", native, StringComparison.Ordinal);
        Assert.Contains("has_next_resource", native, StringComparison.Ordinal);
        Assert.DoesNotContain("nextResource", types, StringComparison.Ordinal);
        Assert.DoesNotContain("snapshot->nextResource", native, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_device_get_dev_resource_snapshot_safe", generated, StringComparison.Ordinal);
        Assert.Contains("NativeCudaDevResourceSnapshot", generated, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSnapshotIsTypedValueOnlyAndDoesNotExposeVendorPointers()
    {
        string snapshot = ReadSource("src", "JYPPX.CudaSharp", "Devices", "CudaDevResourceSnapshot.cs");
        string device = ReadSource(
            "src", "JYPPX.CudaSharp", "Devices", "CudaDevice.GraphResources.cs");
        string stream = ReadSource("src", "JYPPX.CudaSharp", "Streams", "CudaStream.cs");
        string context = ReadSource("src", "JYPPX.CudaSharp", "Devices", "CudaPrimaryExecutionContext.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Devices", "NativeCudaApi.DeviceResources.cs");
        string publicSurface = snapshot + device + stream + context;

        Assert.Contains("public readonly struct CudaDevResourceSnapshot", snapshot, StringComparison.Ordinal);
        Assert.Contains("public enum CudaDevResourceType", snapshot, StringComparison.Ordinal);
        Assert.Contains("public bool HasNextResource", snapshot, StringComparison.Ordinal);
        Assert.Contains("public bool HasOpaqueWorkqueue", snapshot, StringComparison.Ordinal);
        Assert.Contains("GetDevResourceSnapshot", device + stream + context, StringComparison.Ordinal);
        Assert.Contains("NativeCudaDevResourceSnapshot", interop, StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("nextResource", publicSurface, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageUsesRealAliasesAndRetainsDeferredHistory()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string streamDeviceDeferred = ReadSource("native", "manifests", "cuda", "cuda-thirty-fifth-batch-stream-device-boundaries.manifest.json");
        string otherDeferred = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");

        foreach ((string function, string realId, string deferredCoverageId, string deferredManifestId) in new[]
        {
            ("cudaDeviceGetDevResource", "cuda-device-get-dev-resource-copied-snapshot-safe", "cuda-device-get-dev-resource-deferred", "cuda-device-get-dev-resource-deferred"),
            ("cudaExecutionCtxGetDevResource", "cuda-execution-context-get-dev-resource-copied-snapshot-safe", "cuda-execution-ctx-get-dev-resource-deferred", "cuda-execution-ctx-get-dev-resource-deferred"),
            ("cudaStreamGetDevResource", "cuda-stream-get-dev-resource-copied-snapshot-safe", "*stream-get-dev-resource-deferred", "cuda-stream-get-dev-resource-deferred")
        })
        {
            Assert.Contains($"\"{function}\" = @(\"id:{realId}\")", coverage, StringComparison.Ordinal);
            Assert.Contains($"\"{function}\" = @(\"id:{deferredCoverageId}\")", coverage, StringComparison.Ordinal);
            Assert.Contains(deferredManifestId, streamDeviceDeferred + otherDeferred, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void CoverageRowsAreImplementedWithDeferredHistoryOnCuda132()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "cuda-runtime-interface-comparison.csv");
        string[] rows = comparison.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string function in OfficialFunctions)
        {
            Assert.Contains(rows, row =>
                row.Contains("\"13.2\"", StringComparison.Ordinal) &&
                row.Contains($"\"{function}\"", StringComparison.Ordinal) &&
                row.Contains("\"implemented-with-deferred-history\"", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void SmokeRecordsVersionAndDriverBoundariesForResourceQueries()
    {
        string smoke = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");

        Assert.Contains("ProbeCudaDevResourceSnapshots", smoke, StringComparison.Ordinal);
        Assert.Contains("VersionGuard=NotSupported", smoke, StringComparison.Ordinal);
        Assert.Contains("Available=False Status=", smoke, StringComparison.Ordinal);
        Assert.Contains("CudaDevResourceSnapshots", smoke, StringComparison.Ordinal);
        Assert.Contains("HasNextResource", smoke, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
