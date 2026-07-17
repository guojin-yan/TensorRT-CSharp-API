using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaManagedMemoryLocationV2UpliftTests
{
    private static readonly string[] OfficialFunctions =
    {
        "cudaMemAdvise_v2",
        "cudaMemPrefetchAsync_v2"
    };

    [Fact]
    public void ManifestAndNativeUseManagedOwnersAndIndependentVersionGuards()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-fifty-fifth-batch-managed-memory-location-v2.manifest.json");
        string deferred = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string native = ReadSource("native", "src", "cuda", "modules", "memory", "managed_memory_location_v2.inc");

        Assert.Contains("cuda-managed-memory-advise-location-range-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("cuda-managed-memory-prefetch-location-range-async-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CudaMemory*", manifest, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CudaStream*", manifest, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 12030", manifest, StringComparison.Ordinal);
        Assert.Contains("cuda-mem-advise-v2-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("cuda-mem-prefetch-async-v2-deferred", deferred, StringComparison.Ordinal);

        Assert.Contains("jyppx_cuda_managed_memory_advise_location_range_safe", header + native, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_managed_memory_prefetch_location_range_async_safe", header + native, StringComparison.Ordinal);
        Assert.Contains("!memory_object->is_managed", native, StringComparison.Ordinal);
        Assert.Contains("validate_memory_range_query", native, StringComparison.Ordinal);
        Assert.Contains("cudaMemAdvise_v2", native, StringComparison.Ordinal);
        Assert.Contains("cudaMemPrefetchAsync_v2", native, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 13000", native, StringComparison.Ordinal);
        Assert.Contains("cudaMemAdvise(range_pointer", native, StringComparison.Ordinal);
        Assert.Contains("cudaMemPrefetchAsync(range_pointer", native, StringComparison.Ordinal);
        Assert.Contains("require CUDA Toolkit 12.3 or later", native, StringComparison.Ordinal);
        Assert.Contains("native-exception-caught", native, StringComparison.Ordinal);
        Assert.Contains("windows-seh-caught", native, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfaceIsStronglyTypedOwnerBoundAndPointerFree()
    {
        string location = ReadSource("src", "JYPPX.CudaSharp", "CudaMemoryLocation.cs");
        string managedMemory = ReadSource("src", "JYPPX.CudaSharp", "CudaManagedMemory.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Memory", "NativeCudaApi.MemoryRange.cs");
        string publicSurface = location + managedMemory;

        Assert.Contains("public enum CudaMemoryLocationKind", location, StringComparison.Ordinal);
        Assert.Contains("public readonly struct CudaMemoryLocation", location, StringComparison.Ordinal);
        Assert.Contains("public static CudaMemoryLocation Device(int deviceOrdinal)", location, StringComparison.Ordinal);
        Assert.Contains("public static CudaMemoryLocation Host", location, StringComparison.Ordinal);
        Assert.Contains("public static CudaMemoryLocation HostNuma(int numaNodeId)", location, StringComparison.Ordinal);
        Assert.Contains("public static CudaMemoryLocation CurrentHostNuma", location, StringComparison.Ordinal);
        Assert.Contains("location.Validate(nameof(location))", managedMemory, StringComparison.Ordinal);
        Assert.Contains("public void PrefetchAsync(int offset, int count, CudaMemoryLocation location, CudaStream stream)", managedMemory, StringComparison.Ordinal);
        Assert.Contains("public void Advise(int offset, int count, CudaMemoryAdvice advice, CudaMemoryLocation location)", managedMemory, StringComparison.Ordinal);
        Assert.Contains("PrefetchManagedMemoryLocationRangeAsync", interop, StringComparison.Ordinal);
        Assert.Contains("AdviseManagedMemoryLocationRange", interop, StringComparison.Ordinal);

        Assert.DoesNotContain("public IntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", publicSurface, StringComparison.Ordinal);
    }

    [Fact]
    public void CoveragePrioritizesRealAliasesAndRetainsDeferredHistory()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("\"cudaMemAdvise_v2\" = @(\"id:cuda-managed-memory-advise-location-range-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaMemPrefetchAsync_v2\" = @(\"id:cuda-managed-memory-prefetch-location-range-async-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaMemAdvise_v2\" = @(\"id:cuda-mem-advise-v2-deferred\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaMemPrefetchAsync_v2\" = @(\"id:cuda-mem-prefetch-async-v2-deferred\")", coverage, StringComparison.Ordinal);

        int matcherStart = coverage.IndexOf("function Find-CudaManifestApis", StringComparison.Ordinal);
        int explicitStart = coverage.IndexOf("Find-ExplicitCudaManifestApis $ManifestApis $FunctionName", matcherStart, StringComparison.Ordinal);
        int heuristicStart = coverage.IndexOf("$candidates = Get-CudaCandidates $FunctionName", explicitStart, StringComparison.Ordinal);
        Assert.True(matcherStart >= 0 && explicitStart > matcherStart && heuristicStart > explicitStart);
    }

    [Fact]
    public void CoverageRowsAreImplementedWithDeferredHistoryOnCuda12()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "cuda-runtime-interface-comparison.csv");
        string[] rows = comparison.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string version in new[] { "12.3", "12.9" })
        {
            foreach (string function in OfficialFunctions)
            {
                Assert.Contains(rows, row =>
                    row.Contains($"\"{version}\"", StringComparison.Ordinal) &&
                    row.Contains($"\"{function}\"", StringComparison.Ordinal) &&
                    row.Contains("\"implemented-with-deferred-history\"", StringComparison.Ordinal));
            }
        }
    }

    [Fact]
    public void SmokeAndPackageConsumerExerciseTheLocationSurface()
    {
        string smoke = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        Assert.Contains("ProbeCudaManagedMemoryLocationV2", smoke, StringComparison.Ordinal);
        Assert.Contains("CudaMemoryLocation.Device", smoke, StringComparison.Ordinal);
        Assert.Contains("CudaMemoryLocation.Host", smoke, StringComparison.Ordinal);
        Assert.Contains("CudaMemoryLocation.HostNuma", smoke, StringComparison.Ordinal);
        Assert.Contains("CudaMemoryLocation.CurrentHostNuma", smoke, StringComparison.Ordinal);
        Assert.Contains("VersionGuard=NotSupported", smoke, StringComparison.Ordinal);
        Assert.Contains("Constraint=ConcurrentManagedAccessFalse", smoke, StringComparison.Ordinal);
        Assert.Contains("memory.Advise(CudaMemoryAdvice.SetPreferredLocation, host)", smoke, StringComparison.Ordinal);
        Assert.Contains("memory.PrefetchAsync(device, stream)", smoke, StringComparison.Ordinal);

        Assert.Contains("adviseManagedMemoryLocation", consumer, StringComparison.Ordinal);
        Assert.Contains("prefetchManagedMemoryLocation", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaMemoryLocation.HostNuma)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaManagedMemory.PrefetchAsync)", consumer, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
