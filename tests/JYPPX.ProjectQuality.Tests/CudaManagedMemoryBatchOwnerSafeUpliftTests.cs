using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaManagedMemoryBatchOwnerSafeUpliftTests
{
    private static readonly string[] OfficialFunctions =
    {
        "cudaMemPrefetchBatchAsync",
        "cudaMemDiscardBatchAsync",
        "cudaMemDiscardAndPrefetchBatchAsync"
    };

    [Fact]
    public void ManifestAndNativeAbiUseManagedOwnersAndCuda13Guards()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-fifty-third-batch-managed-memory-batch.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string types = ReadSource("native", "include", "jyppx", "cuda", "types.h");
        string native = ReadSource("native", "src", "cuda", "modules", "memory", "managed_memory_batch.inc");
        string objects = ReadSource("native", "src", "cuda", "object.hpp");
        string allocations = ReadSource("native", "src", "cuda", "api.cpp") +
            ReadSource("native", "src", "cuda", "modules", "memory", "async_memory_pool.inc");

        foreach (string id in new[]
        {
            "cuda-managed-memory-prefetch-batch-owner-array-safe",
            "cuda-managed-memory-discard-batch-owner-array-safe",
            "cuda-managed-memory-discard-and-prefetch-batch-owner-array-safe"
        })
        {
            Assert.Contains(id, manifest, StringComparison.Ordinal);
        }

        Assert.Contains("JYPPX_CudaManagedMemoryBatchRange", header + types, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CudaMemory* memory", types, StringComparison.Ordinal);
        Assert.Contains("bool is_managed", objects, StringComparison.Ordinal);
        Assert.Contains("memory->is_managed = true", allocations, StringComparison.Ordinal);
        Assert.Contains("memory->is_managed = false", allocations, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 13000", native, StringComparison.Ordinal);
        Assert.Contains("cudaMemPrefetchBatchAsync", native, StringComparison.Ordinal);
        Assert.Contains("cudaMemDiscardBatchAsync", native, StringComparison.Ordinal);
        Assert.Contains("cudaMemDiscardAndPrefetchBatchAsync", native, StringComparison.Ordinal);
        Assert.Contains("0ULL", native, StringComparison.Ordinal);
        Assert.Contains("memory-not-managed", native, StringComparison.Ordinal);
        Assert.Contains("native-exception-caught", native, StringComparison.Ordinal);
        Assert.Contains("windows-seh-caught", native, StringComparison.Ordinal);
        Assert.Contains("require CUDA Toolkit 13.0 or later", native, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfaceIsTypedOwnerBoundAndPointerFree()
    {
        string surface = ReadSource("src", "JYPPX.CudaSharp", "Memory", "CudaManagedMemoryBatch.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Memory", "NativeCudaApi.ManagedMemoryBatch.cs");
        string nativeStructs = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeStructs.cs");

        Assert.Contains("public readonly struct CudaManagedMemoryRange", surface, StringComparison.Ordinal);
        Assert.Contains("public readonly struct CudaManagedMemoryPrefetchRange", surface, StringComparison.Ordinal);
        Assert.Contains("public static class CudaManagedMemoryBatch", surface, StringComparison.Ordinal);
        Assert.Contains("IReadOnlyList<CudaManagedMemoryPrefetchRange>", surface, StringComparison.Ordinal);
        Assert.Contains("IReadOnlyList<CudaManagedMemoryRange>", surface, StringComparison.Ordinal);
        Assert.Contains("CopyPrefetchRanges(ranges)", surface, StringComparison.Ordinal);
        Assert.Contains("CopyRanges(ranges)", surface, StringComparison.Ordinal);
        Assert.Contains("Keep every range owner", surface, StringComparison.Ordinal);

        Assert.Contains("owner.DangerousAddRef(ref leases[index])", interop, StringComparison.Ordinal);
        Assert.Contains("owner.DangerousGetHandle()", interop, StringComparison.Ordinal);
        Assert.Contains("ReleaseLeases(owners, leases)", interop, StringComparison.Ordinal);
        Assert.Contains("for (int index = owners.Length - 1; index >= 0; --index)", interop, StringComparison.Ordinal);
        Assert.Contains("GCHandleType.Pinned", interop, StringComparison.Ordinal);
        Assert.Contains("internal struct NativeCudaManagedMemoryBatchRange", nativeStructs, StringComparison.Ordinal);

        Assert.DoesNotContain("public IntPtr", surface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", surface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", surface, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", surface, StringComparison.Ordinal);
    }

    [Fact]
    public void CoveragePrioritizesRealAliasesAndRetainsDeferredHistory()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string deferred = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");

        foreach ((string function, string realId, string deferredId) in new[]
        {
            ("cudaMemPrefetchBatchAsync", "cuda-managed-memory-prefetch-batch-owner-array-safe", "cuda-mem-prefetch-batch-async-deferred"),
            ("cudaMemDiscardBatchAsync", "cuda-managed-memory-discard-batch-owner-array-safe", "cuda-mem-discard-batch-async-deferred"),
            ("cudaMemDiscardAndPrefetchBatchAsync", "cuda-managed-memory-discard-and-prefetch-batch-owner-array-safe", "cuda-mem-discard-and-prefetch-batch-async-deferred")
        })
        {
            Assert.Contains($"\"{function}\" = @(\"id:{realId}\")", coverage, StringComparison.Ordinal);
            Assert.Contains($"\"{function}\" = @(\"id:{deferredId}\")", coverage, StringComparison.Ordinal);
            Assert.Contains(deferredId, deferred, StringComparison.Ordinal);
        }

        Assert.Contains("$explicitMatches = @(Find-ExplicitCudaManifestApis $ManifestApis $FunctionName)", coverage, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageRowsAreImplementedWithDeferredHistoryOnlyOnCuda13()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "cuda-runtime-interface-comparison.csv");
        string[] rows = comparison.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string function in OfficialFunctions)
        {
            Assert.Contains(rows, row =>
                row.Contains("\"13.2\"", StringComparison.Ordinal) &&
                row.Contains($"\"{function}\"", StringComparison.Ordinal) &&
                row.Contains("\"implemented-with-deferred-history\"", StringComparison.Ordinal));
            Assert.DoesNotContain(rows, row =>
                !row.Contains("\"13.2\"", StringComparison.Ordinal) &&
                row.Contains($"\"{function}\"", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void SmokeConsumerAndReviewCloseTheOwnerSafeProofLoop()
    {
        string smoke = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string review = ReadSource("artifacts", "interface-coverage", "cuda-managed-memory-batch-owner-candidate-review.md");

        Assert.Contains("ProbeCudaManagedMemoryBatch", smoke, StringComparison.Ordinal);
        Assert.Contains("VersionGuard=NotSupported", smoke, StringComparison.Ordinal);
        Assert.Contains("allDevicesSupportConcurrentManagedAccess", smoke, StringComparison.Ordinal);
        Assert.Contains("CudaManagedMemoryBatch.PrefetchAsync", smoke, StringComparison.Ordinal);
        Assert.Contains("CudaManagedMemoryBatch.DiscardAsync", smoke, StringComparison.Ordinal);
        Assert.Contains("CudaManagedMemoryBatch.DiscardAndPrefetchAsync", smoke, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaManagedMemoryRange)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaManagedMemoryPrefetchRange)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaManagedMemoryBatch.DiscardAndPrefetchAsync)", consumer, StringComparison.Ordinal);
        Assert.Contains("旧 deferred manifest", review, StringComparison.Ordinal);
        Assert.Contains("CUDA 11/12", review, StringComparison.Ordinal);
        Assert.Contains("owner", review, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
