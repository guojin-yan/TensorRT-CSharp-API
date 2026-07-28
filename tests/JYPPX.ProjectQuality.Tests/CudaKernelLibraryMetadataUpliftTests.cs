using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaKernelLibraryMetadataUpliftTests
{
    private static readonly string[] OfficialFunctions =
    {
        "cudaLibraryLoadData",
        "cudaLibraryLoadFromFile",
        "cudaLibraryUnload",
        "cudaLibraryGetKernelCount",
        "cudaLibraryEnumerateKernels",
        "cudaLibraryGetKernel",
    };

    [Fact]
    public void ManifestAndNativeDefineSixOwnerSafeKernelLibraryEntries()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-fifty-first-batch-kernel-library-metadata.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string types = ReadSource("native", "include", "jyppx", "cuda", "types.h");
        string native = ReadSource("native", "src", "cuda", "modules", "deployment", "kernel_library_metadata.inc");

        foreach (string id in new[]
        {
            "cuda-library-load-data-retained-copy-owner-safe",
            "cuda-library-load-from-file-bridge-owned-safe",
            "cuda-library-unload-bridge-owned-safe",
            "cuda-library-get-kernel-count-copied-scalar-safe",
            "cuda-library-enumerate-kernels-copied-inventory-safe",
            "cuda-library-get-kernel-exists-by-name-safe",
        })
        {
            Assert.Contains(id, manifest, StringComparison.Ordinal);
        }

        Assert.Contains("JYPPX_CudaKernelLibrary", types, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CudaKernelLibraryInventory", types, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_kernel_library_load_data_copy_safe", header, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 12090", native, StringComparison.Ordinal);
        Assert.Contains("retained_code.assign", native, StringComparison.Ordinal);
        Assert.Contains("cudaLibraryLoadData", native, StringComparison.Ordinal);
        Assert.Contains("cudaLibraryLoadFromFile", native, StringComparison.Ordinal);
        Assert.Contains("cudaLibraryUnload", native, StringComparison.Ordinal);
        Assert.Contains("cudaLibraryGetKernelCount", native, StringComparison.Ordinal);
        Assert.Contains("cudaLibraryEnumerateKernels", native, StringComparison.Ordinal);
        Assert.Contains("cudaLibraryGetKernel", native, StringComparison.Ordinal);
        Assert.Contains("(void)cudaGetLastError();", native, StringComparison.Ordinal);
        Assert.Contains("native-exception-caught", native, StringComparison.Ordinal);
        Assert.Contains("windows-seh-caught", native, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedOwnerExposesOnlyCopiedMetadataAndNamedExistence()
    {
        string library = ReadSource("src", "JYPPX.CudaSharp", "Kernels", "CudaKernelLibrary.cs");
        string snapshot = ReadSource("src", "JYPPX.CudaSharp", "Kernels", "CudaKernelLibraryInventorySnapshot.cs");
        string handle = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Handles", "SafeCudaKernelLibraryHandle.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeCudaApi.KernelLibrary.cs");
        string publicSurface = library + snapshot;

        Assert.Contains("public sealed class CudaKernelLibrary : IDisposable", library, StringComparison.Ordinal);
        Assert.Contains("public static CudaKernelLibrary Load(byte[] code)", library, StringComparison.Ordinal);
        Assert.Contains("public static CudaKernelLibrary LoadFromFile(string path)", library, StringComparison.Ordinal);
        Assert.Contains("public uint KernelCount", library, StringComparison.Ordinal);
        Assert.Contains("public CudaKernelLibraryInventorySnapshot Inventory", library, StringComparison.Ordinal);
        Assert.Contains("public bool ContainsKernel(string name)", library, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_kernel_library_destroy_safe(handle)", handle, StringComparison.Ordinal);
        Assert.Contains("Utf8Interop.ToNativeString", interop, StringComparison.Ordinal);

        Assert.DoesNotContain("public IntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("cudaKernel_t", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("cudaLibrary_t", publicSurface, StringComparison.Ordinal);
    }

    [Fact]
    public void CoveragePrioritizesRealAliasesAndRetainsDeferredHistory()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string deferred = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");

        Assert.Contains("\"cudaLibraryLoadData\" = @(\"id:cuda-library-load-data-retained-copy-owner-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaLibraryLoadFromFile\" = @(\"id:cuda-library-load-from-file-bridge-owned-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaLibraryUnload\" = @(\"id:cuda-library-unload-bridge-owned-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaLibraryGetKernelCount\" = @(\"id:cuda-library-get-kernel-count-copied-scalar-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaLibraryEnumerateKernels\" = @(\"id:cuda-library-enumerate-kernels-copied-inventory-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaLibraryGetKernel\" = @(\"id:cuda-library-get-kernel-exists-by-name-safe\")", coverage, StringComparison.Ordinal);

        foreach (string id in new[]
        {
            "cuda-library-load-data-deferred",
            "cuda-library-load-from-file-deferred",
            "cuda-library-unload-deferred",
            "cuda-library-get-kernel-count-deferred",
            "cuda-library-enumerate-kernels-deferred",
            "cuda-library-get-kernel-deferred",
        })
        {
            Assert.Contains(id, deferred, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void CoverageRowsAreImplementedWithDeferredHistoryOnSupportedVersions()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "cuda-runtime-interface-comparison.csv");
        string[] rows = comparison.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string version in new[] { "12.9", "13.2" })
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
    public void SmokeConsumerAndCandidateReviewPreserveTheBoundary()
    {
        string smoke = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string review = ReadSource("artifacts", "interface-coverage", "cuda-kernel-library-owner-candidate-review.md");

        Assert.Contains("DataComplete={dataInventory.IsComplete}", smoke, StringComparison.Ordinal);
        Assert.Contains("VersionGuard=NotSupported", smoke, StringComparison.Ordinal);
        Assert.Contains("CudaKernelLibrary.LoadFromFile", smoke, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaKernelLibrary)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaKernelLibraryInventorySnapshot.IsComplete)", consumer, StringComparison.Ordinal);
        Assert.Contains("`cudaLibraryGetGlobal` | 返回 device pointer", review, StringComparison.Ordinal);
        Assert.Contains("`cudaDeviceGetExecutionCtx` | 返回 driver-owned execution context", review, StringComparison.Ordinal);
        Assert.Contains("旧 deferred manifest 全部保留", review, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
