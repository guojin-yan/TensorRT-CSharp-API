using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaKernelLibrarySymbolAndAttributeUpliftTests
{
    private static readonly string[] OfficialFunctions =
    {
        "cudaLibraryGetGlobal",
        "cudaLibraryGetManaged",
        "cudaLibraryGetUnifiedFunction",
        "cudaKernelSetAttributeForDevice"
    };

    [Fact]
    public void ManifestAndNativeKeepPointersInsideTheCuda129CallStack()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-fifty-fourth-batch-kernel-library-symbols.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string native = ReadSource("native", "src", "cuda", "modules", "deployment", "kernel_library_metadata.inc");

        foreach (string id in new[]
        {
            "cuda-library-get-global-size-by-name-copied-scalar-safe",
            "cuda-library-get-managed-size-by-name-copied-scalar-safe",
            "cuda-library-get-unified-function-exists-by-name-safe",
            "cuda-kernel-set-attribute-for-device-library-owner-name-safe"
        })
        {
            Assert.Contains(id, manifest, StringComparison.Ordinal);
        }

        Assert.Contains("jyppx_cuda_kernel_library_try_get_global_size_safe", header, StringComparison.Ordinal);
        Assert.Contains("cudaLibraryGetGlobal(nullptr, &symbol_size", native, StringComparison.Ordinal);
        Assert.Contains("cudaLibraryGetManaged(nullptr, &symbol_size", native, StringComparison.Ordinal);
        Assert.Contains("cudaLibraryGetUnifiedFunction(&function_pointer", native, StringComparison.Ordinal);
        Assert.Contains("cudaLibraryGetKernel(&kernel", native, StringComparison.Ordinal);
        Assert.Contains("cudaKernelSetAttributeForDevice(kernel", native, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 12090", native, StringComparison.Ordinal);
        Assert.Contains("is_mutable_kernel_library_attribute", native, StringComparison.Ordinal);
        Assert.Contains("(void)cudaGetLastError();", native, StringComparison.Ordinal);
        Assert.Contains("windows-seh-caught", native, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfaceIsOwnerBoundCopiedAndPointerFree()
    {
        string library = ReadSource("src", "JYPPX.CudaSharp", "Kernels", "CudaKernelLibrary.cs");
        string attribute = ReadSource("src", "JYPPX.CudaSharp", "Kernels", "CudaKernelAttribute.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeCudaApi.KernelLibrary.cs");
        string publicSurface = library + attribute;

        Assert.Contains("public bool TryGetGlobalSymbolSize(string name, out ulong sizeInBytes)", library, StringComparison.Ordinal);
        Assert.Contains("public bool TryGetManagedSymbolSize(string name, out ulong sizeInBytes)", library, StringComparison.Ordinal);
        Assert.Contains("public bool ContainsUnifiedFunction(string name)", library, StringComparison.Ordinal);
        Assert.Contains("public void SetAttributeForDevice(string kernelName, CudaKernelAttribute attribute, int value, int deviceOrdinal)", library, StringComparison.Ordinal);
        Assert.Contains("public enum CudaKernelAttribute", attribute, StringComparison.Ordinal);
        Assert.Contains("Utf8Interop.ToNativeString", interop, StringComparison.Ordinal);

        Assert.DoesNotContain("public IntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("cudaKernel_t", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("void*", publicSurface, StringComparison.Ordinal);
    }

    [Fact]
    public void CoveragePrioritizesRealAliasesAndRetainsDeferredHistory()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string deferred = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");

        foreach ((string function, string realId, string deferredId) in new[]
        {
            ("cudaLibraryGetGlobal", "cuda-library-get-global-size-by-name-copied-scalar-safe", "cuda-library-get-global-deferred"),
            ("cudaLibraryGetManaged", "cuda-library-get-managed-size-by-name-copied-scalar-safe", "cuda-library-get-managed-deferred"),
            ("cudaLibraryGetUnifiedFunction", "cuda-library-get-unified-function-exists-by-name-safe", "cuda-library-get-unified-function-deferred"),
            ("cudaKernelSetAttributeForDevice", "cuda-kernel-set-attribute-for-device-library-owner-name-safe", "cuda-kernel-set-attribute-for-device-deferred")
        })
        {
            Assert.Contains($"\"{function}\" = @(\"id:{realId}\")", coverage, StringComparison.Ordinal);
            Assert.Contains($"\"{function}\" = @(\"id:{deferredId}\")", coverage, StringComparison.Ordinal);
            Assert.Contains(deferredId, deferred, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void CoverageRowsAreImplementedWithDeferredHistoryOnlyOnSupportedVersions()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "cuda-runtime-interface-comparison.csv");
        string[] rows = comparison.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string function in OfficialFunctions)
        {
            foreach (string version in new[] { "12.9", "13.2" })
            {
                Assert.Contains(rows, row =>
                    row.Contains($"\"{version}\"", StringComparison.Ordinal) &&
                    row.Contains($"\"{function}\"", StringComparison.Ordinal) &&
                    row.Contains("\"implemented-with-deferred-history\"", StringComparison.Ordinal));
            }
            Assert.DoesNotContain(rows, row =>
                !row.Contains("\"12.9\"", StringComparison.Ordinal) &&
                !row.Contains("\"13.2\"", StringComparison.Ordinal) &&
                row.Contains($"\"{function}\"", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void SmokeConsumerAndReviewCloseThePointerFreeProofLoop()
    {
        string smoke = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string review = ReadSource("artifacts", "interface-coverage", "cuda-kernel-library-symbol-and-attribute-candidate-review.md");

        Assert.Contains("Global={globalFound}:{globalSize}", smoke, StringComparison.Ordinal);
        Assert.Contains("ManagedMissing={managedMissing}", smoke, StringComparison.Ordinal);
        Assert.Contains("UnifiedProbe={unifiedProbe}", smoke, StringComparison.Ordinal);
        Assert.Contains("VendorInvalidArgument:Consumed={consumedError}", smoke, StringComparison.Ordinal);
        Assert.Contains("AttributeSet=True", smoke, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaKernelLibrary.TryGetGlobalSymbolSize)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaKernelLibrary.SetAttributeForDevice)", consumer, StringComparison.Ordinal);
        Assert.Contains("cuda-kernel-library-symbol-and-attribute", consumer, StringComparison.Ordinal);
        Assert.Contains("native 将 `dptr` 传 null", review, StringComparison.Ordinal);
        Assert.Contains("旧 deferred manifest 全部保留", review, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
