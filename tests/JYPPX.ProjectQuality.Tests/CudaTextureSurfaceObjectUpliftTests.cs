using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaTextureSurfaceObjectUpliftTests
{
    private static readonly string[] OfficialFunctions =
    {
        "cudaCreateSurfaceObject",
        "cudaDestroySurfaceObject",
        "cudaGetSurfaceObjectResourceDesc",
        "cudaCreateTextureObject",
        "cudaCreateTextureObject_v2",
        "cudaDestroyTextureObject",
        "cudaGetTextureObjectResourceDesc",
        "cudaGetTextureObjectResourceViewDesc",
        "cudaGetTextureObjectTextureDesc",
        "cudaGetTextureObjectTextureDesc_v2"
    };

    [Fact]
    public void ManifestAndNativeAbiDefineTenRealOwnerSafeEntries()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-fiftieth-batch-texture-surface-array-owners.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string types = ReadSource("native", "include", "jyppx", "cuda", "types.h");
        string native = ReadSource("native", "src", "cuda", "modules", "memory", "texture_surface_objects.inc");

        foreach (string id in new[]
        {
            "cuda-create-surface-object-array-owner-safe",
            "cuda-destroy-surface-object-bridge-owned-safe",
            "cuda-get-surface-object-resource-desc-copied-snapshot-safe",
            "cuda-create-texture-object-array-owner-safe",
            "cuda-create-texture-object-v2-array-owner-safe",
            "cuda-destroy-texture-object-bridge-owned-safe",
            "cuda-get-texture-object-resource-desc-copied-snapshot-safe",
            "cuda-get-texture-object-resource-view-desc-copied-snapshot-safe",
            "cuda-get-texture-object-texture-desc-copied-snapshot-safe",
            "cuda-get-texture-object-texture-desc-v2-copied-snapshot-safe"
        })
        {
            Assert.Contains(id, manifest, StringComparison.Ordinal);
        }

        Assert.Contains("JYPPX_CudaTextureObject", types, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CudaSurfaceObject", types, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CudaTextureDescriptor", types, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_texture_object_create_array_owner_safe", header, StringComparison.Ordinal);
        Assert.Contains("cudaCreateTextureObject_v2", native, StringComparison.Ordinal);
        Assert.Contains("cudaGetTextureObjectTextureDesc_v2", native, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 11080 && CUDART_VERSION < 12000", native, StringComparison.Ordinal);
        Assert.Contains("native-exception-caught", native, StringComparison.Ordinal);
        Assert.Contains("windows-seh-caught", native, StringComparison.Ordinal);
        Assert.Contains("has_resource_view", native, StringComparison.Ordinal);
        Assert.Contains("object->has_resource_view", native, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedObjectsRetainArraySafeHandleLeaseAndExposeOnlyTypedSnapshots()
    {
        string texture = ReadSource("src", "JYPPX.CudaSharp", "Memory", "CudaTextureObject.cs");
        string surface = ReadSource("src", "JYPPX.CudaSharp", "Memory", "CudaSurfaceObject.cs");
        string types = ReadSource("src", "JYPPX.CudaSharp", "Memory", "CudaTextureTypes.cs");
        string textureHandle = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Handles", "SafeCudaTextureObjectHandle.cs");
        string surfaceHandle = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Handles", "SafeCudaSurfaceObjectHandle.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Memory", "NativeCudaApi.TextureSurface.cs");
        string publicSurface = texture + surface + types;

        Assert.Contains("public sealed class CudaTextureObject : IDisposable", texture, StringComparison.Ordinal);
        Assert.Contains("public sealed class CudaSurfaceObject : IDisposable", surface, StringComparison.Ordinal);
        Assert.Contains("public CudaArray OwnerArray", texture, StringComparison.Ordinal);
        Assert.Contains("DangerousAddRef", interop, StringComparison.Ordinal);
        Assert.Contains("DangerousRelease", textureHandle, StringComparison.Ordinal);
        Assert.Contains("DangerousRelease", surfaceHandle, StringComparison.Ordinal);
        Assert.Contains("CudaResourceDescriptorSnapshot", types, StringComparison.Ordinal);
        Assert.Contains("CudaTextureResourceViewSnapshot", types, StringComparison.Ordinal);
        Assert.Contains("public bool IsSpecified", types, StringComparison.Ordinal);

        Assert.DoesNotContain("public IntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", publicSurface, StringComparison.Ordinal);
    }

    [Fact]
    public void CoveragePrioritizesRealAliasesAndRetainsDeferredHistory()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string deferred = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");

        Assert.Contains("\"cudaCreateTextureObject\" = @(\"id:cuda-create-texture-object-array-owner-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaCreateTextureObject_v2\" = @(\"id:cuda-create-texture-object-v2-array-owner-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaGetSurfaceObjectResourceDesc\" = @(\"id:cuda-get-surface-object-resource-desc-copied-snapshot-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaGetTextureObjectTextureDesc_v2\" = @(\"id:cuda-get-texture-object-texture-desc-v2-copied-snapshot-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("cuda-create-texture-object-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("cuda-get-texture-object-texture-desc-v2-deferred", deferred, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageRowsAreImplementedWithDeferredHistory()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "cuda-runtime-interface-comparison.csv");
        string[] rows = comparison.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string function in OfficialFunctions)
        {
            Assert.Contains(rows, row =>
                row.Contains($"\"{function}\"", StringComparison.Ordinal) &&
                row.Contains("\"implemented-with-deferred-history\"", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void SmokeConsumerAndReviewPreserveTheOwnerBoundary()
    {
        string smoke = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string review = ReadSource("artifacts", "interface-coverage", "cuda-texture-surface-owner-candidate-review.md");

        Assert.Contains("OwnerDisposedBeforeQuery=True", smoke, StringComparison.Ordinal);
        Assert.Contains("!textureView.IsSpecified", smoke, StringComparison.Ordinal);
        Assert.Contains("CudaTextureObject.CreateCuda11Version2", smoke, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaTextureObject)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaSurfaceObject)", consumer, StringComparison.Ordinal);
        Assert.Contains("linear and pitch2D texture creation | requires a device-pointer lifetime lease", review, StringComparison.Ordinal);
        Assert.Contains("Old deferred manifests remain in place", review, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
