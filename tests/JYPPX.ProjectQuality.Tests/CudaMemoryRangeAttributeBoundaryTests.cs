using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaMemoryRangeAttributeBoundaryTests
{
    [Fact]
    public void CudaMemoryRangeAttributeApisAreLiftedWithoutDeletingDeferredRecords()
    {
        string safeManifest = ReadSource("native", "manifests", "cuda", "cuda-forty-second-batch-memory-range-attributes.manifest.json");
        string accessedByManifest = ReadSource("native", "manifests", "cuda", "cuda-forty-third-batch-memory-range-accessed-by.manifest.json");
        string deferredManifest = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");
        string nativeSource = ReadSource("native", "src", "cuda", "api.cpp");
        string typesHeader = ReadSource("native", "include", "jyppx", "cuda", "types.h");
        string runtimeHeader = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string coverageScript = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("cuda-mem-range-get-attribute-scalar-safe", safeManifest);
        Assert.Contains("cuda-mem-range-get-attributes-scalar-safe", safeManifest);
        Assert.Contains("\"type\": \"JYPPX_CudaMemory*\", \"direction\": \"in\"", safeManifest);
        Assert.Contains("\"type\": \"const int32_t*\", \"direction\": \"in\", \"managedType\": \"IntPtr\"", safeManifest);
        Assert.Contains("\"type\": \"JYPPX_CudaMemRangeAttributeValue*\", \"direction\": \"out\", \"managedType\": \"IntPtr\"", safeManifest);
        Assert.Contains("cuda-mem-range-get-accessed-by-count-safe", accessedByManifest);
        Assert.Contains("cuda-mem-range-copy-accessed-by-devices-safe", accessedByManifest);
        Assert.Contains("\"type\": \"int32_t*\", \"direction\": \"out\", \"managedType\": \"IntPtr\"", accessedByManifest);
        Assert.Contains("\"type\": \"size_t*\", \"direction\": \"out\", \"managedType\": \"out UIntPtr\"", accessedByManifest);

        Assert.Contains("cuda-mem-range-get-attribute-deferred", deferredManifest);
        Assert.Contains("cuda-mem-range-get-attributes-deferred", deferredManifest);

        Assert.Contains("typedef struct JYPPX_CudaMemRangeAttributeValue", typesHeader);
        Assert.Contains("jyppx_cuda_memory_range_get_attribute", runtimeHeader);
        Assert.Contains("jyppx_cuda_memory_range_get_attributes", runtimeHeader);
        Assert.Contains("jyppx_cuda_memory_range_get_accessed_by_count", runtimeHeader);
        Assert.Contains("jyppx_cuda_memory_range_copy_accessed_by_devices", runtimeHeader);
        Assert.Contains("cudaMemRangeGetAttribute", nativeSource);
        Assert.Contains("cudaMemRangeGetAttributes", nativeSource);
        Assert.Contains("validate_scalar_mem_range_attribute", nativeSource);
        Assert.Contains("query_memory_range_accessed_by_devices", nativeSource);
        Assert.Contains("CUDART_VERSION >= 12030", nativeSource);

        Assert.Contains("\"cudaMemRangeGetAttribute\" = @(\"mem-range-get-attribute-deferred\", \"mem-range-get-attribute-scalar-safe\", \"mem-range-get-accessed-by-count-safe\", \"mem-range-copy-accessed-by-devices-safe\")", coverageScript);
        Assert.Contains("\"cudaMemRangeGetAttributes\" = @(\"mem-range-get-attributes-deferred\", \"mem-range-get-attributes-scalar-safe\")", coverageScript);
    }

    [Fact]
    public void ManagedCudaMemoryRangeAttributeApiDoesNotExposeNativePointers()
    {
        string publicTypes = ReadSource("src", "JYPPX.CudaSharp", "CudaMemoryRangeAttribute.cs");
        string cudaMemory = ReadSource("src", "JYPPX.CudaSharp", "CudaMemory.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Memory", "NativeCudaApi.MemoryRange.cs");
        string nativeStructs = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeStructs.cs");

        Assert.Contains("public enum CudaMemoryRangeAttribute", publicTypes);
        Assert.Contains("public readonly struct CudaMemoryRangeAttributeValue", publicTypes);
        Assert.Contains("public sealed class CudaMemoryRangeDiagnosticSummary", publicTypes);
        Assert.Contains("public CudaMemoryRangeAttributeValue GetRangeAttribute", cudaMemory);
        Assert.Contains("public CudaMemoryRangeAttributeValue[] GetRangeAttributes", cudaMemory);
        Assert.Contains("public int[] GetRangeAccessedByDevices", cudaMemory);
        Assert.Contains("public CudaMemoryRangeDiagnosticSummary GetRangeDiagnosticSummary", cudaMemory);
        Assert.Contains("public int CopiedScalarAttributeCount", publicTypes);
        Assert.Contains("public int CopiedAccessedByDeviceCount", publicTypes);
        Assert.Contains("public bool AdviceControlAttempted", publicTypes);
        Assert.Contains("public bool PrefetchControlAttempted", publicTypes);
        Assert.Contains("public bool PointerFreeCopiedSummary", publicTypes);
        Assert.Contains("public bool CanPromoteRuntimeProof", publicTypes);
        Assert.Contains("public bool CanDeleteDeferredRecord", publicTypes);
        Assert.Contains("NativeCudaMemRangeAttributeValue", nativeStructs);
        Assert.Contains("GCHandle.Alloc(attributes, GCHandleType.Pinned)", interop);
        Assert.Contains("GCHandle.Alloc(values, GCHandleType.Pinned)", interop);
        Assert.Contains("GCHandle.Alloc(devices, GCHandleType.Pinned)", interop);
        Assert.Contains("Array.Resize(ref devices, (int)requiredValue)", interop);
        Assert.Contains("AccessedBy returns a device-id array", cudaMemory);
        Assert.DoesNotContain("public IntPtr", publicTypes + cudaMemory);
        Assert.DoesNotContain("public nint", publicTypes + cudaMemory);
    }

    [Fact]
    public void CudaSmokeCoversManagedMemoryRangeAttributeQuery()
    {
        string program = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");

        Assert.Contains("GetRangeAttribute(CudaMemoryRangeAttribute.PreferredLocation)", program);
        Assert.Contains("GetRangeAttributes(", program);
        Assert.Contains("GetRangeAccessedByDevices()", program);
        Assert.Contains("ManagedMemoryRangeAttributes", program);
        Assert.Contains("ManagedMemoryAccessedBy", program);
        Assert.Contains("MemoryRangeSummary=", program);
        Assert.Contains("GetRangeDiagnosticSummary(", program);
        Assert.Contains("Skipped=True Reason=CudaException", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
