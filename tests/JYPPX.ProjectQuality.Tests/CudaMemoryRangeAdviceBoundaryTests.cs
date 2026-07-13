using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaMemoryRangeAdviceBoundaryTests
{
    [Fact]
    public void CudaMemoryRangeAdviceAndPrefetchApisUseOwnedRangeBridge()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-forty-fifth-batch-memory-range-advice-prefetch.manifest.json");
        string deferredManifest = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");
        string nativeSource = ReadSource("native", "src", "cuda", "api.cpp");
        string runtimeHeader = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string coverageScript = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("cuda-memory-prefetch-range-async-safe", manifest);
        Assert.Contains("jyppx_cuda_memory_prefetch_range_async", manifest);
        Assert.Contains("cuda-memory-advise-range-safe", manifest);
        Assert.Contains("jyppx_cuda_memory_advise_range", manifest);
        Assert.Contains("\"name\": \"memory\", \"type\": \"JYPPX_CudaMemory*\", \"direction\": \"in\"", manifest);
        Assert.Contains("\"name\": \"offset\", \"type\": \"size_t\", \"direction\": \"in\"", manifest);
        Assert.Contains("\"name\": \"count\", \"type\": \"size_t\", \"direction\": \"in\"", manifest);
        Assert.Contains("\"name\": \"stream\", \"type\": \"JYPPX_CudaStream*\", \"direction\": \"in\"", manifest);

        Assert.Contains("cuda-mem-advise-v2-deferred", deferredManifest);
        Assert.Contains("cuda-mem-prefetch-async-v2-deferred", deferredManifest);

        Assert.Contains("jyppx_cuda_memory_prefetch_range_async", runtimeHeader);
        Assert.Contains("jyppx_cuda_memory_advise_range", runtimeHeader);
        Assert.Contains("validate_memory_range_query(memory_object, offset, count, \"cudaMemPrefetchAsync\", &range_pointer)", nativeSource);
        Assert.Contains("validate_memory_range_query(memory_object, offset, count, \"cudaMemAdvise\", &range_pointer)", nativeSource);
        Assert.Contains("cudaMemPrefetchAsync(range_pointer, count", nativeSource);
        Assert.Contains("cudaMemAdvise(range_pointer, count", nativeSource);
        Assert.Contains("return jyppx_cuda_memory_prefetch_range_async(memory, 0, size, destination_device, stream);", nativeSource);
        Assert.Contains("return jyppx_cuda_memory_advise_range(memory, 0, size, advice, device);", nativeSource);

        Assert.Contains("\"cudaMemPrefetchAsync\" = @(\"managed-memory-prefetch\", \"memory-prefetch-range-async-safe\")", coverageScript);
        Assert.Contains("\"cudaMemAdvise\" = @(\"managed-memory-advise\", \"memory-advise-range-safe\")", coverageScript);
    }

    [Fact]
    public void ManagedCudaMemoryRangeAdviceApiDoesNotExposeNativePointers()
    {
        string cudaMemory = ReadSource("src", "JYPPX.CudaSharp", "CudaMemory.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Memory", "NativeCudaApi.MemoryRange.cs");
        string smoke = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");

        Assert.Contains("public void PrefetchAsync(int offset, int count, int destinationDevice, CudaStream stream)", cudaMemory);
        Assert.Contains("public void Advise(int offset, int count, CudaMemoryAdvice advice, int device)", cudaMemory);
        Assert.Contains("ValidateRange(offset, count, nameof(offset), nameof(count))", cudaMemory);
        Assert.Contains("ValidateMemoryAdvice(advice, nameof(advice))", cudaMemory);
        Assert.Contains("PrefetchMemoryRangeAsync(_handle, offset, count, destinationDevice, stream.Handle)", cudaMemory);
        Assert.Contains("AdviseMemoryRange(_handle, offset, count, (int)advice, device)", cudaMemory);
        Assert.Contains("NativeMethodsCuda.jyppx_cuda_memory_prefetch_range_async", interop);
        Assert.Contains("NativeMethodsCuda.jyppx_cuda_memory_advise_range", interop);
        Assert.Contains("managedMemory.Advise(0, managedMemory.SizeInBytes, CudaMemoryAdvice.SetPreferredLocation", smoke);
        Assert.Contains("managedMemory.PrefetchAsync(0, managedMemory.SizeInBytes, CudaDevice.Current, stream)", smoke);
        Assert.DoesNotContain("public IntPtr", cudaMemory);
        Assert.DoesNotContain("public nint", cudaMemory);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
