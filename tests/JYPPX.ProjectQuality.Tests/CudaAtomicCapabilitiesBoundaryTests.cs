using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaAtomicCapabilitiesBoundaryTests
{
    [Fact]
    public void NativeCudaAtomicCapabilityQueriesAreRealVersionGuardedEntrypoints()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-thirty-fifth-batch-stream-device-boundaries.manifest.json");
        string nativeSource = ReadSource("native", "src", "cuda", "modules", "device_atomic_capabilities.inc");
        string generatedMethods = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Generated", "NativeMethodsCuda.Generated.g.cs");
        string coverageScript = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("cuda-device-get-host-atomic-capabilities", manifest);
        Assert.Contains("cuda-device-get-p2p-atomic-capabilities", manifest);
        Assert.Contains("\"type\": \"uint32_t*\", \"direction\": \"out\", \"managedType\": \"IntPtr\"", manifest);
        Assert.Contains("\"type\": \"const int32_t*\", \"direction\": \"in\", \"managedType\": \"IntPtr\"", manifest);
        Assert.Contains("cudaDeviceGetHostAtomicCapabilities", nativeSource);
        Assert.Contains("cudaDeviceGetP2PAtomicCapabilities", nativeSource);
        Assert.Contains("CUDART_VERSION >= 13000", nativeSource);
        Assert.Contains("convert_cuda_atomic_operations", nativeSource);
        Assert.Contains("report_cuda_dependency_missing(\"CUDA host atomic capability query\")", nativeSource);
        Assert.Contains("report_cuda_dependency_missing(\"CUDA peer atomic capability query\")", nativeSource);
        Assert.Contains("jyppx_cuda_device_get_host_atomic_capabilities", generatedMethods);
        Assert.Contains("jyppx_cuda_device_get_p2p_atomic_capabilities", generatedMethods);
        Assert.Contains("\"cudaDeviceGetHostAtomicCapabilities\" = @(\"device-get-host-atomic-capabilities\")", coverageScript);
        Assert.Contains("\"cudaDeviceGetP2PAtomicCapabilities\" = @(\"device-get-p2p-atomic-capabilities\")", coverageScript);
    }

    [Fact]
    public void ManagedCudaAtomicCapabilityApiUsesTypedEnumsAndPinnedArrays()
    {
        string flags = ReadSource("src", "JYPPX.CudaSharp", "Core", "CudaFlags.cs");
        string deviceApi = ReadSource("src", "JYPPX.CudaSharp", "Devices", "CudaDevice.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeCudaApi.Deployment.cs");

        Assert.Contains("public enum CudaAtomicOperation", flags);
        Assert.Contains("IntegerAdd = 0", flags);
        Assert.Contains("FloatMax = 12", flags);
        Assert.Contains("public enum CudaAtomicCapability", flags);
        Assert.Contains("Vector32x4 = 1u << 6", flags);
        Assert.Contains("public static CudaAtomicCapability[] GetHostAtomicCapabilities", deviceApi);
        Assert.Contains("public static CudaAtomicCapability[] GetP2PAtomicCapabilities", deviceApi);
        Assert.Contains("IReadOnlyList<CudaAtomicOperation> operations", deviceApi);
        Assert.Contains("CopyAtomicOperations(operations)", deviceApi);
        Assert.Contains("public static CudaAtomicCapability[] GetDeviceHostAtomicCapabilities", interop);
        Assert.Contains("public static CudaAtomicCapability[] GetDeviceP2PAtomicCapabilities", interop);
        Assert.Contains("GCHandle.Alloc(operationValues, GCHandleType.Pinned)", interop);
        Assert.Contains("GCHandle.Alloc(capabilityValues, GCHandleType.Pinned)", interop);
        Assert.Contains("CudaNativeStatus.ThrowIfFailed(query(", interop);
        Assert.DoesNotContain("public IntPtr", flags + deviceApi);
        Assert.DoesNotContain("public nint", flags + deviceApi);
    }

    [Fact]
    public void CudaSmokeContainsAtomicCapabilityProbe()
    {
        string program = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");

        Assert.Contains("GetHostAtomicCapabilities", program);
        Assert.Contains("GetP2PAtomicCapabilities", program);
        Assert.Contains("CudaAtomicOperation.IntegerAdd", program);
        Assert.Contains("AtomicCapabilities", program);
        Assert.Contains("Skipped=True Reason=CudaException", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
