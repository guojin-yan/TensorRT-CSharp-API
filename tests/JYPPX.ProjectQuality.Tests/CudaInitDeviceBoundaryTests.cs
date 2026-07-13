using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaInitDeviceBoundaryTests
{
    [Fact]
    public void NativeCudaInitDevicePromotesDeferredEntryToVersionGuardedScalarBoundary()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");
        string runtimeHeader = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string nativeSource = ReadSource("native", "src", "cuda", "api.cpp");
        string deferredSource = ReadSource("native", "src", "cuda", "modules", "deferred", "thirty_eighth_batch_other_deferred.inc");
        string coverageScript = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("\"id\": \"cuda-init-device\"", manifest);
        Assert.Contains("\"entryPoint\": \"jyppx_cuda_init_device\"", manifest);
        Assert.Contains("\"name\": \"device\"", manifest);
        Assert.Contains("\"name\": \"device_flags\"", manifest);
        Assert.Contains("\"name\": \"flags\"", manifest);
        Assert.DoesNotContain("cuda-init-device-deferred", manifest);

        Assert.Contains("jyppx_cuda_init_device(int32_t device, uint32_t device_flags, uint32_t flags)", runtimeHeader);
        Assert.Contains("cudaInitDevice(device, device_flags, flags)", nativeSource);
        Assert.Contains("CUDART_VERSION >= 12000", nativeSource);
        Assert.Contains("cudaInitDevice requires CUDA runtime 12.0 or later.", nativeSource);
        Assert.DoesNotContain("jyppx_cuda_init_device_deferred", deferredSource);
        Assert.Contains("\"cudaInitDevice\" = @(\"init-device\")", coverageScript);
    }

    [Fact]
    public void ManagedCudaInitDeviceUsesFlagsAndDoesNotExposeNativeHandles()
    {
        string deviceApi = ReadSource("src", "JYPPX.CudaSharp", "CudaDevice.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeCudaApi.Deployment.cs");
        string smoke = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");

        Assert.Contains("public static void InitDevice(int ordinal, CudaDeviceRuntimeFlags deviceFlags = CudaDeviceRuntimeFlags.ScheduleAuto, uint flags = 0)", deviceApi);
        Assert.Contains("throw new ArgumentOutOfRangeException(nameof(ordinal)", deviceApi);
        Assert.Contains("NativeCudaApi.InitDevice(ordinal, (uint)deviceFlags, flags)", deviceApi);
        Assert.Contains("public static void InitDevice(int device, uint deviceFlags, uint flags)", interop);
        Assert.Contains("NativeMethodsCuda.jyppx_cuda_init_device(device, deviceFlags, flags)", interop);
        Assert.Contains("CudaDevice.InitDevice(CudaDevice.Current, CudaDevice.RuntimeFlags)", smoke);
        Assert.Contains("InitDevice Device=", smoke);
        Assert.Contains("InitDevice=Skipped Reason=CudaException", smoke);

        Assert.DoesNotContain("public IntPtr", deviceApi);
        Assert.DoesNotContain("public nint", deviceApi);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
