using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaSetValidDevicesBoundaryTests
{
    [Fact]
    public void NativeCudaSetValidDevicesPromotesDeferredEntryToCallerOwnedArrayBoundary()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");
        string runtimeHeader = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string nativeSource = ReadSource("native", "src", "cuda", "api.cpp");
        string deferredSource = ReadSource("native", "src", "cuda", "modules", "deferred", "thirty_eighth_batch_other_deferred.inc");
        string coverageScript = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("\"id\": \"cuda-set-valid-devices\"", manifest);
        Assert.Contains("\"entryPoint\": \"jyppx_cuda_set_valid_devices\"", manifest);
        Assert.Contains("\"type\": \"const int32_t*\", \"direction\": \"in\", \"managedType\": \"IntPtr\"", manifest);
        Assert.Contains("\"name\": \"count\"", manifest);
        Assert.DoesNotContain("cuda-set-valid-devices-deferred", manifest);

        Assert.Contains("jyppx_cuda_set_valid_devices(const int32_t* devices, uint32_t count)", runtimeHeader);
        Assert.Contains("std::vector<int> copied_devices", nativeSource);
        Assert.Contains("copied_devices.push_back(static_cast<int>(device))", nativeSource);
        Assert.Contains("cudaSetValidDevices(copied_devices.data(), static_cast<int>(copied_devices.size()))", nativeSource);
        Assert.Contains("At least one CUDA device ordinal is required.", nativeSource);
        Assert.DoesNotContain("jyppx_cuda_set_valid_devices_deferred", deferredSource);
        Assert.Contains("\"cudaSetValidDevices\" = @(\"set-valid-devices\")", coverageScript);
    }

    [Fact]
    public void ManagedCudaSetValidDevicesCopiesManagedArrayAndAvoidsPublicPointerExposure()
    {
        string deviceApi = ReadSource(
            "src", "JYPPX.CudaSharp", "Devices", "CudaDevice.InitializationSelection.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeCudaApi.Deployment.cs");

        Assert.Contains("public static void SetValidDevices(IReadOnlyList<int> ordinals)", deviceApi);
        Assert.Contains("NativeCudaApi.SetValidDevices(CopyDeviceOrdinals(ordinals))", deviceApi);
        Assert.Contains("private static int[] CopyDeviceOrdinals(IReadOnlyList<int> ordinals)", deviceApi);
        Assert.Contains("throw new ArgumentNullException(nameof(ordinals))", deviceApi);
        Assert.Contains("throw new ArgumentException(\"At least one CUDA device ordinal is required.\"", deviceApi);
        Assert.Contains("throw new ArgumentOutOfRangeException(nameof(ordinals), ordinal", deviceApi);

        Assert.Contains("public static void SetValidDevices(int[] ordinals)", interop);
        Assert.Contains("GCHandle.Alloc(ordinals, GCHandleType.Pinned)", interop);
        Assert.Contains("NativeMethodsCuda.jyppx_cuda_set_valid_devices(ordinalsHandle.AddrOfPinnedObject(), (uint)ordinals.Length)", interop);
        Assert.Contains("ordinalsHandle.Free()", interop);

        Assert.DoesNotContain("public IntPtr", deviceApi);
        Assert.DoesNotContain("public nint", deviceApi);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
