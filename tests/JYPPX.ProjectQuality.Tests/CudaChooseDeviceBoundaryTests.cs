using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaChooseDeviceBoundaryTests
{
    [Fact]
    public void NativeCudaChooseDevicePromotesDeferredEntryToSafeValueStruct()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-thirty-fifth-batch-stream-device-boundaries.manifest.json");
        string runtimeHeader = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string typesHeader = ReadSource("native", "include", "jyppx", "cuda", "types.h");
        string nativeSource = ReadSource("native", "src", "cuda", "modules", "deferred", "thirty_fifth_batch_stream_device_deferred.inc");

        Assert.Contains("\"id\": \"cuda-choose-device\"", manifest);
        Assert.Contains("\"entryPoint\": \"jyppx_cuda_choose_device\"", manifest);
        Assert.Contains("const JYPPX_CudaDeviceSelectionRequirements*", manifest);
        Assert.Contains("\"moduleManagedType\": \"in NativeCudaDeviceSelectionRequirements\"", manifest);
        Assert.DoesNotContain("cuda-choose-device-deferred", manifest);

        Assert.Contains("JYPPX_CudaDeviceSelectionRequirements", typesHeader);
        Assert.Contains("jyppx_cuda_choose_device(const JYPPX_CudaDeviceSelectionRequirements* requirements, int32_t* out_device)", runtimeHeader);
        Assert.Contains("cudaDeviceProp properties{}", nativeSource);
        Assert.Contains("cudaChooseDevice(&device, &properties)", nativeSource);
        Assert.Contains("properties.multiProcessorCount = requirements->multi_processor_count", nativeSource);
        Assert.Contains("report_cuda_dependency_missing(\"CUDA device selection\")", nativeSource);
        Assert.DoesNotContain("cudaChooseDevice deferred", nativeSource);
    }

    [Fact]
    public void ManagedCudaChooseDeviceUsesTypedRequirementsWithoutPointerExposure()
    {
        string requirements = ReadSource("src", "JYPPX.CudaSharp", "Devices", "CudaDeviceSelectionRequirements.cs");
        string structs = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeStructs.cs");
        string deviceApi = ReadSource("src", "JYPPX.CudaSharp", "Devices", "CudaDevice.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeCudaApi.Deployment.cs");

        Assert.Contains("public sealed class CudaDeviceSelectionRequirements", requirements);
        Assert.Contains("RequireHostMemoryMapping", requirements);
        Assert.Contains("RequireIntegratedGpu", requirements);
        Assert.Contains("internal NativeCudaDeviceSelectionRequirements ToNative()", requirements);
        Assert.Contains("internal struct NativeCudaDeviceSelectionRequirements", structs);
        Assert.Contains("public static int ChooseDevice(CudaDeviceSelectionRequirements requirements)", deviceApi);
        Assert.Contains("throw new ArgumentNullException(nameof(requirements))", deviceApi);
        Assert.Contains("NativeCudaDeviceSelectionRequirements nativeRequirements = requirements.ToNative()", deviceApi);
        Assert.Contains("public static int ChooseDevice(in NativeCudaDeviceSelectionRequirements requirements)", interop);
        Assert.Contains("NativeMethodsCuda.jyppx_cuda_choose_device(in requirements, out int device)", interop);

        Assert.DoesNotContain("public IntPtr", requirements + deviceApi);
        Assert.DoesNotContain("public nint", requirements + deviceApi);
        Assert.DoesNotContain("cudaDeviceProp", requirements);
    }

    [Fact]
    public void CudaSmokeRunnerExercisesChooseDeviceHighLevelWrapper()
    {
        string program = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");

        Assert.Contains("CudaDeviceSelectionRequirements", program);
        Assert.Contains("CudaDevice.ChooseDevice(chooseRequirements)", program);
        Assert.Contains("ChooseDevice Chosen=", program);
        Assert.Contains("ChooseDevice=Skipped Reason=CudaException", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
