using System.Reflection;
using System.Text.Json;
using JYPPX.CudaSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaRuntimeCompilationOwnerTests
{
    [Fact]
    public void ManagedInputsRejectEmbeddedNulAndDuplicates()
    {
        Assert.Throws<ArgumentException>(() => new CudaRtcProgramSource("kernel\0source"));
        Assert.Throws<ArgumentException>(() => new CudaRtcProgramSource(
            "source",
            headers: new[] { new CudaRtcHeader("same.h", "a"), new CudaRtcHeader("same.h", "b") }));
        Assert.Throws<ArgumentException>(() => new CudaRtcProgramSource(
            "source",
            nameExpressions: new[] { "&kernel", "&kernel" }));
        Assert.Throws<ArgumentException>(() => new CudaRtcCompileOptions(additionalOptions: new[] { "--use_fast_math", "--use_fast_math" }));
        Assert.Throws<ArgumentException>(() => new CudaRtcCompileOptions(targetArchitecture: "invalid-target"));
    }

    [Fact]
    public void PublicRtcSurfaceIsPointerFreeAndUsesCopiedArtifacts()
    {
        Type[] rtcTypes =
        {
            typeof(CudaRtcCompiler),
            typeof(CudaRtcProgram),
            typeof(CudaRtcProgramSource),
            typeof(CudaRtcCompileOptions),
            typeof(CudaRtcCompilationResult),
            typeof(CudaRtcArtifact),
            typeof(CudaRtcCapability)
        };

        foreach (Type type in rtcTypes)
        {
            IEnumerable<Type> exposedTypes = type.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static)
                .SelectMany(GetExposedTypes);
            Assert.DoesNotContain(exposedTypes, exposed =>
                exposed == typeof(IntPtr) || exposed == typeof(UIntPtr) ||
                typeof(System.Runtime.InteropServices.SafeHandle).IsAssignableFrom(exposed));
        }

        Assert.NotNull(typeof(CudaRtcArtifact).GetMethod(nameof(CudaRtcArtifact.ToArray)));
        Assert.Null(typeof(CudaRtcArtifact).GetProperty("Handle"));
        Assert.Null(typeof(CudaRtcProgram).GetProperty("Handle"));
    }

    [Fact]
    public void NativeRtcOwnerUsesOptionalLoaderCallerBuffersAndNoThrowGuards()
    {
        string native = ReadSource("native", "src", "cuda", "rtc.cpp");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-sixty-third-batch-runtime-compilation-owner.manifest.json");

        Assert.Contains("JYPPX_NVRTC_LIBRARY", native, StringComparison.Ordinal);
        Assert.Contains("LoadLibraryA", native, StringComparison.Ordinal);
        Assert.Contains("dlopen", native, StringComparison.Ordinal);
        Assert.Contains("run_rtc_noexcept", native, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CUDA_RTC_GUARD", native, StringComparison.Ordinal);
        Assert.Contains("embedded NUL", native, StringComparison.Ordinal);
        Assert.Contains("kMaximumOutputSize", native, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_rtc_program_get_log_safe", header, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_rtc_program_copy_artifact_safe", header, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CUDA_RTC_OPTIONAL_DYNAMIC", manifest, StringComparison.Ordinal);
        Assert.DoesNotContain("target_link_libraries(jyppxtrtbridge PRIVATE CUDA::nvrtc", ReadSource("CMakeLists.txt"), StringComparison.Ordinal);
    }

    [Fact]
    public void SampleSeparatesCompileLoadFromKernelRuntimeProof()
    {
        string sample = ReadSource("samples", "CudaRuntimeCompilation", "Program.cs");
        string readme = ReadSource("samples", "CudaRuntimeCompilation", "README.md");
        Assert.Contains("CudaRtcCompiler.Compile", sample, StringComparison.Ordinal);
        Assert.Contains("CudaKernelLibrary.Load", sample, StringComparison.Ordinal);
        Assert.Contains("CudaRtcResultCode.Compilation", sample, StringComparison.Ordinal);
        Assert.Contains("kernel-launch=False", sample, StringComparison.Ordinal);
        Assert.Contains("not kernel-runtime", readme, StringComparison.Ordinal);
    }

    [Fact]
    public void CapabilityMatrixAndLocalSmokeKeepVersionAndProofBoundaries()
    {
        using JsonDocument matrix = JsonDocument.Parse(ReadSource("artifacts", "cuda-runtime-compilation", "capability-matrix.json"));
        JsonElement root = matrix.RootElement;
        JsonElement[] windows = root.GetProperty("windows").EnumerateArray().ToArray();
        JsonElement[] linux = root.GetProperty("linux").EnumerateArray().ToArray();
        Assert.Equal(new[] { "11.8", "12.1", "12.9", "13.2" }, windows.Select(item => item.GetProperty("toolkitVersion").GetString()));
        Assert.Equal(4, linux.Length);
        Assert.False(windows[0].GetProperty("capabilities").GetProperty("ltoIr").GetBoolean());
        Assert.True(windows[0].GetProperty("capabilities").GetProperty("deprecatedNvvm").GetBoolean());
        Assert.True(windows[2].GetProperty("capabilities").GetProperty("ltoIr").GetBoolean());
        Assert.False(windows[3].GetProperty("capabilities").GetProperty("deprecatedNvvm").GetBoolean());
        Assert.All(linux, item => Assert.Equal("unverified-local-assets-not-found", item.GetProperty("evidenceState").GetString()));
        Assert.False(root.GetProperty("loaderContract").GetProperty("coreBridgeStaticNvrtcLink").GetBoolean());

        using JsonDocument smoke = JsonDocument.Parse(ReadSource("artifacts", "cuda-runtime-compilation", "local-smoke.json"));
        JsonElement[] records = smoke.RootElement.GetProperty("records").EnumerateArray().ToArray();
        Assert.Equal(4, records.Length);
        Assert.All(records, item =>
        {
            Assert.True(item.GetProperty("compileSuccess").GetBoolean());
            Assert.True(item.GetProperty("compileFailureLogCaptured").GetBoolean());
            Assert.True(item.GetProperty("ptxDeterministic").GetBoolean());
            Assert.True(item.GetProperty("cubinCapturedForSm75").GetBoolean());
            Assert.False(item.GetProperty("kernelLaunch").GetBoolean());
            Assert.False(item.GetProperty("gpuReadback").GetBoolean());
            Assert.False(item.GetProperty("correctnessProof").GetBoolean());
        });
        Assert.True(records[2].GetProperty("loadSucceeded").GetBoolean());
        Assert.False(records[3].GetProperty("loadSucceeded").GetBoolean());
        Assert.Contains("cudaErrorUnsupportedPtxVersion", records[3].GetProperty("loadDiagnostic").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public void PackageManifestsKeepBridgeOnlyAndFullRuntimeRtcRolesSeparate()
    {
        using JsonDocument runtime = JsonDocument.Parse(ReadSource("pack", "runtime", "runtime-packages.manifest.json"));
        JsonElement policy = runtime.RootElement.GetProperty("cudaRtcPackaging");
        Assert.False(policy.GetProperty("bridgeOnlyBundlesNvrtc").GetBoolean());
        Assert.Equal(2, policy.GetProperty("windowsAssetsByCudaVersion").GetProperty("11.8").GetArrayLength());
        Assert.Equal(2, policy.GetProperty("windowsAssetsByCudaVersion").GetProperty("13.2").GetArrayLength());
        Assert.Equal("unverified-local-assets-not-found", policy.GetProperty("linuxEvidenceState").GetString());
        Assert.Equal("pending-owner-and-license-review", policy.GetProperty("redistributionApprovalState").GetString());

        using JsonDocument split = JsonDocument.Parse(ReadSource("pack", "runtime-split", "split-runtime-packages.manifest.json"));
        JsonElement role = split.RootElement.GetProperty("cudaRtcSplitRole");
        Assert.Equal("cuda-rtc", role.GetProperty("role").GetString());
        Assert.False(role.GetProperty("bridgePackagesReferenceRole").GetBoolean());
        Assert.True(role.GetProperty("fullRuntimeCollectionsMayReferenceRole").GetBoolean());
        Assert.Equal("planned-not-materialized", role.GetProperty("prototypeState").GetString());
    }

    [Fact]
    public void NativeAbiEvidenceMatchesRtcManifestHeaderAndPeExports()
    {
        using JsonDocument abi = JsonDocument.Parse(ReadSource("artifacts", "cuda-runtime-compilation", "native-abi-surface.json"));
        JsonElement root = abi.RootElement;
        Assert.True(root.GetProperty("passed").GetBoolean());
        Assert.Equal(12, root.GetProperty("manifestEntryPointCount").GetInt32());
        Assert.Equal(12, root.GetProperty("declaredEntryPointCount").GetInt32());
        Assert.Equal(12, root.GetProperty("matchedPeExportCount").GetInt32());
        Assert.Equal(0, root.GetProperty("missingDeclarationCount").GetInt32());
        Assert.Equal(0, root.GetProperty("missingPeExportCount").GetInt32());
    }

    private static IEnumerable<Type> GetExposedTypes(MemberInfo member)
    {
        if (member is PropertyInfo property)
        {
            yield return property.PropertyType;
        }
        else if (member is MethodInfo method)
        {
            yield return method.ReturnType;
            foreach (ParameterInfo parameter in method.GetParameters())
            {
                yield return parameter.ParameterType;
            }
        }
        else if (member is ConstructorInfo constructor)
        {
            foreach (ParameterInfo parameter in constructor.GetParameters())
            {
                yield return parameter.ParameterType;
            }
        }
        else if (member is FieldInfo field)
        {
            yield return field.FieldType;
        }
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}
