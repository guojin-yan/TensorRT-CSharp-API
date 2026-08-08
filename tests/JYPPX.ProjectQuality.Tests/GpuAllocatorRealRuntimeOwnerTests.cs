using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using JYPPX.TensorRtSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class GpuAllocatorRealRuntimeOwnerTests
{
    [Fact]
    public void TensorRt8_10And11ManifestsExposeTheSameOwnerSafeAbi()
    {
        string[] suffixes =
        {
            "gpu_allocator_owner_create",
            "gpu_allocator_owner_attach_runtime",
            "gpu_allocator_owner_attach_builder",
            "gpu_allocator_owner_detach",
            "gpu_allocator_owner_get_info"
        };

        foreach (int line in new[] { 8, 10, 11 })
        {
            string path = Path.Combine(
                RepositoryPaths.Root,
                "native",
                "manifests",
                "tensorrt",
                "v" + line,
                $"trt{line}-gpu-allocator-callback-owner.manifest.json");
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            JsonElement[] apis = document.RootElement.GetProperty("apis").EnumerateArray().ToArray();
            Assert.Equal(5, apis.Length);
            foreach (string suffix in suffixes)
            {
                Assert.Contains(apis, api => api.GetProperty("entryPoint").GetString() == $"jyppx_trt{line}_{suffix}");
            }
        }
    }

    [Fact]
    public void NativeOwnerHasStableNoThrowVTablePrivateCudaLedgerAndDrainOrdering()
    {
        string source = ReadSource("native", "src", "tensorrt", "common", "gpu_allocator_callback_owner.inc");
        string types = ReadSource("native", "include", "jyppx", "tensorrt", "types.h");

        Assert.Contains("final : public nvinfer1::IGpuAllocator", source);
        Assert.Contains("GpuAllocatorCallbackOwner(const GpuAllocatorCallbackOwner&) = delete", source);
        Assert.Contains("GpuAllocatorCallbackOwner(GpuAllocatorCallbackOwner&&) = delete", source);
        Assert.Contains("~GpuAllocatorCallbackOwner() noexcept override", source);
        Assert.Contains("void* allocate(", source);
        Assert.Contains("void* reallocate(", source);
        Assert.Contains("bool deallocate(", source);
        Assert.Contains("void free(void* const memory) noexcept override", source);
        Assert.Contains("#if JYPPX_TENSORRT_VERSION_MAJOR_NUM < 10", source);
        Assert.Contains("void* allocateAsync(", source);
        Assert.Contains("bool deallocateAsync(", source);
        Assert.Contains("cudaMalloc", source);
        Assert.Contains("cudaMemcpy", source);
        Assert.Contains("cudaFree", source);
        Assert.Contains("cudaDeviceSynchronize", source);
        Assert.Contains("std::unordered_map<void*, AllocationRecord> allocations_", source);
        Assert.Contains("callback_drained_.wait", source);
        Assert.Contains("set_runtime_gpu_allocator_with_seh_guard", source);
        Assert.Contains("set_builder_gpu_allocator_with_seh_guard", source);
        Assert.Contains("prepare_for_destroy_noexcept", source);
        Assert.Contains("std::is_nothrow_destructible<GpuAllocatorCallbackOwner>", source);
        Assert.Contains("JYPPX_TensorRtGpuAllocatorCallback", types);
        Assert.DoesNotContain("void* current_memory,\n    void* stream", types);
    }

    [Fact]
    public void PublicGpuAllocatorSurfaceIsPointerFree()
    {
        Assert.NotNull(typeof(TensorRtGpuAllocatorCallbackOwner).GetConstructor(
            new[] { typeof(TensorRtApiLine), typeof(TensorRtGpuAllocatorHandler) }));

        Type[] types =
        {
            typeof(TensorRtGpuAllocatorCallbackOwner),
            typeof(TensorRtGpuAllocatorCallbackRequest),
            typeof(TensorRtGpuAllocatorCallbackKind),
            typeof(TensorRtGpuAllocatorAttachmentTarget),
            typeof(TensorRtGpuAllocatorHandler),
            typeof(TensorRtGpuAllocatorRuntimeSnapshot)
        };
        foreach (Type type in types)
        {
            IEnumerable<MemberInfo> members = typeof(MulticastDelegate).IsAssignableFrom(type)
                ? new MemberInfo[] { type.GetMethod("Invoke")! }
                : type.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly);
            Assert.DoesNotContain(members.SelectMany(GetExposedTypes), IsPointerLike);
        }
    }

    [Fact]
    public void RuntimeBuilderAndEngineImplementInheritedOwnerLease()
    {
        string runtime = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.GpuAllocator.cs");
        string builder = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.GpuAllocator.cs");
        string engine = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.cs");
        string runtimeCore = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.cs");
        string builderCore = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.cs");

        Assert.Contains("public void SetGpuAllocator(TensorRtGpuAllocatorCallbackOwner owner)", runtime);
        Assert.Contains("AttachGpuAllocatorOwnerToRuntime", runtime);
        Assert.Contains("DeserializeWithGpuAllocatorLease", runtime);
        Assert.Contains("public void SetGpuAllocator(TensorRtGpuAllocatorCallbackOwner owner)", builder);
        Assert.Contains("AttachGpuAllocatorOwnerToBuilder", builder);
        Assert.Contains("BuildEngineWithGpuAllocatorLease", builder);
        Assert.Contains("_gpuAllocatorKeepAlive?.AttachEngineBorrower(line)", engine);
        Assert.Contains("_gpuAllocatorKeepAlive?.DetachEngineBorrower()", engine);
        Assert.Contains("ReleaseManagedGpuAllocatorForDispose();", runtimeCore);
        Assert.Contains("ReleaseManagedGpuAllocatorForDispose();", builderCore);

        int engineDestroy = engine.IndexOf("_handle.Dispose();", StringComparison.Ordinal);
        int ownerRelease = engine.IndexOf("_gpuAllocatorKeepAlive?.DetachEngineBorrower();", engineDestroy, StringComparison.Ordinal);
        Assert.True(engineDestroy >= 0 && ownerRelease > engineDestroy);
    }

    [Fact]
    public void SmokeRequiresBuilderCallbacksZeroLeaksAndTwoFailClosedCases()
    {
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");

        Assert.Contains("--gpu-allocator-runtime-smoke-only", smoke);
        Assert.Contains("runtime.SetGpuAllocator(runtimeOwner)", smoke);
        Assert.Contains("builder.SetGpuAllocator(builderOwner)", smoke);
        Assert.Contains("builderReleased.LiveAllocationCount == 0UL", smoke);
        Assert.Contains("rejectedSnapshot.RejectedCount == 0UL", smoke);
        Assert.Contains("exceptionSnapshot.CallbackFailureCount == 0UL", smoke);
        Assert.Contains("GpuAllocatorRealRuntime=Passed", smoke);
        Assert.Contains("RuntimeAttachLifecycle=True", smoke);
        Assert.Contains("RealCallbackRuntime={builderReleased.RealCallbackRuntime}", smoke);
    }

    [Fact]
    public void RuntimeEvidenceMatchesCapturedArtifactsAndCompleteTutorial()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "gpu-allocator-real-runtime-tensorrt10.11-evidence.json");
        using JsonDocument evidence = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = evidence.RootElement;
        Assert.Equal("local-tensorrt-gpu-allocator-runtime", root.GetProperty("evidenceKind").GetString());
        Assert.True(root.GetProperty("runtimeAttachment").GetProperty("passed").GetBoolean());
        Assert.False(root.GetProperty("runtimeAttachment").GetProperty("callbackInvocationClaimed").GetBoolean());
        Assert.True(root.GetProperty("builderPositive").GetProperty("realCallbackRuntime").GetBoolean());
        Assert.Equal(0UL, root.GetProperty("builderPositive").GetProperty("liveAllocationCountAfterEngineDispose").GetUInt64());
        Assert.True(root.GetProperty("rejectionNegative").GetProperty("passed").GetBoolean());
        Assert.True(root.GetProperty("exceptionNegative").GetProperty("passed").GetBoolean());

        foreach (string artifactName in new[] { "stdout", "runtimeResultImage" })
        {
            JsonElement artifact = root.GetProperty("artifacts").GetProperty(artifactName);
            string relativePath = artifact.GetProperty("path").GetString()!;
            string expectedHash = artifact.GetProperty("sha256").GetString()!;
            string path = Path.Combine(RepositoryPaths.Root, relativePath.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(path), $"Missing runtime evidence artifact: {relativePath}");
            string actualHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
            Assert.Equal(expectedHash, actualHash);
            Assert.Equal(artifact.GetProperty("length").GetInt64(), new FileInfo(path).Length);
        }

        string tutorial = ReadSource("docs", "articles", "zh-cn", "gpu-allocator-owner-safe-runtime-tutorial.md");
        foreach (string required in new[]
        {
            "JYPPX.TensorRtSharp",
            "JYPPX.CudaSharp",
            "模型获取与转换说明",
            "官方获取方式：不适用",
            "ONNX 暂存目录：不产生 ONNX 文件",
            "SetGpuAllocator",
            "BuildEngineWithConfig",
            "BuilderCallbacks=8",
            "RejectedCount=1",
            "CallbackFailures=1",
            "gpu-allocator-real-runtime-terminal.png",
            "不创建 tag、不发布 Release、不推送任何包"
        })
        {
            Assert.Contains(required, tutorial);
        }
        Assert.DoesNotContain("E:\\GitSpace\\", tutorial, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("D:\\Program Files\\", tutorial, StringComparison.OrdinalIgnoreCase);
    }

    private static IEnumerable<Type> GetExposedTypes(MemberInfo member)
    {
        switch (member)
        {
            case MethodInfo method:
                yield return method.ReturnType;
                foreach (ParameterInfo parameter in method.GetParameters()) yield return parameter.ParameterType;
                break;
            case ConstructorInfo constructor:
                foreach (ParameterInfo parameter in constructor.GetParameters()) yield return parameter.ParameterType;
                break;
            case PropertyInfo property:
                yield return property.PropertyType;
                break;
            case FieldInfo field:
                yield return field.FieldType;
                break;
            case EventInfo eventInfo when eventInfo.EventHandlerType != null:
                yield return eventInfo.EventHandlerType;
                break;
        }
    }

    private static bool IsPointerLike(Type type)
    {
        Type candidate = type.IsByRef ? type.GetElementType()! : type;
        return candidate == typeof(IntPtr) ||
            candidate == typeof(UIntPtr) ||
            candidate.IsPointer ||
            typeof(SafeHandle).IsAssignableFrom(candidate);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}
