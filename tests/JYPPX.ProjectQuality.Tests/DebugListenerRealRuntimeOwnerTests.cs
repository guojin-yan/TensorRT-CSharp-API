using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerRealRuntimeOwnerTests
{
    [Fact]
    public void TensorRt10And11ManifestsExposeTheSameOwnerSafeAbi()
    {
        string root = FindRepositoryRoot();
        string[] expectedSuffixes =
        {
            "debug_listener_owner_create",
            "debug_listener_owner_attach",
            "debug_listener_owner_detach",
            "debug_listener_owner_get_info"
        };

        foreach (int line in new[] { 10, 11 })
        {
            string path = Path.Combine(
                root,
                "native",
                "manifests",
                "tensorrt",
                "v" + line,
                $"trt{line}-debug-listener-callback-owner.manifest.json");
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            JsonElement[] apis = document.RootElement.GetProperty("apis").EnumerateArray().ToArray();
            Assert.Equal(4, apis.Length);
            foreach (string suffix in expectedSuffixes)
            {
                Assert.Contains(apis, api => api.GetProperty("entryPoint").GetString() == $"jyppx_trt{line}_{suffix}");
            }
        }
    }

    [Fact]
    public void NativeOwnerHasStableNoThrowVTableAndCopiedMetadataBoundary()
    {
        string root = FindRepositoryRoot();
        string source = File.ReadAllText(Path.Combine(
            root,
            "native",
            "src",
            "tensorrt",
            "common",
            "debug_listener_callback_owner.inc"));

        Assert.Contains("final : public nvinfer1::IDebugListener", source);
        Assert.Contains("DebugListenerCallbackOwner(const DebugListenerCallbackOwner&) = delete", source);
        Assert.Contains("DebugListenerCallbackOwner(DebugListenerCallbackOwner&&) = delete", source);
        Assert.Contains("~DebugListenerCallbackOwner() noexcept override", source);
        Assert.Contains("processDebugTensor(", source);
        Assert.Contains("std::array<char, 256> copied_name", source);
        Assert.Contains("std::array<int64_t, 8> copied_shape", source);
        Assert.Contains("(void)addr", source);
        Assert.Contains("(void)stream", source);
        Assert.Contains("callback_drained_.wait", source);
        Assert.Contains("bool detaching_{false};", source);
        Assert.Contains("std::mutex lifecycle_mutex_;", source);
        Assert.Contains("if (!enter_callback())", source);
        Assert.Contains("std::lock_guard<std::mutex> lock(state_mutex_);\n        if (detaching_)", source);
        Assert.Contains("nvinfer1::IExecutionContext* context = nullptr;", source);
        Assert.Contains("prepare_for_destroy_noexcept", source);
        Assert.Contains("owner->prepare_for_destroy_noexcept()", source);
        Assert.DoesNotContain("return detach(&detached) == JYPPX_STATUS_OK", source);
        Assert.Contains("std::is_nothrow_destructible<DebugListenerCallbackOwner>", source);
        Assert.Contains("noexcept(std::declval<DebugListenerCallbackOwner&>().processDebugTensor", source);

        int leaveCallback = source.IndexOf("void leave_callback() noexcept", StringComparison.Ordinal);
        int synchronizedDecrement = source.IndexOf("std::lock_guard<std::mutex> lock(state_mutex_);", leaveCallback, StringComparison.Ordinal);
        int notifyDrained = source.IndexOf("callback_drained_.notify_all();", leaveCallback, StringComparison.Ordinal);
        Assert.True(leaveCallback >= 0 && synchronizedDecrement > leaveCallback && notifyDrained > synchronizedDecrement);
    }

    [Fact]
    public void PublicRuntimeSurfaceRemainsPointerFree()
    {
        Type[] types =
        {
            typeof(TensorRtDebugListenerCallbackOwner),
            typeof(TensorRtDebugListenerHandler),
            typeof(TensorRtDebugListenerRuntimeSnapshot),
            typeof(TensorRtDebugTensorMetadataSnapshot),
            typeof(TensorRtExecutionContext)
        };

        foreach (Type type in types)
        {
            IEnumerable<MemberInfo> publicMembers = typeof(MulticastDelegate).IsAssignableFrom(type)
                ? new MemberInfo[] { type.GetMethod("Invoke")! }
                : type.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly);
            IEnumerable<Type> exposedTypes = publicMembers
                .SelectMany(GetExposedTypes);
            Assert.DoesNotContain(exposedTypes, IsPointerLike);
        }
    }

    [Fact]
    public void ManagedContextOwnsDetachBeforeReleaseLifecycle()
    {
        string root = FindRepositoryRoot();
        string context = File.ReadAllText(Path.Combine(
            root,
            "src",
            "JYPPX.TensorRtSharp",
            "Execution",
            "TensorRtExecutionContext.cs"));
        string outputDebug = File.ReadAllText(Path.Combine(
            root,
            "src",
            "JYPPX.TensorRtSharp",
            "Execution",
            "TensorRtExecutionContext.OutputDebug.cs"));
        string lifecycle = File.ReadAllText(Path.Combine(
            root,
            "src",
            "JYPPX.TensorRtSharp",
            "Callbacks",
            "Debugging",
            "TensorRtDebugListenerCallbackOwner.Lifecycle.cs"));
        string runtimeCallback = File.ReadAllText(Path.Combine(
            root,
            "src",
            "JYPPX.TensorRtSharp",
            "Callbacks",
            "Debugging",
            "TensorRtDebugListenerCallbackOwner.RuntimeCallback.cs"));

        Assert.Contains("NativeBridgeApi.DetachDebugListenerOwner", context);
        Assert.Contains("public void SetDebugListener", outputDebug);
        Assert.Contains("listener.AttachBorrower(Line)", outputDebug);
        Assert.Contains("_debugListenerKeepAlive = listener", outputDebug);
        Assert.Contains("_nativeHandle?.Dispose()", lifecycle);
        Assert.Contains("FreeCallbackState()", lifecycle);
        Assert.Contains("private static int s_runtimeCallbackDepth", runtimeCallback);
        Assert.Contains("s_runtimeCallbackDepth++", runtimeCallback);
        Assert.Contains("s_runtimeCallbackDepth--", runtimeCallback);

        int contextRelease = context.IndexOf("_handle.Dispose();", StringComparison.Ordinal);
        int listenerBorrowRelease = context.IndexOf("debugListener?.DetachBorrower();", StringComparison.Ordinal);
        Assert.True(contextRelease >= 0 && listenerBorrowRelease > contextRelease);
    }

    [Fact]
    public void OptInSmokeBuildsDebugTensorAndRequiresRealInvocation()
    {
        string root = FindRepositoryRoot();
        string source = File.ReadAllText(Path.Combine(
            root,
            "smoke",
            "CallbackAllocatorSafeControlsSmokeRunner",
            "Program.cs"));

        Assert.Contains("network.MarkDebugTensor(output)", source);
        Assert.Contains("context.SetDebugListener(owner)", source);
        Assert.Contains("context.SetTensorDebugState(\"debug_output\", true)", source);
        Assert.Contains("attached.IsRealCallbackRuntimeProof", source);
        Assert.Contains("negativeEnqueueFailed", source);
        Assert.Contains("negativeCallbackRejected", source);
        Assert.Contains("negativeAttached.FailureCount > 0", source);
        Assert.Contains("NegativeFailureCount={negativeAttached.FailureCount}", source);
        Assert.Contains("DebugListenerRealRuntime=Passed", source);
        Assert.Contains("BorrowedPointerExposed={attached.BorrowedPointerExposed}", source);
        Assert.Contains("--debug-listener-runtime-smoke-only", source);
        Assert.Contains("Mode=DebugListenerRuntimeSmokeOnly", source);
    }

    [Fact]
    public void RuntimeEvidenceMatchesCapturedArtifactsAndTutorial()
    {
        string root = FindRepositoryRoot();
        string evidencePath = Path.Combine(
            root,
            "samples",
            "assets",
            "debug-listener-real-runtime-tensorrt10.11-evidence.json");
        using JsonDocument evidence = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement rootElement = evidence.RootElement;
        Assert.Equal("local-tensorrt-debug-listener-runtime", rootElement.GetProperty("evidenceKind").GetString());
        Assert.True(rootElement.GetProperty("positive").GetProperty("passed").GetBoolean());
        Assert.True(rootElement.GetProperty("positive").GetProperty("isRealCallbackRuntimeProof").GetBoolean());
        Assert.True(rootElement.GetProperty("negative").GetProperty("passed").GetBoolean());
        Assert.False(rootElement.GetProperty("negative").GetProperty("enqueueFailed").GetBoolean());

        foreach (string artifactName in new[] { "stdout", "runtimeResultImage" })
        {
            JsonElement artifact = rootElement.GetProperty("artifacts").GetProperty(artifactName);
            string relativePath = artifact.GetProperty("path").GetString()!;
            string expectedHash = artifact.GetProperty("sha256").GetString()!;
            string artifactPath = Path.Combine(root, relativePath.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(artifactPath), $"Missing runtime evidence artifact: {relativePath}");
            using SHA256 sha256 = SHA256.Create();
            string actualHash = Convert.ToHexString(sha256.ComputeHash(File.ReadAllBytes(artifactPath))).ToLowerInvariant();
            Assert.Equal(expectedHash, actualHash);
        }

        string tutorial = File.ReadAllText(Path.Combine(
            root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-real-runtime-tutorial.md"));
        foreach (string requiredText in new[]
        {
            "JYPPX.TensorRtSharp",
            "JYPPX.CudaSharp",
            "模型与转换说明",
            "没有模型下载地址或 ONNX 转换步骤",
            "模型名称、官方获取方式、导出/转换命令",
            "MarkDebugTensor",
            "SetDebugListener",
            "processDebugTensor",
            "InvocationCount=1",
            "NegativeFailureCount=1",
            "NegativeEnqueueFailed",
            "debug-listener-real-runtime-terminal.png",
            "不创建 tag、Release、NuGet 或 GitHub Packages"
        })
        {
            Assert.Contains(requiredText, tutorial);
        }

        Assert.DoesNotContain("E:\\GitSpace\\", tutorial, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("C:\\Users\\", tutorial, StringComparison.OrdinalIgnoreCase);
    }

    private static IEnumerable<Type> GetExposedTypes(MemberInfo member)
    {
        switch (member)
        {
            case MethodInfo method:
                yield return method.ReturnType;
                foreach (ParameterInfo parameter in method.GetParameters())
                {
                    yield return parameter.ParameterType;
                }
                break;
            case ConstructorInfo constructor:
                foreach (ParameterInfo parameter in constructor.GetParameters())
                {
                    yield return parameter.ParameterType;
                }
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

    private static string FindRepositoryRoot()
    {
        DirectoryInfo? directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory != null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "TensorRtSharp.sln")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException("Repository root was not found.");
    }
}
