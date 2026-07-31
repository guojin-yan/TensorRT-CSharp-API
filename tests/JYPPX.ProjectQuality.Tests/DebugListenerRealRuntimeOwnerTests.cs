using System.Reflection;
using System.Runtime.InteropServices;
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
        Assert.Contains("DebugListenerRealRuntime=Passed", source);
        Assert.Contains("BorrowedPointerExposed={attached.BorrowedPointerExposed}", source);
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
