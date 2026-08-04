using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using JYPPX.TensorRtSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OutputAllocatorRealRuntimeOwnerTests
{
    [Fact]
    public void TensorRt8_10And11ManifestsExposeTheSameOwnerSafeAbi()
    {
        string[] suffixes =
        {
            "output_allocator_owner_create",
            "output_allocator_owner_attach",
            "output_allocator_owner_detach",
            "output_allocator_owner_get_info"
        };

        foreach (int line in new[] { 8, 10, 11 })
        {
            string path = Path.Combine(
                RepositoryPaths.Root,
                "native",
                "manifests",
                "tensorrt",
                "v" + line,
                $"trt{line}-output-allocator-callback-owner.manifest.json");
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            JsonElement[] apis = document.RootElement.GetProperty("apis").EnumerateArray().ToArray();
            Assert.Equal(4, apis.Length);
            foreach (string suffix in suffixes)
            {
                Assert.Contains(apis, api => api.GetProperty("entryPoint").GetString() == $"jyppx_trt{line}_{suffix}");
            }
        }
    }

    [Fact]
    public void NativeOwnerHasNoThrowVTableAllocationLedgerAndDrainOrdering()
    {
        string source = ReadSource("native", "src", "tensorrt", "common", "output_allocator_callback_owner.inc");

        Assert.Contains("final : public nvinfer1::IOutputAllocator", source);
        Assert.Contains("OutputAllocatorCallbackOwner(const OutputAllocatorCallbackOwner&) = delete", source);
        Assert.Contains("OutputAllocatorCallbackOwner(OutputAllocatorCallbackOwner&&) = delete", source);
        Assert.Contains("~OutputAllocatorCallbackOwner() noexcept override", source);
        Assert.Contains("reallocateOutput(", source);
        Assert.Contains("reallocateOutputAsync(", source);
        Assert.Contains("notifyShape(", source);
        Assert.Contains("cudaMalloc", source);
        Assert.Contains("cudaFree", source);
        Assert.Contains("cudaDeviceSynchronize", source);
        Assert.Contains("allocations_", source);
        Assert.Contains("callback_drained_.wait", source);
        Assert.Contains("bool detaching_{false};", source);
        Assert.Contains("if (!enter_callback())", source);
        Assert.Contains("prepare_for_destroy_noexcept", source);
        Assert.Contains("std::is_nothrow_destructible<OutputAllocatorCallbackOwner>", source);

        int leaveCallback = source.IndexOf("void leave_callback() noexcept", StringComparison.Ordinal);
        int stateLock = source.IndexOf("std::lock_guard<std::mutex> lock(state_mutex_);", leaveCallback, StringComparison.Ordinal);
        int decrement = source.IndexOf("in_flight_callback_count_.fetch_sub", leaveCallback, StringComparison.Ordinal);
        int notify = source.IndexOf("callback_drained_.notify_all();", leaveCallback, StringComparison.Ordinal);
        Assert.True(leaveCallback >= 0 && stateLock > leaveCallback && decrement > stateLock && notify > decrement);
    }

    [Fact]
    public void PublicRuntimeSurfaceIsPointerFreeAndKeepsOldConstructors()
    {
        Assert.NotNull(typeof(TensorRtOutputAllocatorCallbackOwner).GetConstructor(Type.EmptyTypes));
        Assert.NotNull(typeof(TensorRtOutputAllocatorCallbackOwner).GetConstructor(
            new[] { typeof(TensorRtApiLine), typeof(TensorRtOutputAllocatorHandler) }));
        Assert.NotNull(typeof(TensorRtOutputAllocatorCallbackRequest).GetConstructor(
            new[] { typeof(string), typeof(ulong), typeof(ulong), typeof(long[]), typeof(string), typeof(bool) }));

        Type[] types =
        {
            typeof(TensorRtOutputAllocatorCallbackOwner),
            typeof(TensorRtOutputAllocatorCallbackRequest),
            typeof(TensorRtOutputAllocatorCallbackKind),
            typeof(TensorRtOutputAllocatorHandler),
            typeof(TensorRtOutputAllocatorRuntimeSnapshot),
            typeof(TensorRtExecutionContext)
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
    public void ManagedContextOwnsAttachClearAndDisposeLifecycle()
    {
        string context = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.cs");
        string attach = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.OutputAllocator.cs");
        string clear = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs");
        string lifecycle = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "MemoryAllocation", "TensorRtOutputAllocatorCallbackOwner.Lifecycle.cs");
        string callback = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "MemoryAllocation", "TensorRtOutputAllocatorCallbackOwner.RuntimeCallback.cs");

        Assert.Contains("public void SetOutputAllocator", attach);
        Assert.Contains("owner.AttachBorrower(Line)", attach);
        Assert.Contains("_outputAllocatorKeepAlive.Add", attach);
        Assert.Contains("NativeBridgeApi.DetachOutputAllocatorOwner", clear);
        Assert.Contains("DetachOutputAllocatorsForDispose", context);
        Assert.Contains("_nativeHandle?.Dispose()", lifecycle);
        Assert.Contains("FreeRuntimeCallbackHandles()", lifecycle);
        Assert.Contains("private static int s_runtimeCallbackDepth", callback);
        Assert.Contains("s_runtimeCallbackDepth++", callback);
        Assert.Contains("s_runtimeCallbackDepth--", callback);
    }

    [Fact]
    public void OptInSmokeRequiresPositiveAllocationAndNegativeFailClosedResult()
    {
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");

        Assert.Contains("--output-allocator-runtime-smoke-only", smoke);
        Assert.Contains("positiveContext.SetOutputAllocator", smoke);
        Assert.Contains("positiveAttached.RealCallbackRuntime", smoke);
        Assert.Contains("positiveDetached.LiveAllocationCount == 0UL", smoke);
        Assert.Contains("negativeEnqueueFailed", smoke);
        Assert.Contains("negativeAttached.AllocationCount == 0UL", smoke);
        Assert.Contains("OutputAllocatorRealRuntime=Passed", smoke);
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

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}
