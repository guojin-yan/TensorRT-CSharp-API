using System.Reflection;
using System.Text.Json;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ExecutionContextDeviceMemoryOwnerTests
{
    [Fact]
    public void PublicSurfacePreservesCompatibilityAndAddsExplicitV2WithoutPointers()
    {
        MethodInfo compatibility = AssertMethod("SetDeviceMemory", typeof(CudaMemory));
        MethodInfo v2 = AssertMethod("SetDeviceMemoryV2", typeof(CudaMemory));
        MethodInfo clear = AssertMethod("ClearDeviceMemory");

        Assert.Equal(typeof(void), compatibility.ReturnType);
        Assert.Equal(typeof(void), v2.ReturnType);
        Assert.Equal(typeof(void), clear.ReturnType);
        Assert.Equal(typeof(bool), typeof(TensorRtExecutionContext).GetProperty("HasBoundDeviceMemory")!.PropertyType);
        Assert.Equal(typeof(int), typeof(TensorRtExecutionContext).GetProperty("BoundDeviceMemorySizeInBytes")!.PropertyType);
        Assert.Equal(typeof(int), typeof(TensorRtExecutionContext).GetProperty("RetainedDeviceMemoryLeaseCount")!.PropertyType);
        Assert.DoesNotContain(
            typeof(TensorRtExecutionContext).GetMembers(BindingFlags.Instance | BindingFlags.Public),
            member => ExposedTypes(member).Any(type => type == typeof(IntPtr) || type == typeof(UIntPtr)));
    }

    [Fact]
    public void NativeAndManagedPathsRetainOwnersAndKeepDeferredAbiHistory()
    {
        string managed = Read("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.DeviceMemory.cs");
        string context = Read("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.cs");
        string lease = Read("src", "JYPPX.TensorRtSharp", "Internal", "Handles", "TensorRtDeviceMemoryHandleLease.cs");
        string native10 = Read("native", "src", "tensorrt", "v10", "modules", "context", "deployment_context.inc");
        string native11 = Read("native", "src", "tensorrt", "v11", "api.cpp");
        string deferred10 = Read("native", "src", "tensorrt", "v10", "modules", "deferred", "twenty_third_batch_deferred.inc");
        string deferred11 = Read("native", "src", "tensorrt", "v11", "modules", "deferred", "twenty_third_batch_deferred.inc");

        Assert.Contains("TensorRtDeviceMemoryHandleLease.Create", managed, StringComparison.Ordinal);
        Assert.Contains("_retiredDeviceMemoryLeases.Add", managed, StringComparison.Ordinal);
        Assert.Contains("_handle.Dispose()", context, StringComparison.Ordinal);
        Assert.True(
            context.IndexOf("_handle.Dispose()", StringComparison.Ordinal) <
            context.IndexOf("deviceMemoryLease?.Dispose()", StringComparison.Ordinal));
        Assert.True(
            context.IndexOf("if (_auxiliaryStreamContextDisposed)", StringComparison.Ordinal) <
            context.IndexOf("_deviceMemoryContextDisposed = true", StringComparison.Ordinal));
        Assert.Contains("DangerousAddRef", lease, StringComparison.Ordinal);
        Assert.Contains("DangerousRelease", lease, StringComparison.Ordinal);
        Assert.Contains("~TensorRtDeviceMemoryHandleLease", lease, StringComparison.Ordinal);
        Assert.Contains("setDeviceMemoryV2(memory_payload->pointer", native10, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt10_execution_context_set_device_memory_v2", native10, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt11_execution_context_set_device_memory_v2", native11, StringComparison.Ordinal);
        Assert.Contains("setDeviceMemoryV2(nullptr, 0)", native10 + native11, StringComparison.Ordinal);
        Assert.Contains("set_device_memory_v2_deferred", deferred10, StringComparison.Ordinal);
        Assert.Contains("set_device_memory_v2_deferred", deferred11, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("10")]
    [InlineData("11")]
    public void V2ManifestsDescribeContextBorrowedManagedLease(string line)
    {
        using JsonDocument document = JsonDocument.Parse(Read(
            "native", "manifests", "tensorrt", "v" + line, $"trt{line}-device-memory-v2-owner-safe.manifest.json"));
        JsonElement[] apis = document.RootElement.GetProperty("apis").EnumerateArray().ToArray();
        JsonElement set = Assert.Single(apis, api => api.GetProperty("id").GetString() == $"trt{line}-execution-context-set-device-memory-v2");
        Assert.Equal("context-borrows-memory-managed-lease", set.GetProperty("ownership").GetString());
        Assert.Equal(2, set.GetProperty("parameters").GetArrayLength());
        Assert.Contains("JYPPX_HAS_CUDA_TOOLKIT", set.GetProperty("versionGuard").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public void RuntimeSmokeDisposesCallerWrappersBeforeInferenceAndChecksClearState()
    {
        string smoke = Read("smoke", "NetworkConvolutionScaleSmokeRunner", "Program.cs");
        Assert.Contains("CreateExecutionContextWithoutDeviceMemory", smoke, StringComparison.Ordinal);
        Assert.Contains("compatibilityMemory.Dispose()", smoke, StringComparison.Ordinal);
        Assert.Contains("activeMemory.Dispose()", smoke, StringComparison.Ordinal);
        Assert.Contains("Math.Max(engine.DeviceMemorySizeInBytes, 1UL)", smoke, StringComparison.Ordinal);
        Assert.Contains("RetainedDeviceMemoryLeaseCount != 2", smoke, StringComparison.Ordinal);
        Assert.Contains("context.EnqueueAsync(stream)", smoke, StringComparison.Ordinal);
        Assert.Contains("context.ClearDeviceMemory()", smoke, StringComparison.Ordinal);
        Assert.Contains("ConvolutionScalePaddingOutputMatch=True", smoke, StringComparison.Ordinal);
    }

    [Fact]
    public void LocalRuntimeEvidenceSeparatesPassedLinesFromBlockedAndPublicProof()
    {
        using JsonDocument document = JsonDocument.Parse(Read(
            "artifacts", "interface-coverage", "execution-context-device-memory-owner-local-runtime-evidence.json"));
        JsonElement root = document.RootElement;
        JsonElement[] lines = root.GetProperty("runtimeCases").EnumerateArray().ToArray();

        Assert.Equal("project-reference-local-runtime", root.GetProperty("evidenceClassification").GetString());
        Assert.Equal(new[] { 8, 10 }, root.GetProperty("sourceTreeRuntimePassedLines").EnumerateArray().Select(item => item.GetInt32()));
        Assert.Equal("passed", Assert.Single(lines, item => item.GetProperty("tensorRtLine").GetInt32() == 8).GetProperty("runtimeResult").GetString());
        Assert.Equal("passed", Assert.Single(lines, item => item.GetProperty("tensorRtLine").GetInt32() == 10).GetProperty("runtimeResult").GetString());
        Assert.Equal("blocked-at-create-infer-runtime", Assert.Single(lines, item => item.GetProperty("tensorRtLine").GetInt32() == 11).GetProperty("runtimeResult").GetString());
        Assert.False(root.GetProperty("isAllSupportedLinesRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("usesExternalModel").GetBoolean());
        Assert.False(root.GetProperty("devicePointerRecorded").GetBoolean());
    }

    private static MethodInfo AssertMethod(string name, params Type[] parameters)
    {
        return typeof(TensorRtExecutionContext).GetMethod(name, parameters)
            ?? throw new Xunit.Sdk.XunitException($"Missing public method {name}.");
    }

    private static IEnumerable<Type> ExposedTypes(MemberInfo member)
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
    }

    private static string Read(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}
