using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeAttachEntryMinimalSafetyTests
{
    [Fact]
    public void NativeAttachEntryMinimalSafetyIsScopedSourceVisibleEvidenceOnly()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 16, 16 },
            "quality-debug-listener-native-attach-entry-minimal-safety",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot snapshot = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult scaffold =
            TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.Evaluate(snapshot);
        TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult safety =
            TensorRtDebugListenerNativeAttachEntryMinimalSafety.Evaluate(snapshot, scaffold);
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(snapshot);
        TensorRtDebugListenerRuntimeProofAttemptPreflightResult preflight =
            TensorRtDebugListenerRuntimeProofAttemptPreflight.Evaluate(precheck);

        Assert.Equal("debug-listener-native-attach-entry-minimal-safety", safety.EvidenceKind);
        Assert.Equal("minimal-safety", safety.RuntimeEvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", safety.CallbackKind);
        Assert.False(safety.RealCallbackRuntime);
        Assert.False(safety.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, safety.Line);
        Assert.True(safety.RuntimeScaffoldReady);
        Assert.True(safety.LifecycleGateReady);
        Assert.True(safety.LifecyclePointerFree);
        Assert.True(safety.NativeAttachEntryLocated);
        Assert.True(safety.NativeDetachEntryLocated);
        Assert.True(safety.AttachEntryParameterShapeReady);
        Assert.True(safety.AttachEntryNoThrowReady);
        Assert.True(safety.AttachEntryVersionGuardReady);
        Assert.True(safety.AttachEntryOwnershipDiagnosticsReady);
        Assert.False(safety.SetDebugListenerNonNullEnabled);
        Assert.True(safety.NonNullAttachStillDisabled);
        Assert.True(safety.NativeAttachWouldBeBlocked);
        Assert.True(safety.MinimalSafetyReady);
        Assert.False(safety.ProcessDebugTensorRuntimeReady);
        Assert.False(safety.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(safety.CanImplementNativeAttach);
        Assert.False(safety.CanAttemptRuntimeProof);
        Assert.True(safety.RuntimeProofBlocked);
        Assert.True(safety.DeferredRowsStillRequired);
        Assert.Equal("minimal-safety-ready", safety.Status);
        Assert.Contains("setDebugListener(non-null)", safety.ReasonNativeAttachStillBlocked, StringComparison.Ordinal);
        Assert.Contains("RuntimeEvidenceKind=minimal-safety", safety.Diagnostic);
        Assert.Contains("NativeAttachEntryLocated=True", safety.Diagnostic);
        Assert.Contains("SetDebugListenerNonNullEnabled=False", safety.Diagnostic);
        Assert.Contains("NativeAttachWouldBeBlocked=True", safety.Diagnostic);

        Assert.False(precheck.NativeAttachEntryLocated);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.False(preflight.CanEnableSetDebugListenerNonNull);
        Assert.False(preflight.CanPromoteRealCallbackRuntime);
        Assert.True(preflight.RuntimeProofBlocked);
    }

    [Fact]
    public void PublicMinimalSafetySurfaceDoesNotExposeRawPointers()
    {
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerNativeAttachEntryMinimalSafety));
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult));

        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDebugListenerNativeAttachEntryMinimalSafety.cs");
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public UIntPtr", source);
        Assert.DoesNotContain("public nint", source);
    }

    [Fact]
    public void NativeSmokeReadinessDocsPackageAndDeferredRowsContainMinimalSafetyEvidence()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDebugListenerNativeAttachEntryMinimalSafety.cs");
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "debug_listener_native_attach_entry_minimal_safety.inc");
        string trt8Api = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11Api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string doc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-attach-entry-minimal-safety.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        AssertMinimalSafetySurfaceMarkers(source);
        AssertMinimalSafetySurfaceMarkers(smoke);
        AssertMinimalSafetySurfaceMarkers(readiness);
        AssertMinimalSafetySurfaceMarkers(bridgeConsumer);
        AssertMinimalSafetySurfaceMarkers(packageConsumer);
        AssertMinimalSafetyEvidenceText(doc);
        AssertMinimalSafetyEvidenceText(schema);
        AssertMinimalSafetyEvidenceText(latest);
        AssertMinimalSafetyEvidenceText(trampolineGate);
        AssertMinimalSafetyEvidenceText(smokeReadme);
        AssertMinimalSafetyEvidenceText(runtimeSplitReadme);
        Assert.Contains("debug-listener-native-attach-entry-minimal-safety.md", toc);
        Assert.Contains("debug-listener-native-attach-entry-minimal-safety.md", index);
        Assert.Contains("DebugListenerNativeAttachEntryMinimalSafety final", nativeSource);
        Assert.Contains("can_call_set_debug_listener_non_null", nativeSource);
        Assert.Contains("set_debug_listener_non_null_enabled = false", nativeSource);
        Assert.Contains("debug_listener_native_attach_entry_minimal_safety.inc", trt8Api);
        Assert.Contains("debug_listener_native_attach_entry_minimal_safety.inc", trt10Api);
        Assert.Contains("debug_listener_native_attach_entry_minimal_safety.inc", trt11Api);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static void AssertMinimalSafetySurfaceMarkers(string text)
    {
        Assert.Contains("debug-listener-native-attach-entry-minimal-safety", text);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryMinimalSafety", text);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult", text);
        Assert.Contains("MinimalSafetyReady", text);
        Assert.Contains("NativeAttachEntryLocated", text);
        Assert.Contains("SetDebugListenerNonNullEnabled", text);
        Assert.Contains("NativeAttachWouldBeBlocked", text);
        Assert.Contains("ReasonNativeAttachStillBlocked", text);
    }

    private static void AssertMinimalSafetyEvidenceText(string text)
    {
        Assert.Contains("debug-listener-native-attach-entry-minimal-safety", text);
        Assert.Contains("RuntimeEvidenceKind=minimal-safety", text);
        Assert.Contains("MinimalSafetyReady", text);
        Assert.Contains("NativeAttachEntryLocated=True", text);
        Assert.Contains("SetDebugListenerNonNullEnabled=False", text);
        Assert.Contains("NativeAttachWouldBeBlocked=True", text);
    }

    private static void AssertNoRawPointerTypes(Type type)
    {
        foreach (ConstructorInfo constructor in type.GetConstructors(BindingFlags.Instance | BindingFlags.Public))
        {
            foreach (ParameterInfo parameter in constructor.GetParameters())
            {
                AssertNoRawPointerType(parameter.ParameterType);
            }
        }

        foreach (PropertyInfo property in type.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.Static))
        {
            AssertNoRawPointerType(property.PropertyType);
        }

        foreach (MethodInfo method in type.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.DeclaredOnly))
        {
            AssertNoRawPointerType(method.ReturnType);
            foreach (ParameterInfo parameter in method.GetParameters())
            {
                AssertNoRawPointerType(parameter.ParameterType);
            }
        }
    }

    private static void AssertNoRawPointerType(Type type)
    {
        Assert.NotEqual(typeof(IntPtr), type);
        Assert.NotEqual(typeof(UIntPtr), type);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
