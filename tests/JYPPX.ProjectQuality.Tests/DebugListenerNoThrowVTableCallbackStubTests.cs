using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNoThrowVTableCallbackStubTests
{
    [Fact]
    public void CallbackStubCopiesMetadataAndStaysNonProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 32, 32 },
            "quality-debug-listener-nothrow-vtable-callback-stub",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot snapshot = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult minimalSafety =
            TensorRtDebugListenerNativeAttachEntryMinimalSafety.Evaluate(snapshot);
        TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult scaffold =
            TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.Evaluate(snapshot);
        TensorRtDebugListenerNoThrowVTableCallbackStubResult stub =
            TensorRtDebugListenerNoThrowVTableCallbackStub.Evaluate(snapshot, minimalSafety, scaffold);
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(snapshot);

        Assert.Equal("debug-listener-nothrow-vtable-callback-stub", stub.EvidenceKind);
        Assert.Equal("callback-stub-gate", stub.RuntimeEvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", stub.CallbackKind);
        Assert.False(stub.RealCallbackRuntime);
        Assert.False(stub.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, stub.Line);
        Assert.True(stub.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, stub.LastStatus);
        Assert.Equal("quality_debug_tensor", stub.TensorName);
        Assert.Equal(TensorRtDataType.Float, stub.DataType);
        Assert.Equal(TensorRtTensorLocation.Device, stub.Location);
        Assert.Equal(4, stub.ShapeRank);
        Assert.Contains("32", stub.ShapeSummary);
        Assert.True(stub.IsInput);
        Assert.True(stub.IsExecutionTensor);
        Assert.True(stub.CallbackEntryCount > 0);
        Assert.Equal(stub.CallbackEntryCount, stub.CallbackLeaveCount);
        Assert.Equal(0, stub.FailureCount);
        Assert.True(stub.MinimalSafetyReady);
        Assert.True(stub.NoThrowVTableScaffoldGateReady);
        Assert.True(stub.NoThrowVTableScaffoldReady);
        Assert.True(stub.CallbackStubShapeReady);
        Assert.True(stub.CallbackStubNoThrowReady);
        Assert.True(stub.CallbackMetadataCopyReady);
        Assert.True(stub.CallbackExceptionCaptureReady);
        Assert.True(stub.CallbackStatusMappingReady);
        Assert.True(stub.CallbackInFlightEnterReady);
        Assert.True(stub.CallbackInFlightLeaveReady);
        Assert.True(stub.CallbackInFlightPairingReady);
        Assert.True(stub.CallbackInFlightNeverNegativeReady);
        Assert.True(stub.BorrowedDebugTensorMetadataCopyReady);
        Assert.True(stub.BorrowedDebugTensorPointerEscapeBlocked);
        Assert.False(stub.DebugTensorPointerExposed);
        Assert.False(stub.DebugTensorDataPointerExposed);
        Assert.False(stub.SetDebugListenerNonNullEnabled);
        Assert.True(stub.NativeAttachWouldBeBlocked);
        Assert.False(stub.NativeVTableInstalled);
        Assert.True(stub.CallbackStubGateReady);
        Assert.False(stub.ProcessDebugTensorRuntimeReady);
        Assert.False(stub.CanInstallNativeVTable);
        Assert.False(stub.CanCallProcessDebugTensorRuntime);
        Assert.False(stub.CanAttemptRuntimeProof);
        Assert.True(stub.RuntimeProofBlocked);
        Assert.True(stub.DeferredRowsStillRequired);
        Assert.Contains("callback-stub-gate", stub.ReasonCallbackRuntimeStillBlocked, StringComparison.Ordinal);
        Assert.Equal("callback-stub-gate-ready", stub.Status);
        Assert.Contains("RuntimeEvidenceKind=callback-stub-gate", stub.Diagnostic);
        Assert.Contains("CallbackStubGateReady=True", stub.Diagnostic);
        Assert.Contains("DebugTensorDataPointerExposed=False", stub.Diagnostic);
        Assert.Contains("NativeVTableInstalled=False", stub.Diagnostic);
        Assert.Contains("CanCallProcessDebugTensorRuntime=False", stub.Diagnostic);

        Assert.False(precheck.NativeAttachEntryLocated);
        Assert.False(precheck.NativeVTableReady);
        Assert.False(precheck.ProcessDebugTensorRuntimeReady);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
    }

    [Fact]
    public void PublicCallbackStubSurfaceDoesNotExposeRawPointers()
    {
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerNoThrowVTableCallbackStub));
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerNoThrowVTableCallbackStubResult));

        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNoThrowVTableCallbackStub.cs");
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public UIntPtr", source);
        Assert.DoesNotContain("public nint", source);
        Assert.Contains("DebugTensorPointerExposed", source);
        Assert.Contains("DebugTensorDataPointerExposed", source);
        Assert.Contains("NativeVTableInstalled", source);
    }

    [Fact]
    public void SourceNativeSmokeReadinessPackageAndDocsContainCallbackStubEvidenceButNotRuntimeProof()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNoThrowVTableCallbackStub.cs");
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "debug_listener_nothrow_vtable_callback_stub.inc");
        string trt8Api = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11Api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string doc = ReadSource("docs", "articles", "zh-cn", "debug-listener-nothrow-vtable-callback-stub.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        AssertCallbackStubSurfaceMarkers(source);
        AssertCallbackStubSurfaceMarkers(smoke);
        AssertCallbackStubSurfaceMarkers(readiness);
        AssertCallbackStubSurfaceMarkers(bridgeConsumer);
        AssertCallbackStubSurfaceMarkers(doc);
        AssertPackageConsumerMarkers(packageConsumer);
        AssertCallbackStubEvidenceKindMarkers(readiness);
        AssertCallbackStubEvidenceKindMarkers(doc);
        AssertCallbackStubEvidenceKindMarkers(latest);
        AssertCallbackStubEvidenceKindMarkers(smokeReadme);
        AssertCallbackStubEvidenceKindMarkers(runtimeSplitReadme);
        AssertCallbackStubOverviewMarkers(schema);
        AssertCallbackStubOverviewMarkers(trampolineGate);
        AssertCallbackStubOverviewMarkers(smokeReadme);
        AssertCallbackStubOverviewMarkers(runtimeSplitReadme);
        Assert.Contains("debug-listener-nothrow-vtable-callback-stub.md", toc);
        Assert.Contains("debug-listener-nothrow-vtable-callback-stub.md", index);
        Assert.Contains("DebugListenerNoThrowVTableCallbackStub final", nativeSource);
        Assert.Contains("begin_callback", nativeSource);
        Assert.Contains("complete_callback_success", nativeSource);
        Assert.Contains("complete_callback_failure", nativeSource);
        Assert.Contains("can_return_status_without_throwing", nativeSource);
        Assert.Contains("configure_api_line", nativeSource);
        Assert.Contains("line_supports_debug_listener", nativeSource);
        Assert.Contains("borrowed_pointer_escape_blocked", nativeSource);
        Assert.Contains("metadata_copy_ready", nativeSource);
        Assert.Contains("inflight_never_negative", nativeSource);
        Assert.Contains("debug_listener_nothrow_vtable_callback_stub.inc", trt8Api);
        Assert.Contains("debug_listener_nothrow_vtable_callback_stub.inc", trt10Api);
        Assert.Contains("debug_listener_nothrow_vtable_callback_stub.inc", trt11Api);
        Assert.Contains("DebugListenerNoThrowVTableCallbackStub=", smoke);
        Assert.Contains("hasDebugListenerNoThrowVTableCallbackStub", readiness);
        Assert.Contains("New-DebugListenerNoThrowVTableCallbackStubEvidence", readiness);
        Assert.Contains("callback-stub-gate", packageConsumer);
        Assert.Contains("callback-stub-gate", bridgeConsumer);
        Assert.Contains("not proof", doc);
        Assert.Contains("not proof", schema);
        Assert.Contains("not proof", latest);
        Assert.Contains("not proof", trampolineGate);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static void AssertCallbackStubSurfaceMarkers(string text)
    {
        Assert.Contains("debug-listener-nothrow-vtable-callback-stub", text);
        Assert.Contains("TensorRtDebugListenerNoThrowVTableCallbackStub", text);
        Assert.Contains("TensorRtDebugListenerNoThrowVTableCallbackStubResult", text);
        Assert.Contains("CallbackStubGateReady", text);
        Assert.Contains("CallbackMetadataCopyReady", text);
        Assert.Contains("DebugTensorPointerExposed", text);
        Assert.Contains("DebugTensorDataPointerExposed", text);
        Assert.Contains("SetDebugListenerNonNullEnabled", text);
        Assert.Contains("NativeAttachWouldBeBlocked", text);
        Assert.Contains("NativeVTableInstalled", text);
        Assert.Contains("CanInstallNativeVTable", text);
        Assert.Contains("CanCallProcessDebugTensorRuntime", text);
        Assert.Contains("RuntimeProofBlocked", text);
    }

    private static void AssertCallbackStubEvidenceKindMarkers(string text)
    {
        Assert.Contains("callback-stub-gate", text);
        Assert.Contains("RuntimeEvidenceKind", text);
        Assert.Contains("RealCallbackRuntime=False", text);
        Assert.Contains("IsRealCallbackRuntimeProof=False", text);
    }

    private static void AssertCallbackStubOverviewMarkers(string text)
    {
        Assert.Contains("debug-listener-nothrow-vtable-callback-stub", text);
        Assert.Contains("callback-stub-gate", text);
        Assert.Contains("not proof", text);
    }

    private static void AssertPackageConsumerMarkers(string text)
    {
        Assert.Contains("TensorRtDebugListenerNoThrowVTableCallbackStub", text);
        Assert.Contains("TensorRtDebugListenerNoThrowVTableCallbackStubResult", text);
        Assert.Contains("debug-listener-nothrow-vtable-callback-stub", text);
        Assert.Contains("callback-stub-gate", text);
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
