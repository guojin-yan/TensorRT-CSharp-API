using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeVTableInstallPreflightTests
{
    [Fact]
    public void NativeVTableInstallPreflightStaysPointerFreeBlockedAndNonProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_install_preflight_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 8, 8 },
            "quality-debug-listener-native-vtable-install-preflight",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot snapshot = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerNativeVTableInstallPreflightResult preflight =
            TensorRtDebugListenerNativeVTableInstallPreflight.Evaluate(snapshot);

        Assert.Equal("debug-listener-native-vtable-install-preflight", preflight.EvidenceKind);
        Assert.Equal("native-vtable-install-preflight", preflight.RuntimeEvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", preflight.CallbackKind);
        Assert.False(preflight.RealCallbackRuntime);
        Assert.False(preflight.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, preflight.Line);
        Assert.True(preflight.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, preflight.LastStatus);
        Assert.True(preflight.NativeOwnerLifecycleGateReady);
        Assert.True(preflight.NativeAttachBridgeShapeGateReady);
        Assert.True(preflight.NativeNoThrowVTableScaffoldGateReady);
        Assert.True(preflight.BorrowedDebugTensorMetadataGateReady);
        Assert.True(preflight.VTableInstallShapeReady);
        Assert.True(preflight.VTableInstallVersionGuardReady);
        Assert.True(preflight.VTableInstallNoThrowBoundaryReady);
        Assert.True(preflight.VTableInstallOwnershipDiagnosticsReady);
        Assert.True(preflight.VTableInstallPointerFree);
        Assert.False(preflight.AttachBridgeSetDebugListenerNonNullEnabled);
        Assert.False(preflight.SetDebugListenerNonNullEnabled);
        Assert.True(preflight.NonNullAttachStillDisabled);
        Assert.False(preflight.VTableAddressExposed);
        Assert.False(preflight.VTablePointerProduced);
        Assert.False(preflight.DebugTensorPointerExposed);
        Assert.False(preflight.DebugTensorDataPointerExposed);
        Assert.True(preflight.BorrowedDebugTensorPointerEscapeBlocked);
        Assert.True(preflight.BorrowedDebugTensorDataPointerEscapeBlocked);
        Assert.False(preflight.BorrowedDebugTensorLifetimeReady);
        Assert.False(preflight.BorrowedDebugTensorDataLifetimeReady);
        Assert.True(preflight.NativeVTableInstallPreflightReady);
        Assert.False(preflight.NativeVTableInstalled);
        Assert.False(preflight.NativeVTableInstallRuntimeReady);
        Assert.False(preflight.CanEnableSetDebugListenerNonNull);
        Assert.False(preflight.CanInstallNativeVTable);
        Assert.False(preflight.ProcessDebugTensorRuntimeReady);
        Assert.False(preflight.CanCallProcessDebugTensorRuntime);
        Assert.False(preflight.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(preflight.CanAttemptRuntimeProof);
        Assert.True(preflight.RuntimeProofBlocked);
        Assert.True(preflight.DeferredRowsStillRequired);
        Assert.Equal("native-vtable-install-preflight-ready", preflight.Status);
        Assert.Contains("native-vtable-install-preflight", preflight.ReasonNativeVTableInstallStillBlocked, StringComparison.Ordinal);
        Assert.Contains("NativeVTableInstallPreflightReady=True", preflight.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("NativeVTableInstalled=False", preflight.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("CanInstallNativeVTable=False", preflight.Diagnostic, StringComparison.Ordinal);
        Assert.Contains(preflight.BlockedPrerequisites, item => item.Contains("real-callback-runtime", StringComparison.Ordinal));
    }

    [Fact]
    public void PublicNativeVTableInstallPreflightSurfaceDoesNotExposeRawPointers()
    {
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerNativeVTableInstallPreflight));
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerNativeVTableInstallPreflightResult));

        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDebugListenerNativeVTableInstallPreflight.cs");
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public UIntPtr", source);
        Assert.DoesNotContain("public nint", source);
        Assert.Contains("NativeVTableInstalled", source);
        Assert.Contains("CanInstallNativeVTable", source);
        Assert.Contains("ReasonNativeVTableInstallStillBlocked", source);
    }

    [Fact]
    public void SourceNativeSmokeReadinessPackageAndDocsContainInstallPreflightEvidenceButNotRuntimeProof()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDebugListenerNativeVTableInstallPreflight.cs");
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "debug_listener_native_vtable_install_preflight.inc");
        string trt8Api = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11Api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string doc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-vtable-install-preflight.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        AssertInstallPreflightSurfaceMarkers(source);
        AssertInstallPreflightSurfaceMarkers(smoke);
        AssertInstallPreflightSurfaceMarkers(readiness);
        AssertInstallPreflightSurfaceMarkers(bridgeConsumer);
        AssertInstallPreflightSurfaceMarkers(packageConsumer);
        AssertInstallPreflightSurfaceMarkers(doc);
        AssertInstallPreflightEvidenceKindMarkers(readiness);
        AssertInstallPreflightEvidenceKindMarkers(doc);
        AssertInstallPreflightEvidenceKindMarkers(schema);
        AssertInstallPreflightEvidenceKindMarkers(latest);
        AssertInstallPreflightEvidenceKindMarkers(trampolineGate);
        AssertInstallPreflightEvidenceKindMarkers(smokeReadme);
        AssertInstallPreflightEvidenceKindMarkers(runtimeSplitReadme);
        Assert.Contains("debug-listener-native-vtable-install-preflight.md", toc);
        Assert.Contains("debug-listener-native-vtable-install-preflight.md", index);
        Assert.Contains("DebugListenerNativeVTableInstallPreflight final", nativeSource);
        Assert.Contains("configure_api_line", nativeSource);
        Assert.Contains("configure_preflight", nativeSource);
        Assert.Contains("preflight_shape_ready", nativeSource);
        Assert.Contains("can_enable_set_debug_listener_non_null", nativeSource);
        Assert.Contains("can_install_native_vtable", nativeSource);
        Assert.Contains("native_vtable_installed", nativeSource);
        Assert.Contains("process_debug_tensor_runtime_ready", nativeSource);
        Assert.Contains("debug_listener_native_vtable_install_preflight.inc", trt8Api);
        Assert.Contains("debug_listener_native_vtable_install_preflight.inc", trt10Api);
        Assert.Contains("debug_listener_native_vtable_install_preflight.inc", trt11Api);
        Assert.Contains("DebugListenerNativeVTableInstallPreflight=", smoke);
        Assert.Contains("hasDebugListenerNativeVTableInstallPreflight", readiness);
        Assert.Contains("New-DebugListenerNativeVTableInstallPreflightEvidence", readiness);
        Assert.Contains("debugListenerNativeVTableInstallPreflight", readiness);
        Assert.Contains("native-vtable-install-preflight", packageConsumer);
        Assert.Contains("native-vtable-install-preflight", bridgeConsumer);
        Assert.Contains("not proof", doc);
        Assert.Contains("not proof", schema);
        Assert.Contains("not proof", latest);
        Assert.Contains("not proof", trampolineGate);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static void AssertInstallPreflightSurfaceMarkers(string text)
    {
        Assert.Contains("debug-listener-native-vtable-install-preflight", text);
        Assert.Contains("TensorRtDebugListenerNativeVTableInstallPreflight", text);
        Assert.Contains("TensorRtDebugListenerNativeVTableInstallPreflightResult", text);
        Assert.Contains("NativeVTableInstallPreflightReady", text);
        Assert.Contains("VTableInstallShapeReady", text);
        Assert.Contains("VTableInstallVersionGuardReady", text);
        Assert.Contains("VTableInstallNoThrowBoundaryReady", text);
        Assert.Contains("VTableInstallOwnershipDiagnosticsReady", text);
        Assert.Contains("VTableInstallPointerFree", text);
        Assert.Contains("SetDebugListenerNonNullEnabled", text);
        Assert.Contains("NativeVTableInstalled", text);
        Assert.Contains("NativeVTableInstallRuntimeReady", text);
        Assert.Contains("CanEnableSetDebugListenerNonNull", text);
        Assert.Contains("CanInstallNativeVTable", text);
        Assert.Contains("CanCallProcessDebugTensorRuntime", text);
        Assert.Contains("ReasonNativeVTableInstallStillBlocked", text);
        Assert.Contains("RuntimeProofBlocked", text);
    }

    private static void AssertInstallPreflightEvidenceKindMarkers(string text)
    {
        Assert.Contains("native-vtable-install-preflight", text);
        Assert.Contains("RuntimeEvidenceKind", text);
        Assert.Contains("RealCallbackRuntime=False", text);
        Assert.Contains("IsRealCallbackRuntimeProof=False", text);
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
