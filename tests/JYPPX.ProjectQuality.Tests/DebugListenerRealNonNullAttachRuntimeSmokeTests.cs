using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerRealNonNullAttachRuntimeSmokeTests
{
    [Fact]
    public void RuntimeSmokeDefaultsToSkippedPointerFreeAndNonProof()
    {
        TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult smoke = CreateRuntimeSmoke(optInEnabled: false, fullPackageConsumerReport: false);

        Assert.Equal("debug-listener-real-non-null-attach-runtime-smoke", smoke.EvidenceKind);
        Assert.Equal("runtime-smoke-skipped", smoke.RuntimeEvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", smoke.CallbackKind);
        Assert.Equal(TensorRtApiLine.TensorRt11, smoke.Line);
        Assert.Equal(11, smoke.TensorRtLine);
        Assert.False(smoke.OptInEnabled);
        Assert.False(smoke.FullPackageConsumerReport);
        Assert.False(smoke.RealCallbackRuntime);
        Assert.False(smoke.IsRealCallbackRuntimeProof);
        Assert.False(smoke.AttachGuardReady);
        Assert.False(smoke.NativeVTableReady);
        Assert.False(smoke.CallbackInvocationReady);
        Assert.False(smoke.AttachAttempted);
        Assert.False(smoke.AttachSucceeded);
        Assert.False(smoke.DetachAttempted);
        Assert.False(smoke.DetachSucceeded);
        Assert.False(smoke.RollbackAttempted);
        Assert.False(smoke.RollbackSucceeded);
        Assert.False(smoke.NativeVTableInstalled);
        Assert.False(smoke.ProcessDebugTensorInvoked);
        Assert.Equal(0, smoke.InvocationCount);
        Assert.Equal(0, smoke.AllocationCount);
        Assert.Equal(0, smoke.ReleaseCount);
        Assert.Equal(0, smoke.FailureCount);
        Assert.Equal(0, smoke.InFlightCallbackCount);
        Assert.Equal(BridgeStatusCode.NotReady, smoke.LastStatus);
        Assert.True(smoke.ReportPointerFree);
        Assert.False(smoke.CanAttemptRuntimeProof);
        Assert.False(smoke.CanPromoteRealCallbackRuntime);
        Assert.True(smoke.RuntimeProofBlocked);
        Assert.True(smoke.DeferredRowsStillRequired);
        Assert.Equal("skipped", smoke.Status);
        Assert.Contains("opt-in is disabled", smoke.LastDiagnostic, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("RuntimeEvidenceKind=runtime-smoke-skipped", smoke.Diagnostic, StringComparison.Ordinal);
        Assert.Contains(smoke.BlockedPrerequisites, item => item.Contains("runtime smoke was not explicitly enabled", StringComparison.Ordinal));
    }

    [Fact]
    public void RuntimeSmokeOptInStaysBlockedAndNonProofWithoutAttachPrerequisites()
    {
        TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult smoke = CreateRuntimeSmoke(optInEnabled: true, fullPackageConsumerReport: true);

        Assert.Equal("debug-listener-real-non-null-attach-runtime-smoke", smoke.EvidenceKind);
        Assert.Equal("runtime-smoke-blocked", smoke.RuntimeEvidenceKind);
        Assert.True(smoke.OptInEnabled);
        Assert.True(smoke.FullPackageConsumerReport);
        Assert.False(smoke.RealCallbackRuntime);
        Assert.False(smoke.IsRealCallbackRuntimeProof);
        Assert.False(smoke.AttachGuardReady);
        Assert.False(smoke.NativeVTableReady);
        Assert.False(smoke.BorrowedDebugTensorRuntimeReady);
        Assert.False(smoke.CallbackInvocationReady);
        Assert.False(smoke.AttachAttempted);
        Assert.False(smoke.AttachSucceeded);
        Assert.False(smoke.DetachAttempted);
        Assert.False(smoke.DetachSucceeded);
        Assert.True(smoke.RollbackAttempted);
        Assert.True(smoke.RollbackSucceeded);
        Assert.False(smoke.NativeVTableInstalled);
        Assert.False(smoke.ProcessDebugTensorInvoked);
        Assert.Equal(0, smoke.InvocationCount);
        Assert.Equal(0, smoke.ReleaseCount);
        Assert.Equal(1, smoke.FailureCount);
        Assert.Equal(BridgeStatusCode.NotImplemented, smoke.LastStatus);
        Assert.True(smoke.ReportPointerFree);
        Assert.False(smoke.CanAttemptRuntimeProof);
        Assert.False(smoke.CanPromoteRealCallbackRuntime);
        Assert.True(smoke.RuntimeProofBlocked);
        Assert.Equal("blocked", smoke.Status);
        Assert.Contains("blocked before attach", smoke.LastDiagnostic, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("AttachGuardReady=False", smoke.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("NativeVTableReady=False", smoke.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("CallbackInvocationReady=False", smoke.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("CanPromoteRealCallbackRuntime=False", smoke.Diagnostic, StringComparison.Ordinal);
    }

    [Fact]
    public void PublicRuntimeSmokeSurfaceDoesNotExposeRawPointers()
    {
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerRealNonNullAttachRuntimeSmoke));
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult));

        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs");
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public UIntPtr", source);
        Assert.DoesNotContain("public nint", source);
        Assert.Contains("ReportPointerFree", source);
        Assert.Contains("ProcessDebugTensorInvoked", source);
        Assert.Contains("CanPromoteRealCallbackRuntime", source);
        Assert.Contains("runtime-smoke-skipped", source);
        Assert.Contains("runtime-smoke-blocked", source);
        Assert.Contains("runtime-smoke-attempted", source);
    }

    [Fact]
    public void SourceNativeSmokeReadinessPackageAndDocsContainRuntimeSmokeEvidenceButNotRuntimeProof()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs");
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "debug_listener_real_non_null_attach_runtime_smoke.inc");
        string trt8Api = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11Api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string doc = ReadSource("docs", "articles", "zh-cn", "debug-listener-real-non-null-attach-runtime-smoke.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        AssertRuntimeSmokeSurfaceMarkers(source);
        AssertRuntimeSmokeSurfaceMarkers(smoke);
        AssertRuntimeSmokeSurfaceMarkers(readiness);
        AssertRuntimeSmokeSurfaceMarkers(bridgeConsumer);
        AssertRuntimeSmokeSurfaceMarkers(packageConsumer);
        AssertRuntimeSmokeSurfaceMarkers(doc);
        AssertRuntimeSmokeNonProofMarkers(readiness);
        AssertRuntimeSmokeNonProofMarkers(packageConsumer);
        AssertRuntimeSmokeNonProofMarkers(schema);
        AssertRuntimeSmokeNonProofMarkers(latest);
        AssertRuntimeSmokeNonProofMarkers(trampolineGate);
        AssertRuntimeSmokeNonProofMarkers(smokeReadme);
        AssertRuntimeSmokeNonProofMarkers(runtimeSplitReadme);
        Assert.Contains("debug-listener-real-non-null-attach-runtime-smoke.md", toc);
        Assert.Contains("debug-listener-real-non-null-attach-runtime-smoke.md", index);
        Assert.Contains("DebugListenerRealNonNullAttachRuntimeSmokeAttempt final", nativeSource);
        Assert.Contains("configure_api_line", nativeSource);
        Assert.Contains("configure_opt_in", nativeSource);
        Assert.Contains("configure_prerequisites", nativeSource);
        Assert.Contains("can_attempt_attach", nativeSource);
        Assert.Contains("attach_succeeded", nativeSource);
        Assert.Contains("native_vtable_installed", nativeSource);
        Assert.Contains("process_debug_tensor_invoked", nativeSource);
        Assert.Contains("can_promote_real_callback_runtime", nativeSource);
        Assert.Contains("debug_listener_real_non_null_attach_runtime_smoke.inc", trt8Api);
        Assert.Contains("debug_listener_real_non_null_attach_runtime_smoke.inc", trt10Api);
        Assert.Contains("debug_listener_real_non_null_attach_runtime_smoke.inc", trt11Api);
        Assert.Contains("DebugListenerRealNonNullAttachRuntimeSmoke=", smoke);
        Assert.Contains("New-DebugListenerRealNonNullAttachRuntimeSmokeEvidence", readiness);
        Assert.Contains("debugListenerRealNonNullAttachRuntimeSmoke", readiness);
        Assert.Contains("hasDebugListenerRealNonNullAttachRuntimeSmoke", readiness);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    [Fact]
    public void RuntimeSmokeNonProofMarkersAreNotPromotedByPackageOrReadinessParsers()
    {
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");

        Assert.Contains("\"runtime-smoke-skipped\"", packageConsumer);
        Assert.Contains("\"runtime-smoke-blocked\"", packageConsumer);
        Assert.Contains("\"runtime-smoke-attempted\"", packageConsumer);
        Assert.Contains("\"runtime-smoke-failed\"", packageConsumer);
        Assert.Contains("DebugListenerRealNonNullAttachRuntimeSmoke=", packageConsumer);
        Assert.Contains("IsRealCallbackRuntimeProof = $false", packageConsumer);
        Assert.Contains("RuntimeEvidenceKind=runtime-smoke-skipped", readiness);
        Assert.Contains("RuntimeEvidenceKind=runtime-smoke-blocked", readiness);
        Assert.Contains("RuntimeEvidenceKind=runtime-smoke-attempted", readiness);
        Assert.Contains("RuntimeEvidenceKind=runtime-smoke-failed", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);
        Assert.Contains("runtime-smoke-skipped", runtimeSplitReadme);
        Assert.Contains("remain insufficient on their own", runtimeSplitReadme);
    }

    private static TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult CreateRuntimeSmoke(bool optInEnabled, bool fullPackageConsumerReport)
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_real_non_null_attach_runtime_smoke_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 2, 8, 8 },
            "quality-debug-listener-real-non-null-attach-runtime-smoke",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();

        TensorRtDebugListenerRuntimeProofAttemptPreflightResult preflight =
            TensorRtDebugListenerRuntimeProofAttemptPreflight.Evaluate(owner.GetSnapshot("post-dispose"));
        return TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.Evaluate(
            preflight,
            "quality-runtime-key",
            optInEnabled,
            fullPackageConsumerReport);
    }

    private static void AssertRuntimeSmokeSurfaceMarkers(string text)
    {
        Assert.Contains("debug-listener-real-non-null-attach-runtime-smoke", text);
        Assert.Contains("TensorRtDebugListenerRealNonNullAttachRuntimeSmoke", text);
        Assert.Contains("TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult", text);
        Assert.Contains("RuntimeEvidenceKind", text);
        Assert.Contains("OptInEnabled", text);
        Assert.Contains("FullPackageConsumerReport", text);
        Assert.Contains("AttachGuardReady", text);
        Assert.Contains("NativeVTableReady", text);
        Assert.Contains("BorrowedDebugTensorRuntimeReady", text);
        Assert.Contains("CallbackInvocationReady", text);
        Assert.Contains("AttachAttempted", text);
        Assert.Contains("AttachSucceeded", text);
        Assert.Contains("DetachAttempted", text);
        Assert.Contains("DetachSucceeded", text);
        Assert.Contains("RollbackAttempted", text);
        Assert.Contains("RollbackSucceeded", text);
        Assert.Contains("NativeVTableInstalled", text);
        Assert.Contains("ProcessDebugTensorInvoked", text);
        Assert.Contains("InvocationCount", text);
        Assert.Contains("AllocationCount", text);
        Assert.Contains("ReleaseCount", text);
        Assert.Contains("FailureCount", text);
        Assert.Contains("InFlightCallbackCount", text);
        Assert.Contains("LastStatus", text);
        Assert.Contains("LastDiagnostic", text);
        Assert.Contains("ReportPointerFree", text);
        Assert.Contains("CanAttemptRuntimeProof", text);
        Assert.Contains("CanPromoteRealCallbackRuntime", text);
        Assert.Contains("ReasonRuntimeProofStillBlocked", text);
        Assert.Contains("RuntimeProofBlocked", text);
    }

    private static void AssertRuntimeSmokeNonProofMarkers(string text)
    {
        Assert.Contains("runtime-smoke-skipped", text);
        Assert.Contains("runtime-smoke-blocked", text);
        Assert.Contains("runtime-smoke-attempted", text);
        Assert.Contains("RealCallbackRuntime=False", text);
        Assert.Contains("IsRealCallbackRuntimeProof=False", text);
        Assert.Contains("not proof", text);
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
        return RepositorySourceReader.Read(path);
    }
}
