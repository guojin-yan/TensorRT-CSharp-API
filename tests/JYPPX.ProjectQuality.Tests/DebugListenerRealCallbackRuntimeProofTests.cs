using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerRealCallbackRuntimeProofTests
{
    [Fact]
    public void RealCallbackRuntimeProofGateDefaultsToSkippedPointerFreeAndNonProof()
    {
        TensorRtDebugListenerRealCallbackRuntimeProofResult proof = CreateProof(optInEnabled: false, fullPackageConsumerReport: false);

        Assert.Equal("debug-listener-real-callback-runtime-proof", proof.EvidenceKind);
        Assert.Equal("runtime-smoke-skipped", proof.RuntimeEvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", proof.CallbackKind);
        Assert.Equal(TensorRtApiLine.TensorRt11, proof.Line);
        Assert.Equal(11, proof.TensorRtLine);
        Assert.False(proof.OptInEnabled);
        Assert.False(proof.FullPackageConsumerReport);
        Assert.False(proof.RealCallbackRuntime);
        Assert.False(proof.IsRealCallbackRuntimeProof);
        Assert.False(proof.RuntimeSmokeReady);
        Assert.True(proof.TrampolineShapeReady);
        Assert.False(proof.AttachAttempted);
        Assert.False(proof.AttachSucceeded);
        Assert.False(proof.DetachAttempted);
        Assert.True(proof.DetachSucceeded);
        Assert.False(proof.RollbackAttempted);
        Assert.True(proof.RollbackSucceeded);
        Assert.False(proof.NativeVTableInstalled);
        Assert.False(proof.ProcessDebugTensorInvoked);
        Assert.Equal(0, proof.InvocationCount);
        Assert.Equal(0, proof.FailureCount);
        Assert.Equal(0, proof.InFlightCallbackCount);
        Assert.True(proof.BorrowedDebugTensorMetadataCopied);
        Assert.True(proof.PointerFreeSurfaceReady);
        Assert.False(proof.ProcessDebugTensorRuntimeReady);
        Assert.False(proof.AttemptedNoInvocation);
        Assert.Equal(BridgeStatusCode.NotReady, proof.LastStatus);
        Assert.False(proof.CanAttemptRuntimeProof);
        Assert.False(proof.CanPromoteRealCallbackRuntime);
        Assert.True(proof.RuntimeProofBlocked);
        Assert.True(proof.DeferredRowsStillRequired);
        Assert.Equal("skipped", proof.Status);
        Assert.Contains("opt-in is disabled", proof.LastDiagnostic, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("InvocationCount=0", proof.Diagnostic, StringComparison.Ordinal);
        Assert.Contains(proof.BlockedPrerequisites, item => item.Contains("invocation count", StringComparison.Ordinal));
    }

    [Fact]
    public void RealCallbackRuntimeProofGateOptInStaysBlockedWithoutInvocation()
    {
        TensorRtDebugListenerRealCallbackRuntimeProofResult proof = CreateProof(optInEnabled: true, fullPackageConsumerReport: true);

        Assert.Equal("debug-listener-real-callback-runtime-proof", proof.EvidenceKind);
        Assert.Equal("real-callback-runtime-blocked", proof.RuntimeEvidenceKind);
        Assert.True(proof.OptInEnabled);
        Assert.True(proof.FullPackageConsumerReport);
        Assert.False(proof.RealCallbackRuntime);
        Assert.False(proof.IsRealCallbackRuntimeProof);
        Assert.False(proof.AttachSucceeded);
        Assert.False(proof.NativeVTableInstalled);
        Assert.False(proof.ProcessDebugTensorInvoked);
        Assert.Equal(0, proof.InvocationCount);
        Assert.False(proof.CanPromoteRealCallbackRuntime);
        Assert.True(proof.RuntimeProofBlocked);
        Assert.Equal("blocked", proof.Status);
        Assert.Contains("real callback runtime proof is blocked", proof.LastDiagnostic, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void PublicRealCallbackRuntimeProofSurfaceDoesNotExposeRawPointers()
    {
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerRealCallbackRuntimeProof));
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerRealCallbackRuntimeProofResult));

        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRealCallbackRuntimeProof.cs");
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public UIntPtr", source);
        Assert.DoesNotContain("public nint", source);
        Assert.Contains("PointerFreeSurfaceReady", source);
        Assert.Contains("invocationCount > 0", source);
        Assert.Contains("AttemptedNoInvocation", source);
        Assert.Contains("real-callback-runtime-blocked", source);
        Assert.Contains("attempted-no-invocation", source);
    }

    [Fact]
    public void SourceNativeSmokeReadinessPackageAndDocsContainProofGateEvidenceButNotRuntimeProof()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRealCallbackRuntimeProof.cs");
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "debug_listener_real_callback_runtime_proof.inc");
        string trt8Api = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11Api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string doc = ReadSource("docs", "articles", "zh-cn", "debug-listener-real-callback-runtime-proof.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        AssertProofGateSurfaceMarkers(source);
        AssertProofGateSurfaceMarkers(smoke);
        AssertProofGateSurfaceMarkers(readiness);
        AssertProofGateSurfaceMarkers(bridgeConsumer);
        AssertProofGateSurfaceMarkers(packageConsumer);
        AssertProofGateSurfaceMarkers(doc);
        Assert.Contains("DebugListenerRealCallbackRuntimeProofGate final", nativeSource);
        Assert.Contains("DebugListenerRealCallbackRuntimeProofReport", nativeSource);
        Assert.Contains("configure_api_line", nativeSource);
        Assert.Contains("configure_prerequisites", nativeSource);
        Assert.Contains("configure_attempt", nativeSource);
        Assert.Contains("configure_invocation", nativeSource);
        Assert.Contains("can_attempt_runtime_proof", nativeSource);
        Assert.Contains("can_promote_real_callback_runtime", nativeSource);
        Assert.Contains("debug_listener_real_callback_runtime_proof.inc", trt8Api);
        Assert.Contains("debug_listener_real_callback_runtime_proof.inc", trt10Api);
        Assert.Contains("debug_listener_real_callback_runtime_proof.inc", trt11Api);
        Assert.Contains("DebugListenerRealCallbackRuntimeProof=", smoke);
        Assert.Contains("New-DebugListenerRealCallbackRuntimeProofEvidence", readiness);
        Assert.Contains("debugListenerRealCallbackRuntimeProof", readiness);
        Assert.Contains("hasDebugListenerRealCallbackRuntimeProof", readiness);
        Assert.Contains("debug-listener-real-callback-runtime-proof.md", toc);
        Assert.Contains("debug-listener-real-callback-runtime-proof.md", index);
        Assert.Contains("debug-listener-real-callback-runtime-proof", schema);
        Assert.Contains("debug-listener-real-callback-runtime-proof", latest);
        Assert.Contains("debug-listener-real-callback-runtime-proof", trampolineGate);
        Assert.Contains("debug-listener-real-callback-runtime-proof", smokeReadme);
        Assert.Contains("debug-listener-real-callback-runtime-proof", runtimeSplitReadme);
        Assert.Contains("InvocationCount>0", packageConsumer);
        Assert.Contains("InvocationCount>0", doc);
        Assert.Contains("not proof", doc);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static TensorRtDebugListenerRealCallbackRuntimeProofResult CreateProof(bool optInEnabled, bool fullPackageConsumerReport)
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_real_callback_runtime_proof_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 8, 8 },
            "quality-debug-listener-real-callback-runtime-proof",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        return TensorRtDebugListenerRealCallbackRuntimeProof.Evaluate(
            owner.GetSnapshot("post-dispose"),
            "quality-runtime-key",
            optInEnabled,
            fullPackageConsumerReport);
    }

    private static void AssertProofGateSurfaceMarkers(string text)
    {
        Assert.Contains("debug-listener-real-callback-runtime-proof", text);
        Assert.Contains("TensorRtDebugListenerRealCallbackRuntimeProof", text);
        Assert.Contains("TensorRtDebugListenerRealCallbackRuntimeProofResult", text);
        Assert.Contains("RuntimeEvidenceKind", text);
        Assert.Contains("real-callback-runtime-blocked", text);
        Assert.Contains("attempted-no-invocation", text);
        Assert.Contains("RuntimeSmokeReady", text);
        Assert.Contains("TrampolineShapeReady", text);
        Assert.Contains("AttachAttempted", text);
        Assert.Contains("AttachSucceeded", text);
        Assert.Contains("DetachAttempted", text);
        Assert.Contains("DetachSucceeded", text);
        Assert.Contains("RollbackAttempted", text);
        Assert.Contains("RollbackSucceeded", text);
        Assert.Contains("NativeVTableInstalled", text);
        Assert.Contains("ProcessDebugTensorInvoked", text);
        Assert.Contains("InvocationCount", text);
        Assert.Contains("FailureCount", text);
        Assert.Contains("InFlightCallbackCount", text);
        Assert.Contains("BorrowedDebugTensorMetadataCopied", text);
        Assert.Contains("PointerFreeSurfaceReady", text);
        Assert.Contains("ProcessDebugTensorRuntimeReady", text);
        Assert.Contains("AttemptedNoInvocation", text);
        Assert.Contains("CanPromoteRealCallbackRuntime", text);
        Assert.Contains("RuntimeProofBlocked", text);
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
