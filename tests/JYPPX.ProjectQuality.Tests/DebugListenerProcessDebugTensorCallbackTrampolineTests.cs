using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerProcessDebugTensorCallbackTrampolineTests
{
    [Fact]
    public void CallbackTrampolineShapeStaysPointerFreeAndNonProof()
    {
        TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult trampoline = CreateTrampoline();

        Assert.Equal("debug-listener-process-debug-tensor-callback-trampoline", trampoline.EvidenceKind);
        Assert.Equal("callback-trampoline-shape", trampoline.RuntimeEvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", trampoline.CallbackKind);
        Assert.False(trampoline.RealCallbackRuntime);
        Assert.False(trampoline.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, trampoline.Line);
        Assert.Equal(11, trampoline.TensorRtLine);
        Assert.True(trampoline.TrampolineShapeReady);
        Assert.True(trampoline.NativeCallbackEntryLocated);
        Assert.True(trampoline.NoThrowCallbackEntryReady);
        Assert.True(trampoline.ExceptionCaptureReady);
        Assert.True(trampoline.CallbackStatusMappingReady);
        Assert.True(trampoline.InFlightAccountingReady);
        Assert.True(trampoline.DetachBeforeReleaseReady);
        Assert.True(trampoline.BorrowedDebugTensorMetadataCopyReady);
        Assert.False(trampoline.BorrowedDebugTensorPointerExposed);
        Assert.False(trampoline.BorrowedDebugTensorDataPointerExposed);
        Assert.True(trampoline.PointerFreeSurfaceReady);
        Assert.False(trampoline.ProcessDebugTensorRuntimeReady);
        Assert.False(trampoline.OptInEnabled);
        Assert.False(trampoline.FullPackageConsumerReport);
        Assert.False(trampoline.AttachAttempted);
        Assert.False(trampoline.AttachSucceeded);
        Assert.False(trampoline.NativeVTableInstalled);
        Assert.False(trampoline.ProcessDebugTensorInvoked);
        Assert.Equal(0, trampoline.InvocationCount);
        Assert.True(trampoline.CallbackStubEntryCount > 0);
        Assert.Equal(trampoline.CallbackStubEntryCount, trampoline.CallbackStubLeaveCount);
        Assert.Equal(0, trampoline.FailureCount);
        Assert.Equal(0, trampoline.InFlightCallbackCount);
        Assert.Equal(BridgeStatusCode.NotReady, trampoline.LastStatus);
        Assert.False(trampoline.CanAttemptRuntimeProof);
        Assert.False(trampoline.CanPromoteRealCallbackRuntime);
        Assert.True(trampoline.RuntimeProofBlocked);
        Assert.True(trampoline.DeferredRowsStillRequired);
        Assert.Equal("quality_process_debug_tensor_trampoline", trampoline.Metadata.TensorName);
        Assert.True(trampoline.Metadata.MetadataCopied);
        Assert.Contains("callback-trampoline-shape", trampoline.LastDiagnostic, StringComparison.Ordinal);
        Assert.Contains("ProcessDebugTensorRuntimeReady=False", trampoline.Diagnostic, StringComparison.Ordinal);
        Assert.Contains(trampoline.BlockedPrerequisites, item => item.Contains("callback-trampoline-shape", StringComparison.Ordinal));
    }

    [Fact]
    public void PublicCallbackTrampolineSurfaceDoesNotExposeRawPointers()
    {
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerProcessDebugTensorCallbackTrampoline));
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult));
        AssertNoRawPointerTypes(typeof(TensorRtDebugTensorMetadataSnapshot));

        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.cs");
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public UIntPtr", source);
        Assert.DoesNotContain("public nint", source);
        Assert.Contains("BorrowedDebugTensorPointerExposed", source);
        Assert.Contains("BorrowedDebugTensorDataPointerExposed", source);
        Assert.Contains("PointerFreeSurfaceReady", source);
        Assert.Contains("TensorRtDebugTensorMetadataSnapshot", source);
    }

    [Fact]
    public void SourceNativeSmokeReadinessPackageAndDocsContainTrampolineEvidenceButNotRuntimeProof()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.cs");
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "debug_listener_process_debug_tensor_callback_trampoline.inc");
        string trt8Api = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11Api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string doc = ReadSource("docs", "articles", "zh-cn", "debug-listener-process-debug-tensor-callback-trampoline.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        AssertTrampolineSurfaceMarkers(source);
        AssertTrampolineSmokeMarkers(smoke);
        AssertTrampolineSurfaceMarkers(readiness);
        AssertTrampolineSurfaceMarkers(bridgeConsumer);
        AssertTrampolineSurfaceMarkers(packageConsumer);
        AssertTrampolineSurfaceMarkers(doc);
        Assert.Contains("DebugListenerProcessDebugTensorCallbackTrampoline final", nativeSource);
        Assert.Contains("DebugListenerProcessDebugTensorCallbackReport", nativeSource);
        Assert.Contains("DebugListenerBorrowedDebugTensorMetadataCopy", nativeSource);
        Assert.Contains("DebugListenerCallbackInFlightScope", nativeSource);
        Assert.Contains("begin_callback", nativeSource);
        Assert.Contains("complete_callback_success", nativeSource);
        Assert.Contains("complete_callback_failure", nativeSource);
        Assert.Contains("can_return_status_without_throwing", nativeSource);
        Assert.Contains("trampoline_shape_ready", nativeSource);
        Assert.Contains("make_report", nativeSource);
        Assert.Contains("debug_listener_process_debug_tensor_callback_trampoline.inc", trt8Api);
        Assert.Contains("debug_listener_process_debug_tensor_callback_trampoline.inc", trt10Api);
        Assert.Contains("debug_listener_process_debug_tensor_callback_trampoline.inc", trt11Api);
        Assert.Contains("DebugListenerProcessDebugTensorCallbackTrampoline=", smoke);
        Assert.Contains("New-DebugListenerProcessDebugTensorCallbackTrampolineEvidence", readiness);
        Assert.Contains("debugListenerProcessDebugTensorCallbackTrampoline", readiness);
        Assert.Contains("hasDebugListenerProcessDebugTensorCallbackTrampoline", readiness);
        Assert.Contains("callback-trampoline-shape", schema);
        Assert.Contains("callback-trampoline-shape", latest);
        Assert.Contains("callback-trampoline-shape", trampolineGate);
        Assert.Contains("callback-trampoline-shape", smokeReadme);
        Assert.Contains("callback-trampoline-shape", runtimeSplitReadme);
        Assert.Contains("debug-listener-process-debug-tensor-callback-trampoline.md", toc);
        Assert.Contains("debug-listener-process-debug-tensor-callback-trampoline.md", index);
        Assert.Contains("not proof", doc);
        Assert.Contains("not proof", schema);
        Assert.Contains("not proof", latest);
        Assert.Contains("not proof", trampolineGate);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    [Fact]
    public void CallbackTrampolineShapeIsListedAsNonProofForPackageAndReadinessParsers()
    {
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");

        Assert.Contains("\"callback-trampoline-shape\"", packageConsumer);
        Assert.Contains("DebugListenerProcessDebugTensorCallbackTrampoline=", packageConsumer);
        Assert.Contains("RuntimeEvidenceKind=callback-trampoline-shape", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);
        Assert.Contains("canPromoteRealCallbackRuntime = $false", readiness);
        Assert.Contains("callback-trampoline-shape", runtimeSplitReadme);
    }

    private static TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult CreateTrampoline()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_process_debug_tensor_trampoline",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 16, 16 },
            "quality-debug-listener-process-debug-tensor-callback-trampoline",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot snapshot = owner.GetSnapshot("post-dispose");
        return TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.Evaluate(
            snapshot,
            "quality-runtime-key",
            optInEnabled: false,
            fullPackageConsumerReport: false);
    }

    private static void AssertTrampolineSurfaceMarkers(string text)
    {
        Assert.Contains("debug-listener-process-debug-tensor-callback-trampoline", text);
        Assert.Contains("TensorRtDebugListenerProcessDebugTensorCallbackTrampoline", text);
        Assert.Contains("TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult", text);
        Assert.Contains("TensorRtDebugTensorMetadataSnapshot", text);
        Assert.Contains("callback-trampoline-shape", text);
        Assert.Contains("TrampolineShapeReady", text);
        Assert.Contains("NativeCallbackEntryLocated", text);
        Assert.Contains("NoThrowCallbackEntryReady", text);
        Assert.Contains("ExceptionCaptureReady", text);
        Assert.Contains("CallbackStatusMappingReady", text);
        Assert.Contains("InFlightAccountingReady", text);
        Assert.Contains("DetachBeforeReleaseReady", text);
        Assert.Contains("BorrowedDebugTensorMetadataCopyReady", text);
        Assert.Contains("BorrowedDebugTensorPointerExposed", text);
        Assert.Contains("BorrowedDebugTensorDataPointerExposed", text);
        Assert.Contains("PointerFreeSurfaceReady", text);
        Assert.Contains("ProcessDebugTensorRuntimeReady", text);
        Assert.Contains("CallbackStubEntryCount", text);
        Assert.Contains("CallbackStubLeaveCount", text);
        Assert.Contains("CanPromoteRealCallbackRuntime", text);
        Assert.Contains("RuntimeProofBlocked", text);
    }

    private static void AssertTrampolineSmokeMarkers(string text)
    {
        Assert.Contains("debug-listener-process-debug-tensor-callback-trampoline", text);
        Assert.Contains("DebugListenerProcessDebugTensorCallbackTrampoline=", text);
        Assert.Contains("callback-trampoline-shape", text);
        Assert.Contains("TrampolineShapeReady", text);
        Assert.Contains("NativeCallbackEntryLocated", text);
        Assert.Contains("NoThrowCallbackEntryReady", text);
        Assert.Contains("ExceptionCaptureReady", text);
        Assert.Contains("CallbackStatusMappingReady", text);
        Assert.Contains("InFlightAccountingReady", text);
        Assert.Contains("BorrowedDebugTensorMetadataCopyReady", text);
        Assert.Contains("PointerFreeSurfaceReady", text);
        Assert.Contains("ProcessDebugTensorRuntimeReady", text);
        Assert.Contains("CallbackStubEntryCount", text);
        Assert.Contains("CallbackStubLeaveCount", text);
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
