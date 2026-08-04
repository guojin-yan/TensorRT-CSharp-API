using System.Reflection;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeOwnerVTableInstallExperimentTests
{
    [Fact]
    public void NativeOwnerVTableInstallExperimentStaysDisabledPointerFreeAndNonProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_owner_vtable_experiment_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 4, 4 },
            "quality-debug-listener-native-owner-vtable-install-experiment",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot snapshot = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult experiment =
            TensorRtDebugListenerNativeOwnerVTableInstallExperiment.Evaluate(snapshot);

        Assert.Equal("debug-listener-native-owner-vtable-install-experiment", experiment.EvidenceKind);
        Assert.Equal("native-owner-vtable-install-experiment", experiment.RuntimeEvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", experiment.CallbackKind);
        Assert.False(experiment.RealCallbackRuntime);
        Assert.False(experiment.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, experiment.Line);
        Assert.True(experiment.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, experiment.LastStatus);
        Assert.True(experiment.NativeOwnerLifecycleGateReady);
        Assert.True(experiment.NativeAttachBridgeShapeGateReady);
        Assert.True(experiment.NativeNoThrowVTableScaffoldGateReady);
        Assert.True(experiment.BorrowedDebugTensorMetadataGateReady);
        Assert.True(experiment.NativeVTableInstallPreflightReady);
        Assert.True(experiment.ExperimentShapeReady);
        Assert.True(experiment.InstallAttemptGuardReady);
        Assert.False(experiment.NonNullAttachEnabled);
        Assert.False(experiment.RuntimeProofEnabled);
        Assert.False(experiment.NativeVTableInstallAttempted);
        Assert.False(experiment.NativeVTableInstalled);
        Assert.True(experiment.RollbackReady);
        Assert.True(experiment.DetachBeforeReleaseReady);
        Assert.True(experiment.FailureStatusMappingReady);
        Assert.True(experiment.PointerFree);
        Assert.False(experiment.VTableAddressExposed);
        Assert.False(experiment.VTablePointerProduced);
        Assert.False(experiment.DebugTensorPointerExposed);
        Assert.False(experiment.DebugTensorDataPointerExposed);
        Assert.False(experiment.CanEnableSetDebugListenerNonNull);
        Assert.False(experiment.CanInstallNativeVTable);
        Assert.False(experiment.ProcessDebugTensorRuntimeReady);
        Assert.False(experiment.CanCallProcessDebugTensorRuntime);
        Assert.False(experiment.CanAttemptRuntimeProof);
        Assert.True(experiment.RuntimeProofBlocked);
        Assert.True(experiment.DeferredRowsStillRequired);
        Assert.Equal("native-owner-vtable-install-experiment-ready", experiment.Status);
        Assert.Contains("native-owner-vtable-install-experiment", experiment.ReasonNativeOwnerVTableInstallStillBlocked, StringComparison.Ordinal);
        Assert.Contains("ExperimentShapeReady=True", experiment.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("NativeVTableInstallAttempted=False", experiment.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("NativeVTableInstalled=False", experiment.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("CanInstallNativeVTable=False", experiment.Diagnostic, StringComparison.Ordinal);
        Assert.Contains(experiment.BlockedPrerequisites, item => item.Contains("real-callback-runtime", StringComparison.Ordinal));
    }

    [Fact]
    public void PublicNativeOwnerVTableInstallExperimentSurfaceDoesNotExposeRawPointers()
    {
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerNativeOwnerVTableInstallExperiment));
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult));

        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeOwnerVTableInstallExperiment.cs");
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public UIntPtr", source);
        Assert.DoesNotContain("public nint", source);
        Assert.Contains("NativeVTableInstallAttempted", source);
        Assert.Contains("CanInstallNativeVTable", source);
        Assert.Contains("ReasonNativeOwnerVTableInstallStillBlocked", source);
    }

    [Fact]
    public void SourceNativeSmokeReadinessPackageAndDocsContainExperimentEvidenceButNotRuntimeProof()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeOwnerVTableInstallExperiment.cs");
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "debug_listener_native_owner_vtable_install_experiment.inc");
        string trt8Api = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11Api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string doc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-owner-vtable-install-experiment.md");
        string preflightDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-vtable-install-preflight.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        AssertExperimentSurfaceMarkers(source);
        AssertExperimentSurfaceMarkers(smoke);
        AssertExperimentSurfaceMarkers(readiness);
        AssertExperimentSurfaceMarkers(bridgeConsumer);
        AssertExperimentSurfaceMarkers(packageConsumer);
        AssertExperimentSurfaceMarkers(doc);
        AssertExperimentEvidenceKindMarkers(readiness);
        AssertExperimentEvidenceKindMarkers(doc);
        AssertExperimentEvidenceKindMarkers(schema);
        AssertExperimentEvidenceKindMarkers(latest);
        AssertExperimentEvidenceKindMarkers(trampolineGate);
        AssertExperimentEvidenceKindMarkers(smokeReadme);
        Assert.Contains("debug-listener-native-owner-vtable-install-experiment.md", toc);
        Assert.Contains("debug-listener-native-owner-vtable-install-experiment.md", index);
        Assert.Contains("DebugListener Native Owner VTable Install Experiment", preflightDoc);
        Assert.Contains("DebugListenerNativeOwnerVTableInstallExperiment final", nativeSource);
        Assert.Contains("configure_api_line", nativeSource);
        Assert.Contains("configure_prerequisites", nativeSource);
        Assert.Contains("configure_disabled_reason", nativeSource);
        Assert.Contains("experiment_shape_ready", nativeSource);
        Assert.Contains("install_attempt_guard_ready", nativeSource);
        Assert.Contains("native_vtable_install_attempted", nativeSource);
        Assert.Contains("native_vtable_installed", nativeSource);
        Assert.Contains("rollback_ready", nativeSource);
        Assert.Contains("detach_before_release_ready", nativeSource);
        Assert.Contains("failure_status_mapping_ready", nativeSource);
        Assert.Contains("can_attempt_runtime_proof", nativeSource);
        Assert.Contains("debug_listener_native_owner_vtable_install_experiment.inc", trt8Api);
        Assert.Contains("debug_listener_native_owner_vtable_install_experiment.inc", trt10Api);
        Assert.Contains("debug_listener_native_owner_vtable_install_experiment.inc", trt11Api);
        Assert.Contains("DebugListenerNativeOwnerVTableInstallExperiment=", smoke);
        Assert.Contains("hasDebugListenerNativeOwnerVTableInstallExperiment", readiness);
        Assert.Contains("New-DebugListenerNativeOwnerVTableInstallExperimentEvidence", readiness);
        Assert.Contains("debugListenerNativeOwnerVTableInstallExperiment", readiness);
        Assert.Contains("native-owner-vtable-install-experiment", packageConsumer);
        Assert.Contains("native-owner-vtable-install-experiment", bridgeConsumer);
        Assert.Contains("not proof", doc);
        Assert.Contains("not proof", schema);
        Assert.Contains("not proof", latest);
        Assert.Contains("not proof", trampolineGate);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static void AssertExperimentSurfaceMarkers(string text)
    {
        Assert.Contains("debug-listener-native-owner-vtable-install-experiment", text);
        Assert.Contains("TensorRtDebugListenerNativeOwnerVTableInstallExperiment", text);
        Assert.Contains("TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult", text);
        Assert.Contains("ExperimentShapeReady", text);
        Assert.Contains("InstallAttemptGuardReady", text);
        Assert.Contains("NonNullAttachEnabled", text);
        Assert.Contains("RuntimeProofEnabled", text);
        Assert.Contains("NativeVTableInstallAttempted", text);
        Assert.Contains("NativeVTableInstalled", text);
        Assert.Contains("RollbackReady", text);
        Assert.Contains("DetachBeforeReleaseReady", text);
        Assert.Contains("FailureStatusMappingReady", text);
        Assert.Contains("PointerFree", text);
        Assert.Contains("CanEnableSetDebugListenerNonNull", text);
        Assert.Contains("CanInstallNativeVTable", text);
        Assert.Contains("CanCallProcessDebugTensorRuntime", text);
        Assert.Contains("ReasonNativeOwnerVTableInstallStillBlocked", text);
        Assert.Contains("RuntimeProofBlocked", text);
    }

    private static void AssertExperimentEvidenceKindMarkers(string text)
    {
        Assert.Contains("native-owner-vtable-install-experiment", text);
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
        return RepositorySourceReader.Read(path);
    }
}
