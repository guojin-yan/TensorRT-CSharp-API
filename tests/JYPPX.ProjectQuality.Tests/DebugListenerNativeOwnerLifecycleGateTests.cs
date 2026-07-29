using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeOwnerLifecycleGateTests
{
    [Fact]
    public void NativeOwnerLifecycleGatePromotesSourceVisibleLifecycleScaffoldWithoutRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-owner-lifecycle-gate",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerNativeNoThrowDestructorResult destructor =
            TensorRtDebugListenerNativeNoThrowDestructor.Evaluate(disposed);
        TensorRtDebugListenerNativeOwnerLifecycleGateResult lifecycle =
            TensorRtDebugListenerNativeOwnerLifecycleGate.Evaluate(disposed, destructor);

        Assert.Equal("debug-listener-native-owner-lifecycle-gate", lifecycle.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", lifecycle.CallbackKind);
        Assert.Equal("lifecycle-gate", lifecycle.RuntimeEvidenceKind);
        Assert.False(lifecycle.RealCallbackRuntime);
        Assert.False(lifecycle.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, lifecycle.Line);
        Assert.True(lifecycle.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, lifecycle.LastStatus);
        Assert.True(lifecycle.NativeNoThrowDestructorGateReady);
        Assert.True(lifecycle.NativeOwnerNonCopyableReady);
        Assert.True(lifecycle.NativeOwnerCopyBlocked);
        Assert.True(lifecycle.NativeOwnerMoveBlocked);
        Assert.False(lifecycle.NativeOwnerAddressExposed);
        Assert.False(lifecycle.NativeOwnerPointerProduced);
        Assert.True(lifecycle.DestructorNoThrowScaffoldReady);
        Assert.True(lifecycle.DestructorExceptionEscapeBlocked);
        Assert.False(lifecycle.DestructorAddressExposed);
        Assert.False(lifecycle.DestructorPointerProduced);
        Assert.True(lifecycle.ManagedDisposeSnapshotReady);
        Assert.True(lifecycle.LifecycleScaffoldReady);
        Assert.True(lifecycle.ReleaseHookOrderingGateReady);
        Assert.True(lifecycle.DisposeIdempotencyGateReady);
        Assert.True(lifecycle.InFlightDrainGateReady);
        Assert.True(lifecycle.CallbackStateUnpinAfterDetachGateReady);
        Assert.True(lifecycle.DelegateUnpinAfterDetachGateReady);
        Assert.False(lifecycle.LifecycleAddressExposed);
        Assert.False(lifecycle.LifecyclePointerProduced);
        Assert.False(lifecycle.NativeAttachEntryLocated);
        Assert.True(lifecycle.NativeDetachEntryLocated);
        Assert.True(lifecycle.NoThrowNativeDestructorReady);
        Assert.False(lifecycle.ReleaseHookOrderingReady);
        Assert.False(lifecycle.DisposeIdempotencyReady);
        Assert.False(lifecycle.InFlightDrainBeforeReleaseReady);
        Assert.False(lifecycle.CallbackStateUnpinAfterDetachReady);
        Assert.False(lifecycle.DelegateUnpinAfterDetachReady);
        Assert.True(lifecycle.LifecycleGateReady);
        Assert.False(lifecycle.NativeOwnerLifecycleReady);
        Assert.False(lifecycle.NativeVTableDesignReady);
        Assert.False(lifecycle.ProcessDebugTensorRuntimeReady);
        Assert.False(lifecycle.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(lifecycle.CanImplementNativeAttach);
        Assert.False(lifecycle.CanAttemptRuntimeProof);
        Assert.True(lifecycle.RuntimeProofBlocked);
        Assert.True(lifecycle.DeferredRowsStillRequired);
        Assert.Equal("lifecycle-gate-ready", lifecycle.Status);
        Assert.Contains(lifecycle.BlockedPrerequisites, item => item.Contains("setDebugListener(non-null)", StringComparison.Ordinal));
        Assert.Contains(lifecycle.BlockedPrerequisites, item => item.Contains("native IDebugListener vtable", StringComparison.Ordinal));
        Assert.Contains(lifecycle.BlockedPrerequisites, item => item.Contains("processDebugTensor", StringComparison.Ordinal));
        Assert.Contains("RuntimeEvidenceKind=lifecycle-gate", lifecycle.Diagnostic);
        Assert.Contains("NativeNoThrowDestructorGateReady=True", lifecycle.Diagnostic);
        Assert.Contains("ManagedDisposeSnapshotReady=True", lifecycle.Diagnostic);
        Assert.Contains("LifecycleScaffoldReady=True", lifecycle.Diagnostic);
        Assert.Contains("ReleaseHookOrderingGateReady=True", lifecycle.Diagnostic);
        Assert.Contains("DisposeIdempotencyGateReady=True", lifecycle.Diagnostic);
        Assert.Contains("InFlightDrainGateReady=True", lifecycle.Diagnostic);
        Assert.Contains("CallbackStateUnpinAfterDetachGateReady=True", lifecycle.Diagnostic);
        Assert.Contains("DelegateUnpinAfterDetachGateReady=True", lifecycle.Diagnostic);
        Assert.Contains("LifecycleAddressExposed=False", lifecycle.Diagnostic);
        Assert.Contains("LifecyclePointerProduced=False", lifecycle.Diagnostic);
        Assert.Contains("LifecycleGateReady=True", lifecycle.Diagnostic);
        Assert.Contains("NativeOwnerLifecycleReady=False", lifecycle.Diagnostic);
        Assert.Contains("NativeVTableDesignReady=False", lifecycle.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", lifecycle.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", lifecycle.Diagnostic);
    }

    [Fact]
    public void RuntimeProofPrecheckConsumesNativeOwnerLifecycleGateAndStaysBlocked()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-owner-lifecycle-gate-precheck",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(owner.GetSnapshot("post-dispose"));

        Assert.True(precheck.NativeNoThrowDestructorGateReady);
        Assert.True(precheck.NoThrowNativeDestructorReady);
        Assert.True(precheck.NativeOwnerLifecycleGateReady);
        Assert.True(precheck.ManagedDisposeSnapshotReady);
        Assert.True(precheck.LifecycleScaffoldReady);
        Assert.True(precheck.ReleaseHookOrderingGateReady);
        Assert.True(precheck.DisposeIdempotencyGateReady);
        Assert.True(precheck.InFlightDrainGateReady);
        Assert.True(precheck.CallbackStateUnpinAfterDetachGateReady);
        Assert.True(precheck.DelegateUnpinAfterDetachGateReady);
        Assert.False(precheck.LifecycleAddressExposed);
        Assert.False(precheck.LifecyclePointerProduced);
        Assert.False(precheck.NativeOwnerLifecycleReady);
        Assert.False(precheck.CanImplementNativeAttach);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.Contains("NativeOwnerLifecycleGateReady=True", precheck.Diagnostic);
        Assert.Contains("ManagedDisposeSnapshotReady=True", precheck.Diagnostic);
        Assert.Contains("LifecycleScaffoldReady=True", precheck.Diagnostic);
        Assert.Contains("ReleaseHookOrderingGateReady=True", precheck.Diagnostic);
        Assert.Contains("DisposeIdempotencyGateReady=True", precheck.Diagnostic);
        Assert.Contains("InFlightDrainGateReady=True", precheck.Diagnostic);
        Assert.Contains("CallbackStateUnpinAfterDetachGateReady=True", precheck.Diagnostic);
        Assert.Contains("DelegateUnpinAfterDetachGateReady=True", precheck.Diagnostic);
        Assert.Contains("LifecycleAddressExposed=False", precheck.Diagnostic);
        Assert.Contains("LifecyclePointerProduced=False", precheck.Diagnostic);
    }

    [Fact]
    public void PublicNativeOwnerLifecycleGateSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerNativeOwnerLifecycleGate),
            typeof(TensorRtDebugListenerNativeOwnerLifecycleGateResult)
        };

        foreach (Type type in publicTypes)
        {
            foreach (ConstructorInfo constructor in type.GetConstructors(BindingFlags.Instance | BindingFlags.Public))
            {
                foreach (ParameterInfo parameter in constructor.GetParameters())
                {
                    Assert.NotEqual(typeof(IntPtr), parameter.ParameterType);
                    Assert.NotEqual(typeof(UIntPtr), parameter.ParameterType);
                }
            }

            foreach (PropertyInfo property in type.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.Static))
            {
                Assert.NotEqual(typeof(IntPtr), property.PropertyType);
                Assert.NotEqual(typeof(UIntPtr), property.PropertyType);
            }

            foreach (MethodInfo method in type.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.DeclaredOnly))
            {
                Assert.NotEqual(typeof(IntPtr), method.ReturnType);
                Assert.NotEqual(typeof(UIntPtr), method.ReturnType);
                foreach (ParameterInfo parameter in method.GetParameters())
                {
                    Assert.NotEqual(typeof(IntPtr), parameter.ParameterType);
                    Assert.NotEqual(typeof(UIntPtr), parameter.ParameterType);
                }
            }
        }
    }

    [Fact]
    public void SourceSmokeDocsAndReadinessKeepNativeOwnerLifecycleGateSeparateFromRuntimeProof()
    {
        string lifecycleSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeOwnerLifecycleGate.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string nativeLifecycle = ReadSource("native", "src", "tensorrt", "common", "debug_listener_native_owner_lifecycle_gate.inc");
        string trt8 = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10 = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11 = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string lifecycleDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-owner-lifecycle-gate.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerNativeOwnerLifecycleGate", lifecycleSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerNativeOwnerLifecycleGateResult", lifecycleSource);
        Assert.Contains("EvidenceKind => \"debug-listener-native-owner-lifecycle-gate\"", lifecycleSource);
        Assert.Contains("RuntimeEvidenceKind => \"lifecycle-gate\"", lifecycleSource);
        Assert.Contains("RealCallbackRuntime => false", lifecycleSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", lifecycleSource);
        Assert.Contains("LifecycleGateReady", lifecycleSource);
        Assert.Contains("NativeOwnerLifecycleReady", lifecycleSource);
        Assert.Contains("NativeVTableDesignReady", lifecycleSource);
        Assert.DoesNotContain("public IntPtr", lifecycleSource);
        Assert.DoesNotContain("public nint", lifecycleSource);

        Assert.Contains("TensorRtDebugListenerNativeOwnerLifecycleGate", precheckSource);
        Assert.Contains("NativeOwnerLifecycleGateReady", precheckSource);
        Assert.Contains("LifecycleScaffoldReady", precheckSource);
        Assert.Contains("LifecycleAddressExposed", precheckSource);
        Assert.Contains("LifecyclePointerProduced", precheckSource);

        Assert.Contains("debug-listener-native-owner-lifecycle-gate", smokeProgram);
        Assert.Contains("DebugListenerNativeOwnerLifecycleGate=", smokeProgram);
        Assert.Contains("NativeOwnerLifecycleGateReady", smokeProgram);

        Assert.Contains("struct DebugListenerNativeOwnerLifecycleGate final", nativeLifecycle);
        Assert.Contains("DebugListenerNativeOwnerLifecycleGate(const DebugListenerNativeOwnerLifecycleGate&) = delete", nativeLifecycle);
        Assert.Contains("operator=(const DebugListenerNativeOwnerLifecycleGate&) = delete", nativeLifecycle);
        Assert.Contains("DebugListenerNativeOwnerLifecycleGate(DebugListenerNativeOwnerLifecycleGate&&) = delete", nativeLifecycle);
        Assert.Contains("operator=(DebugListenerNativeOwnerLifecycleGate&&) = delete", nativeLifecycle);
        Assert.Contains("request_detach_before_release", nativeLifecycle);
        Assert.Contains("request_release", nativeLifecycle);
        Assert.Contains("can_unpin_after_detach", nativeLifecycle);
        Assert.Contains("std::is_nothrow_destructible", nativeLifecycle);
        Assert.Contains("debug_listener_native_owner_lifecycle_gate.inc", trt8);
        Assert.Contains("debug_listener_native_owner_lifecycle_gate.inc", trt10);
        Assert.Contains("debug_listener_native_owner_lifecycle_gate.inc", trt11);

        Assert.Contains("New-DebugListenerNativeOwnerLifecycleGateEvidence", readiness);
        Assert.Contains("debugListenerNativeOwnerLifecycleGate", readiness);
        Assert.Contains("hasDebugListenerNativeOwnerLifecycleGate", readiness);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", readiness);
        Assert.Contains("LifecycleGateReady=True", readiness);
        Assert.Contains("NativeOwnerLifecycleReady=False", readiness);
        Assert.Contains("NativeVTableDesignReady=False", readiness);
        Assert.Contains("Debug listener native owner lifecycle gate missing evidence", readiness);

        Assert.Contains("debug-listener-native-owner-lifecycle-gate", packageConsumer);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeOwnerLifecycleGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeOwnerLifecycleGateResult", bridgeConsumer);
        Assert.Contains("NativeOwnerLifecycleGateReady", bridgeConsumer);
        Assert.Contains("LifecyclePointerProduced", bridgeConsumer);

        Assert.Contains("debug-listener-native-owner-lifecycle-gate", lifecycleDoc);
        Assert.Contains("RuntimeEvidenceKind=lifecycle-gate", lifecycleDoc);
        Assert.Contains("NativeNoThrowDestructorGateReady=True", lifecycleDoc);
        Assert.Contains("ManagedDisposeSnapshotReady=True", lifecycleDoc);
        Assert.Contains("LifecycleScaffoldReady=True", lifecycleDoc);
        Assert.Contains("ReleaseHookOrderingGateReady=True", lifecycleDoc);
        Assert.Contains("DisposeIdempotencyGateReady=True", lifecycleDoc);
        Assert.Contains("InFlightDrainGateReady=True", lifecycleDoc);
        Assert.Contains("CallbackStateUnpinAfterDetachGateReady=True", lifecycleDoc);
        Assert.Contains("DelegateUnpinAfterDetachGateReady=True", lifecycleDoc);
        Assert.Contains("LifecycleAddressExposed=False", lifecycleDoc);
        Assert.Contains("LifecyclePointerProduced=False", lifecycleDoc);
        Assert.Contains("LifecycleGateReady=True", lifecycleDoc);
        Assert.Contains("NativeOwnerLifecycleReady=False", lifecycleDoc);
        Assert.Contains("NativeVTableDesignReady=False", lifecycleDoc);
        Assert.Contains("CanImplementNativeAttach=False", lifecycleDoc);
        Assert.Contains("RuntimeProofBlocked=True", lifecycleDoc);
        Assert.Contains("not proof", lifecycleDoc);

        Assert.Contains("debug-listener-native-owner-lifecycle-gate", precheckDoc);
        Assert.Contains("NativeOwnerLifecycleGateReady=True", precheckDoc);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", trampolineGate);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", schema);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", latest);
        Assert.Contains("debugListenerNativeOwnerLifecycleGate", runtimeSplitReadme);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", smokeReadme);

        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }
}
