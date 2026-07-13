using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerRuntimeProofPrecheckTests
{
    [Fact]
    public void PrecheckCopiesOwnerDesignStateAndStaysBlockedBeforeRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-runtime-proof-precheck",
            isInput: true,
            isExecutionTensor: true);

        TensorRtDebugListenerCallbackOwnerSnapshot diagnostic = owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        Assert.True(diagnostic.DebugTensorMetadataCopied);

        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(disposed);

        Assert.Equal("debug-listener-runtime-proof-precheck", precheck.EvidenceKind);
        Assert.Equal("debug-listener", precheck.CallbackKind);
        Assert.Equal("runtime-gate", precheck.RuntimeEvidenceKind);
        Assert.False(precheck.RealCallbackRuntime);
        Assert.False(precheck.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, precheck.Line);
        Assert.True(precheck.OwnerDesignReady);
        Assert.True(precheck.DebugTensorMetadataCopied);
        Assert.True(precheck.DisposeReleaseReady);
        Assert.True(precheck.PointerFreeSurfaceReady);
        Assert.True(precheck.AttachDetachDesignGateReady);
        Assert.False(precheck.AttachControlAvailable);
        Assert.True(precheck.DetachClearControlAvailable);
        Assert.True(precheck.ManagedOwnerStateMachineReady);
        Assert.False(precheck.LineSpecificAttachDetachReady);
        Assert.False(precheck.StableNativeOwnerAddressReady);
        Assert.False(precheck.NoThrowNativeVTableReady);
        Assert.False(precheck.NativeVTableReady);
        Assert.False(precheck.ExceptionToStatusMappingReady);
        Assert.True(precheck.BorrowedTensorSafetyGateReady);
        Assert.True(precheck.AttachVTableSafetyGateReady);
        Assert.True(precheck.NativeAttachNoThrowPreflightReady);
        Assert.True(precheck.NativeOwnerAddressDesignGateReady);
        Assert.True(precheck.NativeNoThrowVTableDesignGateReady);
        Assert.True(precheck.NativeAttachEntryDesignGateReady);
        Assert.True(precheck.NativeDetachBeforeReleaseDesignGateReady);
        Assert.True(precheck.NativeOwnerLifecycleDryRunReady);
        Assert.True(precheck.NativeAttachEntryRuntimeScaffoldReady);
        Assert.True(precheck.NativeOwnerStableIdentityReady);
        Assert.True(precheck.OwnerIdentityDiagnosticsReady);
        Assert.True(precheck.OwnerIdentityPointerFree);
        Assert.True(precheck.NativeOwnerNonCopyableStorageReady);
        Assert.True(precheck.NativeOwnerCopyBlocked);
        Assert.True(precheck.NativeOwnerMoveBlocked);
        Assert.False(precheck.NativeOwnerAddressExposed);
        Assert.False(precheck.NativeOwnerPointerProduced);
        Assert.True(precheck.AttachEntryParameterShapeReady);
        Assert.True(precheck.AttachEntryNoThrowBoundaryReady);
        Assert.True(precheck.AttachEntryOwnershipDiagnosticsReady);
        Assert.False(precheck.NativeAttachEntryLocated);
        Assert.True(precheck.NativeDetachEntryLocated);
        Assert.False(precheck.LineSpecificAttachEntryDesignReady);
        Assert.False(precheck.AttachEntryNoThrowReady);
        Assert.False(precheck.AttachEntryVersionGuardReady);
        Assert.False(precheck.AttachEntryOwnershipReady);
        Assert.False(precheck.DetachBeforeReleaseReady);
        Assert.False(precheck.ReleaseHookOrderingReady);
        Assert.False(precheck.DisposeIdempotencyReady);
        Assert.False(precheck.InFlightDrainBeforeReleaseReady);
        Assert.False(precheck.CallbackStateUnpinAfterDetachReady);
        Assert.False(precheck.DelegateUnpinAfterDetachReady);
        Assert.False(precheck.StableNativeOwnerAddressDesignReady);
        Assert.True(precheck.ManagedCallbackKeepAliveDesignReady);
        Assert.True(precheck.NativeOwnerNonCopyableReady);
        Assert.False(precheck.NativeOwnerDisposeOrderReady);
        Assert.False(precheck.NativeOwnerReleaseHookReady);
        Assert.False(precheck.NativeOwnerInFlightDrainReady);
        Assert.True(precheck.NativeNoThrowDestructorGateReady);
        Assert.True(precheck.DestructorNoThrowScaffoldReady);
        Assert.True(precheck.DestructorExceptionEscapeBlocked);
        Assert.False(precheck.DestructorAddressExposed);
        Assert.False(precheck.DestructorPointerProduced);
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
        Assert.False(precheck.NoThrowVTableDesignReady);
        Assert.False(precheck.ExceptionToStatusMappingDesignReady);
        Assert.False(precheck.NativeVTableTrampolineReady);
        Assert.False(precheck.CallbackExceptionCaptureReady);
        Assert.False(precheck.CallbackStatusMappingReady);
        Assert.False(precheck.CallbackInFlightAccountingReady);
        Assert.False(precheck.CanImplementNativeAttach);
        Assert.True(precheck.BorrowedDebugTensorPointerEscapeBlocked);
        Assert.False(precheck.BorrowedDebugTensorLifetimeReady);
        Assert.False(precheck.BorrowedDebugTensorDataLifetimeReady);
        Assert.False(precheck.ProcessDebugTensorRuntimeReady);
        Assert.False(precheck.FullPackageConsumerRuntimeEvidenceReady);
        Assert.True(precheck.DeferredRowsStillRequired);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.Equal("precheck-blocked", precheck.Status);
        Assert.True(precheck.BlockedPrerequisiteCount >= 4);
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("setDebugListener", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("no-throw vtable", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("lifetime", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("real-callback-runtime", StringComparison.Ordinal));
        Assert.Contains("RealCallbackRuntime=False", precheck.Diagnostic);
        Assert.Contains("IsRealCallbackRuntimeProof=False", precheck.Diagnostic);
        Assert.Contains("AttachDetachDesignGateReady=True", precheck.Diagnostic);
        Assert.Contains("AttachControlAvailable=False", precheck.Diagnostic);
        Assert.Contains("DetachClearControlAvailable=True", precheck.Diagnostic);
        Assert.Contains("BorrowedTensorSafetyGateReady=True", precheck.Diagnostic);
        Assert.Contains("NativeAttachNoThrowPreflightReady=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerAddressDesignGateReady=True", precheck.Diagnostic);
        Assert.Contains("NativeNoThrowVTableDesignGateReady=True", precheck.Diagnostic);
        Assert.Contains("NativeAttachEntryDesignGateReady=True", precheck.Diagnostic);
        Assert.Contains("NativeDetachBeforeReleaseDesignGateReady=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerLifecycleDryRunReady=True", precheck.Diagnostic);
        Assert.Contains("NativeAttachEntryRuntimeScaffoldReady=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerStableIdentityReady=True", precheck.Diagnostic);
        Assert.Contains("OwnerIdentityDiagnosticsReady=True", precheck.Diagnostic);
        Assert.Contains("OwnerIdentityPointerFree=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableStorageReady=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerCopyBlocked=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerMoveBlocked=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerAddressExposed=False", precheck.Diagnostic);
        Assert.Contains("NativeOwnerPointerProduced=False", precheck.Diagnostic);
        Assert.Contains("AttachEntryParameterShapeReady=True", precheck.Diagnostic);
        Assert.Contains("AttachEntryNoThrowBoundaryReady=True", precheck.Diagnostic);
        Assert.Contains("AttachEntryOwnershipDiagnosticsReady=True", precheck.Diagnostic);
        Assert.Contains("NativeAttachEntryLocated=False", precheck.Diagnostic);
        Assert.Contains("NativeDetachEntryLocated=True", precheck.Diagnostic);
        Assert.Contains("LineSpecificAttachEntryDesignReady=False", precheck.Diagnostic);
        Assert.Contains("AttachEntryNoThrowReady=False", precheck.Diagnostic);
        Assert.Contains("AttachEntryVersionGuardReady=False", precheck.Diagnostic);
        Assert.Contains("AttachEntryOwnershipReady=False", precheck.Diagnostic);
        Assert.Contains("DetachBeforeReleaseReady=False", precheck.Diagnostic);
        Assert.Contains("ReleaseHookOrderingReady=False", precheck.Diagnostic);
        Assert.Contains("DisposeIdempotencyReady=False", precheck.Diagnostic);
        Assert.Contains("InFlightDrainBeforeReleaseReady=False", precheck.Diagnostic);
        Assert.Contains("CallbackStateUnpinAfterDetachReady=False", precheck.Diagnostic);
        Assert.Contains("DelegateUnpinAfterDetachReady=False", precheck.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableReady=True", precheck.Diagnostic);
        Assert.Contains("NativeNoThrowDestructorGateReady=True", precheck.Diagnostic);
        Assert.Contains("DestructorNoThrowScaffoldReady=True", precheck.Diagnostic);
        Assert.Contains("DestructorExceptionEscapeBlocked=True", precheck.Diagnostic);
        Assert.Contains("DestructorAddressExposed=False", precheck.Diagnostic);
        Assert.Contains("DestructorPointerProduced=False", precheck.Diagnostic);
        Assert.Contains("NoThrowNativeDestructorReady=True", precheck.Diagnostic);
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
        Assert.Contains("NativeOwnerLifecycleReady=False", precheck.Diagnostic);
        Assert.Contains("NoThrowVTableDesignReady=False", precheck.Diagnostic);
        Assert.Contains("ExceptionToStatusMappingDesignReady=False", precheck.Diagnostic);
        Assert.Contains("NativeVTableTrampolineReady=False", precheck.Diagnostic);
        Assert.Contains("CallbackExceptionCaptureReady=False", precheck.Diagnostic);
        Assert.Contains("CallbackStatusMappingReady=False", precheck.Diagnostic);
        Assert.Contains("CallbackInFlightAccountingReady=False", precheck.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", precheck.Diagnostic);
        Assert.Contains("BorrowedDebugTensorPointerEscapeBlocked=True", precheck.Diagnostic);
        Assert.Contains("BorrowedDebugTensorDataLifetimeReady=False", precheck.Diagnostic);
        Assert.Contains("ProcessDebugTensorRuntimeReady=False", precheck.Diagnostic);
    }

    [Fact]
    public void PublicPrecheckSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerRuntimeProofPrecheck),
            typeof(TensorRtDebugListenerRuntimeProofPrecheckResult)
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
    public void ReadinessSmokeAndDocsKeepPrecheckSeparateFromRuntimeProof()
    {
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string attachDetachGateSource = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDebugListenerAttachDetachDesignGate.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string attachDetachGateDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-attach-detach-design-gate.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerRuntimeProofPrecheck", precheckSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerRuntimeProofPrecheckResult", precheckSource);
        Assert.Contains("RuntimeEvidenceKind => \"runtime-gate\"", precheckSource);
        Assert.Contains("RealCallbackRuntime => false", precheckSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", precheckSource);
        Assert.Contains("AttachDetachDesignGateReady", precheckSource);
        Assert.Contains("AttachControlAvailable", precheckSource);
        Assert.Contains("DetachClearControlAvailable", precheckSource);
        Assert.Contains("LineSpecificAttachDetachReady", precheckSource);
        Assert.Contains("StableNativeOwnerAddressReady", precheckSource);
        Assert.Contains("NoThrowNativeVTableReady", precheckSource);
        Assert.Contains("NativeVTableReady", precheckSource);
        Assert.Contains("TensorRtDebugListenerAttachVTableSafetyGate", precheckSource);
        Assert.Contains("AttachVTableSafetyGateReady", precheckSource);
        Assert.Contains("ExceptionToStatusMappingReady", precheckSource);
        Assert.Contains("TensorRtDebugListenerNativeAttachNoThrowPreflight", precheckSource);
        Assert.Contains("NativeAttachNoThrowPreflightReady", precheckSource);
        Assert.Contains("TensorRtDebugListenerNativeOwnerAddressDesignGate", precheckSource);
        Assert.Contains("NativeOwnerAddressDesignGateReady", precheckSource);
        Assert.Contains("TensorRtDebugListenerNativeNoThrowVTableDesignGate", precheckSource);
        Assert.Contains("NativeNoThrowVTableDesignGateReady", precheckSource);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryDesignGate", precheckSource);
        Assert.Contains("NativeAttachEntryDesignGateReady", precheckSource);
        Assert.Contains("TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate", precheckSource);
        Assert.Contains("NativeDetachBeforeReleaseDesignGateReady", precheckSource);
        Assert.Contains("TensorRtDebugListenerNativeOwnerLifecycleDryRun", precheckSource);
        Assert.Contains("NativeOwnerLifecycleDryRunReady", precheckSource);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryRuntimeScaffold", precheckSource);
        Assert.Contains("NativeAttachEntryRuntimeScaffoldReady", precheckSource);
        Assert.Contains("TensorRtDebugListenerNativeOwnerStableIdentity", precheckSource);
        Assert.Contains("NativeOwnerStableIdentityReady", precheckSource);
        Assert.Contains("OwnerIdentityDiagnosticsReady", precheckSource);
        Assert.Contains("OwnerIdentityPointerFree", precheckSource);
        Assert.Contains("TensorRtDebugListenerNativeOwnerLifecycleGate", precheckSource);
        Assert.Contains("NativeOwnerLifecycleGateReady", precheckSource);
        Assert.Contains("LifecycleScaffoldReady", precheckSource);
        Assert.Contains("LifecycleAddressExposed", precheckSource);
        Assert.Contains("LifecyclePointerProduced", precheckSource);
        Assert.Contains("AttachEntryParameterShapeReady", precheckSource);
        Assert.Contains("AttachEntryNoThrowBoundaryReady", precheckSource);
        Assert.Contains("AttachEntryOwnershipDiagnosticsReady", precheckSource);
        Assert.Contains("LineSpecificAttachEntryDesignReady", precheckSource);
        Assert.Contains("AttachEntryNoThrowReady", precheckSource);
        Assert.Contains("AttachEntryVersionGuardReady", precheckSource);
        Assert.Contains("AttachEntryOwnershipReady", precheckSource);
        Assert.Contains("DetachBeforeReleaseReady", precheckSource);
        Assert.Contains("NativeOwnerLifecycleReady", precheckSource);
        Assert.Contains("NativeAttachEntryLocated", precheckSource);
        Assert.Contains("NativeDetachEntryLocated", precheckSource);
        Assert.Contains("NoThrowVTableDesignReady", precheckSource);
        Assert.Contains("ExceptionToStatusMappingDesignReady", precheckSource);
        Assert.Contains("NativeVTableTrampolineReady", precheckSource);
        Assert.Contains("CallbackExceptionCaptureReady", precheckSource);
        Assert.Contains("CallbackStatusMappingReady", precheckSource);
        Assert.Contains("CallbackInFlightAccountingReady", precheckSource);
        Assert.Contains("CanImplementNativeAttach", precheckSource);
        Assert.Contains("TensorRtDebugListenerBorrowedTensorSafetyGate", precheckSource);
        Assert.Contains("BorrowedTensorSafetyGateReady", precheckSource);
        Assert.Contains("BorrowedDebugTensorPointerEscapeBlocked", precheckSource);
        Assert.Contains("BorrowedDebugTensorDataLifetimeReady", precheckSource);
        Assert.Contains("ProcessDebugTensorRuntimeReady", precheckSource);
        Assert.Contains("CanAttemptRuntimeProof", precheckSource);
        Assert.DoesNotContain("public IntPtr", precheckSource);
        Assert.DoesNotContain("public nint", precheckSource);
        Assert.Contains("TensorRtDebugListenerAttachDetachDesignGate", attachDetachGateSource);

        Assert.Contains("debug-listener-runtime-proof-precheck", smokeProgram);
        Assert.Contains("DebugListenerRuntimeProofPrecheck=", smokeProgram);
        Assert.Contains("debug-listener-attach-detach-design-gate", smokeProgram);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", smokeProgram);
        Assert.Contains("DebugListenerBorrowedTensorSafetyGate=", smokeProgram);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", smokeProgram);
        Assert.Contains("DebugListenerAttachVTableSafetyGate=", smokeProgram);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", smokeProgram);
        Assert.Contains("DebugListenerNativeAttachNoThrowPreflight=", smokeProgram);
        Assert.Contains("debug-listener-native-owner-address-design-gate", smokeProgram);
        Assert.Contains("DebugListenerNativeOwnerAddressDesignGate=", smokeProgram);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", smokeProgram);
        Assert.Contains("DebugListenerNativeNoThrowVTableDesignGate=", smokeProgram);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", smokeProgram);
        Assert.Contains("DebugListenerNativeAttachEntryDesignGate=", smokeProgram);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", smokeProgram);
        Assert.Contains("DebugListenerNativeDetachBeforeReleaseDesignGate=", smokeProgram);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", smokeProgram);
        Assert.Contains("DebugListenerNativeOwnerLifecycleDryRun=", smokeProgram);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", smokeProgram);
        Assert.Contains("DebugListenerNativeAttachEntryRuntimeScaffold=", smokeProgram);
        Assert.Contains("debug-listener-native-owner-stable-identity", smokeProgram);
        Assert.Contains("DebugListenerNativeOwnerStableIdentity=", smokeProgram);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", smokeProgram);
        Assert.Contains("DebugListenerNativeOwnerLifecycleGate=", smokeProgram);
        Assert.Contains("AttachControlAvailable", smokeProgram);
        Assert.Contains("DetachClearControlAvailable", smokeProgram);
        Assert.Contains("NativeAttachNoThrowPreflightReady", smokeProgram);
        Assert.Contains("NativeOwnerAddressDesignGateReady", smokeProgram);
        Assert.Contains("NativeNoThrowVTableDesignGateReady", smokeProgram);
        Assert.Contains("NativeAttachEntryDesignGateReady", smokeProgram);
        Assert.Contains("NativeDetachBeforeReleaseDesignGateReady", smokeProgram);
        Assert.Contains("NativeOwnerLifecycleDryRunReady", smokeProgram);
        Assert.Contains("NativeAttachEntryRuntimeScaffoldReady", smokeProgram);
        Assert.Contains("NativeOwnerStableIdentityReady", smokeProgram);
        Assert.Contains("OwnerIdentityDiagnosticsReady", smokeProgram);
        Assert.Contains("OwnerIdentityPointerFree", smokeProgram);
        Assert.Contains("NativeOwnerLifecycleGateReady", smokeProgram);
        Assert.Contains("ManagedDisposeSnapshotReady", smokeProgram);
        Assert.Contains("LifecycleScaffoldReady", smokeProgram);
        Assert.Contains("LifecycleAddressExposed", smokeProgram);
        Assert.Contains("LifecyclePointerProduced", smokeProgram);
        Assert.Contains("AttachEntryParameterShapeReady", smokeProgram);
        Assert.Contains("AttachEntryNoThrowBoundaryReady", smokeProgram);
        Assert.Contains("AttachEntryOwnershipDiagnosticsReady", smokeProgram);
        Assert.Contains("LineSpecificAttachEntryDesignReady", smokeProgram);
        Assert.Contains("AttachEntryNoThrowReady", smokeProgram);
        Assert.Contains("AttachEntryVersionGuardReady", smokeProgram);
        Assert.Contains("AttachEntryOwnershipReady", smokeProgram);
        Assert.Contains("DetachBeforeReleaseReady", smokeProgram);
        Assert.Contains("ReleaseHookOrderingReady", smokeProgram);
        Assert.Contains("DisposeIdempotencyReady", smokeProgram);
        Assert.Contains("InFlightDrainBeforeReleaseReady", smokeProgram);
        Assert.Contains("CallbackStateUnpinAfterDetachReady", smokeProgram);
        Assert.Contains("DelegateUnpinAfterDetachReady", smokeProgram);
        Assert.Contains("NativeOwnerLifecycleReady", smokeProgram);
        Assert.Contains("CanImplementNativeAttach", smokeProgram);
        Assert.Contains("BorrowedTensorSafetyGateReady", smokeProgram);
        Assert.Contains("BorrowedDebugTensorDataLifetimeReady", smokeProgram);
        Assert.Contains("ProcessDebugTensorRuntimeReady", smokeProgram);
        Assert.Contains("CanAttemptRuntimeProof", smokeProgram);
        Assert.Contains("RuntimeProofBlocked", smokeProgram);

        Assert.Contains("debugListenerRuntimeProofPrecheck", readiness);
        Assert.Contains("New-DebugListenerRuntimeProofPrecheckEvidence", readiness);
        Assert.Contains("debugListenerAttachDetachDesignGate", readiness);
        Assert.Contains("New-DebugListenerAttachDetachDesignGateEvidence", readiness);
        Assert.Contains("debugListenerBorrowedTensorSafetyGate", readiness);
        Assert.Contains("New-DebugListenerBorrowedTensorSafetyGateEvidence", readiness);
        Assert.Contains("debugListenerAttachVTableSafetyGate", readiness);
        Assert.Contains("New-DebugListenerAttachVTableSafetyGateEvidence", readiness);
        Assert.Contains("debugListenerNativeAttachNoThrowPreflight", readiness);
        Assert.Contains("New-DebugListenerNativeAttachNoThrowPreflightEvidence", readiness);
        Assert.Contains("debugListenerNativeOwnerAddressDesignGate", readiness);
        Assert.Contains("New-DebugListenerNativeOwnerAddressDesignGateEvidence", readiness);
        Assert.Contains("debug-listener-native-owner-address-design-gate", readiness);
        Assert.Contains("debugListenerNativeNoThrowVTableDesignGate", readiness);
        Assert.Contains("New-DebugListenerNativeNoThrowVTableDesignGateEvidence", readiness);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", readiness);
        Assert.Contains("debugListenerNativeAttachEntryDesignGate", readiness);
        Assert.Contains("New-DebugListenerNativeAttachEntryDesignGateEvidence", readiness);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", readiness);
        Assert.Contains("debugListenerNativeDetachBeforeReleaseDesignGate", readiness);
        Assert.Contains("New-DebugListenerNativeDetachBeforeReleaseDesignGateEvidence", readiness);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", readiness);
        Assert.Contains("debugListenerNativeOwnerLifecycleDryRun", readiness);
        Assert.Contains("New-DebugListenerNativeOwnerLifecycleDryRunEvidence", readiness);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", readiness);
        Assert.Contains("debugListenerNativeAttachEntryRuntimeScaffold", readiness);
        Assert.Contains("New-DebugListenerNativeAttachEntryRuntimeScaffoldEvidence", readiness);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", readiness);
        Assert.Contains("debugListenerNativeOwnerStableIdentity", readiness);
        Assert.Contains("New-DebugListenerNativeOwnerStableIdentityEvidence", readiness);
        Assert.Contains("debug-listener-native-owner-stable-identity", readiness);
        Assert.Contains("debugListenerNativeOwnerLifecycleGate", readiness);
        Assert.Contains("New-DebugListenerNativeOwnerLifecycleGateEvidence", readiness);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", readiness);
        Assert.Contains("runtime-gate-precheck", readiness);
        Assert.Contains("precheck-ready", readiness);
        Assert.Contains("NativeAttachNoThrowPreflightReady", readiness);
        Assert.Contains("NativeOwnerAddressDesignGateReady", readiness);
        Assert.Contains("NativeNoThrowVTableDesignGateReady", readiness);
        Assert.Contains("NativeAttachEntryDesignGateReady", readiness);
        Assert.Contains("NativeDetachBeforeReleaseDesignGateReady", readiness);
        Assert.Contains("NativeOwnerLifecycleDryRunReady", readiness);
        Assert.Contains("NativeAttachEntryRuntimeScaffoldReady", readiness);
        Assert.Contains("NativeOwnerStableIdentityReady", readiness);
        Assert.Contains("OwnerIdentityDiagnosticsReady", readiness);
        Assert.Contains("OwnerIdentityPointerFree", readiness);
        Assert.Contains("NativeOwnerLifecycleGateReady", readiness);
        Assert.Contains("LifecycleScaffoldReady", readiness);
        Assert.Contains("LifecycleAddressExposed=False", readiness);
        Assert.Contains("LifecyclePointerProduced=False", readiness);
        Assert.Contains("AttachEntryParameterShapeReady=True", readiness);
        Assert.Contains("AttachEntryNoThrowBoundaryReady=True", readiness);
        Assert.Contains("AttachEntryOwnershipDiagnosticsReady=True", readiness);
        Assert.Contains("LineSpecificAttachEntryDesignReady=False", readiness);
        Assert.Contains("AttachEntryNoThrowReady=False", readiness);
        Assert.Contains("AttachEntryVersionGuardReady=False", readiness);
        Assert.Contains("AttachEntryOwnershipReady=False", readiness);
        Assert.Contains("DetachBeforeReleaseReady=False", readiness);
        Assert.Contains("ReleaseHookOrderingReady=False", readiness);
        Assert.Contains("DisposeIdempotencyReady=False", readiness);
        Assert.Contains("InFlightDrainBeforeReleaseReady=False", readiness);
        Assert.Contains("CallbackStateUnpinAfterDetachReady=False", readiness);
        Assert.Contains("DelegateUnpinAfterDetachReady=False", readiness);
        Assert.Contains("NativeOwnerLifecycleReady=False", readiness);
        Assert.Contains("NativeVTableTrampolineReady=False", readiness);
        Assert.Contains("CallbackExceptionCaptureReady=False", readiness);
        Assert.Contains("CallbackStatusMappingReady=False", readiness);
        Assert.Contains("CallbackInFlightAccountingReady=False", readiness);
        Assert.Contains("CanImplementNativeAttach=False", readiness);
        Assert.Contains("canAttemptRuntimeProof = $false", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);

        Assert.Contains("debug-listener-runtime-proof-precheck", packageConsumer);
        Assert.Contains("debug-listener-attach-detach-design-gate", packageConsumer);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", packageConsumer);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", packageConsumer);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", packageConsumer);
        Assert.Contains("debug-listener-native-owner-address-design-gate", packageConsumer);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", packageConsumer);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", packageConsumer);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", packageConsumer);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", packageConsumer);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", packageConsumer);
        Assert.Contains("debug-listener-native-owner-stable-identity", packageConsumer);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", packageConsumer);
        Assert.Contains("debug-listener-runtime-proof-precheck", bridgeConsumer);
        Assert.Contains("debug-listener-attach-detach-design-gate", bridgeConsumer);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", bridgeConsumer);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", bridgeConsumer);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", bridgeConsumer);
        Assert.Contains("debug-listener-native-owner-address-design-gate", bridgeConsumer);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", bridgeConsumer);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", bridgeConsumer);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", bridgeConsumer);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", bridgeConsumer);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", bridgeConsumer);
        Assert.Contains("debug-listener-native-owner-stable-identity", bridgeConsumer);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerRuntimeProofPrecheck", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerAttachDetachDesignGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerBorrowedTensorSafetyGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerAttachVTableSafetyGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeAttachNoThrowPreflight", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeOwnerAddressDesignGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeNoThrowVTableDesignGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryDesignGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeOwnerLifecycleDryRun", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryRuntimeScaffold", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeOwnerStableIdentity", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeOwnerLifecycleGate", bridgeConsumer);

        Assert.Contains("debug-listener-runtime-proof-precheck", precheckDoc);
        Assert.Contains("RuntimeEvidenceKind=runtime-gate", precheckDoc);
        Assert.Contains("AttachDetachDesignGateReady=True", precheckDoc);
        Assert.Contains("BorrowedTensorSafetyGateReady=True", precheckDoc);
        Assert.Contains("AttachVTableSafetyGateReady=True", precheckDoc);
        Assert.Contains("NativeAttachNoThrowPreflightReady=True", precheckDoc);
        Assert.Contains("NativeOwnerAddressDesignGateReady=True", precheckDoc);
        Assert.Contains("NativeNoThrowVTableDesignGateReady=True", precheckDoc);
        Assert.Contains("NativeAttachEntryDesignGateReady=True", precheckDoc);
        Assert.Contains("NativeDetachBeforeReleaseDesignGateReady=True", precheckDoc);
        Assert.Contains("NativeOwnerLifecycleDryRunReady=True", precheckDoc);
        Assert.Contains("NativeAttachEntryRuntimeScaffoldReady=True", precheckDoc);
        Assert.Contains("NativeOwnerStableIdentityReady=True", precheckDoc);
        Assert.Contains("OwnerIdentityDiagnosticsReady=True", precheckDoc);
        Assert.Contains("OwnerIdentityPointerFree=True", precheckDoc);
        Assert.Contains("NativeOwnerLifecycleGateReady=True", precheckDoc);
        Assert.Contains("LifecycleScaffoldReady=True", precheckDoc);
        Assert.Contains("LifecycleAddressExposed=False", precheckDoc);
        Assert.Contains("LifecyclePointerProduced=False", precheckDoc);
        Assert.Contains("AttachEntryParameterShapeReady=True", precheckDoc);
        Assert.Contains("AttachEntryNoThrowBoundaryReady=True", precheckDoc);
        Assert.Contains("AttachEntryOwnershipDiagnosticsReady=True", precheckDoc);
        Assert.Contains("NativeAttachEntryLocated=False", precheckDoc);
        Assert.Contains("NativeDetachEntryLocated=True", precheckDoc);
        Assert.Contains("LineSpecificAttachEntryDesignReady=False", precheckDoc);
        Assert.Contains("AttachEntryNoThrowReady=False", precheckDoc);
        Assert.Contains("AttachEntryVersionGuardReady=False", precheckDoc);
        Assert.Contains("AttachEntryOwnershipReady=False", precheckDoc);
        Assert.Contains("DetachBeforeReleaseReady=False", precheckDoc);
        Assert.Contains("ReleaseHookOrderingReady=False", precheckDoc);
        Assert.Contains("DisposeIdempotencyReady=False", precheckDoc);
        Assert.Contains("InFlightDrainBeforeReleaseReady=False", precheckDoc);
        Assert.Contains("CallbackStateUnpinAfterDetachReady=False", precheckDoc);
        Assert.Contains("DelegateUnpinAfterDetachReady=False", precheckDoc);
        Assert.Contains("NativeOwnerLifecycleReady=False", precheckDoc);
        Assert.Contains("NoThrowVTableDesignReady=False", precheckDoc);
        Assert.Contains("ExceptionToStatusMappingDesignReady=False", precheckDoc);
        Assert.Contains("NativeVTableTrampolineReady=False", precheckDoc);
        Assert.Contains("CallbackExceptionCaptureReady=False", precheckDoc);
        Assert.Contains("CallbackStatusMappingReady=False", precheckDoc);
        Assert.Contains("CallbackInFlightAccountingReady=False", precheckDoc);
        Assert.Contains("CanImplementNativeAttach=False", precheckDoc);
        Assert.Contains("BorrowedDebugTensorPointerEscapeBlocked=True", precheckDoc);
        Assert.Contains("BorrowedDebugTensorDataLifetimeReady=False", precheckDoc);
        Assert.Contains("ProcessDebugTensorRuntimeReady=False", precheckDoc);
        Assert.Contains("AttachControlAvailable=False", precheckDoc);
        Assert.Contains("DetachClearControlAvailable=True", precheckDoc);
        Assert.Contains("ExceptionToStatusMappingReady=False", precheckDoc);
        Assert.Contains("CanAttemptRuntimeProof=False", precheckDoc);
        Assert.Contains("RuntimeProofBlocked=True", precheckDoc);
        Assert.Contains("setDebugListener", precheckDoc);
        Assert.Contains("not proof", precheckDoc);
        Assert.Contains("debug-listener-attach-detach-design-gate", attachDetachGateDoc);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", attachDetachGateDoc);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", attachDetachGateDoc);
        Assert.Contains("RuntimeEvidenceKind=design-gate", attachDetachGateDoc);
        Assert.Contains("debug-listener-runtime-proof-precheck", trampolineGate);
        Assert.Contains("debug-listener-attach-detach-design-gate", trampolineGate);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", trampolineGate);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", trampolineGate);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", trampolineGate);
        Assert.Contains("debug-listener-native-owner-address-design-gate", trampolineGate);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", trampolineGate);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", trampolineGate);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", trampolineGate);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", trampolineGate);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", trampolineGate);
        Assert.Contains("debug-listener-native-owner-stable-identity", trampolineGate);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", trampolineGate);
        Assert.Contains("debug-listener-runtime-proof-precheck", schema);
        Assert.Contains("debug-listener-attach-detach-design-gate", schema);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", schema);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", schema);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", schema);
        Assert.Contains("debug-listener-native-owner-address-design-gate", schema);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", schema);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", schema);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", schema);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", schema);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", schema);
        Assert.Contains("debug-listener-native-owner-stable-identity", schema);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", schema);
        Assert.Contains("debug-listener-runtime-proof-precheck", latest);
        Assert.Contains("debug-listener-attach-detach-design-gate", latest);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", latest);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", latest);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", latest);
        Assert.Contains("debug-listener-native-owner-address-design-gate", latest);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", latest);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", latest);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", latest);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", latest);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", latest);
        Assert.Contains("debug-listener-native-owner-stable-identity", latest);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", latest);
        Assert.Contains("debugListenerRuntimeProofPrecheck", runtimeSplitReadme);
        Assert.Contains("debugListenerAttachDetachDesignGate", runtimeSplitReadme);
        Assert.Contains("debugListenerBorrowedTensorSafetyGate", runtimeSplitReadme);
        Assert.Contains("debugListenerAttachVTableSafetyGate", runtimeSplitReadme);
        Assert.Contains("debugListenerNativeAttachNoThrowPreflight", runtimeSplitReadme);
        Assert.Contains("debugListenerNativeOwnerAddressDesignGate", runtimeSplitReadme);
        Assert.Contains("debugListenerNativeNoThrowVTableDesignGate", runtimeSplitReadme);
        Assert.Contains("debugListenerNativeAttachEntryDesignGate", runtimeSplitReadme);
        Assert.Contains("debugListenerNativeDetachBeforeReleaseDesignGate", runtimeSplitReadme);
        Assert.Contains("debugListenerNativeOwnerLifecycleDryRun", runtimeSplitReadme);
        Assert.Contains("debugListenerNativeAttachEntryRuntimeScaffold", runtimeSplitReadme);
        Assert.Contains("debugListenerNativeOwnerStableIdentity", runtimeSplitReadme);
        Assert.Contains("debugListenerNativeOwnerLifecycleGate", runtimeSplitReadme);
        Assert.Contains("debug-listener-runtime-proof-precheck", smokeReadme);
        Assert.Contains("debug-listener-attach-detach-design-gate", smokeReadme);
        Assert.Contains("debug-listener-borrowed-tensor-safety-gate", smokeReadme);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", smokeReadme);
        Assert.Contains("debug-listener-native-attach-nothrow-preflight", smokeReadme);
        Assert.Contains("debug-listener-native-owner-address-design-gate", smokeReadme);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", smokeReadme);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", smokeReadme);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", smokeReadme);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", smokeReadme);
        Assert.Contains("debug-listener-native-attach-entry-runtime-scaffold", smokeReadme);
        Assert.Contains("debug-listener-native-owner-stable-identity", smokeReadme);
        Assert.Contains("debug-listener-native-owner-lifecycle-gate", smokeReadme);

        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"notifyShape\",\"IOutputAllocator::notifyShape\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"reallocateOutput\",\"IOutputAllocator::reallocateOutput\",\"other\",\"deferred-only\"", comparison);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
