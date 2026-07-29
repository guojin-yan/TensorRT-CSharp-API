using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerAttachBridgeVTableBatchTests
{
    [Fact]
    public void AttachBridgeMappingAccountingAndVTableScaffoldGatesStayPointerFreeAndNonProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-attach-bridge-vtable-batch",
            isInput: true,
            isExecutionTensor: true);

        TensorRtDebugListenerCallbackOwnerSnapshot diagnostic = owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        Assert.True(diagnostic.DebugTensorMetadataCopied);
        Assert.True(diagnostic.ProcessDebugTensorCount > 0);

        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerNativeOwnerLifecycleGateResult lifecycle =
            TensorRtDebugListenerNativeOwnerLifecycleGate.Evaluate(disposed);
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult attachBridge =
            TensorRtDebugListenerNativeAttachBridgeShapeGate.Evaluate(disposed, lifecycle);
        TensorRtDebugListenerExceptionStatusMappingGateResult mapping =
            TensorRtDebugListenerExceptionStatusMappingGate.Evaluate(disposed, attachBridge);
        TensorRtDebugListenerInFlightAccountingGateResult accounting =
            TensorRtDebugListenerInFlightAccountingGate.Evaluate(disposed, mapping);
        TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult vtable =
            TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.Evaluate(disposed, attachBridge, mapping, accounting);
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(disposed);

        AssertAttachBridgeGate(attachBridge);
        AssertExceptionStatusMappingGate(mapping);
        AssertInFlightAccountingGate(accounting);
        AssertVTableScaffoldGate(vtable);
        AssertPrecheckConsumesTheBatch(precheck);
    }

    [Fact]
    public void PublicBatchSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerNativeAttachBridgeShapeGate),
            typeof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult),
            typeof(TensorRtDebugListenerExceptionStatusMappingGate),
            typeof(TensorRtDebugListenerExceptionStatusMappingGateResult),
            typeof(TensorRtDebugListenerInFlightAccountingGate),
            typeof(TensorRtDebugListenerInFlightAccountingGateResult),
            typeof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGate),
            typeof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult)
        };

        foreach (Type type in publicTypes)
        {
            AssertNoRawPointerTypes(type);
        }
    }

    [Fact]
    public void SourceNativeSmokeReadinessPackageAndDocsContainBatchEvidenceButNotRuntimeProof()
    {
        string attachBridgeSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeAttachBridgeShapeGate.cs");
        string mappingSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerExceptionStatusMappingGate.cs");
        string accountingSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerInFlightAccountingGate.cs");
        string vtableSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string nativeAttachBridge = ReadSource("native", "src", "tensorrt", "common", "debug_listener_native_attach_bridge_shape_gate.inc");
        string nativeMapping = ReadSource("native", "src", "tensorrt", "common", "debug_listener_exception_status_mapping_gate.inc");
        string nativeAccounting = ReadSource("native", "src", "tensorrt", "common", "debug_listener_inflight_accounting_gate.inc");
        string nativeVTable = ReadSource("native", "src", "tensorrt", "common", "debug_listener_native_nothrow_vtable_scaffold_gate.inc");
        string trt8 = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10 = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11 = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string attachBridgeDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-attach-bridge-shape-gate.md");
        string mappingDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-exception-status-mapping-gate.md");
        string accountingDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-inflight-accounting-gate.md");
        string vtableDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-nothrow-vtable-scaffold-gate.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerNativeAttachBridgeShapeGate", attachBridgeSource);
        Assert.Contains("RuntimeEvidenceKind => \"attach-bridge-shape-gate\"", attachBridgeSource);
        Assert.Contains("SetDebugListenerNonNullEnabled", attachBridgeSource);
        Assert.Contains("NonNullAttachStillDisabled", attachBridgeSource);
        Assert.Contains("NativeAttachEntryLocated", attachBridgeSource);
        Assert.Contains("CanAttemptRuntimeProof", attachBridgeSource);
        Assert.Contains("RuntimeProofBlocked", attachBridgeSource);
        Assert.DoesNotContain("public IntPtr", attachBridgeSource);
        Assert.DoesNotContain("public UIntPtr", attachBridgeSource);
        Assert.DoesNotContain("public nint", attachBridgeSource);

        Assert.Contains("public static class TensorRtDebugListenerExceptionStatusMappingGate", mappingSource);
        Assert.Contains("RuntimeEvidenceKind => \"exception-status-gate\"", mappingSource);
        Assert.Contains("NativeCallbackExceptionCaptureReady", mappingSource);
        Assert.Contains("CallbackStatusMappingGateReady", mappingSource);
        Assert.Contains("MappingAddressExposed", mappingSource);
        Assert.Contains("MappingPointerProduced", mappingSource);
        Assert.DoesNotContain("public IntPtr", mappingSource);
        Assert.DoesNotContain("public UIntPtr", mappingSource);
        Assert.DoesNotContain("public nint", mappingSource);

        Assert.Contains("public static class TensorRtDebugListenerInFlightAccountingGate", accountingSource);
        Assert.Contains("RuntimeEvidenceKind => \"inflight-accounting-gate\"", accountingSource);
        Assert.Contains("CallbackEnterAccountingGateReady", accountingSource);
        Assert.Contains("CallbackLeaveAccountingGateReady", accountingSource);
        Assert.Contains("ReleaseAfterDrainGateReady", accountingSource);
        Assert.Contains("AccountingAddressExposed", accountingSource);
        Assert.DoesNotContain("public IntPtr", accountingSource);
        Assert.DoesNotContain("public UIntPtr", accountingSource);
        Assert.DoesNotContain("public nint", accountingSource);

        Assert.Contains("public static class TensorRtDebugListenerNativeNoThrowVTableScaffoldGate", vtableSource);
        Assert.Contains("RuntimeEvidenceKind => \"vtable-scaffold-gate\"", vtableSource);
        Assert.Contains("NoThrowVTableScaffoldReady", vtableSource);
        Assert.Contains("VTableDestructorNoThrowReady", vtableSource);
        Assert.Contains("ProcessDebugTensorCallbackStubNoThrowReady", vtableSource);
        Assert.Contains("VTableAddressExposed", vtableSource);
        Assert.Contains("VTablePointerProduced", vtableSource);
        Assert.DoesNotContain("public IntPtr", vtableSource);
        Assert.DoesNotContain("public UIntPtr", vtableSource);
        Assert.DoesNotContain("public nint", vtableSource);

        Assert.Contains("TensorRtDebugListenerNativeAttachBridgeShapeGate", precheckSource);
        Assert.Contains("TensorRtDebugListenerExceptionStatusMappingGate", precheckSource);
        Assert.Contains("TensorRtDebugListenerInFlightAccountingGate", precheckSource);
        Assert.Contains("TensorRtDebugListenerNativeNoThrowVTableScaffoldGate", precheckSource);
        Assert.Contains("NativeAttachBridgeShapeGateReady", precheckSource);
        Assert.Contains("ExceptionStatusMappingGateReady", precheckSource);
        Assert.Contains("InFlightAccountingGateReady", precheckSource);
        Assert.Contains("NativeNoThrowVTableScaffoldGateReady", precheckSource);
        Assert.Contains("CanAttemptRuntimeProof", precheckSource);
        Assert.Contains("FullPackageConsumerRuntimeEvidenceReady => false", precheckSource);

        Assert.Contains("struct DebugListenerNativeAttachBridgeShapeGate final", nativeAttachBridge);
        Assert.Contains("DebugListenerNativeAttachBridgeShapeGate(const DebugListenerNativeAttachBridgeShapeGate&) = delete", nativeAttachBridge);
        Assert.Contains("configure_shape", nativeAttachBridge);
        Assert.Contains("noexcept", nativeAttachBridge);
        Assert.Contains("non_null_attach_enabled = false", nativeAttachBridge);
        Assert.Contains("struct DebugListenerExceptionStatusMappingGate final", nativeMapping);
        Assert.Contains("map_exception_to_status", nativeMapping);
        Assert.Contains("exception_escape_blocked", nativeMapping);
        Assert.Contains("struct DebugListenerInFlightAccountingGate final", nativeAccounting);
        Assert.Contains("enter_callback", nativeAccounting);
        Assert.Contains("leave_callback", nativeAccounting);
        Assert.Contains("can_release_after_drain", nativeAccounting);
        Assert.Contains("struct DebugListenerNativeNoThrowVTableScaffoldGate final", nativeVTable);
        Assert.Contains("process_debug_tensor_stub", nativeVTable);
        Assert.Contains("borrowed_pointer_escape_blocked", nativeVTable);
        Assert.Contains("debug_listener_native_attach_bridge_shape_gate.inc", trt8);
        Assert.Contains("debug_listener_exception_status_mapping_gate.inc", trt8);
        Assert.Contains("debug_listener_inflight_accounting_gate.inc", trt8);
        Assert.Contains("debug_listener_native_nothrow_vtable_scaffold_gate.inc", trt8);
        Assert.Contains("debug_listener_native_attach_bridge_shape_gate.inc", trt10);
        Assert.Contains("debug_listener_exception_status_mapping_gate.inc", trt10);
        Assert.Contains("debug_listener_inflight_accounting_gate.inc", trt10);
        Assert.Contains("debug_listener_native_nothrow_vtable_scaffold_gate.inc", trt10);
        Assert.Contains("debug_listener_native_attach_bridge_shape_gate.inc", trt11);
        Assert.Contains("debug_listener_exception_status_mapping_gate.inc", trt11);
        Assert.Contains("debug_listener_inflight_accounting_gate.inc", trt11);
        Assert.Contains("debug_listener_native_nothrow_vtable_scaffold_gate.inc", trt11);

        AssertBatchMarkers(smokeProgram);
        Assert.Contains("DebugListenerNativeAttachBridgeShapeGate=", smokeProgram);
        Assert.Contains("DebugListenerExceptionStatusMappingGate=", smokeProgram);
        Assert.Contains("DebugListenerInFlightAccountingGate=", smokeProgram);
        Assert.Contains("DebugListenerNativeNoThrowVTableScaffoldGate=", smokeProgram);
        Assert.Contains("VTableAddressExposed", smokeProgram);
        Assert.Contains("VTablePointerProduced", smokeProgram);

        AssertBatchMarkers(readiness);
        Assert.Contains("New-DebugListenerNativeAttachBridgeShapeGateEvidence", readiness);
        Assert.Contains("New-DebugListenerExceptionStatusMappingGateEvidence", readiness);
        Assert.Contains("New-DebugListenerInFlightAccountingGateEvidence", readiness);
        Assert.Contains("New-DebugListenerNativeNoThrowVTableScaffoldGateEvidence", readiness);
        Assert.Contains("hasDebugListenerNativeAttachBridgeShapeGate", readiness);
        Assert.Contains("hasDebugListenerExceptionStatusMappingGate", readiness);
        Assert.Contains("hasDebugListenerInFlightAccountingGate", readiness);
        Assert.Contains("hasDebugListenerNativeNoThrowVTableScaffoldGate", readiness);
        Assert.Contains("NativeAttachBridgeShapeGateReady=True", readiness);
        Assert.Contains("ExceptionStatusMappingGateReady=True", readiness);
        Assert.Contains("InFlightAccountingGateReady=True", readiness);
        Assert.Contains("NativeNoThrowVTableScaffoldGateReady=True", readiness);
        Assert.Contains("VTableAddressExposed=False", readiness);
        Assert.Contains("VTablePointerProduced=False", readiness);

        AssertBatchMarkers(packageConsumer);
        AssertBatchMarkers(bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeAttachBridgeShapeGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerExceptionStatusMappingGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerInFlightAccountingGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeNoThrowVTableScaffoldGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult", bridgeConsumer);

        Assert.Contains("debug-listener-native-attach-bridge-shape-gate", attachBridgeDoc);
        Assert.Contains("debug-listener-exception-status-mapping-gate", mappingDoc);
        Assert.Contains("debug-listener-inflight-accounting-gate", accountingDoc);
        Assert.Contains("debug-listener-native-nothrow-vtable-scaffold-gate", vtableDoc);
        AssertBatchMarkers(precheckDoc);
        AssertBatchMarkers(schema);
        AssertBatchMarkers(trampolineGate);
        AssertBatchMarkers(latest);
        AssertBatchMarkers(runtimeSplitReadme);
        AssertBatchMarkers(smokeReadme);
        Assert.Contains("not proof", attachBridgeDoc);
        Assert.Contains("not proof", mappingDoc);
        Assert.Contains("not proof", accountingDoc);
        Assert.Contains("not proof", vtableDoc);
        Assert.Contains("not proof", schema);

        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static void AssertAttachBridgeGate(TensorRtDebugListenerNativeAttachBridgeShapeGateResult gate)
    {
        Assert.Equal("debug-listener-native-attach-bridge-shape-gate", gate.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", gate.CallbackKind);
        Assert.Equal("attach-bridge-shape-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, gate.LastStatus);
        Assert.True(gate.NativeOwnerLifecycleGateReady);
        Assert.True(gate.AttachBridgeShapeReady);
        Assert.True(gate.AttachBridgeNoThrowBoundaryReady);
        Assert.True(gate.AttachBridgeVersionGuardReady);
        Assert.True(gate.AttachBridgeOwnershipDiagnosticsReady);
        Assert.True(gate.AttachBridgePointerFree);
        Assert.False(gate.SetDebugListenerNonNullEnabled);
        Assert.True(gate.NonNullAttachStillDisabled);
        Assert.False(gate.NativeAttachEntryLocated);
        Assert.True(gate.NativeDetachEntryLocated);
        Assert.False(gate.NativeOwnerLifecycleReady);
        Assert.False(gate.NativeVTableDesignReady);
        Assert.True(gate.AttachBridgeShapeGateReady);
        Assert.False(gate.ProcessDebugTensorRuntimeReady);
        Assert.False(gate.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(gate.CanImplementNativeAttach);
        Assert.False(gate.CanAttemptRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.Equal("attach-bridge-shape-gate-ready", gate.Status);
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("setDebugListener(non-null)", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("processDebugTensor", StringComparison.Ordinal));
        Assert.Contains("RealCallbackRuntime=False", gate.Diagnostic);
        Assert.Contains("NativeAttachEntryLocated=False", gate.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", gate.Diagnostic);
    }

    private static void AssertExceptionStatusMappingGate(TensorRtDebugListenerExceptionStatusMappingGateResult gate)
    {
        Assert.Equal("debug-listener-exception-status-mapping-gate", gate.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", gate.CallbackKind);
        Assert.Equal("exception-status-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, gate.LastStatus);
        Assert.True(gate.AttachBridgeShapeGateReady);
        Assert.True(gate.ManagedCallbackExceptionCaptureReady);
        Assert.True(gate.NativeCallbackExceptionCaptureReady);
        Assert.True(gate.CallbackStatusMappingGateReady);
        Assert.True(gate.ExceptionEscapeBlocked);
        Assert.True(gate.DiagnosticCopyReady);
        Assert.False(gate.MappingAddressExposed);
        Assert.False(gate.MappingPointerProduced);
        Assert.False(gate.NativeAttachEntryLocated);
        Assert.True(gate.ExceptionStatusMappingGateReady);
        Assert.False(gate.ProcessDebugTensorRuntimeReady);
        Assert.False(gate.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(gate.CanAttemptRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.Equal("exception-status-gate-ready", gate.Status);
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("processDebugTensor", StringComparison.Ordinal));
        Assert.Contains("ExceptionStatusMappingGateReady=True", gate.Diagnostic);
        Assert.Contains("MappingAddressExposed=False", gate.Diagnostic);
    }

    private static void AssertInFlightAccountingGate(TensorRtDebugListenerInFlightAccountingGateResult gate)
    {
        Assert.Equal("debug-listener-inflight-accounting-gate", gate.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", gate.CallbackKind);
        Assert.Equal("inflight-accounting-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, gate.LastStatus);
        Assert.True(gate.ProcessDebugTensorCount > 0);
        Assert.Equal(0, gate.InFlightCallbackCount);
        Assert.True(gate.MaxInFlightCallbackCount > 0);
        Assert.True(gate.ReleaseHookCount > 0);
        Assert.False(gate.CallbackStatePinned);
        Assert.False(gate.DelegatePinned);
        Assert.True(gate.DisposeRequested);
        Assert.True(gate.ExceptionStatusMappingGateReady);
        Assert.True(gate.CallbackEnterAccountingGateReady);
        Assert.True(gate.CallbackLeaveAccountingGateReady);
        Assert.True(gate.CallbackInFlightNeverNegativeReady);
        Assert.True(gate.ReleaseAfterDrainGateReady);
        Assert.True(gate.CallbackStateUnpinAfterDrainGateReady);
        Assert.False(gate.AccountingAddressExposed);
        Assert.False(gate.AccountingPointerProduced);
        Assert.False(gate.NativeAttachEntryLocated);
        Assert.True(gate.InFlightAccountingGateReady);
        Assert.False(gate.ProcessDebugTensorRuntimeReady);
        Assert.False(gate.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(gate.CanAttemptRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.Equal("inflight-accounting-gate-ready", gate.Status);
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("processDebugTensor", StringComparison.Ordinal));
        Assert.Contains("InFlightAccountingGateReady=True", gate.Diagnostic);
        Assert.Contains("AccountingPointerProduced=False", gate.Diagnostic);
    }

    private static void AssertVTableScaffoldGate(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult gate)
    {
        Assert.Equal("debug-listener-native-nothrow-vtable-scaffold-gate", gate.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", gate.CallbackKind);
        Assert.Equal("vtable-scaffold-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, gate.LastStatus);
        Assert.True(gate.NativeAttachBridgeShapeGateReady);
        Assert.True(gate.ExceptionStatusMappingGateReady);
        Assert.True(gate.InFlightAccountingGateReady);
        Assert.True(gate.NoThrowVTableScaffoldReady);
        Assert.True(gate.VTableDestructorNoThrowReady);
        Assert.True(gate.ProcessDebugTensorCallbackStubNoThrowReady);
        Assert.True(gate.ExceptionEscapeBlocked);
        Assert.True(gate.CallbackExceptionCaptureGateReady);
        Assert.True(gate.CallbackStatusMappingGateReady);
        Assert.True(gate.CallbackInFlightAccountingGateReady);
        Assert.True(gate.BorrowedDebugTensorPointerEscapeBlocked);
        Assert.False(gate.VTableAddressExposed);
        Assert.False(gate.VTablePointerProduced);
        Assert.False(gate.NativeAttachEntryLocated);
        Assert.False(gate.NativeVTableDesignReady);
        Assert.True(gate.VTableScaffoldGateReady);
        Assert.False(gate.ProcessDebugTensorRuntimeReady);
        Assert.False(gate.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(gate.CanImplementNativeAttach);
        Assert.False(gate.CanAttemptRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.Equal("vtable-scaffold-gate-ready", gate.Status);
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("native IDebugListener vtable implementation", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("processDebugTensor", StringComparison.Ordinal));
        Assert.Contains("VTableAddressExposed=False", gate.Diagnostic);
        Assert.Contains("VTablePointerProduced=False", gate.Diagnostic);
    }

    private static void AssertPrecheckConsumesTheBatch(TensorRtDebugListenerRuntimeProofPrecheckResult precheck)
    {
        Assert.Equal("debug-listener-runtime-proof-precheck", precheck.EvidenceKind);
        Assert.Equal("runtime-gate", precheck.RuntimeEvidenceKind);
        Assert.False(precheck.RealCallbackRuntime);
        Assert.False(precheck.IsRealCallbackRuntimeProof);
        Assert.True(precheck.NativeAttachBridgeShapeGateReady);
        Assert.True(precheck.AttachBridgeShapeReady);
        Assert.True(precheck.AttachBridgeNoThrowBoundaryReady);
        Assert.True(precheck.AttachBridgeVersionGuardReady);
        Assert.True(precheck.AttachBridgeOwnershipDiagnosticsReady);
        Assert.True(precheck.AttachBridgePointerFree);
        Assert.True(precheck.NonNullAttachStillDisabled);
        Assert.True(precheck.ExceptionStatusMappingGateReady);
        Assert.True(precheck.NativeCallbackExceptionCaptureReady);
        Assert.True(precheck.CallbackStatusMappingGateReady);
        Assert.True(precheck.ExceptionEscapeBlocked);
        Assert.True(precheck.DiagnosticCopyReady);
        Assert.True(precheck.InFlightAccountingGateReady);
        Assert.True(precheck.CallbackEnterAccountingGateReady);
        Assert.True(precheck.CallbackLeaveAccountingGateReady);
        Assert.True(precheck.CallbackInFlightNeverNegativeReady);
        Assert.True(precheck.ReleaseAfterDrainGateReady);
        Assert.True(precheck.CallbackStateUnpinAfterDrainGateReady);
        Assert.True(precheck.NativeNoThrowVTableScaffoldGateReady);
        Assert.True(precheck.NoThrowVTableScaffoldReady);
        Assert.True(precheck.VTableDestructorNoThrowReady);
        Assert.True(precheck.ProcessDebugTensorCallbackStubNoThrowReady);
        Assert.False(precheck.VTableAddressExposed);
        Assert.False(precheck.VTablePointerProduced);
        Assert.False(precheck.NativeAttachEntryLocated);
        Assert.False(precheck.NativeVTableReady);
        Assert.False(precheck.CanImplementNativeAttach);
        Assert.False(precheck.ProcessDebugTensorRuntimeReady);
        Assert.False(precheck.FullPackageConsumerRuntimeEvidenceReady);
        Assert.True(precheck.DeferredRowsStillRequired);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.Equal("precheck-blocked", precheck.Status);
        Assert.Contains("NativeAttachBridgeShapeGateReady=True", precheck.Diagnostic);
        Assert.Contains("ExceptionStatusMappingGateReady=True", precheck.Diagnostic);
        Assert.Contains("InFlightAccountingGateReady=True", precheck.Diagnostic);
        Assert.Contains("NativeNoThrowVTableScaffoldGateReady=True", precheck.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", precheck.Diagnostic);
        Assert.Contains("CanAttemptRuntimeProof=False", precheck.Diagnostic);
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("real-callback-runtime", StringComparison.Ordinal));
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

    private static void AssertBatchMarkers(string text)
    {
        Assert.Contains("debug-listener-native-attach-bridge-shape-gate", text);
        Assert.Contains("debug-listener-exception-status-mapping-gate", text);
        Assert.Contains("debug-listener-inflight-accounting-gate", text);
        Assert.Contains("debug-listener-native-nothrow-vtable-scaffold-gate", text);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }
}
