using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates whether the DebugListener owner design snapshot is ready for the next real callback runtime proof stage.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This precheck consumes copied metadata only. It does not attach a debug listener to TensorRT, does not expose a
/// callback owner pointer, and does not prove that <c>IDebugListener::processDebugTensor</c> has been invoked by a real
/// TensorRT build/enqueue path.
/// 该说明强调当前结果属于 DebugListener 安全门禁或 precheck，不代表 TensorRT 已真实触发 callback。
/// </remarks>
public static partial class TensorRtDebugListenerRuntimeProofPrecheck
{
    /// <summary>
    /// Evaluates the current DebugListener runtime proof precheck from copied owner, lifecycle gate, attach bridge,
    /// exception/status mapping, in-flight accounting, and vtable scaffold evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="attachVTableSafetyGate">The copied attach/vtable safety gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeAttachNoThrowPreflight">The copied native attach/no-throw preflight result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeOwnerAddressDesignGate">The copied native owner address design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeNoThrowVTableDesignGate">The copied native no-throw vtable design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeAttachEntryDesignGate">The copied native attach entry design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeDetachBeforeReleaseDesignGate">The copied native detach-before-release design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeOwnerLifecycleDryRun">The copied native owner lifecycle dry-run result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeAttachEntryRuntimeScaffold">The copied native attach entry runtime scaffold result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeOwnerStableIdentity">The copied native owner stable identity result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeOwnerNonCopyableStorage">The copied native owner non-copyable storage result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeNoThrowDestructor">The copied native no-throw destructor result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeOwnerLifecycleGate">The copied native owner lifecycle gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeAttachBridgeShapeGate">The copied native attach bridge shape gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="exceptionStatusMappingGate">The copied exception/status mapping gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="inFlightAccountingGate">The copied in-flight accounting gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeNoThrowVTableScaffoldGate">The copied native no-throw vtable scaffold gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free precheck result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerRuntimeProofPrecheckResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate,
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableSafetyGate,
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight,
        TensorRtDebugListenerNativeOwnerAddressDesignGateResult nativeOwnerAddressDesignGate,
        TensorRtDebugListenerNativeNoThrowVTableDesignGateResult nativeNoThrowVTableDesignGate,
        TensorRtDebugListenerNativeAttachEntryDesignGateResult nativeAttachEntryDesignGate,
        TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult nativeDetachBeforeReleaseDesignGate,
        TensorRtDebugListenerNativeOwnerLifecycleDryRunResult nativeOwnerLifecycleDryRun,
        TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult nativeAttachEntryRuntimeScaffold,
        TensorRtDebugListenerNativeOwnerStableIdentityResult nativeOwnerStableIdentity,
        TensorRtDebugListenerNativeOwnerNonCopyableStorageResult nativeOwnerNonCopyableStorage,
        TensorRtDebugListenerNativeNoThrowDestructorResult nativeNoThrowDestructor,
        TensorRtDebugListenerNativeOwnerLifecycleGateResult nativeOwnerLifecycleGate,
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult nativeAttachBridgeShapeGate,
        TensorRtDebugListenerExceptionStatusMappingGateResult exceptionStatusMappingGate,
        TensorRtDebugListenerInFlightAccountingGateResult inFlightAccountingGate,
        TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult nativeNoThrowVTableScaffoldGate)
    {
        bool ownerDesignReady =
            string.Equals(ownerDesignSnapshot.EvidenceKind, "debug-listener-callback-owner-design", StringComparison.Ordinal) &&
            string.Equals(ownerDesignSnapshot.RuntimeEvidenceKind, "not-present", StringComparison.Ordinal) &&
            !ownerDesignSnapshot.RealCallbackRuntime &&
            !ownerDesignSnapshot.IsRealCallbackRuntimeProof &&
            ownerDesignSnapshot.LastStatus == BridgeStatusCode.Ok &&
            ownerDesignSnapshot.FailureCount == 0 &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool metadataCopyReady =
            ownerDesignSnapshot.DebugTensorMetadataCopied &&
            !string.IsNullOrWhiteSpace(ownerDesignSnapshot.TensorName) &&
            ownerDesignSnapshot.ShapeRank >= 0;
        bool disposeReleaseReady =
            ownerDesignSnapshot.DisposeRequested &&
            ownerDesignSnapshot.ReleaseHookCount > 0 &&
            !ownerDesignSnapshot.CallbackStatePinned &&
            !ownerDesignSnapshot.DelegatePinned &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool pointerFreeSurfaceReady =
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;

        List<string> blockers = new List<string>();
        if (!attachDetachDesignGate.LineSupportsDebugListener)
        {
            blockers.Add("TensorRT 10 or 11 IDebugListener line support has not been selected.");
        }

        if (!ownerDesignReady)
        {
            blockers.Add("debug-listener-callback-owner-design snapshot is not clean owner-design evidence.");
        }

        if (!metadataCopyReady)
        {
            blockers.Add("debug tensor copied metadata is incomplete.");
        }

        if (!disposeReleaseReady)
        {
            blockers.Add("dispose release hook evidence is not present on the owner design snapshot.");
        }

        if (!pointerFreeSurfaceReady)
        {
            blockers.Add("public DebugListener surface still exposes or produces a borrowed debug tensor pointer.");
        }

        foreach (string blocker in attachDetachDesignGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in borrowedTensorSafetyGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in attachVTableSafetyGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in nativeAttachNoThrowPreflight.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in nativeOwnerAddressDesignGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in nativeNoThrowVTableDesignGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in nativeAttachEntryDesignGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in nativeDetachBeforeReleaseDesignGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in nativeOwnerLifecycleDryRun.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in nativeAttachEntryRuntimeScaffold.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in nativeOwnerStableIdentity.BlockedPrerequisites)
        {
            if (nativeOwnerNonCopyableStorage.NativeOwnerNonCopyableReady &&
                blocker.IndexOf("non-copyable storage", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                continue;
            }

            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in nativeOwnerNonCopyableStorage.BlockedPrerequisites)
        {
            if (nativeNoThrowDestructor.NoThrowNativeDestructorReady &&
                blocker.IndexOf("no-throw destructor", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                continue;
            }

            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in nativeNoThrowDestructor.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in nativeOwnerLifecycleGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in nativeAttachBridgeShapeGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in exceptionStatusMappingGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in inFlightAccountingGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        foreach (string blocker in nativeNoThrowVTableScaffoldGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        if (!blockers.Contains("full package consumer smoke has not emitted real-callback-runtime evidence."))
        {
            blockers.Add("full package consumer smoke has not emitted real-callback-runtime evidence.");
        }

        return new TensorRtDebugListenerRuntimeProofPrecheckResult(
            ownerDesignSnapshot.Line,
            ownerDesignReady,
            metadataCopyReady,
            disposeReleaseReady,
            pointerFreeSurfaceReady,
            attachDetachDesignGate.DesignGateReady,
            attachDetachDesignGate.AttachControlAvailable,
            attachDetachDesignGate.DetachClearControlAvailable,
            attachDetachDesignGate.ManagedOwnerStateMachineReady,
            attachVTableSafetyGate.StableNativeOwnerAddressReady,
            attachVTableSafetyGate.NoThrowNativeVTableReady,
            attachVTableSafetyGate.ExceptionToStatusMappingReady,
            borrowedTensorSafetyGate.SafetyGateReady,
            attachVTableSafetyGate.SafetyGateReady,
            nativeAttachNoThrowPreflight.PreflightReady,
            nativeOwnerAddressDesignGate.DesignGateReady,
            nativeNoThrowVTableDesignGate.DesignGateReady,
            nativeAttachEntryDesignGate.DesignGateReady,
            nativeDetachBeforeReleaseDesignGate.DesignGateReady,
            nativeOwnerLifecycleDryRun.DryRunReady,
            nativeAttachEntryRuntimeScaffold.RuntimeScaffoldReady,
            nativeOwnerStableIdentity.StableNativeOwnerIdentityReady,
            nativeOwnerStableIdentity.OwnerIdentityDiagnosticsReady,
            nativeOwnerStableIdentity.OwnerIdentityPointerFree,
            nativeOwnerNonCopyableStorage.NativeOwnerNonCopyableReady,
            nativeOwnerNonCopyableStorage.NativeOwnerCopyBlocked,
            nativeOwnerNonCopyableStorage.NativeOwnerMoveBlocked,
            nativeOwnerNonCopyableStorage.NativeOwnerAddressExposed,
            nativeOwnerNonCopyableStorage.NativeOwnerPointerProduced,
            nativeNoThrowDestructor.NoThrowNativeDestructorReady,
            nativeNoThrowDestructor.DestructorNoThrowScaffoldReady,
            nativeNoThrowDestructor.DestructorExceptionEscapeBlocked,
            nativeNoThrowDestructor.DestructorAddressExposed,
            nativeNoThrowDestructor.DestructorPointerProduced,
            nativeOwnerLifecycleGate.LifecycleGateReady,
            nativeOwnerLifecycleGate.ManagedDisposeSnapshotReady,
            nativeOwnerLifecycleGate.LifecycleScaffoldReady,
            nativeOwnerLifecycleGate.ReleaseHookOrderingGateReady,
            nativeOwnerLifecycleGate.DisposeIdempotencyGateReady,
            nativeOwnerLifecycleGate.InFlightDrainGateReady,
            nativeOwnerLifecycleGate.CallbackStateUnpinAfterDetachGateReady,
            nativeOwnerLifecycleGate.DelegateUnpinAfterDetachGateReady,
            nativeOwnerLifecycleGate.LifecycleAddressExposed,
            nativeOwnerLifecycleGate.LifecyclePointerProduced,
            nativeAttachBridgeShapeGate.AttachBridgeShapeGateReady,
            nativeAttachBridgeShapeGate.AttachBridgeShapeReady,
            nativeAttachBridgeShapeGate.AttachBridgeNoThrowBoundaryReady,
            nativeAttachBridgeShapeGate.AttachBridgeVersionGuardReady,
            nativeAttachBridgeShapeGate.AttachBridgeOwnershipDiagnosticsReady,
            nativeAttachBridgeShapeGate.AttachBridgePointerFree,
            nativeAttachBridgeShapeGate.NonNullAttachStillDisabled,
            exceptionStatusMappingGate.ExceptionStatusMappingGateReady,
            exceptionStatusMappingGate.NativeCallbackExceptionCaptureReady,
            exceptionStatusMappingGate.CallbackStatusMappingGateReady,
            exceptionStatusMappingGate.ExceptionEscapeBlocked,
            exceptionStatusMappingGate.DiagnosticCopyReady,
            inFlightAccountingGate.InFlightAccountingGateReady,
            inFlightAccountingGate.CallbackEnterAccountingGateReady,
            inFlightAccountingGate.CallbackLeaveAccountingGateReady,
            inFlightAccountingGate.CallbackInFlightNeverNegativeReady,
            inFlightAccountingGate.ReleaseAfterDrainGateReady,
            inFlightAccountingGate.CallbackStateUnpinAfterDrainGateReady,
            nativeNoThrowVTableScaffoldGate.VTableScaffoldGateReady,
            nativeNoThrowVTableScaffoldGate.NoThrowVTableScaffoldReady,
            nativeNoThrowVTableScaffoldGate.VTableDestructorNoThrowReady,
            nativeNoThrowVTableScaffoldGate.ProcessDebugTensorCallbackStubNoThrowReady,
            nativeNoThrowVTableScaffoldGate.VTableAddressExposed,
            nativeNoThrowVTableScaffoldGate.VTablePointerProduced,
            nativeAttachEntryRuntimeScaffold.AttachEntryParameterShapeReady,
            nativeAttachEntryRuntimeScaffold.AttachEntryNoThrowBoundaryReady,
            nativeAttachEntryRuntimeScaffold.AttachEntryOwnershipDiagnosticsReady,
            borrowedTensorSafetyGate.BorrowedDebugTensorPointerEscapeBlocked,
            nativeAttachEntryRuntimeScaffold.NativeAttachEntryLocated,
            nativeAttachEntryRuntimeScaffold.NativeDetachEntryLocated,
            nativeDetachBeforeReleaseDesignGate.LineSpecificAttachEntryDesignReady,
            nativeDetachBeforeReleaseDesignGate.AttachEntryNoThrowReady,
            nativeDetachBeforeReleaseDesignGate.AttachEntryVersionGuardReady,
            nativeDetachBeforeReleaseDesignGate.AttachEntryOwnershipReady,
            nativeOwnerLifecycleDryRun.DetachBeforeReleaseReady,
            nativeOwnerLifecycleDryRun.ReleaseHookOrderingReady,
            nativeOwnerLifecycleDryRun.DisposeIdempotencyReady,
            nativeOwnerLifecycleDryRun.InFlightDrainBeforeReleaseReady,
            nativeOwnerLifecycleDryRun.CallbackStateUnpinAfterDetachReady,
            nativeOwnerLifecycleDryRun.DelegateUnpinAfterDetachReady,
            nativeOwnerAddressDesignGate.StableNativeOwnerAddressDesignReady,
            nativeOwnerAddressDesignGate.ManagedCallbackKeepAliveDesignReady,
            nativeOwnerNonCopyableStorage.NativeOwnerNonCopyableReady,
            nativeOwnerLifecycleDryRun.NativeOwnerDisposeOrderReady,
            nativeOwnerLifecycleDryRun.NativeOwnerReleaseHookReady,
            nativeOwnerLifecycleDryRun.NativeOwnerInFlightDrainReady,
            nativeNoThrowDestructor.NoThrowNativeDestructorReady,
            nativeNoThrowVTableDesignGate.NoThrowVTableDesignReady,
            nativeNoThrowVTableDesignGate.ExceptionToStatusMappingDesignReady,
            nativeNoThrowVTableDesignGate.NativeVTableTrampolineReady,
            nativeNoThrowVTableDesignGate.CallbackExceptionCaptureReady,
            nativeNoThrowVTableDesignGate.CallbackStatusMappingReady,
            nativeNoThrowVTableDesignGate.CallbackInFlightAccountingReady,
            nativeOwnerLifecycleGate.CanImplementNativeAttach &&
                nativeAttachBridgeShapeGate.CanImplementNativeAttach &&
                nativeNoThrowVTableScaffoldGate.CanImplementNativeAttach,
            borrowedTensorSafetyGate.BorrowedDebugTensorLifetimeReady,
            borrowedTensorSafetyGate.BorrowedDebugTensorDataLifetimeReady,
            borrowedTensorSafetyGate.ProcessDebugTensorRuntimeReady,
            blockers.ToArray());
    }
}
