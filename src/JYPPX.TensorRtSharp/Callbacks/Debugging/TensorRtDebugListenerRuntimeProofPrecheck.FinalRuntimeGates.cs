using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public static partial class TensorRtDebugListenerRuntimeProofPrecheck
{
    /// <summary>
    /// Evaluates the current DebugListener runtime proof precheck from copied owner, gate, preflight, owner address,
    /// no-throw vtable, attach entry, detach-before-release design, native owner lifecycle dry-run, native attach entry
    /// runtime scaffold, stable owner identity, and non-copyable storage evidence.
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
        TensorRtDebugListenerNativeOwnerNonCopyableStorageResult nativeOwnerNonCopyableStorage)
    {
        TensorRtDebugListenerNativeNoThrowDestructorResult nativeNoThrowDestructor =
            TensorRtDebugListenerNativeNoThrowDestructor.Evaluate(ownerDesignSnapshot, nativeOwnerNonCopyableStorage);
        return Evaluate(
            ownerDesignSnapshot,
            attachDetachDesignGate,
            borrowedTensorSafetyGate,
            attachVTableSafetyGate,
            nativeAttachNoThrowPreflight,
            nativeOwnerAddressDesignGate,
            nativeNoThrowVTableDesignGate,
            nativeAttachEntryDesignGate,
            nativeDetachBeforeReleaseDesignGate,
            nativeOwnerLifecycleDryRun,
            nativeAttachEntryRuntimeScaffold,
            nativeOwnerStableIdentity,
            nativeOwnerNonCopyableStorage,
            nativeNoThrowDestructor);
    }

    /// <summary>
    /// Evaluates the current DebugListener runtime proof precheck from copied owner, gate, preflight, owner address,
    /// no-throw vtable, attach entry, detach-before-release design, native owner lifecycle dry-run, native attach entry
    /// runtime scaffold, stable owner identity, non-copyable storage, and no-throw destructor evidence.
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
        TensorRtDebugListenerNativeNoThrowDestructorResult nativeNoThrowDestructor)
    {
        TensorRtDebugListenerNativeOwnerLifecycleGateResult nativeOwnerLifecycleGate =
            TensorRtDebugListenerNativeOwnerLifecycleGate.Evaluate(ownerDesignSnapshot, nativeNoThrowDestructor);
        return Evaluate(
            ownerDesignSnapshot,
            attachDetachDesignGate,
            borrowedTensorSafetyGate,
            attachVTableSafetyGate,
            nativeAttachNoThrowPreflight,
            nativeOwnerAddressDesignGate,
            nativeNoThrowVTableDesignGate,
            nativeAttachEntryDesignGate,
            nativeDetachBeforeReleaseDesignGate,
            nativeOwnerLifecycleDryRun,
            nativeAttachEntryRuntimeScaffold,
            nativeOwnerStableIdentity,
            nativeOwnerNonCopyableStorage,
            nativeNoThrowDestructor,
            nativeOwnerLifecycleGate);
    }

    /// <summary>
    /// Evaluates the current DebugListener runtime proof precheck from copied owner, gate, preflight, owner address,
    /// no-throw vtable, attach entry, detach-before-release design, native owner lifecycle dry-run, native attach entry
    /// runtime scaffold, stable owner identity, non-copyable storage, no-throw destructor, and lifecycle gate evidence.
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
        TensorRtDebugListenerNativeOwnerLifecycleGateResult nativeOwnerLifecycleGate)
    {
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult nativeAttachBridgeShapeGate =
            TensorRtDebugListenerNativeAttachBridgeShapeGate.Evaluate(ownerDesignSnapshot, nativeOwnerLifecycleGate);
        TensorRtDebugListenerExceptionStatusMappingGateResult exceptionStatusMappingGate =
            TensorRtDebugListenerExceptionStatusMappingGate.Evaluate(ownerDesignSnapshot, nativeAttachBridgeShapeGate);
        TensorRtDebugListenerInFlightAccountingGateResult inFlightAccountingGate =
            TensorRtDebugListenerInFlightAccountingGate.Evaluate(ownerDesignSnapshot, exceptionStatusMappingGate);
        TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult nativeNoThrowVTableScaffoldGate =
            TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.Evaluate(
                ownerDesignSnapshot,
                nativeAttachBridgeShapeGate,
                exceptionStatusMappingGate,
                inFlightAccountingGate);
        return Evaluate(
            ownerDesignSnapshot,
            attachDetachDesignGate,
            borrowedTensorSafetyGate,
            attachVTableSafetyGate,
            nativeAttachNoThrowPreflight,
            nativeOwnerAddressDesignGate,
            nativeNoThrowVTableDesignGate,
            nativeAttachEntryDesignGate,
            nativeDetachBeforeReleaseDesignGate,
            nativeOwnerLifecycleDryRun,
            nativeAttachEntryRuntimeScaffold,
            nativeOwnerStableIdentity,
            nativeOwnerNonCopyableStorage,
            nativeNoThrowDestructor,
            nativeOwnerLifecycleGate,
            nativeAttachBridgeShapeGate,
            exceptionStatusMappingGate,
            inFlightAccountingGate,
            nativeNoThrowVTableScaffoldGate);
    }

}
