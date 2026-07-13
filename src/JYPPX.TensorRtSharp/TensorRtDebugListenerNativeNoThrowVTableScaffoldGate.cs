using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener native no-throw vtable scaffold gate evidence.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This gate reports source-visible vtable scaffold, exception/status mapping, and in-flight accounting evidence. It
/// does not install a native <c>IDebugListener</c> into TensorRT and is not proof that TensorRT invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerNativeNoThrowVTableScaffoldGate
{
    /// <summary>
    /// Evaluates no-throw vtable scaffold evidence from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free no-throw vtable scaffold gate result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult attachBridgeShapeGate =
            TensorRtDebugListenerNativeAttachBridgeShapeGate.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerExceptionStatusMappingGateResult exceptionStatusMappingGate =
            TensorRtDebugListenerExceptionStatusMappingGate.Evaluate(ownerDesignSnapshot, attachBridgeShapeGate);
        TensorRtDebugListenerInFlightAccountingGateResult inFlightAccountingGate =
            TensorRtDebugListenerInFlightAccountingGate.Evaluate(ownerDesignSnapshot, exceptionStatusMappingGate);
        return Evaluate(ownerDesignSnapshot, attachBridgeShapeGate, exceptionStatusMappingGate, inFlightAccountingGate);
    }

    /// <summary>
    /// Evaluates no-throw vtable scaffold evidence from copied owner, attach bridge, mapping, and accounting gates.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachBridgeShapeGate">The copied attach bridge shape gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="exceptionStatusMappingGate">The copied exception/status mapping gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="inFlightAccountingGate">The copied in-flight accounting gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free no-throw vtable scaffold gate result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeAttachBridgeShapeGateResult attachBridgeShapeGate,
        TensorRtDebugListenerExceptionStatusMappingGateResult exceptionStatusMappingGate,
        TensorRtDebugListenerInFlightAccountingGateResult inFlightAccountingGate)
    {
        bool nativeAttachBridgeShapeGateReady = attachBridgeShapeGate.AttachBridgeShapeGateReady;
        bool exceptionStatusMappingGateReady = exceptionStatusMappingGate.ExceptionStatusMappingGateReady;
        bool inFlightAccountingGateReady = inFlightAccountingGate.InFlightAccountingGateReady;
        bool noThrowVTableScaffoldReady =
            nativeAttachBridgeShapeGateReady &&
            exceptionStatusMappingGateReady &&
            inFlightAccountingGateReady;
        bool vTableDestructorNoThrowReady = noThrowVTableScaffoldReady;
        bool processDebugTensorCallbackStubNoThrowReady = noThrowVTableScaffoldReady;
        bool exceptionEscapeBlocked =
            exceptionStatusMappingGate.ExceptionEscapeBlocked &&
            processDebugTensorCallbackStubNoThrowReady;
        bool callbackExceptionCaptureGateReady = exceptionStatusMappingGate.NativeCallbackExceptionCaptureReady;
        bool callbackStatusMappingGateReady = exceptionStatusMappingGate.CallbackStatusMappingGateReady;
        bool callbackInFlightAccountingGateReady = inFlightAccountingGate.InFlightAccountingGateReady;
        const bool borrowedDebugTensorPointerEscapeBlocked = true;
        const bool vtableAddressExposed = false;
        const bool vtablePointerProduced = false;
        const bool nativeVTableDesignReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, nativeAttachBridgeShapeGateReady, "debug-listener-native-attach-bridge-shape-gate is not ready for no-throw vtable scaffold evaluation.");
        AddBlockerIfFalse(blockers, exceptionStatusMappingGateReady, "debug-listener-exception-status-mapping-gate is not ready for no-throw vtable scaffold evaluation.");
        AddBlockerIfFalse(blockers, inFlightAccountingGateReady, "debug-listener-inflight-accounting-gate is not ready for no-throw vtable scaffold evaluation.");
        AddBlockerIfFalse(blockers, noThrowVTableScaffoldReady, "native IDebugListener no-throw vtable scaffold is incomplete.");
        AddBlockerIfFalse(blockers, vTableDestructorNoThrowReady, "native IDebugListener vtable destructor no-throw scaffold is incomplete.");
        AddBlockerIfFalse(blockers, processDebugTensorCallbackStubNoThrowReady, "native IDebugListener processDebugTensor no-throw callback stub scaffold is incomplete.");
        AddBlockerIfFalse(blockers, exceptionEscapeBlocked, "native IDebugListener callback exception escape is not blocked.");
        AddBlockerIfFalse(blockers, callbackExceptionCaptureGateReady, "native IDebugListener callback exception capture gate is incomplete.");
        AddBlockerIfFalse(blockers, callbackStatusMappingGateReady, "native IDebugListener callback status mapping gate is incomplete.");
        AddBlockerIfFalse(blockers, callbackInFlightAccountingGateReady, "native IDebugListener callback in-flight accounting gate is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorPointerEscapeBlocked, "borrowed debug tensor pointer escape is not blocked.");
        AddBlockerIfFalse(blockers, !vtableAddressExposed, "native IDebugListener vtable scaffold exposes a native address.");
        AddBlockerIfFalse(blockers, !vtablePointerProduced, "native IDebugListener vtable scaffold produces a native pointer.");
        AddBlockerIfFalse(blockers, attachBridgeShapeGate.NativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, nativeVTableDesignReady, "native IDebugListener vtable implementation is not complete beyond scaffold evidence.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime vtable scaffold evidence.");

        foreach (string blocker in attachBridgeShapeGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in exceptionStatusMappingGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in inFlightAccountingGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult(
            ownerDesignSnapshot.Line,
            attachBridgeShapeGate.OwnerId,
            ownerDesignSnapshot.LastStatus,
            nativeAttachBridgeShapeGateReady,
            exceptionStatusMappingGateReady,
            inFlightAccountingGateReady,
            noThrowVTableScaffoldReady,
            vTableDestructorNoThrowReady,
            processDebugTensorCallbackStubNoThrowReady,
            exceptionEscapeBlocked,
            callbackExceptionCaptureGateReady,
            callbackStatusMappingGateReady,
            callbackInFlightAccountingGateReady,
            borrowedDebugTensorPointerEscapeBlocked,
            vtableAddressExposed,
            vtablePointerProduced,
            attachBridgeShapeGate.NativeAttachEntryLocated,
            nativeVTableDesignReady,
            processDebugTensorRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady,
            blockers.ToArray());
    }

    private static void AddBlockerIfFalse(List<string> blockers, bool condition, string blocker)
    {
        if (!condition)
        {
            AddBlocker(blockers, blocker);
        }
    }

    private static void AddBlocker(List<string> blockers, string blocker)
    {
        if (!string.IsNullOrWhiteSpace(blocker) && !blockers.Contains(blocker))
        {
            blockers.Add(blocker);
        }
    }
}

/// <summary>
/// Reports copied DebugListener native no-throw vtable scaffold diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        bool nativeAttachBridgeShapeGateReady,
        bool exceptionStatusMappingGateReady,
        bool inFlightAccountingGateReady,
        bool noThrowVTableScaffoldReady,
        bool vTableDestructorNoThrowReady,
        bool processDebugTensorCallbackStubNoThrowReady,
        bool exceptionEscapeBlocked,
        bool callbackExceptionCaptureGateReady,
        bool callbackStatusMappingGateReady,
        bool callbackInFlightAccountingGateReady,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool vtableAddressExposed,
        bool vtablePointerProduced,
        bool nativeAttachEntryLocated,
        bool nativeVTableDesignReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        NativeAttachBridgeShapeGateReady = nativeAttachBridgeShapeGateReady;
        ExceptionStatusMappingGateReady = exceptionStatusMappingGateReady;
        InFlightAccountingGateReady = inFlightAccountingGateReady;
        NoThrowVTableScaffoldReady = noThrowVTableScaffoldReady;
        VTableDestructorNoThrowReady = vTableDestructorNoThrowReady;
        ProcessDebugTensorCallbackStubNoThrowReady = processDebugTensorCallbackStubNoThrowReady;
        ExceptionEscapeBlocked = exceptionEscapeBlocked;
        CallbackExceptionCaptureGateReady = callbackExceptionCaptureGateReady;
        CallbackStatusMappingGateReady = callbackStatusMappingGateReady;
        CallbackInFlightAccountingGateReady = callbackInFlightAccountingGateReady;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        VTableAddressExposed = vtableAddressExposed;
        VTablePointerProduced = vtablePointerProduced;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NativeVTableDesignReady = nativeVTableDesignReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-native-nothrow-vtable-scaffold-gate";

    /// <summary>Gets the callback kind represented by this gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "vtable-scaffold-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied diagnostic owner id. This is not a pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the copied last status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets whether attach bridge shape gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachBridgeShapeGateReady { get; }

    /// <summary>Gets whether exception/status mapping gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ExceptionStatusMappingGateReady { get; }

    /// <summary>Gets whether in-flight accounting gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool InFlightAccountingGateReady { get; }

    /// <summary>Gets whether source-visible no-throw vtable scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowVTableScaffoldReady { get; }

    /// <summary>Gets whether vtable destructor no-throw scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTableDestructorNoThrowReady { get; }

    /// <summary>Gets whether processDebugTensor callback stub no-throw scaffold evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorCallbackStubNoThrowReady { get; }

    /// <summary>Gets whether callback exceptions are blocked from crossing the C ABI. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ExceptionEscapeBlocked { get; }

    /// <summary>Gets whether callback exception capture gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackExceptionCaptureGateReady { get; }

    /// <summary>Gets whether callback status mapping gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStatusMappingGateReady { get; }

    /// <summary>Gets whether callback in-flight accounting gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CallbackInFlightAccountingGateReady { get; }

    /// <summary>Gets whether borrowed debug tensor pointer escape remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether the vtable scaffold exposes a native address. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTableAddressExposed { get; }

    /// <summary>Gets whether the vtable scaffold produces a native pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTablePointerProduced { get; }

    /// <summary>Gets whether a native non-null DebugListener attach entry has been located. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether native IDebugListener vtable implementation is complete. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableDesignReady { get; }

    /// <summary>Gets whether vtable scaffold gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool VTableScaffoldGateReady =>
        NativeAttachBridgeShapeGateReady &&
        ExceptionStatusMappingGateReady &&
        InFlightAccountingGateReady &&
        NoThrowVTableScaffoldReady &&
        VTableDestructorNoThrowReady &&
        ProcessDebugTensorCallbackStubNoThrowReady &&
        ExceptionEscapeBlocked &&
        CallbackExceptionCaptureGateReady &&
        CallbackStatusMappingGateReady &&
        CallbackInFlightAccountingGateReady &&
        BorrowedDebugTensorPointerEscapeBlocked &&
        !VTableAddressExposed &&
        !VTablePointerProduced &&
        LastStatus == BridgeStatusCode.Ok;

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether the native attach bridge can be implemented without unresolved blockers. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanImplementNativeAttach =>
        VTableScaffoldGateReady &&
        NativeAttachEntryLocated &&
        NativeVTableDesignReady;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof =>
        CanImplementNativeAttach &&
        ProcessDebugTensorRuntimeReady &&
        FullPackageConsumerRuntimeEvidenceReady;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the vtable scaffold gate status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => VTableScaffoldGateReady ? "vtable-scaffold-gate-ready" : "vtable-scaffold-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-native-nothrow-vtable-scaffold-gate; RuntimeEvidenceKind=vtable-scaffold-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; VTableScaffoldGateReady=" + VTableScaffoldGateReady + "; " +
        "NativeAttachBridgeShapeGateReady=" + NativeAttachBridgeShapeGateReady + "; " +
        "ExceptionStatusMappingGateReady=" + ExceptionStatusMappingGateReady + "; " +
        "InFlightAccountingGateReady=" + InFlightAccountingGateReady + "; " +
        "NoThrowVTableScaffoldReady=" + NoThrowVTableScaffoldReady + "; " +
        "VTableDestructorNoThrowReady=" + VTableDestructorNoThrowReady + "; " +
        "ProcessDebugTensorCallbackStubNoThrowReady=" + ProcessDebugTensorCallbackStubNoThrowReady + "; " +
        "ExceptionEscapeBlocked=" + ExceptionEscapeBlocked + "; " +
        "CallbackExceptionCaptureGateReady=" + CallbackExceptionCaptureGateReady + "; " +
        "CallbackStatusMappingGateReady=" + CallbackStatusMappingGateReady + "; " +
        "CallbackInFlightAccountingGateReady=" + CallbackInFlightAccountingGateReady + "; " +
        "BorrowedDebugTensorPointerEscapeBlocked=" + BorrowedDebugTensorPointerEscapeBlocked + "; " +
        "VTableAddressExposed=" + VTableAddressExposed + "; " +
        "VTablePointerProduced=" + VTablePointerProduced + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NativeVTableDesignReady=" + NativeVTableDesignReady + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:vtable={NativeVTableDesignReady}:proof={IsRealCallbackRuntimeProof}";
    }
}
