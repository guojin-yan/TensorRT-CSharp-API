using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates copied DebugListener attach/vtable prerequisites before a native no-throw callback bridge exists.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This safety gate is pointer-free. It does not attach a non-null <c>IDebugListener</c>, does not expose a native
/// listener owner address, and does not prove that TensorRT invoked <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerAttachVTableSafetyGate
{
    /// <summary>
    /// Evaluates attach/vtable safety from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free attach/vtable safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerAttachVTableSafetyGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachGate =
            TensorRtDebugListenerAttachDetachDesignGate.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorGate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachGate);
        return Evaluate(ownerDesignSnapshot, attachDetachGate, borrowedTensorGate);
    }

    /// <summary>
    /// Evaluates attach/vtable safety from copied owner and attach/detach design evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free attach/vtable safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerAttachVTableSafetyGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate)
    {
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorGate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachDesignGate);
        return Evaluate(ownerDesignSnapshot, attachDetachDesignGate, borrowedTensorGate);
    }

    /// <summary>
    /// Evaluates attach/vtable safety from copied owner, attach/detach, and borrowed tensor safety evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free attach/vtable safety result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerAttachVTableSafetyGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate)
    {
        bool ownerDesignReady =
            string.Equals(ownerDesignSnapshot.EvidenceKind, "debug-listener-callback-owner-design", StringComparison.Ordinal) &&
            string.Equals(ownerDesignSnapshot.RuntimeEvidenceKind, "not-present", StringComparison.Ordinal) &&
            !ownerDesignSnapshot.RealCallbackRuntime &&
            !ownerDesignSnapshot.IsRealCallbackRuntimeProof &&
            ownerDesignSnapshot.LastStatus == BridgeStatusCode.Ok &&
            ownerDesignSnapshot.FailureCount == 0 &&
            ownerDesignSnapshot.InFlightCallbackCount == 0 &&
            ownerDesignSnapshot.ProcessDebugTensorCount > 0;
        bool debugTensorMetadataCopied =
            ownerDesignSnapshot.DebugTensorMetadataCopied &&
            attachDetachDesignGate.DebugTensorMetadataCopied &&
            borrowedTensorSafetyGate.DebugTensorMetadataCopied &&
            !string.IsNullOrWhiteSpace(ownerDesignSnapshot.TensorName) &&
            ownerDesignSnapshot.ShapeRank >= 0;
        bool pointerFreeSurfaceReady =
            attachDetachDesignGate.PointerFreeSurfaceReady &&
            borrowedTensorSafetyGate.PointerFreeSurfaceReady &&
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;

        bool lineSupportsDebugListener = attachDetachDesignGate.LineSupportsDebugListener;
        bool attachDetachDesignGateReady = attachDetachDesignGate.DesignGateReady;
        bool borrowedTensorSafetyGateReady = borrowedTensorSafetyGate.SafetyGateReady;
        bool managedOwnerStateMachineReady = attachDetachDesignGate.ManagedOwnerStateMachineReady;
        bool borrowedDebugTensorPointerEscapeBlocked = borrowedTensorSafetyGate.BorrowedDebugTensorPointerEscapeBlocked;
        bool detachClearControlAvailable = attachDetachDesignGate.DetachClearControlAvailable;

        const bool attachControlAvailable = false;
        const bool stableNativeOwnerAddressReady = false;
        const bool noThrowNativeVTableReady = false;
        const bool exceptionToStatusMappingReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, lineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, ownerDesignReady, "debug-listener-callback-owner-design snapshot is not clean owner-design evidence.");
        AddBlockerIfFalse(blockers, attachDetachDesignGateReady, "debug-listener-attach-detach-design-gate is not ready for attach/vtable safety evaluation.");
        AddBlockerIfFalse(blockers, borrowedTensorSafetyGateReady, "debug-listener-borrowed-tensor-safety-gate is not ready for attach/vtable safety evaluation.");
        AddBlockerIfFalse(blockers, managedOwnerStateMachineReady, "managed DebugListener owner state machine has not shown dispose, release hook, unpin, and in-flight drain evidence.");
        AddBlockerIfFalse(blockers, debugTensorMetadataCopied, "debug tensor copied metadata is incomplete.");
        AddBlockerIfFalse(blockers, pointerFreeSurfaceReady, "public DebugListener attach/vtable surface still exposes, produces, or leaks a debug tensor pointer.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorPointerEscapeBlocked, "borrowed debug tensor pointer escape is not blocked.");
        AddBlockerIfFalse(blockers, detachClearControlAvailable, "line-specific setDebugListener(nullptr) detach/clear control is not available.");
        AddBlockerIfFalse(blockers, attachControlAvailable, "line-specific setDebugListener(non-null) attach bridge is not implemented.");
        AddBlockerIfFalse(blockers, stableNativeOwnerAddressReady, "stable native DebugListener owner address is not implemented.");
        AddBlockerIfFalse(blockers, noThrowNativeVTableReady, "native IDebugListener no-throw vtable trampoline is not implemented.");
        AddBlockerIfFalse(blockers, exceptionToStatusMappingReady, "exception-to-status mapping for the native DebugListener vtable bridge is not proven.");
        AddBlockerIfFalse(blockers, borrowedTensorSafetyGate.BorrowedDebugTensorLifetimeReady, "borrowed debug tensor pointer lifetime has not been proven against a real TensorRT callback.");
        AddBlockerIfFalse(blockers, borrowedTensorSafetyGate.BorrowedDebugTensorDataLifetimeReady, "borrowed debug tensor data buffer lifetime has not been proven against a real TensorRT callback.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime debug listener attach/vtable evidence.");

        foreach (string blocker in attachDetachDesignGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in borrowedTensorSafetyGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerAttachVTableSafetyGateResult(
            ownerDesignSnapshot.Line,
            lineSupportsDebugListener,
            ownerDesignReady,
            attachDetachDesignGateReady,
            borrowedTensorSafetyGateReady,
            managedOwnerStateMachineReady,
            debugTensorMetadataCopied,
            pointerFreeSurfaceReady,
            borrowedDebugTensorPointerEscapeBlocked,
            detachClearControlAvailable,
            attachControlAvailable,
            stableNativeOwnerAddressReady,
            noThrowNativeVTableReady,
            exceptionToStatusMappingReady,
            borrowedTensorSafetyGate.BorrowedDebugTensorLifetimeReady,
            borrowedTensorSafetyGate.BorrowedDebugTensorDataLifetimeReady,
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
/// Reports copied DebugListener attach/vtable safety diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This result is a safety gate. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> remain <see langword="false"/> until a native no-throw vtable bridge and
/// full package consumer <c>real-callback-runtime</c> smoke evidence exist.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public readonly struct TensorRtDebugListenerAttachVTableSafetyGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerAttachVTableSafetyGateResult(
        TensorRtApiLine line,
        bool lineSupportsDebugListener,
        bool ownerDesignReady,
        bool attachDetachDesignGateReady,
        bool borrowedTensorSafetyGateReady,
        bool managedOwnerStateMachineReady,
        bool debugTensorMetadataCopied,
        bool pointerFreeSurfaceReady,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool detachClearControlAvailable,
        bool attachControlAvailable,
        bool stableNativeOwnerAddressReady,
        bool noThrowNativeVTableReady,
        bool exceptionToStatusMappingReady,
        bool borrowedDebugTensorLifetimeReady,
        bool borrowedDebugTensorDataLifetimeReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        LineSupportsDebugListener = lineSupportsDebugListener;
        OwnerDesignReady = ownerDesignReady;
        AttachDetachDesignGateReady = attachDetachDesignGateReady;
        BorrowedTensorSafetyGateReady = borrowedTensorSafetyGateReady;
        ManagedOwnerStateMachineReady = managedOwnerStateMachineReady;
        DebugTensorMetadataCopied = debugTensorMetadataCopied;
        PointerFreeSurfaceReady = pointerFreeSurfaceReady;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        DetachClearControlAvailable = detachClearControlAvailable;
        AttachControlAvailable = attachControlAvailable;
        StableNativeOwnerAddressReady = stableNativeOwnerAddressReady;
        NoThrowNativeVTableReady = noThrowNativeVTableReady;
        ExceptionToStatusMappingReady = exceptionToStatusMappingReady;
        BorrowedDebugTensorLifetimeReady = borrowedDebugTensorLifetimeReady;
        BorrowedDebugTensorDataLifetimeReady = borrowedDebugTensorDataLifetimeReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this safety gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-attach-vtable-safety-gate";

    /// <summary>Gets the callback kind represented by this safety gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-attach-vtable";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected TensorRT line has DebugListener API support. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool LineSupportsDebugListener { get; }

    /// <summary>Gets whether owner design evidence was clean. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool OwnerDesignReady { get; }

    /// <summary>Gets whether attach/detach design evidence was ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttachDetachDesignGateReady { get; }

    /// <summary>Gets whether borrowed tensor safety evidence was ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedTensorSafetyGateReady { get; }

    /// <summary>Gets whether the managed owner state machine has clean dispose and in-flight drain evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ManagedOwnerStateMachineReady { get; }

    /// <summary>Gets whether debug tensor metadata was copied before the gate was evaluated. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorMetadataCopied { get; }

    /// <summary>Gets whether the public surface remains pointer-free. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool PointerFreeSurfaceReady { get; }

    /// <summary>Gets whether borrowed debug tensor pointers are blocked from escaping public APIs. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether TensorRT 10/11 detach/clear control exists through <c>setDebugListener(nullptr)</c>. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DetachClearControlAvailable { get; }

    /// <summary>Gets whether a non-null DebugListener attach bridge is available. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttachControlAvailable { get; }

    /// <summary>Gets whether the line-specific attach/detach pair is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool LineSpecificAttachDetachReady => AttachControlAvailable && DetachClearControlAvailable;

    /// <summary>Gets whether a stable native owner address is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool StableNativeOwnerAddressReady { get; }

    /// <summary>Gets whether a no-throw native vtable trampoline is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowNativeVTableReady { get; }

    /// <summary>Gets whether native vtable readiness is complete. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableReady => StableNativeOwnerAddressReady && NoThrowNativeVTableReady;

    /// <summary>Gets whether managed exceptions are mapped to native status without crossing the C ABI. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ExceptionToStatusMappingReady { get; }

    /// <summary>Gets whether borrowed debug tensor pointer lifetime is runtime ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorLifetimeReady { get; }

    /// <summary>Gets whether borrowed debug tensor data buffer lifetime is runtime ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataLifetimeReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether this safety gate has enough copied evidence to feed the runtime-proof precheck. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool SafetyGateReady =>
        LineSupportsDebugListener &&
        OwnerDesignReady &&
        AttachDetachDesignGateReady &&
        BorrowedTensorSafetyGateReady &&
        ManagedOwnerStateMachineReady &&
        DebugTensorMetadataCopied &&
        PointerFreeSurfaceReady &&
        BorrowedDebugTensorPointerEscapeBlocked &&
        DetachClearControlAvailable;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof =>
        SafetyGateReady &&
        LineSpecificAttachDetachReady &&
        NativeVTableReady &&
        ExceptionToStatusMappingReady &&
        BorrowedDebugTensorLifetimeReady &&
        BorrowedDebugTensorDataLifetimeReady &&
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

    /// <summary>Gets the safety gate status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => SafetyGateReady ? "safety-gate-ready" : "safety-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-attach-vtable-safety-gate; RuntimeEvidenceKind=design-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; SafetyGateReady=" + SafetyGateReady + "; " +
        "AttachControlAvailable=" + AttachControlAvailable + "; " +
        "StableNativeOwnerAddressReady=" + StableNativeOwnerAddressReady + "; " +
        "NoThrowNativeVTableReady=" + NoThrowNativeVTableReady + "; " +
        "ExceptionToStatusMappingReady=" + ExceptionToStatusMappingReady + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:attach={AttachControlAvailable}:vtable={NativeVTableReady}:proof={IsRealCallbackRuntimeProof}";
    }
}
