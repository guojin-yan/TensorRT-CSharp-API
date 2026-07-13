using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free DebugListener attach/detach design gate before any real TensorRT callback bridge exists.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This gate describes lifecycle readiness only. It does not call <c>setDebugListener</c> with a non-null listener,
/// does not expose a native listener pointer, and is not proof that <c>IDebugListener::processDebugTensor</c> ran.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public static class TensorRtDebugListenerAttachDetachDesignGate
{
    /// <summary>
    /// Evaluates the attach/detach design gate from a copied DebugListener owner design snapshot.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free attach/detach design result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerAttachDetachDesignGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        bool lineSupportsDebugListener =
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt10 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt11;
        bool ownerDesignReady =
            string.Equals(ownerDesignSnapshot.EvidenceKind, "debug-listener-callback-owner-design", StringComparison.Ordinal) &&
            string.Equals(ownerDesignSnapshot.RuntimeEvidenceKind, "not-present", StringComparison.Ordinal) &&
            !ownerDesignSnapshot.RealCallbackRuntime &&
            !ownerDesignSnapshot.IsRealCallbackRuntimeProof &&
            ownerDesignSnapshot.LastStatus == BridgeStatusCode.Ok &&
            ownerDesignSnapshot.FailureCount == 0 &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool managedOwnerStateMachineReady =
            ownerDesignSnapshot.DisposeRequested &&
            ownerDesignSnapshot.ReleaseHookCount > 0 &&
            !ownerDesignSnapshot.CallbackStatePinned &&
            !ownerDesignSnapshot.DelegatePinned &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool metadataCopyReady =
            ownerDesignSnapshot.DebugTensorMetadataCopied &&
            !string.IsNullOrWhiteSpace(ownerDesignSnapshot.TensorName) &&
            ownerDesignSnapshot.ShapeRank >= 0;
        bool pointerFreeSurfaceReady =
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;

        bool detachClearControlAvailable = lineSupportsDebugListener;
        const bool attachControlAvailable = false;
        const bool stableNativeOwnerAddressReady = false;
        const bool noThrowNativeVTableReady = false;
        const bool borrowedDebugTensorLifetimeReady = false;

        List<string> blockers = new List<string>();
        if (!lineSupportsDebugListener)
        {
            blockers.Add("TensorRT 10 or 11 IDebugListener line support has not been selected.");
        }

        if (!ownerDesignReady)
        {
            blockers.Add("debug-listener-callback-owner-design snapshot is not clean owner-design evidence.");
        }

        if (!managedOwnerStateMachineReady)
        {
            blockers.Add("managed DebugListener owner state machine has not shown dispose, release hook, unpin, and in-flight drain evidence.");
        }

        if (!metadataCopyReady)
        {
            blockers.Add("debug tensor metadata copy-out is incomplete.");
        }

        if (!pointerFreeSurfaceReady)
        {
            blockers.Add("public DebugListener attach/detach surface still exposes or produces a borrowed debug tensor pointer.");
        }

        if (!detachClearControlAvailable)
        {
            blockers.Add("line-specific setDebugListener(nullptr) detach/clear control is not available.");
        }

        if (!attachControlAvailable)
        {
            blockers.Add("line-specific setDebugListener(non-null) attach bridge is not implemented.");
        }

        if (!stableNativeOwnerAddressReady)
        {
            blockers.Add("native DebugListener owner stable address is not implemented.");
        }

        if (!noThrowNativeVTableReady)
        {
            blockers.Add("native IDebugListener no-throw vtable trampoline is not implemented.");
        }

        if (!borrowedDebugTensorLifetimeReady)
        {
            blockers.Add("borrowed debug tensor pointer and data buffer lifetime rules are not implemented.");
        }

        return new TensorRtDebugListenerAttachDetachDesignGateResult(
            ownerDesignSnapshot.Line,
            lineSupportsDebugListener,
            ownerDesignReady,
            managedOwnerStateMachineReady,
            metadataCopyReady,
            pointerFreeSurfaceReady,
            detachClearControlAvailable,
            attachControlAvailable,
            stableNativeOwnerAddressReady,
            noThrowNativeVTableReady,
            borrowedDebugTensorLifetimeReady,
            blockers.ToArray());
    }
}

/// <summary>
/// Reports copied DebugListener attach/detach design diagnostics without exposing native pointers.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This result is a design gate. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> remain <see langword="false"/> until full package consumer smoke produces
/// complete <c>real-callback-runtime</c> evidence.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public readonly struct TensorRtDebugListenerAttachDetachDesignGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerAttachDetachDesignGateResult(
        TensorRtApiLine line,
        bool lineSupportsDebugListener,
        bool ownerDesignReady,
        bool managedOwnerStateMachineReady,
        bool debugTensorMetadataCopied,
        bool pointerFreeSurfaceReady,
        bool detachClearControlAvailable,
        bool attachControlAvailable,
        bool stableNativeOwnerAddressReady,
        bool noThrowNativeVTableReady,
        bool borrowedDebugTensorLifetimeReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        LineSupportsDebugListener = lineSupportsDebugListener;
        OwnerDesignReady = ownerDesignReady;
        ManagedOwnerStateMachineReady = managedOwnerStateMachineReady;
        DebugTensorMetadataCopied = debugTensorMetadataCopied;
        PointerFreeSurfaceReady = pointerFreeSurfaceReady;
        DetachClearControlAvailable = detachClearControlAvailable;
        AttachControlAvailable = attachControlAvailable;
        StableNativeOwnerAddressReady = stableNativeOwnerAddressReady;
        NoThrowNativeVTableReady = noThrowNativeVTableReady;
        BorrowedDebugTensorLifetimeReady = borrowedDebugTensorLifetimeReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string EvidenceKind => "debug-listener-attach-detach-design-gate";

    /// <summary>Gets the callback kind represented by this gate. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string CallbackKind => "debug-listener-attach-detach";

    /// <summary>Gets the runtime evidence kind. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this gate proves a real TensorRT callback runtime. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this gate as real callback runtime proof. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected TensorRT line has DebugListener API support. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool LineSupportsDebugListener { get; }

    /// <summary>Gets whether the owner design snapshot was clean owner-design evidence. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool OwnerDesignReady { get; }

    /// <summary>Gets whether the managed owner state machine has clean dispose and in-flight drain evidence. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool ManagedOwnerStateMachineReady { get; }

    /// <summary>Gets whether debug tensor metadata was copied before the gate was evaluated. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool DebugTensorMetadataCopied { get; }

    /// <summary>Gets whether the public surface remains pointer-free. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool PointerFreeSurfaceReady { get; }

    /// <summary>Gets whether TensorRT 10/11 detach/clear control exists through <c>setDebugListener(nullptr)</c>. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool DetachClearControlAvailable { get; }

    /// <summary>Gets whether a non-null DebugListener attach bridge is available. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachControlAvailable { get; }

    /// <summary>Gets whether the line-specific attach/detach pair is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool LineSpecificAttachDetachReady => AttachControlAvailable && DetachClearControlAvailable;

    /// <summary>Gets whether a stable native owner address is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool StableNativeOwnerAddressReady { get; }

    /// <summary>Gets whether a no-throw native vtable trampoline is ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NoThrowNativeVTableReady { get; }

    /// <summary>Gets whether native vtable readiness is complete. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeVTableReady => StableNativeOwnerAddressReady && NoThrowNativeVTableReady;

    /// <summary>Gets whether borrowed debug tensor/data lifetime rules are ready. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool BorrowedDebugTensorLifetimeReady { get; }

    /// <summary>Gets whether this design gate has enough copied evidence to feed the runtime-proof precheck. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool DesignGateReady =>
        LineSupportsDebugListener &&
        OwnerDesignReady &&
        ManagedOwnerStateMachineReady &&
        DebugTensorMetadataCopied &&
        PointerFreeSurfaceReady &&
        DetachClearControlAvailable;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool CanAttemptRuntimeProof =>
        DesignGateReady &&
        LineSpecificAttachDetachReady &&
        NativeVTableReady &&
        BorrowedDebugTensorLifetimeReady;

    /// <summary>Gets whether real runtime proof is still blocked. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets the copied list of prerequisites that still block real runtime proof. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied number of blocked prerequisites. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the design gate status. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Status => DesignGateReady ? "design-gate-ready" : "design-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Diagnostic =>
        "debug-listener-attach-detach-design-gate; RuntimeEvidenceKind=design-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; AttachControlAvailable=" + AttachControlAvailable + "; " +
        "DetachClearControlAvailable=" + DetachClearControlAvailable + "; " +
        "LineSpecificAttachDetachReady=" + LineSpecificAttachDetachReady + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    /// <returns>A diagnostic string. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:attach={AttachControlAvailable}:detach={DetachClearControlAvailable}:proof={IsRealCallbackRuntimeProof}";
    }
}
