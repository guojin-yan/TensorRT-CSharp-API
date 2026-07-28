using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates whether the OutputAllocator owner design snapshot is ready for the next real callback runtime proof stage.
/// 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。
/// </summary>
/// <remarks>
/// This precheck consumes copied diagnostics only. It does not attach an output allocator to TensorRT, does not expose
/// an output buffer or device pointer, and does not prove that <c>IOutputAllocator::notifyShape</c> or
/// <c>IOutputAllocator::reallocateOutput</c> has been invoked by a real TensorRT build/enqueue path.
/// 该说明强调当前结果只是 OutputAllocator 安全门禁或 precheck，不代表真实 TensorRT callback runtime proof。
/// </remarks>
public static class TensorRtOutputAllocatorRuntimeProofPrecheck
{
    /// <summary>
    /// Evaluates the current OutputAllocator runtime proof precheck from a copied owner design snapshot.
    /// 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied OutputAllocator owner design snapshot. 该参数传入已复制的 OutputAllocator owner、gate 或 runtime proof precheck evidence。</param>
    /// <returns>A pointer-free precheck result. 返回不暴露 native output buffer 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtOutputAllocatorRuntimeProofPrecheckResult Evaluate(
        TensorRtOutputAllocatorCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtOutputAllocatorAttachDetachDesignGateResult attachDetachDesignGate =
            TensorRtOutputAllocatorAttachDetachDesignGate.Evaluate(ownerDesignSnapshot);
        TensorRtOutputBufferOwnershipSafetyGateResult outputBufferOwnershipSafetyGate =
            TensorRtOutputBufferOwnershipSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachDesignGate);
        return Evaluate(ownerDesignSnapshot, attachDetachDesignGate, outputBufferOwnershipSafetyGate);
    }

    /// <summary>
    /// Evaluates the current OutputAllocator runtime proof precheck from copied owner and attach/detach design evidence.
    /// 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied OutputAllocator owner design snapshot. 该参数传入已复制的 OutputAllocator owner、gate 或 runtime proof precheck evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 OutputAllocator owner、gate 或 runtime proof precheck evidence。</param>
    /// <returns>A pointer-free precheck result. 返回不暴露 native output buffer 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtOutputAllocatorRuntimeProofPrecheckResult Evaluate(
        TensorRtOutputAllocatorCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtOutputAllocatorAttachDetachDesignGateResult attachDetachDesignGate)
    {
        TensorRtOutputBufferOwnershipSafetyGateResult outputBufferOwnershipSafetyGate =
            TensorRtOutputBufferOwnershipSafetyGate.Evaluate(ownerDesignSnapshot, attachDetachDesignGate);
        return Evaluate(ownerDesignSnapshot, attachDetachDesignGate, outputBufferOwnershipSafetyGate);
    }

    /// <summary>
    /// Evaluates the current OutputAllocator runtime proof precheck from copied owner, attach/detach, and ownership safety evidence.
    /// 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied OutputAllocator owner design snapshot. 该参数传入已复制的 OutputAllocator owner、gate 或 runtime proof precheck evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 OutputAllocator owner、gate 或 runtime proof precheck evidence。</param>
    /// <param name="outputBufferOwnershipSafetyGate">The copied output-buffer ownership safety gate result. 该参数传入已复制的 OutputAllocator owner、gate 或 runtime proof precheck evidence。</param>
    /// <returns>A pointer-free precheck result. 返回不暴露 native output buffer 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtOutputAllocatorRuntimeProofPrecheckResult Evaluate(
        TensorRtOutputAllocatorCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtOutputAllocatorAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtOutputBufferOwnershipSafetyGateResult outputBufferOwnershipSafetyGate)
    {
        bool ownerDesignReady =
            string.Equals(ownerDesignSnapshot.EvidenceKind, "output-allocator-callback-owner-design", StringComparison.Ordinal) &&
            string.Equals(ownerDesignSnapshot.RuntimeEvidenceKind, "not-present", StringComparison.Ordinal) &&
            !ownerDesignSnapshot.RealCallbackRuntime &&
            !ownerDesignSnapshot.IsRealCallbackRuntimeProof &&
            ownerDesignSnapshot.RuntimeGateStatus == BridgeStatusCode.Ok &&
            ownerDesignSnapshot.FailureCount == 0 &&
            ownerDesignSnapshot.InFlightCallbackCount == 0 &&
            ownerDesignSnapshot.NotifyShapeCount > 0 &&
            ownerDesignSnapshot.ReallocateOutputCount > 0;
        bool nativeLedgerDesignReady =
            ownerDesignSnapshot.NativeLedgerAvailable &&
            ownerDesignSnapshot.NativeLedgerStatus == BridgeStatusCode.Ok &&
            ownerDesignSnapshot.StateTransitionCount >= 2UL &&
            ownerDesignSnapshot.LedgerAllocationCount > 0UL &&
            ownerDesignSnapshot.LedgerAllocationCount == ownerDesignSnapshot.LedgerReleaseCount &&
            ownerDesignSnapshot.LedgerFailureCount == 0UL &&
            !ownerDesignSnapshot.HasLiveAllocation;
        bool disposeReleaseReady =
            ownerDesignSnapshot.DisposeRequested &&
            ownerDesignSnapshot.ReleaseHookCount > 0 &&
            !ownerDesignSnapshot.CallbackStatePinned &&
            !ownerDesignSnapshot.DelegatePinned &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool pointerFreeSurfaceReady =
            !ownerDesignSnapshot.OutputBufferPointerExposed &&
            !ownerDesignSnapshot.OutputBufferPointerProduced;
        const bool devicePointerLedgerRuntimeReady = false;
        const bool streamLifetimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        if (!attachDetachDesignGate.LineSupportsOutputAllocator)
        {
            blockers.Add("TensorRT 8, 10, or 11 IOutputAllocator line support has not been selected.");
        }

        if (!ownerDesignReady)
        {
            blockers.Add("output-allocator-callback-owner-design snapshot is not clean owner-design evidence.");
        }

        if (!nativeLedgerDesignReady)
        {
            blockers.Add("native allocator owner ledger dry-run evidence is not available or not clean.");
        }

        if (!disposeReleaseReady)
        {
            blockers.Add("dispose release hook evidence is not present on the owner design snapshot.");
        }

        if (!pointerFreeSurfaceReady)
        {
            blockers.Add("public OutputAllocator surface still exposes or produces an output buffer pointer.");
        }

        foreach (string blocker in attachDetachDesignGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        if (!devicePointerLedgerRuntimeReady)
        {
            blockers.Add("runtime device pointer ownership ledger is not implemented.");
        }

        if (!streamLifetimeReady)
        {
            blockers.Add("CUDA stream lifetime and async allocation semantics are not implemented.");
        }

        foreach (string blocker in outputBufferOwnershipSafetyGate.BlockedPrerequisites)
        {
            if (!blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }

        if (!fullPackageConsumerRuntimeEvidenceReady)
        {
            blockers.Add("full package consumer smoke has not emitted real-callback-runtime evidence.");
        }

        return new TensorRtOutputAllocatorRuntimeProofPrecheckResult(
            ownerDesignSnapshot.Line,
            attachDetachDesignGate.LineSupportsOutputAllocator,
            ownerDesignReady,
            nativeLedgerDesignReady,
            disposeReleaseReady,
            pointerFreeSurfaceReady,
            attachDetachDesignGate.DesignGateReady,
            attachDetachDesignGate.AttachControlAvailable,
            attachDetachDesignGate.DetachClearControlAvailable,
            attachDetachDesignGate.ManagedOwnerStateMachineReady,
            attachDetachDesignGate.StableNativeOwnerAddressReady,
            attachDetachDesignGate.NoThrowNativeVTableReady,
            outputBufferOwnershipSafetyGate.SafetyGateReady,
            outputBufferOwnershipSafetyGate.OutputBufferOwnershipRuntimeReady,
            outputBufferOwnershipSafetyGate.CurrentMemoryReusePolicyReady,
            outputBufferOwnershipSafetyGate.BorrowedPointerEscapeBlocked,
            outputBufferOwnershipSafetyGate.OwnedDevicePointerReleasePolicyReady,
            outputBufferOwnershipSafetyGate.ShapeNotificationOrderingReady,
            outputBufferOwnershipSafetyGate.ReallocateOutputRuntimeReady,
            blockers.ToArray());
    }
}

/// <summary>
/// Reports copied OutputAllocator runtime proof precheck diagnostics.
/// 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。
/// </summary>
/// <remarks>
/// This result is a runtime gate precheck only. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> are always <see langword="false"/> until a full package consumer smoke
/// produces complete <c>real-callback-runtime</c> evidence.
/// 该说明强调当前结果只是 OutputAllocator 安全门禁或 precheck，不代表真实 TensorRT callback runtime proof。
/// </remarks>
public readonly struct TensorRtOutputAllocatorRuntimeProofPrecheckResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtOutputAllocatorRuntimeProofPrecheckResult(
        TensorRtApiLine line,
        bool lineSupportsOutputAllocator,
        bool ownerDesignReady,
        bool nativeLedgerDesignReady,
        bool disposeReleaseReady,
        bool pointerFreeSurfaceReady,
        bool attachDetachDesignGateReady,
        bool attachControlAvailable,
        bool detachClearControlAvailable,
        bool managedOwnerStateMachineReady,
        bool stableNativeOwnerAddressReady,
        bool noThrowNativeVTableReady,
        bool outputBufferOwnershipSafetyGateReady,
        bool outputBufferOwnershipRuntimeReady,
        bool currentMemoryReusePolicyReady,
        bool borrowedPointerEscapeBlocked,
        bool ownedDevicePointerReleasePolicyReady,
        bool shapeNotificationOrderingReady,
        bool reallocateOutputRuntimeReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        LineSupportsOutputAllocator = lineSupportsOutputAllocator;
        OwnerDesignReady = ownerDesignReady;
        NativeLedgerDesignReady = nativeLedgerDesignReady;
        DisposeReleaseReady = disposeReleaseReady;
        PointerFreeSurfaceReady = pointerFreeSurfaceReady;
        AttachDetachDesignGateReady = attachDetachDesignGateReady;
        AttachControlAvailable = attachControlAvailable;
        DetachClearControlAvailable = detachClearControlAvailable;
        ManagedOwnerStateMachineReady = managedOwnerStateMachineReady;
        StableNativeOwnerAddressReady = stableNativeOwnerAddressReady;
        NoThrowNativeVTableReady = noThrowNativeVTableReady;
        OutputBufferOwnershipSafetyGateReady = outputBufferOwnershipSafetyGateReady;
        OutputBufferOwnershipRuntimeReady = outputBufferOwnershipRuntimeReady;
        CurrentMemoryReusePolicyReady = currentMemoryReusePolicyReady;
        BorrowedPointerEscapeBlocked = borrowedPointerEscapeBlocked;
        OwnedDevicePointerReleasePolicyReady = ownedDevicePointerReleasePolicyReady;
        ShapeNotificationOrderingReady = shapeNotificationOrderingReady;
        ReallocateOutputRuntimeReady = reallocateOutputRuntimeReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this precheck. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public string EvidenceKind => "output-allocator-runtime-proof-precheck";

    /// <summary>Gets the callback kind represented by this precheck. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public string CallbackKind => "output-allocator";

    /// <summary>Gets the runtime evidence kind. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public string RuntimeEvidenceKind => "runtime-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the owner design snapshot. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected TensorRT line has OutputAllocator API support. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool LineSupportsOutputAllocator { get; }

    /// <summary>Gets whether the owner design snapshot was clean owner-design evidence. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool OwnerDesignReady { get; }

    /// <summary>Gets whether native allocator owner ledger dry-run evidence was clean. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool NativeLedgerDesignReady { get; }

    /// <summary>Gets whether dispose release hook evidence was present before the precheck. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool DisposeReleaseReady { get; }

    /// <summary>Gets whether the public surface remains pointer-free. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool PointerFreeSurfaceReady { get; }

    /// <summary>Gets whether the attach/detach design gate has copied evidence ready for precheck consumption. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool AttachDetachDesignGateReady { get; }

    /// <summary>Gets whether a non-null OutputAllocator attach bridge is available. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool AttachControlAvailable { get; }

    /// <summary>Gets whether the TensorRT 8/10/11 detach/clear control is available. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool DetachClearControlAvailable { get; }

    /// <summary>Gets whether the managed owner state machine has clean dispose and in-flight drain evidence. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool ManagedOwnerStateMachineReady { get; }

    /// <summary>Gets whether line-specific execution context attach/detach is ready. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool LineSpecificAttachDetachReady => AttachControlAvailable && DetachClearControlAvailable;

    /// <summary>Gets whether a stable native owner address is ready. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool StableNativeOwnerAddressReady { get; }

    /// <summary>Gets whether a no-throw native vtable trampoline is ready. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool NoThrowNativeVTableReady { get; }

    /// <summary>Gets whether the native OutputAllocator owner and no-throw vtable are ready. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool NativeVTableReady => StableNativeOwnerAddressReady && NoThrowNativeVTableReady;

    /// <summary>Gets whether runtime device pointer ownership ledgering is ready. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool DevicePointerLedgerRuntimeReady => false;

    /// <summary>Gets whether CUDA stream lifetime and async allocation semantics are ready. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool StreamLifetimeReady => false;

    /// <summary>Gets whether the output-buffer ownership safety gate has copied evidence ready for precheck consumption. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool OutputBufferOwnershipSafetyGateReady { get; }

    /// <summary>Gets whether output buffer ownership and current-memory reuse rules are ready. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool OutputBufferOwnershipRuntimeReady { get; }

    /// <summary>Gets whether currentMemory reuse policy is runtime ready. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool CurrentMemoryReusePolicyReady { get; }

    /// <summary>Gets whether borrowed output buffer pointers are blocked from escaping public APIs. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool BorrowedPointerEscapeBlocked { get; }

    /// <summary>Gets whether owned device pointer release policy is runtime ready. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool OwnedDevicePointerReleasePolicyReady { get; }

    /// <summary>Gets whether notifyShape/reallocateOutput ordering has been proven in a real runtime callback. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool ShapeNotificationOrderingReady { get; }

    /// <summary>Gets whether reallocateOutput runtime callback execution is ready. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool ReallocateOutputRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady => false;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool CanAttemptRuntimeProof =>
        OwnerDesignReady &&
        NativeLedgerDesignReady &&
        DisposeReleaseReady &&
        PointerFreeSurfaceReady &&
        LineSupportsOutputAllocator &&
        AttachDetachDesignGateReady &&
        ManagedOwnerStateMachineReady &&
        LineSpecificAttachDetachReady &&
        NativeVTableReady &&
        DevicePointerLedgerRuntimeReady &&
        StreamLifetimeReady &&
        OutputBufferOwnershipSafetyGateReady &&
        OutputBufferOwnershipRuntimeReady &&
        CurrentMemoryReusePolicyReady &&
        BorrowedPointerEscapeBlocked &&
        OwnedDevicePointerReleasePolicyReady &&
        ShapeNotificationOrderingReady &&
        ReallocateOutputRuntimeReady &&
        FullPackageConsumerRuntimeEvidenceReady;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets the copied list of prerequisites that still block real runtime proof. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied number of blocked prerequisites. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the precheck status. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public string Status => CanAttemptRuntimeProof ? "can-attempt-runtime-proof" : "precheck-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    public string Diagnostic =>
        "output-allocator-runtime-proof-precheck; RuntimeEvidenceKind=runtime-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "AttachDetachDesignGateReady=" + AttachDetachDesignGateReady + "; " +
        "AttachControlAvailable=" + AttachControlAvailable + "; " +
        "DetachClearControlAvailable=" + DetachClearControlAvailable + "; " +
        "StableNativeOwnerAddressReady=" + StableNativeOwnerAddressReady + "; " +
        "NoThrowNativeVTableReady=" + NoThrowNativeVTableReady + "; " +
        "OutputBufferOwnershipSafetyGateReady=" + OutputBufferOwnershipSafetyGateReady + "; " +
        "OutputBufferOwnershipRuntimeReady=" + OutputBufferOwnershipRuntimeReady + "; " +
        "CurrentMemoryReusePolicyReady=" + CurrentMemoryReusePolicyReady + "; " +
        "BorrowedPointerEscapeBlocked=" + BorrowedPointerEscapeBlocked + "; " +
        "OwnedDevicePointerReleasePolicyReady=" + OwnedDevicePointerReleasePolicyReady + "; " +
        "ShapeNotificationOrderingReady=" + ShapeNotificationOrderingReady + "; " +
        "ReallocateOutputRuntimeReady=" + ReallocateOutputRuntimeReady + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 OutputAllocator callback 安全边界的只读诊断信息；不会暴露 native output buffer 指针。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native output buffer 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:blocked={BlockedPrerequisiteCount}:proof={IsRealCallbackRuntimeProof}";
    }
}
