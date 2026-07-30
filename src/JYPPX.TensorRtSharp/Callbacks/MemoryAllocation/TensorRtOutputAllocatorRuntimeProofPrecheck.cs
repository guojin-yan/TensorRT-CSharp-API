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
