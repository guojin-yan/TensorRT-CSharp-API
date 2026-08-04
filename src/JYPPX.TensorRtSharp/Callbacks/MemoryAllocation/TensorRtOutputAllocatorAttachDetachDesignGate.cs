using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free OutputAllocator attach/detach design gate before any real TensorRT callback bridge exists.
/// 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This gate describes lifecycle readiness only. It does not call <c>setOutputAllocator</c> with a non-null allocator,
/// does not expose a native allocator pointer, and is not proof that <c>IOutputAllocator::notifyShape</c> or
/// <c>IOutputAllocator::reallocateOutput</c> ran.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 OutputAllocator 回调执行证据。
/// </remarks>
public static class TensorRtOutputAllocatorAttachDetachDesignGate
{
    /// <summary>
    /// Evaluates the attach/detach design gate from a copied OutputAllocator owner design snapshot.
    /// 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied owner design snapshot. 中文：该参数是复制后的 OutputAllocator 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free attach/detach design result. 中文：返回复制后的 OutputAllocator 无裸指针诊断结果。</returns>
    public static TensorRtOutputAllocatorAttachDetachDesignGateResult Evaluate(
        TensorRtOutputAllocatorCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        bool lineSupportsOutputAllocator =
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt8 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt10 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt11;
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
        bool managedOwnerStateMachineReady =
            ownerDesignSnapshot.DisposeRequested &&
            ownerDesignSnapshot.ReleaseHookCount > 0 &&
            !ownerDesignSnapshot.CallbackStatePinned &&
            !ownerDesignSnapshot.DelegatePinned &&
            ownerDesignSnapshot.InFlightCallbackCount == 0;
        bool pointerFreeSurfaceReady =
            !ownerDesignSnapshot.OutputBufferPointerExposed &&
            !ownerDesignSnapshot.OutputBufferPointerProduced;

        bool detachClearControlAvailable = lineSupportsOutputAllocator;
        const bool attachControlAvailable = false;
        const bool stableNativeOwnerAddressReady = false;
        const bool noThrowNativeVTableReady = false;
        const bool outputBufferOwnershipRuntimeReady = false;

        List<string> blockers = new List<string>();
        if (!lineSupportsOutputAllocator)
        {
            blockers.Add("TensorRT 8, 10, or 11 IOutputAllocator line support has not been selected.");
        }

        if (!ownerDesignReady)
        {
            blockers.Add("output-allocator-callback-owner-design snapshot is not clean owner-design evidence.");
        }

        if (!managedOwnerStateMachineReady)
        {
            blockers.Add("managed OutputAllocator owner state machine has not shown dispose, release hook, unpin, and in-flight drain evidence.");
        }

        if (!pointerFreeSurfaceReady)
        {
            blockers.Add("public OutputAllocator attach/detach surface still exposes or produces an output buffer pointer.");
        }

        if (!detachClearControlAvailable)
        {
            blockers.Add("line-specific setOutputAllocator(nullptr) detach/clear control is not available.");
        }

        if (!attachControlAvailable)
        {
            blockers.Add("line-specific setOutputAllocator(non-null) attach bridge is not implemented.");
        }

        if (!stableNativeOwnerAddressReady)
        {
            blockers.Add("native OutputAllocator owner stable address is not implemented.");
        }

        if (!noThrowNativeVTableReady)
        {
            blockers.Add("native IOutputAllocator no-throw vtable trampoline is not implemented.");
        }

        if (!outputBufferOwnershipRuntimeReady)
        {
            blockers.Add("output buffer ownership, current-memory reuse, and borrowed/owned device pointer rules are not implemented.");
        }

        return new TensorRtOutputAllocatorAttachDetachDesignGateResult(
            ownerDesignSnapshot.Line,
            lineSupportsOutputAllocator,
            ownerDesignReady,
            managedOwnerStateMachineReady,
            pointerFreeSurfaceReady,
            detachClearControlAvailable,
            attachControlAvailable,
            stableNativeOwnerAddressReady,
            noThrowNativeVTableReady,
            outputBufferOwnershipRuntimeReady,
            blockers.ToArray());
    }
}
