using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates copied OutputAllocator output-buffer ownership rules before any real TensorRT callback runtime proof exists.
/// 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。
/// </summary>
/// <remarks>
/// This gate models ownership policy only. It does not allocate device memory, does not consume or return
/// <c>currentMemory</c>, and does not expose a native output buffer pointer.
/// 该说明强调当前结果只是输出缓冲区所有权安全门禁，不代表真实 TensorRT callback runtime proof。
/// </remarks>
public static class TensorRtOutputBufferOwnershipSafetyGate
{
    /// <summary>
    /// Evaluates output-buffer ownership safety from a copied OutputAllocator owner design snapshot.
    /// 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied OutputAllocator owner design snapshot. 该参数传入已复制的 OutputAllocator owner 或 attach/detach evidence。</param>
    /// <returns>A pointer-free output-buffer ownership safety result. 返回不暴露 native output buffer 指针的诊断结果。</returns>
    public static TensorRtOutputBufferOwnershipSafetyGateResult Evaluate(
        TensorRtOutputAllocatorCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtOutputAllocatorAttachDetachDesignGateResult attachDetachGate =
            TensorRtOutputAllocatorAttachDetachDesignGate.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, attachDetachGate);
    }

    /// <summary>
    /// Evaluates output-buffer ownership safety from copied owner and attach/detach design evidence.
    /// 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied OutputAllocator owner design snapshot. 该参数传入已复制的 OutputAllocator owner 或 attach/detach evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 OutputAllocator owner 或 attach/detach evidence。</param>
    /// <returns>A pointer-free output-buffer ownership safety result. 返回不暴露 native output buffer 指针的诊断结果。</returns>
    public static TensorRtOutputBufferOwnershipSafetyGateResult Evaluate(
        TensorRtOutputAllocatorCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtOutputAllocatorAttachDetachDesignGateResult attachDetachDesignGate)
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
        bool pointerFreeSurfaceReady =
            !ownerDesignSnapshot.OutputBufferPointerExposed &&
            !ownerDesignSnapshot.OutputBufferPointerProduced;
        bool copiedCurrentMemoryMetadataReady = ownerDesignReady;
        bool copiedShapeMetadataReady =
            ownerDesignSnapshot.ShapeRank >= 0 &&
            !string.IsNullOrWhiteSpace(ownerDesignSnapshot.TensorName);
        bool copiedRequestMetadataReady =
            ownerDesignSnapshot.Alignment > 0UL;

        const bool outputBufferOwnershipRuntimeReady = false;
        const bool currentMemoryReusePolicyReady = false;
        const bool borrowedPointerEscapeBlocked = true;
        const bool ownedDevicePointerReleasePolicyReady = false;
        const bool shapeNotificationOrderingReady = false;
        const bool reallocateOutputRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        if (!ownerDesignReady)
        {
            blockers.Add("output-allocator-callback-owner-design snapshot is not clean owner-design evidence.");
        }

        if (!attachDetachDesignGate.DesignGateReady)
        {
            blockers.Add("output-allocator-attach-detach-design-gate is not ready for ownership safety evaluation.");
        }

        if (!pointerFreeSurfaceReady)
        {
            blockers.Add("public OutputAllocator ownership surface still exposes or produces an output buffer pointer.");
        }

        if (!copiedCurrentMemoryMetadataReady)
        {
            blockers.Add("copied currentMemory metadata is not available from a clean owner design snapshot.");
        }

        if (!copiedShapeMetadataReady)
        {
            blockers.Add("copied output tensor shape metadata is not available.");
        }

        if (!copiedRequestMetadataReady)
        {
            blockers.Add("copied output allocator size/alignment request metadata is not valid.");
        }

        if (!currentMemoryReusePolicyReady)
        {
            blockers.Add("currentMemory reuse policy has not been proven against a real TensorRT callback.");
        }

        if (!ownedDevicePointerReleasePolicyReady)
        {
            blockers.Add("owned device pointer release policy is not implemented.");
        }

        if (!shapeNotificationOrderingReady)
        {
            blockers.Add("notifyShape before reallocateOutput ordering has not been proven by a real TensorRT callback.");
        }

        if (!reallocateOutputRuntimeReady)
        {
            blockers.Add("IOutputAllocator::reallocateOutput runtime callback has not been implemented.");
        }

        if (!outputBufferOwnershipRuntimeReady)
        {
            blockers.Add("output buffer ownership, current-memory reuse, and borrowed/owned device pointer rules are not runtime ready.");
        }

        if (!fullPackageConsumerRuntimeEvidenceReady)
        {
            blockers.Add("full package consumer smoke has not emitted real-callback-runtime output allocator ownership evidence.");
        }

        return new TensorRtOutputBufferOwnershipSafetyGateResult(
            ownerDesignSnapshot.Line,
            ownerDesignSnapshot.TensorName,
            ownerDesignSnapshot.RequestedSize,
            ownerDesignSnapshot.Alignment,
            ownerDesignSnapshot.ShapeRank,
            ownerDesignSnapshot.HasCurrentMemory,
            ownerDesignSnapshot.NotifyShapeCount,
            ownerDesignSnapshot.ReallocateOutputCount,
            attachDetachDesignGate.DesignGateReady,
            ownerDesignReady,
            pointerFreeSurfaceReady,
            copiedCurrentMemoryMetadataReady,
            copiedShapeMetadataReady,
            copiedRequestMetadataReady,
            outputBufferOwnershipRuntimeReady,
            currentMemoryReusePolicyReady,
            borrowedPointerEscapeBlocked,
            ownedDevicePointerReleasePolicyReady,
            shapeNotificationOrderingReady,
            reallocateOutputRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady,
            blockers.ToArray());
    }
}
