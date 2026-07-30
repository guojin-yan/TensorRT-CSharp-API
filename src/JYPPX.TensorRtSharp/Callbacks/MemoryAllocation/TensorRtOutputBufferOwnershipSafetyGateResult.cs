using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied OutputAllocator output-buffer ownership diagnostics without exposing native pointers.
/// 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。
/// </summary>
/// <remarks>
/// This result is a safety gate. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> remain <see langword="false"/> until real TensorRT callback smoke proves
/// <c>IOutputAllocator::reallocateOutput</c> ownership behavior.
/// 该说明强调当前结果只是输出缓冲区所有权安全门禁，不代表真实 TensorRT callback runtime proof。
/// </remarks>
public readonly struct TensorRtOutputBufferOwnershipSafetyGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtOutputBufferOwnershipSafetyGateResult(
        TensorRtApiLine line,
        string tensorName,
        ulong requestedSize,
        ulong alignment,
        int shapeRank,
        bool hasCurrentMemory,
        long notifyShapeCount,
        long reallocateOutputCount,
        bool attachDetachDesignGateReady,
        bool ownerDesignReady,
        bool pointerFreeSurfaceReady,
        bool copiedCurrentMemoryMetadataReady,
        bool copiedShapeMetadataReady,
        bool copiedRequestMetadataReady,
        bool outputBufferOwnershipRuntimeReady,
        bool currentMemoryReusePolicyReady,
        bool borrowedPointerEscapeBlocked,
        bool ownedDevicePointerReleasePolicyReady,
        bool shapeNotificationOrderingReady,
        bool reallocateOutputRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        TensorName = tensorName ?? string.Empty;
        RequestedSize = requestedSize;
        Alignment = alignment;
        ShapeRank = shapeRank;
        HasCurrentMemory = hasCurrentMemory;
        NotifyShapeCount = notifyShapeCount;
        ReallocateOutputCount = reallocateOutputCount;
        AttachDetachDesignGateReady = attachDetachDesignGateReady;
        OwnerDesignReady = ownerDesignReady;
        PointerFreeSurfaceReady = pointerFreeSurfaceReady;
        CopiedCurrentMemoryMetadataReady = copiedCurrentMemoryMetadataReady;
        CopiedShapeMetadataReady = copiedShapeMetadataReady;
        CopiedRequestMetadataReady = copiedRequestMetadataReady;
        OutputBufferOwnershipRuntimeReady = outputBufferOwnershipRuntimeReady;
        CurrentMemoryReusePolicyReady = currentMemoryReusePolicyReady;
        BorrowedPointerEscapeBlocked = borrowedPointerEscapeBlocked;
        OwnedDevicePointerReleasePolicyReady = ownedDevicePointerReleasePolicyReady;
        ShapeNotificationOrderingReady = shapeNotificationOrderingReady;
        ReallocateOutputRuntimeReady = reallocateOutputRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this safety gate. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public string EvidenceKind => "output-buffer-ownership-safety-gate";

    /// <summary>Gets the callback kind represented by this safety gate. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public string CallbackKind => "output-allocator-output-buffer-ownership";

    /// <summary>Gets the runtime evidence kind. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied tensor name. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public string TensorName { get; }

    /// <summary>Gets the copied requested output buffer size. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public ulong RequestedSize { get; }

    /// <summary>Gets the copied requested output buffer alignment. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public ulong Alignment { get; }

    /// <summary>Gets the copied output shape rank. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public int ShapeRank { get; }

    /// <summary>Gets whether copied metadata reported a currentMemory value. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool HasCurrentMemory { get; }

    /// <summary>Gets the copied notifyShape diagnostic count. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public long NotifyShapeCount { get; }

    /// <summary>Gets the copied reallocateOutput diagnostic count. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public long ReallocateOutputCount { get; }

    /// <summary>Gets whether attach/detach design evidence was ready. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool AttachDetachDesignGateReady { get; }

    /// <summary>Gets whether owner design evidence was clean. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool OwnerDesignReady { get; }

    /// <summary>Gets whether the public surface remains pointer-free. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool PointerFreeSurfaceReady { get; }

    /// <summary>Gets whether copied currentMemory metadata is available without exposing the pointer value. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool CopiedCurrentMemoryMetadataReady { get; }

    /// <summary>Gets whether copied shape metadata is available. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool CopiedShapeMetadataReady { get; }

    /// <summary>Gets whether copied size/alignment request metadata is available. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool CopiedRequestMetadataReady { get; }

    /// <summary>Gets whether output buffer ownership is runtime ready. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool OutputBufferOwnershipRuntimeReady { get; }

    /// <summary>Gets whether currentMemory reuse policy is runtime ready. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool CurrentMemoryReusePolicyReady { get; }

    /// <summary>Gets whether borrowed output buffer pointers are blocked from escaping public APIs. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool BorrowedPointerEscapeBlocked { get; }

    /// <summary>Gets whether owned device pointer release policy is runtime ready. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool OwnedDevicePointerReleasePolicyReady { get; }

    /// <summary>Gets whether notifyShape/reallocateOutput ordering has been proven in a real runtime callback. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool ShapeNotificationOrderingReady { get; }

    /// <summary>Gets whether reallocateOutput runtime callback execution is ready. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool ReallocateOutputRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether this safety gate has enough copied evidence to feed the runtime-proof precheck. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool SafetyGateReady =>
        AttachDetachDesignGateReady &&
        OwnerDesignReady &&
        PointerFreeSurfaceReady &&
        CopiedCurrentMemoryMetadataReady &&
        CopiedShapeMetadataReady &&
        CopiedRequestMetadataReady &&
        BorrowedPointerEscapeBlocked;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool CanAttemptRuntimeProof =>
        SafetyGateReady &&
        OutputBufferOwnershipRuntimeReady &&
        CurrentMemoryReusePolicyReady &&
        OwnedDevicePointerReleasePolicyReady &&
        ShapeNotificationOrderingReady &&
        ReallocateOutputRuntimeReady &&
        FullPackageConsumerRuntimeEvidenceReady;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets copied blocked prerequisites. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the safety gate status. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public string Status => SafetyGateReady ? "safety-gate-ready" : "safety-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    public string Diagnostic =>
        "output-buffer-ownership-safety-gate; RuntimeEvidenceKind=design-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; SafetyGateReady=" + SafetyGateReady + "; " +
        "OutputBufferOwnershipRuntimeReady=" + OutputBufferOwnershipRuntimeReady + "; " +
        "CurrentMemoryReusePolicyReady=" + CurrentMemoryReusePolicyReady + "; " +
        "BorrowedPointerEscapeBlocked=" + BorrowedPointerEscapeBlocked + "; " +
        "OwnedDevicePointerReleasePolicyReady=" + OwnedDevicePointerReleasePolicyReady + "; " +
        "ShapeNotificationOrderingReady=" + ShapeNotificationOrderingReady + "; " +
        "ReallocateOutputRuntimeReady=" + ReallocateOutputRuntimeReady + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 OutputAllocator 输出缓冲区所有权边界的只读诊断信息；不会暴露 native 指针。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native output buffer 指针的诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:ownership={OutputBufferOwnershipRuntimeReady}:proof={IsRealCallbackRuntimeProof}";
    }
}
