using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports allocator interface-info design status without exposing allocator pointers.
/// 报告 allocator interface-info 设计状态，不暴露 allocator 指针。
/// </summary>
public readonly struct TensorRtAllocatorInterfaceInfoDesignGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtAllocatorInterfaceInfoDesignGateResult(
        TensorRtApiLine line,
        bool lineSupportsAllocatorInterfaceInfo,
        bool copiedInterfaceInfoMetadataReady,
        bool temporaryStorageAllocatorSnapshotAvailable,
        bool outputAllocatorSnapshotAvailable,
        bool allocatorOwnerLifetimeModeled,
        bool deviceMemoryOwnershipModeled,
        bool asyncStreamLifetimeModeled,
        string[] blockedPrerequisites)
    {
        Line = line;
        LineSupportsAllocatorInterfaceInfo = lineSupportsAllocatorInterfaceInfo;
        CopiedInterfaceInfoMetadataReady = copiedInterfaceInfoMetadataReady;
        TemporaryStorageAllocatorSnapshotAvailable = temporaryStorageAllocatorSnapshotAvailable;
        OutputAllocatorSnapshotAvailable = outputAllocatorSnapshotAvailable;
        AllocatorOwnerLifetimeModeled = allocatorOwnerLifetimeModeled;
        DeviceMemoryOwnershipModeled = deviceMemoryOwnershipModeled;
        AsyncStreamLifetimeModeled = asyncStreamLifetimeModeled;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 获取该设计门的证据标记。</summary>
    public string EvidenceKind => "allocator-interface-info-design-gate";

    /// <summary>Gets the diagnostics kind represented by this gate. 获取该设计门代表的诊断类型。</summary>
    public string DiagnosticsKind => "allocator-interface-info";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this gate is runtime execution evidence. 获取该设计门是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this gate may be promoted as runtime execution proof. 获取该设计门是否可晋级为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected TensorRT line supports allocator interface-info metadata. 获取当前 TensorRT line 是否支持 allocator interface-info metadata。</summary>
    public bool LineSupportsAllocatorInterfaceInfo { get; }

    /// <summary>Gets whether copied interface-info metadata shape is ready. 获取 copied interface-info metadata 形态是否就绪。</summary>
    public bool CopiedInterfaceInfoMetadataReady { get; }

    /// <summary>Gets whether temporary-storage allocator copied metadata exists. 获取 temporary-storage allocator copied metadata 是否存在。</summary>
    public bool TemporaryStorageAllocatorSnapshotAvailable { get; }

    /// <summary>Gets whether output allocator copied metadata exists. 获取 output allocator copied metadata 是否存在。</summary>
    public bool OutputAllocatorSnapshotAvailable { get; }

    /// <summary>Gets whether allocator owner lifetime has been modeled. 获取 allocator owner lifetime 是否已建模。</summary>
    public bool AllocatorOwnerLifetimeModeled { get; }

    /// <summary>Gets whether device memory ownership has been modeled. 获取 device memory ownership 是否已建模。</summary>
    public bool DeviceMemoryOwnershipModeled { get; }

    /// <summary>Gets whether async stream lifetime has been modeled. 获取 async stream lifetime 是否已建模。</summary>
    public bool AsyncStreamLifetimeModeled { get; }

    /// <summary>Gets whether public APIs expose allocator pointers. 获取 public API 是否暴露 allocator 指针。</summary>
    public bool AllocatorPointerExposed => false;

    /// <summary>Gets whether public APIs expose device memory pointers through allocator callbacks. 获取 public API 是否通过 allocator callback 暴露 device memory 指针。</summary>
    public bool DeviceMemoryPointerExposed => false;

    /// <summary>Gets whether public APIs can invoke allocation callbacks. 获取 public API 是否可调用 allocation callback。</summary>
    public bool AllocationCallbackInvocationEnabled => false;

    /// <summary>Gets whether direct allocator callback rows intentionally remain deferred. 获取 direct allocator callback 行是否继续 deferred。</summary>
    public bool DirectAllocatorCallbackRowsDeferred => true;

    /// <summary>Gets whether copied allocator metadata shape is ready. 获取 copied allocator metadata 形态是否就绪。</summary>
    public bool CopiedMetadataShapeReady =>
        CopiedInterfaceInfoMetadataReady &&
        TemporaryStorageAllocatorSnapshotAvailable &&
        OutputAllocatorSnapshotAvailable;

    /// <summary>Gets whether the public surface remains pointer-free. 获取 public surface 是否保持无裸指针。</summary>
    public bool PointerFreeSurfaceReady =>
        !AllocatorPointerExposed &&
        !DeviceMemoryPointerExposed &&
        !AllocationCallbackInvocationEnabled;

    /// <summary>Gets whether this design gate is ready as non-proof evidence. 获取该 design gate 作为非 proof 证据是否就绪。</summary>
    public bool DesignGateReady =>
        LineSupportsAllocatorInterfaceInfo &&
        CopiedMetadataShapeReady &&
        PointerFreeSurfaceReady;

    /// <summary>Gets whether this gate can be promoted without runtime proof. 获取该结果是否可在无 runtime proof 情况下晋级。</summary>
    public bool CanPromoteWithoutRuntimeProof => false;

    /// <summary>Gets whether this result can be promoted as runtime proof. 获取该结果是否可晋级为 runtime proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether runtime proof remains blocked. 获取 runtime proof 是否仍被阻塞。</summary>
    public bool RuntimeProofBlocked => !CanPromoteRuntimeProof;

    /// <summary>Gets whether direct deferred rows are still required. 获取 direct deferred 行是否仍必须保留。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets the candidate TensorRT interfaces covered by this design gate. 获取该设计门覆盖的候选 TensorRT 接口。</summary>
    public ReadOnlyCollection<string> CandidateInterfaces =>
        Array.AsReadOnly(new[]
        {
            "IGpuAllocator",
            "IGpuAsyncAllocator",
            "IOutputAllocator"
        });

    /// <summary>Gets the direct deferred methods covered by this design gate. 获取该设计门覆盖的 direct deferred 方法。</summary>
    public ReadOnlyCollection<string> CandidateMethods =>
        Array.AsReadOnly(new[]
        {
            "IGpuAllocator::getInterfaceInfo",
            "IGpuAsyncAllocator::getInterfaceInfo",
            "IOutputAllocator::getInterfaceInfo"
        });

    /// <summary>Gets the copied output mode required before implementation. 获取实现前要求的 copied 输出模式。</summary>
    public string RequiredOutputMode => "copied allocator interface metadata snapshot";

    /// <summary>Gets the next safe implementation step. 获取下一步安全实现动作。</summary>
    public string NextSafeImplementationStep =>
        "Keep direct allocator callbacks deferred while using owner-scoped copied metadata for temporary-storage and output allocator interface info.";

    /// <summary>Gets the direct candidate method count. 获取 direct 候选方法数量。</summary>
    public int CandidateMethodCount => CandidateMethods.Count;

    /// <summary>Gets copied blocked prerequisites. 获取已复制的阻塞前置项。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 获取已复制阻塞前置项数量。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the design gate status. 获取设计门状态。</summary>
    public string Status => DesignGateReady ? "design-gate-ready" : "design-gate-blocked";

    /// <summary>Gets a compact diagnostic summary. 获取简短诊断摘要。</summary>
    public string Diagnostic =>
        "allocator-interface-info-design-gate; RuntimeEvidenceKind=design-gate; " +
        "IsRuntimeExecutionEvidence=False; IsRuntimeExecutionProof=False; " +
        "CopiedMetadataShapeReady=" + CopiedMetadataShapeReady + "; " +
        "PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; " +
        "AllocatorPointerExposed=False; DeviceMemoryPointerExposed=False; " +
        "AllocationCallbackInvocationEnabled=False; DirectAllocatorCallbackRowsDeferred=True; " +
        "RequiredOutputMode=" + RequiredOutputMode + "; CandidateMethodCount=" + CandidateMethodCount + "; " +
        "CanPromoteWithoutRuntimeProof=False; RuntimeProofBlocked=True; " +
        "DeferredRowsStillRequired=True; BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 返回简短诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:pointerFree={PointerFreeSurfaceReady}:proof={IsRuntimeExecutionProof}";
    }
}
