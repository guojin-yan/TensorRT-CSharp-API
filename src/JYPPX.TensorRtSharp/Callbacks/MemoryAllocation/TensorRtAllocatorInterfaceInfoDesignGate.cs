using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free TensorRT allocator interface-info metadata design gate.
/// 评估 TensorRT allocator interface-info metadata 的无裸指针设计门。
/// </summary>
/// <remarks>
/// Allocators are application-owned callback objects. This gate only records copied interface metadata
/// requirements and does not expose allocator handles or enable allocation callbacks.
/// Allocator 是应用侧拥有的 callback 对象；该门禁只记录 copied interface metadata 要求，不暴露 allocator
/// handle，也不启用 allocation callback。
/// </remarks>
public static class TensorRtAllocatorInterfaceInfoDesignGate
{
    /// <summary>
    /// Evaluates the known public allocator interface-info design surface for a TensorRT API line.
    /// 基于已知 public allocator interface-info 设计边界评估指定 TensorRT API line。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <returns>A pointer-free allocator interface-info design gate result. 无裸指针 allocator interface-info 设计门结果。</returns>
    public static TensorRtAllocatorInterfaceInfoDesignGateResult EvaluateKnownSurface(TensorRtApiLine line)
    {
        return Evaluate(
            line,
            copiedInterfaceInfoMetadataReady: true,
            temporaryStorageAllocatorSnapshotAvailable: line == TensorRtApiLine.TensorRt10 || line == TensorRtApiLine.TensorRt11,
            outputAllocatorSnapshotAvailable: line == TensorRtApiLine.TensorRt10 || line == TensorRtApiLine.TensorRt11,
            allocatorOwnerLifetimeModeled: false,
            deviceMemoryOwnershipModeled: false,
            asyncStreamLifetimeModeled: false);
    }

    /// <summary>
    /// Evaluates allocator interface-info readiness from explicit capability flags.
    /// 根据显式能力标记评估 allocator interface-info 就绪状态。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="copiedInterfaceInfoMetadataReady">Whether copied interface-info metadata shape is ready. copied interface-info metadata 形态是否就绪。</param>
    /// <param name="temporaryStorageAllocatorSnapshotAvailable">Whether temporary-storage allocator copied metadata exists. temporary-storage allocator copied metadata 是否存在。</param>
    /// <param name="outputAllocatorSnapshotAvailable">Whether output allocator copied metadata exists. output allocator copied metadata 是否存在。</param>
    /// <param name="allocatorOwnerLifetimeModeled">Whether allocator owner lifetime has been modeled. allocator owner lifetime 是否已建模。</param>
    /// <param name="deviceMemoryOwnershipModeled">Whether device memory ownership has been modeled. device memory ownership 是否已建模。</param>
    /// <param name="asyncStreamLifetimeModeled">Whether async stream lifetime has been modeled. async stream lifetime 是否已建模。</param>
    /// <returns>A pointer-free allocator interface-info design gate result. 无裸指针 allocator interface-info 设计门结果。</returns>
    public static TensorRtAllocatorInterfaceInfoDesignGateResult Evaluate(
        TensorRtApiLine line,
        bool copiedInterfaceInfoMetadataReady,
        bool temporaryStorageAllocatorSnapshotAvailable,
        bool outputAllocatorSnapshotAvailable,
        bool allocatorOwnerLifetimeModeled,
        bool deviceMemoryOwnershipModeled,
        bool asyncStreamLifetimeModeled)
    {
        bool lineSupportsAllocatorInterfaceInfo =
            line == TensorRtApiLine.TensorRt10 ||
            line == TensorRtApiLine.TensorRt11;

        List<string> blockers = new List<string>();
        if (!lineSupportsAllocatorInterfaceInfo)
        {
            blockers.Add("TensorRT 10 or 11 allocator interface-info line support has not been selected.");
        }

        if (!copiedInterfaceInfoMetadataReady)
        {
            blockers.Add("copied allocator interface-info metadata shape is not ready.");
        }

        if (!temporaryStorageAllocatorSnapshotAvailable)
        {
            blockers.Add("temporary-storage allocator copied metadata snapshot is not available.");
        }

        if (!outputAllocatorSnapshotAvailable)
        {
            blockers.Add("output allocator copied metadata snapshot is not available.");
        }

        if (!allocatorOwnerLifetimeModeled)
        {
            blockers.Add("allocator owner lifetime is not modeled.");
        }

        if (!deviceMemoryOwnershipModeled)
        {
            blockers.Add("device memory allocation and release ownership remains deferred.");
        }

        if (!asyncStreamLifetimeModeled)
        {
            blockers.Add("async allocator stream lifetime remains deferred.");
        }

        blockers.Add("direct allocator pointer access remains deferred by design.");
        blockers.Add("allocate/deallocate/reallocate/notifyShape callbacks remain deferred by design.");
        blockers.Add("full package consumer runtime execution proof has not been promoted from this design gate.");

        return new TensorRtAllocatorInterfaceInfoDesignGateResult(
            line,
            lineSupportsAllocatorInterfaceInfo,
            copiedInterfaceInfoMetadataReady,
            temporaryStorageAllocatorSnapshotAvailable,
            outputAllocatorSnapshotAvailable,
            allocatorOwnerLifetimeModeled,
            deviceMemoryOwnershipModeled,
            asyncStreamLifetimeModeled,
            blockers.ToArray());
    }
}
