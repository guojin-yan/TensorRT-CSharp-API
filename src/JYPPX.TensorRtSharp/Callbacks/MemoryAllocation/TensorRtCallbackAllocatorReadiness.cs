using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Aggregates managed callback and allocator readiness evidence without invoking TensorRT or exposing native pointers.
/// 聚合托管 callback 与 allocator readiness 证据；不会调用 TensorRT，也不会暴露 native pointer。
/// </summary>
/// <remarks>
/// This helper consumes existing copied gate and precheck results. It is a release-readiness summary for managed surfaces,
/// not proof that TensorRT has invoked allocator, output allocator, or debug listener callbacks.
/// 该工具只消费已有的复制式 gate / precheck 结果；它是 managed surface 的发布 readiness 摘要，
/// 不是 TensorRT 已真实触发 allocator、output allocator 或 debug listener callback 的证明。
/// </remarks>
public static class TensorRtCallbackAllocatorReadiness
{
    /// <summary>
    /// Aggregates callback and allocator readiness from copied gate and precheck evidence.
    /// 从复制式 gate 与 precheck 证据聚合 callback / allocator readiness。
    /// </summary>
    /// <param name="allocatorLedgerSafetyGate">Copied allocator ledger safety gate evidence. 复制出的 allocator ledger 安全门禁证据。</param>
    /// <param name="outputAllocatorRuntimeProofPrecheck">Copied OutputAllocator runtime proof precheck evidence. 复制出的 OutputAllocator runtime proof precheck 证据。</param>
    /// <param name="debugListenerRuntimeProofPrecheck">Copied DebugListener runtime proof precheck evidence. 复制出的 DebugListener runtime proof precheck 证据。</param>
    /// <param name="loggerCallbackReady">Whether the managed logger callback wrapper is available. managed logger callback wrapper 是否可用。</param>
    /// <param name="profilerCallbackReady">Whether the managed profiler callback wrapper is available. managed profiler callback wrapper 是否可用。</param>
    /// <param name="progressMonitorCallbackReady">Whether the managed progress monitor callback wrapper is available. managed progress monitor callback wrapper 是否可用。</param>
    /// <returns>A pointer-free readiness snapshot. 不暴露 pointer 的 readiness 快照。</returns>
    public static TensorRtCallbackAllocatorReadinessSnapshot Evaluate(
        TensorRtAllocatorLedgerSafetyGateResult allocatorLedgerSafetyGate,
        TensorRtOutputAllocatorRuntimeProofPrecheckResult outputAllocatorRuntimeProofPrecheck,
        TensorRtDebugListenerRuntimeProofPrecheckResult debugListenerRuntimeProofPrecheck,
        bool loggerCallbackReady = true,
        bool profilerCallbackReady = true,
        bool progressMonitorCallbackReady = true)
    {
        List<string> blockers = new List<string>();
        AddRange(blockers, allocatorLedgerSafetyGate.BlockedPrerequisites);
        AddRange(blockers, outputAllocatorRuntimeProofPrecheck.BlockedPrerequisites);
        AddRange(blockers, debugListenerRuntimeProofPrecheck.BlockedPrerequisites);

        bool allocatorOwnerDryRunReady =
            allocatorLedgerSafetyGate.InternalPrototypeStatus == JYPPX.TensorRtSharp.Shared.Interop.BridgeStatusCode.Ok &&
            allocatorLedgerSafetyGate.InvocationCount > 0 &&
            allocatorLedgerSafetyGate.FailureCount == 0 &&
            allocatorLedgerSafetyGate.PointerFreeSurfaceReady;
        bool allocatorLedgerSafetyGateReady =
            allocatorLedgerSafetyGate.NativeLedgerDesignReady &&
            allocatorLedgerSafetyGate.PointerFreeSurfaceReady &&
            !allocatorLedgerSafetyGate.IsRealCallbackRuntimeProof;
        bool outputAllocatorOwnerDesignReady =
            outputAllocatorRuntimeProofPrecheck.OwnerDesignReady &&
            outputAllocatorRuntimeProofPrecheck.PointerFreeSurfaceReady;
        bool outputAllocatorRuntimeGateReady =
            outputAllocatorRuntimeProofPrecheck.OwnerDesignReady &&
            outputAllocatorRuntimeProofPrecheck.AttachDetachDesignGateReady &&
            outputAllocatorRuntimeProofPrecheck.OutputBufferOwnershipSafetyGateReady &&
            outputAllocatorRuntimeProofPrecheck.BorrowedPointerEscapeBlocked;
        bool debugListenerOwnerDesignReady =
            debugListenerRuntimeProofPrecheck.OwnerDesignReady &&
            debugListenerRuntimeProofPrecheck.DebugTensorMetadataCopied &&
            debugListenerRuntimeProofPrecheck.PointerFreeSurfaceReady;
        bool debugListenerNoThrowVTableGateReady =
            debugListenerRuntimeProofPrecheck.NativeNoThrowVTableScaffoldGateReady &&
            debugListenerRuntimeProofPrecheck.NoThrowVTableScaffoldReady &&
            debugListenerRuntimeProofPrecheck.ProcessDebugTensorCallbackStubNoThrowReady &&
            debugListenerRuntimeProofPrecheck.ExceptionEscapeBlocked;
        bool debugListenerRuntimeProofPrecheckReady =
            debugListenerRuntimeProofPrecheck.PointerFreeSurfaceReady &&
            debugListenerRuntimeProofPrecheck.AttachDetachDesignGateReady &&
            debugListenerRuntimeProofPrecheck.BorrowedTensorSafetyGateReady &&
            debugListenerRuntimeProofPrecheck.NativeAttachEntryRuntimeScaffoldReady;
        bool realCallbackInvocationProofReady =
            allocatorLedgerSafetyGate.IsRealCallbackRuntimeProof ||
            outputAllocatorRuntimeProofPrecheck.IsRealCallbackRuntimeProof ||
            debugListenerRuntimeProofPrecheck.IsRealCallbackRuntimeProof;
        bool runtimeProofBlocked =
            allocatorLedgerSafetyGate.RuntimeProofBlocked ||
            outputAllocatorRuntimeProofPrecheck.RuntimeProofBlocked ||
            debugListenerRuntimeProofPrecheck.RuntimeProofBlocked ||
            !realCallbackInvocationProofReady;
        bool publishSafeForManagedCallbacks =
            loggerCallbackReady &&
            profilerCallbackReady &&
            progressMonitorCallbackReady &&
            allocatorOwnerDryRunReady &&
            allocatorLedgerSafetyGate.PointerFreeSurfaceReady &&
            outputAllocatorOwnerDesignReady &&
            outputAllocatorRuntimeGateReady &&
            debugListenerOwnerDesignReady &&
            debugListenerRuntimeProofPrecheckReady;
        bool runtimeInvocationProofComplete =
            realCallbackInvocationProofReady &&
            !runtimeProofBlocked &&
            allocatorLedgerSafetyGate.CanAttemptRuntimeProof &&
            outputAllocatorRuntimeProofPrecheck.CanAttemptRuntimeProof &&
            debugListenerRuntimeProofPrecheck.CanAttemptRuntimeProof;

        return new TensorRtCallbackAllocatorReadinessSnapshot(
            loggerCallbackReady,
            profilerCallbackReady,
            progressMonitorCallbackReady,
            allocatorOwnerDryRunReady,
            allocatorLedgerSafetyGateReady,
            outputAllocatorOwnerDesignReady,
            outputAllocatorRuntimeGateReady,
            debugListenerOwnerDesignReady,
            debugListenerNoThrowVTableGateReady,
            debugListenerRuntimeProofPrecheckReady,
            realCallbackInvocationProofReady,
            publishSafeForManagedCallbacks,
            runtimeInvocationProofComplete,
            runtimeProofBlocked,
            blockers.ToArray());
    }

    private static void AddRange(List<string> blockers, IEnumerable<string> source)
    {
        foreach (string blocker in source)
        {
            if (!string.IsNullOrWhiteSpace(blocker) && !blockers.Contains(blocker))
            {
                blockers.Add(blocker);
            }
        }
    }
}
