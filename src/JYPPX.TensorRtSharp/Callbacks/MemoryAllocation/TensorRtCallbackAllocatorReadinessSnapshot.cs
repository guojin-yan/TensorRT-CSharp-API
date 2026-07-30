using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Pointer-free readiness summary for managed callback and allocator safety surfaces.
/// managed callback 与 allocator 安全面的无指针 readiness 摘要。
/// </summary>
/// <remarks>
/// This snapshot intentionally reports real callback invocation proof separately from managed readiness. A positive
/// managed readiness result must not be read as proof that TensorRT invoked a native callback.
/// 该快照有意将真实 callback invocation proof 与 managed readiness 分开；managed readiness 为正不能解读为
/// TensorRT 已经触发 native callback。
/// </remarks>
public sealed class TensorRtCallbackAllocatorReadinessSnapshot
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtCallbackAllocatorReadinessSnapshot(
        bool loggerCallbackReady,
        bool profilerCallbackReady,
        bool progressMonitorCallbackReady,
        bool allocatorOwnerDryRunReady,
        bool allocatorLedgerSafetyGateReady,
        bool outputAllocatorOwnerDesignReady,
        bool outputAllocatorRuntimeGateReady,
        bool debugListenerOwnerDesignReady,
        bool debugListenerNoThrowVTableGateReady,
        bool debugListenerRuntimeProofPrecheckReady,
        bool realCallbackInvocationProofReady,
        bool isPublishSafeForManagedCallbacks,
        bool isRuntimeInvocationProofComplete,
        bool runtimeProofBlocked,
        string[] blockedPrerequisites)
    {
        LoggerCallbackReady = loggerCallbackReady;
        ProfilerCallbackReady = profilerCallbackReady;
        ProgressMonitorCallbackReady = progressMonitorCallbackReady;
        AllocatorOwnerDryRunReady = allocatorOwnerDryRunReady;
        AllocatorLedgerSafetyGateReady = allocatorLedgerSafetyGateReady;
        OutputAllocatorOwnerDesignReady = outputAllocatorOwnerDesignReady;
        OutputAllocatorRuntimeGateReady = outputAllocatorRuntimeGateReady;
        DebugListenerOwnerDesignReady = debugListenerOwnerDesignReady;
        DebugListenerNoThrowVTableGateReady = debugListenerNoThrowVTableGateReady;
        DebugListenerRuntimeProofPrecheckReady = debugListenerRuntimeProofPrecheckReady;
        RealCallbackInvocationProofReady = realCallbackInvocationProofReady;
        IsPublishSafeForManagedCallbacks = isPublishSafeForManagedCallbacks;
        IsRuntimeInvocationProofComplete = isRuntimeInvocationProofComplete;
        RuntimeProofBlocked = runtimeProofBlocked;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness tools. 获取 readiness 工具使用的 marker。</summary>
    public string EvidenceKind => "callback-allocator-readiness-snapshot";

    /// <summary>Gets the runtime evidence kind. 获取 runtime 证据类型。</summary>
    public string RuntimeEvidenceKind => "managed-readiness";

    /// <summary>Gets whether this snapshot proves a real TensorRT callback runtime. 获取该快照是否证明真实 TensorRT callback runtime。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this snapshot as real callback runtime proof. 获取 readiness 是否可将该快照提升为真实 callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets whether managed logger callback wrapper evidence is present. 获取 managed logger callback wrapper 证据是否存在。</summary>
    public bool LoggerCallbackReady { get; }

    /// <summary>Gets whether managed profiler callback wrapper evidence is present. 获取 managed profiler callback wrapper 证据是否存在。</summary>
    public bool ProfilerCallbackReady { get; }

    /// <summary>Gets whether managed progress monitor callback wrapper evidence is present. 获取 managed progress monitor callback wrapper 证据是否存在。</summary>
    public bool ProgressMonitorCallbackReady { get; }

    /// <summary>Gets whether allocator owner dry-run evidence is ready. 获取 allocator owner dry-run 证据是否就绪。</summary>
    public bool AllocatorOwnerDryRunReady { get; }

    /// <summary>Gets whether allocator ledger safety gate evidence is ready. 获取 allocator ledger safety gate 证据是否就绪。</summary>
    public bool AllocatorLedgerSafetyGateReady { get; }

    /// <summary>Gets whether OutputAllocator owner design evidence is ready. 获取 OutputAllocator owner design 证据是否就绪。</summary>
    public bool OutputAllocatorOwnerDesignReady { get; }

    /// <summary>Gets whether OutputAllocator runtime gate evidence is ready for managed readiness. 获取 OutputAllocator runtime gate 证据是否满足 managed readiness。</summary>
    public bool OutputAllocatorRuntimeGateReady { get; }

    /// <summary>Gets whether DebugListener owner design evidence is ready. 获取 DebugListener owner design 证据是否就绪。</summary>
    public bool DebugListenerOwnerDesignReady { get; }

    /// <summary>Gets whether DebugListener no-throw vtable gate evidence is ready. 获取 DebugListener no-throw vtable gate 证据是否就绪。</summary>
    public bool DebugListenerNoThrowVTableGateReady { get; }

    /// <summary>Gets whether DebugListener runtime proof precheck evidence is ready. 获取 DebugListener runtime proof precheck 证据是否就绪。</summary>
    public bool DebugListenerRuntimeProofPrecheckReady { get; }

    /// <summary>Gets whether any consumed evidence proves real callback invocation. 获取已消费证据是否证明真实 callback invocation。</summary>
    public bool RealCallbackInvocationProofReady { get; }

    /// <summary>Gets whether the managed callback surfaces are publish-safe as managed wrappers. 获取 managed callback surface 作为托管 wrapper 是否具备发布安全性。</summary>
    public bool IsPublishSafeForManagedCallbacks { get; }

    /// <summary>Gets whether real runtime invocation proof is complete. 获取真实 runtime invocation proof 是否完成。</summary>
    public bool IsRuntimeInvocationProofComplete { get; }

    /// <summary>Gets whether real runtime callback proof remains blocked. 获取真实 runtime callback proof 是否仍被阻断。</summary>
    public bool RuntimeProofBlocked { get; }

    /// <summary>Gets copied blocked prerequisites from the consumed gates and prechecks. 获取从已消费 gate / precheck 复制出的阻断前置条件。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 获取复制出的阻断前置条件数量。</summary>
    public int BlockedReasonCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets a compact status string. 获取紧凑状态字符串。</summary>
    public string Status => IsRuntimeInvocationProofComplete ? "runtime-proof-complete" : "runtime-proof-blocked";

    /// <summary>Gets a copied diagnostic summary. 获取复制出的诊断摘要。</summary>
    public string Summary =>
        "callback-allocator-readiness-snapshot; RuntimeEvidenceKind=managed-readiness; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; IsPublishSafeForManagedCallbacks=" + IsPublishSafeForManagedCallbacks + "; " +
        "IsRuntimeInvocationProofComplete=" + IsRuntimeInvocationProofComplete + "; RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "LoggerCallbackReady=" + LoggerCallbackReady + "; ProfilerCallbackReady=" + ProfilerCallbackReady + "; " +
        "ProgressMonitorCallbackReady=" + ProgressMonitorCallbackReady + "; AllocatorOwnerDryRunReady=" + AllocatorOwnerDryRunReady + "; " +
        "AllocatorLedgerSafetyGateReady=" + AllocatorLedgerSafetyGateReady + "; OutputAllocatorOwnerDesignReady=" + OutputAllocatorOwnerDesignReady + "; " +
        "OutputAllocatorRuntimeGateReady=" + OutputAllocatorRuntimeGateReady + "; DebugListenerOwnerDesignReady=" + DebugListenerOwnerDesignReady + "; " +
        "DebugListenerNoThrowVTableGateReady=" + DebugListenerNoThrowVTableGateReady + "; " +
        "DebugListenerRuntimeProofPrecheckReady=" + DebugListenerRuntimeProofPrecheckReady + "; " +
        "RealCallbackInvocationProofReady=" + RealCallbackInvocationProofReady + "; BlockedReasonCount=" + BlockedReasonCount + ".";

    /// <summary>Returns a compact diagnostic representation. 返回紧凑诊断表示。</summary>
    /// <returns>A pointer-free diagnostic string. 无指针诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:status={Status}:managed={IsPublishSafeForManagedCallbacks}:runtime={IsRuntimeInvocationProofComplete}:blocked={BlockedReasonCount}";
    }
}
