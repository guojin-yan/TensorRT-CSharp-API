using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports DebugListener callback proof gaps without exposing native pointers.
/// 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
public sealed class TensorRtDebugListenerCallbackProofGapReportResult
{
    private readonly string[] _gapReasons;

    internal TensorRtDebugListenerCallbackProofGapReportResult(
        TensorRtApiLine line,
        string runtimePackageKey,
        bool nonNullAttachStillDisabled,
        bool nativeAttachEntryReady,
        bool nativeVTableInstallBlocked,
        bool noThrowCallbackEntryReady,
        bool exceptionStatusMappingReady,
        bool inFlightAccountingReady,
        bool borrowedDebugTensorMetadataCopied,
        bool detachRollbackReady,
        bool processDebugTensorRuntimeInvoked,
        bool fullPackageConsumerRuntimeProofReady,
        bool pointerFreeSurfaceReady,
        bool attemptedNoInvocation,
        int invocationCount,
        int failureCount,
        int inFlightCallbackCount,
        bool canPromoteRealCallbackRuntime,
        string[] gapReasons)
    {
        Line = line;
        RuntimePackageKey = runtimePackageKey ?? string.Empty;
        NonNullAttachStillDisabled = nonNullAttachStillDisabled;
        NativeAttachEntryReady = nativeAttachEntryReady;
        NativeVTableInstallBlocked = nativeVTableInstallBlocked;
        NoThrowCallbackEntryReady = noThrowCallbackEntryReady;
        ExceptionStatusMappingReady = exceptionStatusMappingReady;
        InFlightAccountingReady = inFlightAccountingReady;
        BorrowedDebugTensorMetadataCopied = borrowedDebugTensorMetadataCopied;
        DetachRollbackReady = detachRollbackReady;
        ProcessDebugTensorRuntimeInvoked = processDebugTensorRuntimeInvoked;
        FullPackageConsumerRuntimeProofReady = fullPackageConsumerRuntimeProofReady;
        PointerFreeSurfaceReady = pointerFreeSurfaceReady;
        AttemptedNoInvocation = attemptedNoInvocation;
        InvocationCount = invocationCount;
        FailureCount = failureCount;
        InFlightCallbackCount = inFlightCallbackCount;
        CanPromoteRealCallbackRuntime = canPromoteRealCallbackRuntime;
        _gapReasons = gapReasons == null ? Array.Empty<string>() : (string[])gapReasons.Clone();
    }

    /// <summary>Gets the evidence marker. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-callback-proof-gap-report";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "proof-gap-report";

    /// <summary>Gets whether this report is real callback runtime evidence. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => CanPromoteRealCallbackRuntime;

    /// <summary>Gets whether this report is promotable real callback runtime proof. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => CanPromoteRealCallbackRuntime;

    /// <summary>Gets the callback kind. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the TensorRT API line. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the TensorRT API line as an integer. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int TensorRtLine => (int)Line;

    /// <summary>Gets the runtime package key, when known. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimePackageKey { get; }

    /// <summary>Gets whether non-null attach remains disabled. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NonNullAttachStillDisabled { get; }

    /// <summary>Gets whether the native attach entry is ready. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryReady { get; }

    /// <summary>Gets whether native vtable install remains blocked. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstallBlocked { get; }

    /// <summary>Gets whether the no-throw callback entry is ready. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowCallbackEntryReady { get; }

    /// <summary>Gets whether exception/status mapping is ready. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ExceptionStatusMappingReady { get; }

    /// <summary>Gets whether in-flight accounting is ready. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool InFlightAccountingReady { get; }

    /// <summary>Gets whether borrowed debug tensor metadata is copied into managed state. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorMetadataCopied { get; }

    /// <summary>Gets whether detach/rollback ordering is ready. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DetachRollbackReady { get; }

    /// <summary>Gets whether TensorRT invoked processDebugTensor. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeInvoked { get; }

    /// <summary>Gets whether full package consumer proof is ready. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeProofReady { get; }

    /// <summary>Gets whether the report is pointer-free. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool PointerFreeSurfaceReady { get; }

    /// <summary>Gets whether an attempt observed no invocation. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttemptedNoInvocation { get; }

    /// <summary>Gets the copied invocation count. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int InvocationCount { get; }

    /// <summary>Gets the copied failure count. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int FailureCount { get; }

    /// <summary>Gets the copied in-flight callback count. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int InFlightCallbackCount { get; }

    /// <summary>Gets whether the evidence can be promoted to real callback runtime proof. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanPromoteRealCallbackRuntime { get; }

    /// <summary>Gets whether runtime proof can be attempted. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof => CanPromoteRealCallbackRuntime;

    /// <summary>Gets whether proof remains blocked. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => !CanPromoteRealCallbackRuntime;

    /// <summary>Gets whether direct callback deferred rows must remain deferred. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => !CanPromoteRealCallbackRuntime;

    /// <summary>Gets copied gap reasons. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> GapReasons =>
        Array.AsReadOnly(_gapReasons ?? Array.Empty<string>());

    /// <summary>Gets the gap reason count. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int GapReasonCount => (_gapReasons ?? Array.Empty<string>()).Length;

    /// <summary>Gets the first copied gap reason, when present. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string PrimaryGapReason => GapReasonCount == 0 ? string.Empty : _gapReasons[0];

    /// <summary>Gets a stable blocker category for release-readiness dashboards. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeProofBlockerCategory => GetRuntimeProofBlockerCategory();

    /// <summary>Gets whether package-consumer runtime proof is still required. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool PackageConsumerRuntimeProofRequired => !FullPackageConsumerRuntimeProofReady;

    /// <summary>Gets whether a real TensorRT processDebugTensor invocation is still required. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeInvocationRequired => !ProcessDebugTensorRuntimeInvoked || InvocationCount == 0;

    /// <summary>Gets the copied evidence source classification. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceSource => "copied-preflight-smoke-trampoline-proof-gate";

    /// <summary>Gets the next owner action needed to close the proof gap. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string NextOwnerAction => GetNextOwnerAction();

    /// <summary>Gets the report status. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => CanPromoteRealCallbackRuntime ? "ready" : "blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback proof 缺口的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-callback-proof-gap-report; RuntimeEvidenceKind=proof-gap-report; " +
        "RealCallbackRuntime=" + RealCallbackRuntime + "; IsRealCallbackRuntimeProof=" + IsRealCallbackRuntimeProof + "; " +
        "NonNullAttachStillDisabled=" + NonNullAttachStillDisabled + "; NativeAttachEntryReady=" + NativeAttachEntryReady + "; " +
        "NativeVTableInstallBlocked=" + NativeVTableInstallBlocked + "; NoThrowCallbackEntryReady=" + NoThrowCallbackEntryReady + "; " +
        "ExceptionStatusMappingReady=" + ExceptionStatusMappingReady + "; InFlightAccountingReady=" + InFlightAccountingReady + "; " +
        "BorrowedDebugTensorMetadataCopied=" + BorrowedDebugTensorMetadataCopied + "; DetachRollbackReady=" + DetachRollbackReady + "; " +
        "ProcessDebugTensorRuntimeInvoked=" + ProcessDebugTensorRuntimeInvoked + "; " +
        "FullPackageConsumerRuntimeProofReady=" + FullPackageConsumerRuntimeProofReady + "; " +
        "PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; AttemptedNoInvocation=" + AttemptedNoInvocation + "; " +
        "InvocationCount=" + InvocationCount + "; FailureCount=" + FailureCount + "; InFlightCallbackCount=" + InFlightCallbackCount + "; " +
        "CanPromoteRealCallbackRuntime=" + CanPromoteRealCallbackRuntime + "; RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "DeferredRowsStillRequired=" + DeferredRowsStillRequired + "; GapReasonCount=" + GapReasonCount + "; " +
        "PrimaryGapReason=" + PrimaryGapReason + "; RuntimeProofBlockerCategory=" + RuntimeProofBlockerCategory + "; " +
        "PackageConsumerRuntimeProofRequired=" + PackageConsumerRuntimeProofRequired + "; RuntimeInvocationRequired=" + RuntimeInvocationRequired + "; " +
        "EvidenceSource=" + EvidenceSource + "; NextOwnerAction=" + NextOwnerAction + ".";

    private string GetRuntimeProofBlockerCategory()
    {
        if (CanPromoteRealCallbackRuntime)
        {
            return "none";
        }

        if (NonNullAttachStillDisabled)
        {
            return "non-null-attach-disabled";
        }

        if (!NativeAttachEntryReady)
        {
            return "native-attach-entry-not-ready";
        }

        if (NativeVTableInstallBlocked)
        {
            return "native-vtable-install-blocked";
        }

        if (!NoThrowCallbackEntryReady || !ExceptionStatusMappingReady || !InFlightAccountingReady)
        {
            return "callback-trampoline-safety-incomplete";
        }

        if (!BorrowedDebugTensorMetadataCopied || !PointerFreeSurfaceReady)
        {
            return "borrowed-debug-tensor-copy-incomplete";
        }

        if (!DetachRollbackReady)
        {
            return "detach-rollback-incomplete";
        }

        if (RuntimeInvocationRequired || AttemptedNoInvocation)
        {
            return "runtime-callback-invocation-missing";
        }

        if (PackageConsumerRuntimeProofRequired)
        {
            return "full-package-consumer-proof-missing";
        }

        return "real-callback-runtime-promotion-blocked";
    }

    private string GetNextOwnerAction()
    {
        switch (RuntimeProofBlockerCategory)
        {
            case "none":
                return "no-action-required";
            case "non-null-attach-disabled":
                return "enable-and-verify-non-null-debug-listener-attach-under-version-guards";
            case "native-attach-entry-not-ready":
                return "complete-native-set-debug-listener-attach-entry-with-no-throw-boundary";
            case "native-vtable-install-blocked":
                return "install-owned-no-throw-debug-listener-vtable-with-detach-before-release";
            case "callback-trampoline-safety-incomplete":
                return "complete-no-throw-callback-status-mapping-and-inflight-accounting";
            case "borrowed-debug-tensor-copy-incomplete":
                return "copy-borrowed-debug-tensor-metadata-before-returning-from-callback";
            case "detach-rollback-incomplete":
                return "prove-detach-rollback-and-dispose-idempotency-before-release";
            case "runtime-callback-invocation-missing":
                return "run-full-package-consumer-smoke-that-triggers-process-debug-tensor";
            case "full-package-consumer-proof-missing":
                return "collect-compatible-host-package-consumer-runtime-proof-with-invocation-count";
            default:
                return "inspect-gap-reasons-and-refresh-runtime-readiness-evidence";
        }
    }
}
