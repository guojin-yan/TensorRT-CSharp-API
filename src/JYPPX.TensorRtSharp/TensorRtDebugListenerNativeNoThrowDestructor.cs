using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener native owner no-throw destructor evidence before native attach is enabled.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This destructor gate reports source-visible destructor <c>noexcept</c> scaffold diagnostics only. It does not create a
/// native owner, does not call <c>setDebugListener(non-null)</c>, and is not proof that TensorRT invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerNativeNoThrowDestructor
{
    /// <summary>
    /// Evaluates native no-throw destructor evidence from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native no-throw destructor result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeNoThrowDestructorResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeOwnerNonCopyableStorageResult nativeOwnerNonCopyableStorage =
            TensorRtDebugListenerNativeOwnerNonCopyableStorage.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, nativeOwnerNonCopyableStorage);
    }

    /// <summary>
    /// Evaluates native no-throw destructor evidence from copied owner and non-copyable storage evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeOwnerNonCopyableStorage">The copied native owner non-copyable storage result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native no-throw destructor result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeNoThrowDestructorResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeOwnerNonCopyableStorageResult nativeOwnerNonCopyableStorage)
    {
        const bool destructorNoThrowScaffoldReady = true;
        const bool destructorExceptionEscapeBlocked = true;
        const bool destructorAddressExposed = false;
        const bool destructorPointerProduced = false;
        const bool nativeOwnerLifecycleReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        bool nativeOwnerNonCopyableStorageReady = nativeOwnerNonCopyableStorage.NativeOwnerNonCopyableReady;
        bool noThrowNativeDestructorReady =
            nativeOwnerNonCopyableStorageReady &&
            nativeOwnerNonCopyableStorage.NativeOwnerCopyBlocked &&
            nativeOwnerNonCopyableStorage.NativeOwnerMoveBlocked &&
            !nativeOwnerNonCopyableStorage.NativeOwnerAddressExposed &&
            !nativeOwnerNonCopyableStorage.NativeOwnerPointerProduced &&
            destructorNoThrowScaffoldReady &&
            destructorExceptionEscapeBlocked &&
            !destructorAddressExposed &&
            !destructorPointerProduced;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, nativeOwnerNonCopyableStorageReady, "debug-listener-native-owner-noncopyable-storage is not ready for no-throw destructor evaluation.");
        AddBlockerIfFalse(blockers, nativeOwnerNonCopyableStorage.NativeOwnerCopyBlocked, "native DebugListener owner copy construction/assignment is not blocked.");
        AddBlockerIfFalse(blockers, nativeOwnerNonCopyableStorage.NativeOwnerMoveBlocked, "native DebugListener owner move construction/assignment is not blocked.");
        AddBlockerIfFalse(blockers, !nativeOwnerNonCopyableStorage.NativeOwnerAddressExposed, "native DebugListener owner address is exposed by the public surface.");
        AddBlockerIfFalse(blockers, !nativeOwnerNonCopyableStorage.NativeOwnerPointerProduced, "native DebugListener owner pointer is produced by the public surface.");
        AddBlockerIfFalse(blockers, destructorNoThrowScaffoldReady, "native DebugListener owner destructor no-throw scaffold is not source-visible.");
        AddBlockerIfFalse(blockers, destructorExceptionEscapeBlocked, "native DebugListener owner destructor exception escape is not blocked.");
        AddBlockerIfFalse(blockers, !destructorAddressExposed, "native DebugListener destructor gate exposes a native owner address.");
        AddBlockerIfFalse(blockers, !destructorPointerProduced, "native DebugListener destructor gate produces a native owner pointer.");
        AddBlockerIfFalse(blockers, nativeOwnerNonCopyableStorage.NativeDetachEntryLocated, "line-specific setDebugListener(nullptr) detach/clear entry is not available.");
        AddBlockerIfFalse(blockers, nativeOwnerNonCopyableStorage.NativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleReady, "native DebugListener owner lifecycle is not complete.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime debug listener no-throw destructor evidence.");

        foreach (string blocker in nativeOwnerNonCopyableStorage.BlockedPrerequisites)
        {
            if (noThrowNativeDestructorReady &&
                blocker.IndexOf("no-throw destructor", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                continue;
            }

            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeNoThrowDestructorResult(
            ownerDesignSnapshot.Line,
            ownerDesignSnapshot.OwnerId,
            ownerDesignSnapshot.LastStatus,
            ownerDesignSnapshot.LastDiagnostic,
            ownerDesignSnapshot.ReleaseDiagnostic,
            nativeOwnerNonCopyableStorageReady,
            nativeOwnerNonCopyableStorage.NativeOwnerNonCopyableReady,
            nativeOwnerNonCopyableStorage.NativeOwnerCopyBlocked,
            nativeOwnerNonCopyableStorage.NativeOwnerMoveBlocked,
            nativeOwnerNonCopyableStorage.NativeOwnerAddressExposed,
            nativeOwnerNonCopyableStorage.NativeOwnerPointerProduced,
            destructorNoThrowScaffoldReady,
            destructorExceptionEscapeBlocked,
            destructorAddressExposed,
            destructorPointerProduced,
            nativeOwnerNonCopyableStorage.NativeAttachEntryLocated,
            nativeOwnerNonCopyableStorage.NativeDetachEntryLocated,
            noThrowNativeDestructorReady,
            nativeOwnerLifecycleReady,
            processDebugTensorRuntimeReady,
            fullPackageConsumerRuntimeEvidenceReady,
            blockers.ToArray());
    }

    private static void AddBlockerIfFalse(List<string> blockers, bool condition, string blocker)
    {
        if (!condition)
        {
            AddBlocker(blockers, blocker);
        }
    }

    private static void AddBlocker(List<string> blockers, string blocker)
    {
        if (!string.IsNullOrWhiteSpace(blocker) && !blockers.Contains(blocker))
        {
            blockers.Add(blocker);
        }
    }
}

/// <summary>
/// Reports copied DebugListener native owner no-throw destructor diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This result is a destructor gate. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> remain <see langword="false"/> until a real native owner, non-null attach
/// bridge, complete owner lifecycle, and full package consumer callback runtime evidence exist.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public readonly struct TensorRtDebugListenerNativeNoThrowDestructorResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerNativeNoThrowDestructorResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        string lastDiagnostic,
        string releaseDiagnostic,
        bool nativeOwnerNonCopyableStorageReady,
        bool nativeOwnerNonCopyableReady,
        bool nativeOwnerCopyBlocked,
        bool nativeOwnerMoveBlocked,
        bool nativeOwnerAddressExposed,
        bool nativeOwnerPointerProduced,
        bool destructorNoThrowScaffoldReady,
        bool destructorExceptionEscapeBlocked,
        bool destructorAddressExposed,
        bool destructorPointerProduced,
        bool nativeAttachEntryLocated,
        bool nativeDetachEntryLocated,
        bool noThrowNativeDestructorReady,
        bool nativeOwnerLifecycleReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        LastDiagnostic = lastDiagnostic ?? string.Empty;
        ReleaseDiagnostic = releaseDiagnostic ?? string.Empty;
        NativeOwnerNonCopyableStorageReady = nativeOwnerNonCopyableStorageReady;
        NativeOwnerNonCopyableReady = nativeOwnerNonCopyableReady;
        NativeOwnerCopyBlocked = nativeOwnerCopyBlocked;
        NativeOwnerMoveBlocked = nativeOwnerMoveBlocked;
        NativeOwnerAddressExposed = nativeOwnerAddressExposed;
        NativeOwnerPointerProduced = nativeOwnerPointerProduced;
        DestructorNoThrowScaffoldReady = destructorNoThrowScaffoldReady;
        DestructorExceptionEscapeBlocked = destructorExceptionEscapeBlocked;
        DestructorAddressExposed = destructorAddressExposed;
        DestructorPointerProduced = destructorPointerProduced;
        NativeAttachEntryLocated = nativeAttachEntryLocated;
        NativeDetachEntryLocated = nativeDetachEntryLocated;
        NoThrowNativeDestructorReady = noThrowNativeDestructorReady;
        NativeOwnerLifecycleReady = nativeOwnerLifecycleReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this destructor gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-native-nothrow-destructor";

    /// <summary>Gets the callback kind represented by this destructor gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "destructor-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied diagnostic owner id. This is not a pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the copied last status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the copied last diagnostic. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets the copied release diagnostic. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string ReleaseDiagnostic { get; }

    /// <summary>Gets whether non-copyable storage evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerNonCopyableStorageReady { get; }

    /// <summary>Gets whether native owner storage is non-copyable in source-visible scaffold evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerNonCopyableReady { get; }

    /// <summary>Gets whether native owner copy construction and assignment are blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerCopyBlocked { get; }

    /// <summary>Gets whether native owner move construction and assignment are blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerMoveBlocked { get; }

    /// <summary>Gets whether the public surface exposes a native owner address. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerAddressExposed { get; }

    /// <summary>Gets whether the public surface produces a native owner pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerPointerProduced { get; }

    /// <summary>Gets whether the source-visible destructor scaffold is no-throw. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DestructorNoThrowScaffoldReady { get; }

    /// <summary>Gets whether destructor exception escape is blocked by the scaffold boundary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DestructorExceptionEscapeBlocked { get; }

    /// <summary>Gets whether the destructor gate exposes a native owner address. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DestructorAddressExposed { get; }

    /// <summary>Gets whether the destructor gate produces a native owner pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DestructorPointerProduced { get; }

    /// <summary>Gets whether a native non-null DebugListener attach entry has been located. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeAttachEntryLocated { get; }

    /// <summary>Gets whether a native DebugListener detach/clear entry has been located. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeDetachEntryLocated { get; }

    /// <summary>Gets whether the native owner destructor has source-visible no-throw scaffold evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NoThrowNativeDestructorReady { get; }

    /// <summary>Gets whether native owner lifecycle implementation is complete. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool NativeOwnerLifecycleReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether the native attach bridge can be implemented without unresolved owner blockers. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanImplementNativeAttach =>
        NativeOwnerNonCopyableReady &&
        NoThrowNativeDestructorReady &&
        NativeAttachEntryLocated &&
        NativeOwnerLifecycleReady;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof =>
        CanImplementNativeAttach &&
        ProcessDebugTensorRuntimeReady &&
        FullPackageConsumerRuntimeEvidenceReady;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the destructor gate status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => NoThrowNativeDestructorReady ? "destructor-gate-ready" : "destructor-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-native-nothrow-destructor; RuntimeEvidenceKind=destructor-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; NativeOwnerNonCopyableStorageReady=" + NativeOwnerNonCopyableStorageReady + "; " +
        "NativeOwnerNonCopyableReady=" + NativeOwnerNonCopyableReady + "; " +
        "NativeOwnerCopyBlocked=" + NativeOwnerCopyBlocked + "; " +
        "NativeOwnerMoveBlocked=" + NativeOwnerMoveBlocked + "; " +
        "NativeOwnerAddressExposed=" + NativeOwnerAddressExposed + "; " +
        "NativeOwnerPointerProduced=" + NativeOwnerPointerProduced + "; " +
        "DestructorNoThrowScaffoldReady=" + DestructorNoThrowScaffoldReady + "; " +
        "DestructorExceptionEscapeBlocked=" + DestructorExceptionEscapeBlocked + "; " +
        "DestructorAddressExposed=" + DestructorAddressExposed + "; " +
        "DestructorPointerProduced=" + DestructorPointerProduced + "; " +
        "NativeAttachEntryLocated=" + NativeAttachEntryLocated + "; " +
        "NativeDetachEntryLocated=" + NativeDetachEntryLocated + "; " +
        "NoThrowNativeDestructorReady=" + NoThrowNativeDestructorReady + "; " +
        "NativeOwnerLifecycleReady=" + NativeOwnerLifecycleReady + "; " +
        "CanImplementNativeAttach=" + CanImplementNativeAttach + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:owner={OwnerId}:proof={IsRealCallbackRuntimeProof}";
    }
}
