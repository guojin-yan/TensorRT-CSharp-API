using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener native owner lifecycle gate evidence before native attach is enabled.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This lifecycle gate reports source-visible release/detach/drain/unpin scaffold diagnostics only. It does not create a
/// native owner, does not call <c>setDebugListener(non-null)</c>, and is not proof that TensorRT invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁或 precheck，不代表 TensorRT 已真实触发 callback。
/// </remarks>
public static class TensorRtDebugListenerNativeOwnerLifecycleGate
{
    /// <summary>
    /// Evaluates native owner lifecycle gate evidence from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free native owner lifecycle gate result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerLifecycleGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeNoThrowDestructorResult nativeNoThrowDestructor =
            TensorRtDebugListenerNativeNoThrowDestructor.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, nativeNoThrowDestructor);
    }

    /// <summary>
    /// Evaluates native owner lifecycle gate evidence from copied owner and no-throw destructor evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeNoThrowDestructor">The copied native no-throw destructor result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free native owner lifecycle gate result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerLifecycleGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeNoThrowDestructorResult nativeNoThrowDestructor)
    {
        const bool lifecycleScaffoldReady = true;
        const bool releaseHookOrderingGateReady = true;
        const bool disposeIdempotencyGateReady = true;
        const bool inFlightDrainGateReady = true;
        const bool callbackStateUnpinAfterDetachGateReady = true;
        const bool delegateUnpinAfterDetachGateReady = true;
        const bool lifecycleAddressExposed = false;
        const bool lifecyclePointerProduced = false;
        const bool nativeAttachEntryLocated = false;
        const bool nativeOwnerLifecycleReady = false;
        const bool nativeVTableDesignReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        bool nativeNoThrowDestructorGateReady = nativeNoThrowDestructor.NoThrowNativeDestructorReady;
        bool managedDisposeSnapshotReady =
            ownerDesignSnapshot.DisposeRequested &&
            ownerDesignSnapshot.ReleaseHookCount > 0 &&
            ownerDesignSnapshot.InFlightCallbackCount == 0 &&
            !ownerDesignSnapshot.CallbackStatePinned &&
            !ownerDesignSnapshot.DelegatePinned;
        bool lifecycleGateReady =
            nativeNoThrowDestructorGateReady &&
            nativeNoThrowDestructor.NativeOwnerNonCopyableReady &&
            nativeNoThrowDestructor.NativeOwnerCopyBlocked &&
            nativeNoThrowDestructor.NativeOwnerMoveBlocked &&
            !nativeNoThrowDestructor.NativeOwnerAddressExposed &&
            !nativeNoThrowDestructor.NativeOwnerPointerProduced &&
            nativeNoThrowDestructor.DestructorNoThrowScaffoldReady &&
            nativeNoThrowDestructor.DestructorExceptionEscapeBlocked &&
            !nativeNoThrowDestructor.DestructorAddressExposed &&
            !nativeNoThrowDestructor.DestructorPointerProduced &&
            managedDisposeSnapshotReady &&
            lifecycleScaffoldReady &&
            releaseHookOrderingGateReady &&
            disposeIdempotencyGateReady &&
            inFlightDrainGateReady &&
            callbackStateUnpinAfterDetachGateReady &&
            delegateUnpinAfterDetachGateReady &&
            !lifecycleAddressExposed &&
            !lifecyclePointerProduced;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, nativeNoThrowDestructorGateReady, "debug-listener-native-nothrow-destructor is not ready for owner lifecycle gate evaluation.");
        AddBlockerIfFalse(blockers, nativeNoThrowDestructor.NativeOwnerNonCopyableReady, "native DebugListener owner non-copyable storage is not ready for lifecycle gate evaluation.");
        AddBlockerIfFalse(blockers, nativeNoThrowDestructor.NativeOwnerCopyBlocked, "native DebugListener owner copy construction/assignment is not blocked.");
        AddBlockerIfFalse(blockers, nativeNoThrowDestructor.NativeOwnerMoveBlocked, "native DebugListener owner move construction/assignment is not blocked.");
        AddBlockerIfFalse(blockers, !nativeNoThrowDestructor.NativeOwnerAddressExposed, "native DebugListener owner address is exposed by the public surface.");
        AddBlockerIfFalse(blockers, !nativeNoThrowDestructor.NativeOwnerPointerProduced, "native DebugListener owner pointer is produced by the public surface.");
        AddBlockerIfFalse(blockers, nativeNoThrowDestructor.DestructorNoThrowScaffoldReady, "native DebugListener owner destructor no-throw scaffold is not ready.");
        AddBlockerIfFalse(blockers, nativeNoThrowDestructor.DestructorExceptionEscapeBlocked, "native DebugListener owner destructor exception escape is not blocked.");
        AddBlockerIfFalse(blockers, !nativeNoThrowDestructor.DestructorAddressExposed, "native DebugListener destructor gate exposes a native owner address.");
        AddBlockerIfFalse(blockers, !nativeNoThrowDestructor.DestructorPointerProduced, "native DebugListener destructor gate produces a native owner pointer.");
        AddBlockerIfFalse(blockers, managedDisposeSnapshotReady, "managed DebugListener dispose snapshot is incomplete for lifecycle gate evaluation.");
        AddBlockerIfFalse(blockers, lifecycleScaffoldReady, "native DebugListener owner lifecycle scaffold is not source-visible.");
        AddBlockerIfFalse(blockers, releaseHookOrderingGateReady, "native DebugListener release hook ordering scaffold is not source-visible.");
        AddBlockerIfFalse(blockers, disposeIdempotencyGateReady, "native DebugListener dispose/release idempotency scaffold is not source-visible.");
        AddBlockerIfFalse(blockers, inFlightDrainGateReady, "native DebugListener in-flight drain scaffold is not source-visible.");
        AddBlockerIfFalse(blockers, callbackStateUnpinAfterDetachGateReady, "native DebugListener callback state post-detach unpin scaffold is not source-visible.");
        AddBlockerIfFalse(blockers, delegateUnpinAfterDetachGateReady, "native DebugListener delegate post-detach unpin scaffold is not source-visible.");
        AddBlockerIfFalse(blockers, !lifecycleAddressExposed, "native DebugListener lifecycle gate exposes a native owner address.");
        AddBlockerIfFalse(blockers, !lifecyclePointerProduced, "native DebugListener lifecycle gate produces a native owner pointer.");
        AddBlockerIfFalse(blockers, nativeNoThrowDestructor.NativeDetachEntryLocated, "line-specific setDebugListener(nullptr) detach/clear entry is not available.");
        AddBlockerIfFalse(blockers, nativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleReady, "native DebugListener owner lifecycle is not complete beyond source-visible scaffold evidence.");
        AddBlockerIfFalse(blockers, nativeVTableDesignReady, "native IDebugListener vtable design is not implemented.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime debug listener lifecycle evidence.");

        foreach (string blocker in nativeNoThrowDestructor.BlockedPrerequisites)
        {
            if (lifecycleGateReady &&
                (blocker.IndexOf("native DebugListener owner lifecycle is not complete", StringComparison.OrdinalIgnoreCase) >= 0 ||
                 blocker.IndexOf("native DebugListener owner lifecycle is not complete.", StringComparison.OrdinalIgnoreCase) >= 0))
            {
                continue;
            }

            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeOwnerLifecycleGateResult(
            ownerDesignSnapshot.Line,
            ownerDesignSnapshot.OwnerId,
            ownerDesignSnapshot.LastStatus,
            ownerDesignSnapshot.ReleaseHookCount,
            ownerDesignSnapshot.InFlightCallbackCount,
            ownerDesignSnapshot.CallbackStatePinned,
            ownerDesignSnapshot.DelegatePinned,
            ownerDesignSnapshot.DisposeRequested,
            ownerDesignSnapshot.LastDiagnostic,
            ownerDesignSnapshot.ReleaseDiagnostic,
            nativeNoThrowDestructorGateReady,
            nativeNoThrowDestructor.NativeOwnerNonCopyableReady,
            nativeNoThrowDestructor.NativeOwnerCopyBlocked,
            nativeNoThrowDestructor.NativeOwnerMoveBlocked,
            nativeNoThrowDestructor.NativeOwnerAddressExposed,
            nativeNoThrowDestructor.NativeOwnerPointerProduced,
            nativeNoThrowDestructor.DestructorNoThrowScaffoldReady,
            nativeNoThrowDestructor.DestructorExceptionEscapeBlocked,
            nativeNoThrowDestructor.DestructorAddressExposed,
            nativeNoThrowDestructor.DestructorPointerProduced,
            managedDisposeSnapshotReady,
            lifecycleScaffoldReady,
            releaseHookOrderingGateReady,
            disposeIdempotencyGateReady,
            inFlightDrainGateReady,
            callbackStateUnpinAfterDetachGateReady,
            delegateUnpinAfterDetachGateReady,
            lifecycleAddressExposed,
            lifecyclePointerProduced,
            nativeAttachEntryLocated,
            nativeNoThrowDestructor.NativeDetachEntryLocated,
            nativeNoThrowDestructor.NoThrowNativeDestructorReady,
            nativeOwnerLifecycleReady,
            nativeVTableDesignReady,
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
