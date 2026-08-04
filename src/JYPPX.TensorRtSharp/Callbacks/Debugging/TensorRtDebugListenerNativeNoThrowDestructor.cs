using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

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
