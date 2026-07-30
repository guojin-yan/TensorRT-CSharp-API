using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener native owner non-copyable storage evidence before native attach is enabled.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This storage gate reports source-visible no-copy/no-move owner storage diagnostics only. It does not expose a native
/// owner address, does not call <c>setDebugListener(non-null)</c>, and is not proof that TensorRT invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public static class TensorRtDebugListenerNativeOwnerNonCopyableStorage
{
    /// <summary>
    /// Evaluates native owner non-copyable storage evidence from a copied DebugListener owner design snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native owner non-copyable storage result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerNonCopyableStorageResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeOwnerStableIdentityResult nativeOwnerStableIdentity =
            TensorRtDebugListenerNativeOwnerStableIdentity.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, nativeOwnerStableIdentity);
    }

    /// <summary>
    /// Evaluates native owner non-copyable storage evidence from copied owner and stable identity evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <param name="nativeOwnerStableIdentity">The copied native owner stable identity result. 该参数传入已复制的 DebugListener owner、gate 或 runtime proof evidence。</param>
    /// <returns>A pointer-free native owner non-copyable storage result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerNonCopyableStorageResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeOwnerStableIdentityResult nativeOwnerStableIdentity)
    {
        const bool nativeOwnerCopyBlocked = true;
        const bool nativeOwnerMoveBlocked = true;
        const bool nativeOwnerAddressExposed = false;
        const bool nativeOwnerPointerProduced = false;
        const bool noThrowNativeDestructorReady = false;
        const bool nativeOwnerLifecycleReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        bool nativeOwnerStableIdentityReady = nativeOwnerStableIdentity.StableNativeOwnerIdentityReady;
        bool ownerIdentityDiagnosticsReady = nativeOwnerStableIdentity.OwnerIdentityDiagnosticsReady;
        bool ownerIdentityPointerFree = nativeOwnerStableIdentity.OwnerIdentityPointerFree;
        bool nativeOwnerNonCopyableReady =
            nativeOwnerStableIdentityReady &&
            ownerIdentityDiagnosticsReady &&
            ownerIdentityPointerFree &&
            nativeOwnerCopyBlocked &&
            nativeOwnerMoveBlocked &&
            !nativeOwnerAddressExposed &&
            !nativeOwnerPointerProduced;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, nativeOwnerStableIdentityReady, "debug-listener-native-owner-stable-identity is not ready for non-copyable storage evaluation.");
        AddBlockerIfFalse(blockers, ownerIdentityDiagnosticsReady, "DebugListener owner id and diagnostic identity chain is incomplete.");
        AddBlockerIfFalse(blockers, ownerIdentityPointerFree, "DebugListener owner identity diagnostics still expose or produce a borrowed pointer.");
        AddBlockerIfFalse(blockers, nativeOwnerCopyBlocked, "native DebugListener owner copy construction/assignment is not blocked.");
        AddBlockerIfFalse(blockers, nativeOwnerMoveBlocked, "native DebugListener owner move construction/assignment is not blocked.");
        AddBlockerIfFalse(blockers, !nativeOwnerAddressExposed, "native DebugListener owner address is exposed by the public surface.");
        AddBlockerIfFalse(blockers, !nativeOwnerPointerProduced, "native DebugListener owner pointer is produced by the public surface.");
        AddBlockerIfFalse(blockers, nativeOwnerStableIdentity.NativeDetachEntryLocated, "line-specific setDebugListener(nullptr) detach/clear entry is not available.");
        AddBlockerIfFalse(blockers, nativeOwnerStableIdentity.NativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, noThrowNativeDestructorReady, "native DebugListener owner no-throw destructor has not been promoted beyond scaffold evidence.");
        AddBlockerIfFalse(blockers, nativeOwnerLifecycleReady, "native DebugListener owner lifecycle is not complete.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime debug listener non-copyable storage evidence.");

        foreach (string blocker in nativeOwnerStableIdentity.BlockedPrerequisites)
        {
            if (nativeOwnerNonCopyableReady &&
                blocker.IndexOf("non-copyable storage", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                continue;
            }

            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeOwnerNonCopyableStorageResult(
            ownerDesignSnapshot.Line,
            ownerDesignSnapshot.OwnerId,
            ownerDesignSnapshot.LastStatus,
            ownerDesignSnapshot.LastDiagnostic,
            ownerDesignSnapshot.ReleaseDiagnostic,
            nativeOwnerStableIdentityReady,
            ownerIdentityDiagnosticsReady,
            ownerIdentityPointerFree,
            nativeOwnerNonCopyableReady,
            nativeOwnerCopyBlocked,
            nativeOwnerMoveBlocked,
            nativeOwnerAddressExposed,
            nativeOwnerPointerProduced,
            nativeOwnerStableIdentity.NativeAttachEntryLocated,
            nativeOwnerStableIdentity.NativeDetachEntryLocated,
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
