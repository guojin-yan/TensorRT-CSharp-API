using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free DebugListener native owner stable identity evidence before a native owner is implemented.
/// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This identity gate copies owner id and diagnostics only. It does not create or expose a native owner pointer, does
/// not call <c>setDebugListener(non-null)</c>, and is not proof that TensorRT invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 DebugListener 回调执行证据。
/// </remarks>
public static class TensorRtDebugListenerNativeOwnerStableIdentity
{
    /// <summary>
    /// Evaluates native owner stable identity evidence from a copied DebugListener owner design snapshot.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free native owner stable identity result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerStableIdentityResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult nativeAttachEntryRuntimeScaffold =
            TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, nativeAttachEntryRuntimeScaffold);
    }

    /// <summary>
    /// Evaluates native owner stable identity evidence from copied owner and native attach entry runtime scaffold evidence.
    /// 中文：该成员报告 DebugListener 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <param name="nativeAttachEntryRuntimeScaffold">The copied native attach entry runtime scaffold result. 中文：该参数是复制后的 DebugListener 边界诊断输入，不包含可借用原生指针。</param>
    /// <returns>A pointer-free native owner stable identity result. 中文：返回复制后的 DebugListener 无裸指针诊断结果。</returns>
    public static TensorRtDebugListenerNativeOwnerStableIdentityResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult nativeAttachEntryRuntimeScaffold)
    {
        bool nativeAttachEntryRuntimeScaffoldReady = nativeAttachEntryRuntimeScaffold.RuntimeScaffoldReady;
        bool ownerIdentityPointerFree =
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;
        bool ownerIdentityDiagnosticsReady =
            nativeAttachEntryRuntimeScaffoldReady &&
            ownerDesignSnapshot.OwnerId > 0 &&
            nativeAttachEntryRuntimeScaffold.OwnerId == ownerDesignSnapshot.OwnerId &&
            ownerDesignSnapshot.LastStatus == BridgeStatusCode.Ok &&
            nativeAttachEntryRuntimeScaffold.LastStatus == BridgeStatusCode.Ok &&
            !string.IsNullOrWhiteSpace(ownerDesignSnapshot.LastDiagnostic) &&
            !string.IsNullOrWhiteSpace(ownerDesignSnapshot.ReleaseDiagnostic);
        bool stableNativeOwnerIdentityReady =
            nativeAttachEntryRuntimeScaffoldReady &&
            ownerIdentityPointerFree &&
            ownerIdentityDiagnosticsReady;

        const bool nativeOwnerNonCopyableReady = false;
        const bool noThrowNativeDestructorReady = false;
        const bool processDebugTensorRuntimeReady = false;
        const bool fullPackageConsumerRuntimeEvidenceReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, nativeAttachEntryRuntimeScaffoldReady, "debug-listener-native-attach-entry-runtime-scaffold is not ready for stable owner identity evaluation.");
        AddBlockerIfFalse(blockers, ownerIdentityPointerFree, "DebugListener owner identity diagnostics still expose or produce a borrowed pointer.");
        AddBlockerIfFalse(blockers, ownerIdentityDiagnosticsReady, "DebugListener owner id and diagnostic identity chain is incomplete.");
        AddBlockerIfFalse(blockers, nativeAttachEntryRuntimeScaffold.NativeDetachEntryLocated, "line-specific setDebugListener(nullptr) detach/clear entry is not available.");
        AddBlockerIfFalse(blockers, nativeAttachEntryRuntimeScaffold.NativeAttachEntryLocated, "line-specific setDebugListener(non-null) native attach entry has not been implemented.");
        AddBlockerIfFalse(blockers, nativeOwnerNonCopyableReady, "native DebugListener owner non-copyable storage has not been implemented.");
        AddBlockerIfFalse(blockers, noThrowNativeDestructorReady, "native DebugListener owner no-throw destructor has not been implemented.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlockerIfFalse(blockers, fullPackageConsumerRuntimeEvidenceReady, "full package consumer smoke has not emitted real-callback-runtime debug listener stable identity evidence.");

        foreach (string blocker in nativeAttachEntryRuntimeScaffold.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        return new TensorRtDebugListenerNativeOwnerStableIdentityResult(
            ownerDesignSnapshot.Line,
            ownerDesignSnapshot.OwnerId,
            ownerDesignSnapshot.LastStatus,
            ownerDesignSnapshot.LastDiagnostic,
            ownerDesignSnapshot.ReleaseDiagnostic,
            nativeAttachEntryRuntimeScaffoldReady,
            stableNativeOwnerIdentityReady,
            nativeOwnerNonCopyableReady,
            ownerIdentityDiagnosticsReady,
            ownerIdentityPointerFree,
            nativeAttachEntryRuntimeScaffold.NativeAttachEntryLocated,
            nativeAttachEntryRuntimeScaffold.NativeDetachEntryLocated,
            noThrowNativeDestructorReady,
            nativeAttachEntryRuntimeScaffold.NativeOwnerLifecycleReady,
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
