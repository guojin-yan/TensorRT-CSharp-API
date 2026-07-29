using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates pointer-free borrowed debug tensor metadata runtime gate evidence.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This gate copies debug tensor name, type, location, shape, and flags into managed diagnostics. It does not expose a
/// TensorRT debug tensor pointer, does not expose a debug tensor data pointer, and is not proof that TensorRT invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 该说明强调当前结果属于 DebugListener 安全门禁或 precheck，不代表 TensorRT 已真实触发 callback。
/// </remarks>
public static class TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate
{
    /// <summary>
    /// Evaluates borrowed debug tensor metadata gate evidence from a copied DebugListener owner snapshot.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free borrowed metadata gate result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot)
    {
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(ownerDesignSnapshot);
        TensorRtDebugListenerNoThrowVTableCallbackStubResult callbackStubGate =
            TensorRtDebugListenerNoThrowVTableCallbackStub.Evaluate(ownerDesignSnapshot);
        return Evaluate(ownerDesignSnapshot, borrowedTensorSafetyGate, callbackStubGate);
    }

    /// <summary>
    /// Evaluates borrowed debug tensor metadata gate evidence from copied owner, safety gate, and callback stub evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="callbackStubGate">The copied no-throw vtable callback stub result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free borrowed metadata gate result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate,
        TensorRtDebugListenerNoThrowVTableCallbackStubResult callbackStubGate)
    {
        bool lineSupportsDebugListener =
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt10 ||
            ownerDesignSnapshot.Line == TensorRtApiLine.TensorRt11;
        bool tensorNameCopied = !string.IsNullOrWhiteSpace(ownerDesignSnapshot.TensorName);
        string tensorName = ownerDesignSnapshot.TensorName ?? string.Empty;
        string shapeSummary = ownerDesignSnapshot.ShapeSummary ?? "[]";
        int tensorNameLength = tensorName.Length;
        bool tensorTypeCopied = Enum.IsDefined(typeof(TensorRtDataType), ownerDesignSnapshot.DataType);
        bool tensorLocationCopied = Enum.IsDefined(typeof(TensorRtTensorLocation), ownerDesignSnapshot.Location);
        bool tensorShapeCopied = ownerDesignSnapshot.ShapeRank >= 0 && !string.IsNullOrWhiteSpace(ownerDesignSnapshot.ShapeSummary);
        bool tensorFlagsCopied = true;
        bool borrowedDebugTensorPointerEscapeBlocked =
            borrowedTensorSafetyGate.BorrowedDebugTensorPointerEscapeBlocked &&
            callbackStubGate.BorrowedDebugTensorPointerEscapeBlocked &&
            !ownerDesignSnapshot.DebugTensorPointerExposed &&
            !ownerDesignSnapshot.DebugTensorPointerProduced &&
            !ownerDesignSnapshot.BorrowedDebugTensorPointerEscaped;
        const bool borrowedDebugTensorDataPointerEscapeBlocked = true;
        bool borrowedDebugTensorMetadataCopyReady =
            ownerDesignSnapshot.DebugTensorMetadataCopied &&
            borrowedTensorSafetyGate.DebugTensorMetadataCopied &&
            callbackStubGate.BorrowedDebugTensorMetadataCopyReady &&
            tensorNameCopied &&
            tensorTypeCopied &&
            tensorLocationCopied &&
            tensorShapeCopied &&
            tensorFlagsCopied &&
            borrowedDebugTensorPointerEscapeBlocked &&
            borrowedDebugTensorDataPointerEscapeBlocked;
        bool metadataGateReady =
            lineSupportsDebugListener &&
            borrowedTensorSafetyGate.SafetyGateReady &&
            callbackStubGate.CallbackStubGateReady &&
            borrowedDebugTensorMetadataCopyReady;

        const bool debugTensorPointerExposed = false;
        const bool debugTensorDataPointerExposed = false;
        const bool borrowedDebugTensorLifetimeReady = false;
        const bool borrowedDebugTensorDataLifetimeReady = false;
        const bool setDebugListenerNonNullEnabled = false;
        const bool nativeVTableInstalled = false;
        const bool processDebugTensorRuntimeReady = false;

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, lineSupportsDebugListener, "TensorRT 10 or 11 IDebugListener line support has not been selected.");
        AddBlockerIfFalse(blockers, borrowedTensorSafetyGate.SafetyGateReady, "debug-listener-borrowed-tensor-safety-gate is not ready for metadata runtime gate evaluation.");
        AddBlockerIfFalse(blockers, callbackStubGate.CallbackStubGateReady, "debug-listener-nothrow-vtable-callback-stub is not ready for metadata runtime gate evaluation.");
        AddBlockerIfFalse(blockers, tensorNameCopied, "debug tensor name was not copied.");
        AddBlockerIfFalse(blockers, tensorTypeCopied, "debug tensor type metadata was not copied.");
        AddBlockerIfFalse(blockers, tensorLocationCopied, "debug tensor location metadata was not copied.");
        AddBlockerIfFalse(blockers, tensorShapeCopied, "debug tensor shape metadata was not copied.");
        AddBlockerIfFalse(blockers, tensorFlagsCopied, "debug tensor flags were not copied.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorMetadataCopyReady, "borrowed debug tensor metadata copy gate is incomplete.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorPointerEscapeBlocked, "borrowed debug tensor pointer escape is not blocked.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorDataPointerEscapeBlocked, "borrowed debug tensor data pointer escape is not blocked.");
        AddBlockerIfFalse(blockers, !debugTensorPointerExposed, "borrowed debug tensor metadata gate exposes a debug tensor pointer.");
        AddBlockerIfFalse(blockers, !debugTensorDataPointerExposed, "borrowed debug tensor metadata gate exposes a debug tensor data pointer.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorLifetimeReady, "borrowed debug tensor lifetime has not been proven against real TensorRT callback execution.");
        AddBlockerIfFalse(blockers, borrowedDebugTensorDataLifetimeReady, "borrowed debug tensor data lifetime has not been proven against real TensorRT callback execution.");
        AddBlockerIfFalse(blockers, !setDebugListenerNonNullEnabled, "setDebugListener(non-null) unexpectedly appears enabled during metadata gate evaluation.");
        AddBlockerIfFalse(blockers, !nativeVTableInstalled, "native IDebugListener vtable is unexpectedly installed during metadata gate evaluation.");
        AddBlockerIfFalse(blockers, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime callback has not been implemented.");
        AddBlocker(blockers, "borrowed-debug-tensor-metadata-gate is non-proof evidence and must not be promoted to real-callback-runtime.");
        AddBlocker(blockers, "full package consumer smoke has not emitted real-callback-runtime borrowed debug tensor metadata evidence.");

        foreach (string blocker in borrowedTensorSafetyGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        foreach (string blocker in callbackStubGate.BlockedPrerequisites)
        {
            AddBlocker(blockers, blocker);
        }

        string reasonMetadataRuntimeStillBlocked = BuildMetadataRuntimeBlockedReason(
            borrowedDebugTensorLifetimeReady,
            borrowedDebugTensorDataLifetimeReady,
            setDebugListenerNonNullEnabled,
            nativeVTableInstalled,
            processDebugTensorRuntimeReady);

        return new TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult(
            ownerDesignSnapshot.Line,
            ownerDesignSnapshot.OwnerId,
            ownerDesignSnapshot.LastStatus,
            tensorName,
            tensorNameLength,
            ownerDesignSnapshot.DataType,
            ownerDesignSnapshot.Location,
            ownerDesignSnapshot.ShapeRank,
            shapeSummary,
            ownerDesignSnapshot.IsInput,
            ownerDesignSnapshot.IsOutput,
            ownerDesignSnapshot.IsShapeTensor,
            ownerDesignSnapshot.IsExecutionTensor,
            borrowedTensorSafetyGate.SafetyGateReady,
            callbackStubGate.CallbackStubGateReady,
            metadataGateReady,
            tensorNameCopied,
            tensorTypeCopied,
            tensorLocationCopied,
            tensorShapeCopied,
            tensorFlagsCopied,
            borrowedDebugTensorMetadataCopyReady,
            borrowedDebugTensorPointerEscapeBlocked,
            borrowedDebugTensorDataPointerEscapeBlocked,
            debugTensorPointerExposed,
            debugTensorDataPointerExposed,
            borrowedDebugTensorLifetimeReady,
            borrowedDebugTensorDataLifetimeReady,
            setDebugListenerNonNullEnabled,
            nativeVTableInstalled,
            processDebugTensorRuntimeReady,
            reasonMetadataRuntimeStillBlocked,
            blockers.ToArray());
    }

    private static string BuildMetadataRuntimeBlockedReason(
        bool borrowedDebugTensorLifetimeReady,
        bool borrowedDebugTensorDataLifetimeReady,
        bool setDebugListenerNonNullEnabled,
        bool nativeVTableInstalled,
        bool processDebugTensorRuntimeReady)
    {
        List<string> reasons = new List<string>();
        AddBlockerIfFalse(reasons, borrowedDebugTensorLifetimeReady, "borrowed debug tensor lifetime remains unproven.");
        AddBlockerIfFalse(reasons, borrowedDebugTensorDataLifetimeReady, "borrowed debug tensor data lifetime remains unproven.");
        AddBlockerIfFalse(reasons, setDebugListenerNonNullEnabled, "setDebugListener(non-null) remains disabled by design.");
        AddBlockerIfFalse(reasons, nativeVTableInstalled, "native IDebugListener vtable has not been installed.");
        AddBlockerIfFalse(reasons, processDebugTensorRuntimeReady, "IDebugListener::processDebugTensor runtime execution is not implemented.");
        AddBlocker(reasons, "borrowed-debug-tensor-metadata-gate is not real-callback-runtime proof.");
        return string.Join(" ", reasons);
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
