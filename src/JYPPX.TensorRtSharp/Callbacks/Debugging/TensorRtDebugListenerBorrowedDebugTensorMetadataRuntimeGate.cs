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

/// <summary>
/// Reports copied borrowed debug tensor metadata runtime gate diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
/// </summary>
public readonly struct TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult(
        TensorRtApiLine line,
        long ownerId,
        BridgeStatusCode lastStatus,
        string tensorName,
        int tensorNameLength,
        TensorRtDataType dataType,
        TensorRtTensorLocation location,
        int tensorShapeRank,
        string shapeSummary,
        bool isInput,
        bool isOutput,
        bool isShapeTensor,
        bool isExecutionTensor,
        bool borrowedTensorSafetyGateReady,
        bool callbackStubGateReady,
        bool metadataGateReady,
        bool tensorNameCopied,
        bool tensorTypeCopied,
        bool tensorLocationCopied,
        bool tensorShapeCopied,
        bool tensorFlagsCopied,
        bool borrowedDebugTensorMetadataCopyReady,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool borrowedDebugTensorDataPointerEscapeBlocked,
        bool debugTensorPointerExposed,
        bool debugTensorDataPointerExposed,
        bool borrowedDebugTensorLifetimeReady,
        bool borrowedDebugTensorDataLifetimeReady,
        bool setDebugListenerNonNullEnabled,
        bool nativeVTableInstalled,
        bool processDebugTensorRuntimeReady,
        string reasonMetadataRuntimeStillBlocked,
        string[] blockedPrerequisites)
    {
        Line = line;
        OwnerId = ownerId;
        LastStatus = lastStatus;
        TensorName = tensorName ?? string.Empty;
        TensorNameLength = tensorNameLength;
        DataType = dataType;
        Location = location;
        TensorShapeRank = tensorShapeRank;
        ShapeSummary = shapeSummary ?? "[]";
        IsInput = isInput;
        IsOutput = isOutput;
        IsShapeTensor = isShapeTensor;
        IsExecutionTensor = isExecutionTensor;
        BorrowedTensorSafetyGateReady = borrowedTensorSafetyGateReady;
        CallbackStubGateReady = callbackStubGateReady;
        MetadataGateReady = metadataGateReady;
        TensorNameCopied = tensorNameCopied;
        TensorTypeCopied = tensorTypeCopied;
        TensorLocationCopied = tensorLocationCopied;
        TensorShapeCopied = tensorShapeCopied;
        TensorFlagsCopied = tensorFlagsCopied;
        BorrowedDebugTensorMetadataCopyReady = borrowedDebugTensorMetadataCopyReady;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        BorrowedDebugTensorDataPointerEscapeBlocked = borrowedDebugTensorDataPointerEscapeBlocked;
        DebugTensorPointerExposed = debugTensorPointerExposed;
        DebugTensorDataPointerExposed = debugTensorDataPointerExposed;
        BorrowedDebugTensorLifetimeReady = borrowedDebugTensorLifetimeReady;
        BorrowedDebugTensorDataLifetimeReady = borrowedDebugTensorDataLifetimeReady;
        SetDebugListenerNonNullEnabled = setDebugListenerNonNullEnabled;
        NativeVTableInstalled = nativeVTableInstalled;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        ReasonMetadataRuntimeStillBlocked = reasonMetadataRuntimeStillBlocked ?? string.Empty;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this metadata gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-borrowed-debug-tensor-metadata-runtime-gate";

    /// <summary>Gets the callback kind represented by this gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-process-debug-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "borrowed-debug-tensor-metadata-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied diagnostic owner id. This is not a pointer. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the copied last status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the copied debug tensor name. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string TensorName { get; }

    /// <summary>Gets the copied debug tensor name length. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int TensorNameLength { get; }

    /// <summary>Gets the copied debug tensor data type. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public TensorRtDataType DataType { get; }

    /// <summary>Gets the copied debug tensor location. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public TensorRtTensorLocation Location { get; }

    /// <summary>Gets the copied debug tensor shape rank. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int TensorShapeRank { get; }

    /// <summary>Gets the copied debug tensor shape summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string ShapeSummary { get; }

    /// <summary>Gets whether copied metadata describes an input tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsInput { get; }

    /// <summary>Gets whether copied metadata describes an output tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsOutput { get; }

    /// <summary>Gets whether copied metadata describes a shape tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsShapeTensor { get; }

    /// <summary>Gets whether copied metadata describes an execution tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool IsExecutionTensor { get; }

    /// <summary>Gets whether borrowed tensor safety gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedTensorSafetyGateReady { get; }

    /// <summary>Gets whether callback stub gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CallbackStubGateReady { get; }

    /// <summary>Gets whether metadata gate evidence is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool MetadataGateReady { get; }

    /// <summary>Gets whether tensor name metadata was copied. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool TensorNameCopied { get; }

    /// <summary>Gets whether tensor type metadata was copied. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool TensorTypeCopied { get; }

    /// <summary>Gets whether tensor location metadata was copied. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool TensorLocationCopied { get; }

    /// <summary>Gets whether tensor shape metadata was copied. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool TensorShapeCopied { get; }

    /// <summary>Gets whether tensor flags were copied. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool TensorFlagsCopied { get; }

    /// <summary>Gets whether borrowed debug tensor metadata was copied into pointer-free state. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorMetadataCopyReady { get; }

    /// <summary>Gets whether borrowed debug tensor pointer escape remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether borrowed debug tensor data pointer escape remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataPointerEscapeBlocked { get; }

    /// <summary>Gets whether a debug tensor pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorPointerExposed { get; }

    /// <summary>Gets whether a debug tensor data pointer is exposed. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorDataPointerExposed { get; }

    /// <summary>Gets whether borrowed debug tensor lifetime is runtime ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorLifetimeReady { get; }

    /// <summary>Gets whether borrowed debug tensor data lifetime is runtime ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataLifetimeReady { get; }

    /// <summary>Gets whether non-null setDebugListener is enabled. This must remain false for metadata-gate evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool SetDebugListenerNonNullEnabled { get; }

    /// <summary>Gets whether a native vtable is installed. This must remain false for metadata-gate evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool NativeVTableInstalled { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether the callback runtime can be called by TensorRT. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanCallProcessDebugTensorRuntime => false;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof => false;

    /// <summary>Gets whether real runtime proof is still blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool RuntimeProofBlocked => true;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets why metadata runtime remains blocked. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string ReasonMetadataRuntimeStillBlocked { get; }

    /// <summary>Gets copied blocked prerequisites. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the metadata gate status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Status => MetadataGateReady ? "borrowed-debug-tensor-metadata-gate-ready" : "borrowed-debug-tensor-metadata-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-borrowed-debug-tensor-metadata-runtime-gate; RuntimeEvidenceKind=borrowed-debug-tensor-metadata-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; MetadataGateReady=" + MetadataGateReady + "; " +
        "BorrowedTensorSafetyGateReady=" + BorrowedTensorSafetyGateReady + "; " +
        "CallbackStubGateReady=" + CallbackStubGateReady + "; " +
        "TensorNameCopied=" + TensorNameCopied + "; " +
        "TensorNameLength=" + TensorNameLength + "; " +
        "TensorTypeCopied=" + TensorTypeCopied + "; " +
        "TensorLocationCopied=" + TensorLocationCopied + "; " +
        "TensorShapeCopied=" + TensorShapeCopied + "; " +
        "TensorShapeRank=" + TensorShapeRank + "; " +
        "TensorFlagsCopied=" + TensorFlagsCopied + "; " +
        "BorrowedDebugTensorMetadataCopyReady=" + BorrowedDebugTensorMetadataCopyReady + "; " +
        "BorrowedDebugTensorPointerEscapeBlocked=" + BorrowedDebugTensorPointerEscapeBlocked + "; " +
        "BorrowedDebugTensorDataPointerEscapeBlocked=" + BorrowedDebugTensorDataPointerEscapeBlocked + "; " +
        "DebugTensorPointerExposed=" + DebugTensorPointerExposed + "; " +
        "DebugTensorDataPointerExposed=" + DebugTensorDataPointerExposed + "; " +
        "BorrowedDebugTensorLifetimeReady=" + BorrowedDebugTensorLifetimeReady + "; " +
        "BorrowedDebugTensorDataLifetimeReady=" + BorrowedDebugTensorDataLifetimeReady + "; " +
        "SetDebugListenerNonNullEnabled=" + SetDebugListenerNonNullEnabled + "; " +
        "NativeVTableInstalled=" + NativeVTableInstalled + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "CanCallProcessDebugTensorRuntime=" + CanCallProcessDebugTensorRuntime + "; " +
        "CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; " +
        "ReasonMetadataRuntimeStillBlocked=" + ReasonMetadataRuntimeStillBlocked + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:metadata={MetadataGateReady}:proof={IsRealCallbackRuntimeProof}";
    }
}
