using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;
/// <summary>
/// Reports copied DebugListener borrowed tensor diagnostics without exposing native pointers.
/// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。
/// </summary>
/// <remarks>
/// This result is a safety gate. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> remain <see langword="false"/> until real TensorRT callback smoke proves
/// <c>IDebugListener::processDebugTensor</c> borrowed tensor lifetime behavior.
/// 该说明强调当前结果属于 DebugListener 安全门禁、runtime smoke 记录或 proof precheck，不代表未满足条件时已有真实 callback proof。
/// </remarks>
public readonly struct TensorRtDebugListenerBorrowedTensorSafetyGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDebugListenerBorrowedTensorSafetyGateResult(
        TensorRtApiLine line,
        string tensorName,
        TensorRtDataType dataType,
        TensorRtTensorLocation location,
        int shapeRank,
        string shapeSummary,
        bool isInput,
        bool isOutput,
        bool isShapeTensor,
        bool isExecutionTensor,
        long processDebugTensorCount,
        bool attachDetachDesignGateReady,
        bool ownerDesignReady,
        bool debugTensorMetadataCopied,
        bool pointerFreeSurfaceReady,
        bool borrowedDebugTensorPointerEscapeBlocked,
        bool borrowedDebugTensorLifetimeReady,
        bool borrowedDebugTensorDataLifetimeReady,
        bool processDebugTensorRuntimeReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        TensorName = tensorName ?? string.Empty;
        DataType = dataType;
        Location = location;
        ShapeRank = shapeRank;
        ShapeSummary = shapeSummary ?? "[]";
        IsInput = isInput;
        IsOutput = isOutput;
        IsShapeTensor = isShapeTensor;
        IsExecutionTensor = isExecutionTensor;
        ProcessDebugTensorCount = processDebugTensorCount;
        AttachDetachDesignGateReady = attachDetachDesignGateReady;
        OwnerDesignReady = ownerDesignReady;
        DebugTensorMetadataCopied = debugTensorMetadataCopied;
        PointerFreeSurfaceReady = pointerFreeSurfaceReady;
        BorrowedDebugTensorPointerEscapeBlocked = borrowedDebugTensorPointerEscapeBlocked;
        BorrowedDebugTensorLifetimeReady = borrowedDebugTensorLifetimeReady;
        BorrowedDebugTensorDataLifetimeReady = borrowedDebugTensorDataLifetimeReady;
        ProcessDebugTensorRuntimeReady = processDebugTensorRuntimeReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this safety gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string EvidenceKind => "debug-listener-borrowed-tensor-safety-gate";

    /// <summary>Gets the callback kind represented by this safety gate. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string CallbackKind => "debug-listener-borrowed-tensor";

    /// <summary>Gets the runtime evidence kind. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this result proves a real TensorRT callback runtime. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this result as real callback runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied debug tensor name. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string TensorName { get; }

    /// <summary>Gets the copied debug tensor data type. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtDataType DataType { get; }

    /// <summary>Gets the copied debug tensor location. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public TensorRtTensorLocation Location { get; }

    /// <summary>Gets the copied debug tensor shape rank. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public int ShapeRank { get; }

    /// <summary>Gets the copied debug tensor shape summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string ShapeSummary { get; }

    /// <summary>Gets whether the copied metadata describes an input tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsInput { get; }

    /// <summary>Gets whether the copied metadata describes an output tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsOutput { get; }

    /// <summary>Gets whether the copied metadata describes a shape tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsShapeTensor { get; }

    /// <summary>Gets whether the copied metadata describes an execution tensor. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool IsExecutionTensor { get; }

    /// <summary>Gets the copied processDebugTensor diagnostic count. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public long ProcessDebugTensorCount { get; }

    /// <summary>Gets whether attach/detach design evidence was ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool AttachDetachDesignGateReady { get; }

    /// <summary>Gets whether owner design evidence was clean. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool OwnerDesignReady { get; }

    /// <summary>Gets whether debug tensor metadata was copied. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool DebugTensorMetadataCopied { get; }

    /// <summary>Gets whether the public surface remains pointer-free. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool PointerFreeSurfaceReady { get; }

    /// <summary>Gets whether borrowed debug tensor pointers are blocked from escaping public APIs. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorPointerEscapeBlocked { get; }

    /// <summary>Gets whether borrowed debug tensor pointer lifetime is runtime ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorLifetimeReady { get; }

    /// <summary>Gets whether borrowed debug tensor data buffer lifetime is runtime ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool BorrowedDebugTensorDataLifetimeReady { get; }

    /// <summary>Gets whether processDebugTensor runtime callback execution is ready. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool ProcessDebugTensorRuntimeReady { get; }

    /// <summary>Gets whether full package consumer smoke has produced real callback runtime evidence. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether this safety gate has enough copied evidence to feed the runtime-proof precheck. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool SafetyGateReady =>
        AttachDetachDesignGateReady &&
        OwnerDesignReady &&
        DebugTensorMetadataCopied &&
        PointerFreeSurfaceReady &&
        BorrowedDebugTensorPointerEscapeBlocked;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public bool CanAttemptRuntimeProof =>
        SafetyGateReady &&
        BorrowedDebugTensorLifetimeReady &&
        BorrowedDebugTensorDataLifetimeReady &&
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

    /// <summary>Gets the safety gate status. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Status => SafetyGateReady ? "safety-gate-ready" : "safety-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    public string Diagnostic =>
        "debug-listener-borrowed-tensor-safety-gate; RuntimeEvidenceKind=design-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; SafetyGateReady=" + SafetyGateReady + "; " +
        "BorrowedDebugTensorPointerEscapeBlocked=" + BorrowedDebugTensorPointerEscapeBlocked + "; " +
        "BorrowedDebugTensorLifetimeReady=" + BorrowedDebugTensorLifetimeReady + "; " +
        "BorrowedDebugTensorDataLifetimeReady=" + BorrowedDebugTensorDataLifetimeReady + "; " +
        "ProcessDebugTensorRuntimeReady=" + ProcessDebugTensorRuntimeReady + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能替代真实 TensorRT callback runtime proof。</summary>
    /// <returns>A diagnostic string. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:lifetime={BorrowedDebugTensorLifetimeReady}:proof={IsRealCallbackRuntimeProof}";
    }
}
