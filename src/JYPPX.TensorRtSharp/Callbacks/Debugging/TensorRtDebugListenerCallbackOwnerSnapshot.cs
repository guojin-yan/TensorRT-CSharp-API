using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied diagnostics for the debug-listener callback owner design gate.
/// 表示 debug-listener callback owner 设计门禁复制出的诊断信息。
/// </summary>
/// <remarks>
/// This snapshot intentionally exposes no native handle, callback owner pointer, borrowed TensorRT tensor pointer, debug
/// tensor data pointer, or tensor buffer ownership. It is not proof that TensorRT has invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 该快照有意不暴露 native handle、callback owner pointer、borrowed TensorRT tensor pointer、debug tensor data pointer 或
/// tensor buffer ownership。它不证明 TensorRT 已调用 <c>IDebugListener::processDebugTensor</c>。
/// </remarks>
public readonly struct TensorRtDebugListenerCallbackOwnerSnapshot
{
    internal TensorRtDebugListenerCallbackOwnerSnapshot(
        long ownerId,
        string operation,
        TensorRtApiLine line,
        BridgeStatusCode lastStatus,
        string tensorName,
        TensorRtDataType dataType,
        TensorRtTensorLocation location,
        int shapeRank,
        string shapeSummary,
        bool isInput,
        bool isOutput,
        bool isShapeTensor,
        bool isExecutionTensor,
        long invocationCount,
        long processDebugTensorCount,
        long failureCount,
        long inFlightCallbackCount,
        long maxInFlightCallbackCount,
        int activeGateCallCount,
        long releaseHookCount,
        bool callbackStatePinned,
        bool delegatePinned,
        bool disposeRequested,
        string lastDiagnostic,
        string releaseDiagnostic)
    {
        OwnerId = ownerId;
        Operation = operation ?? string.Empty;
        Line = line;
        LastStatus = lastStatus;
        TensorName = tensorName ?? string.Empty;
        DataType = dataType;
        Location = location;
        ShapeRank = shapeRank;
        ShapeSummary = shapeSummary ?? "[]";
        IsInput = isInput;
        IsOutput = isOutput;
        IsShapeTensor = isShapeTensor;
        IsExecutionTensor = isExecutionTensor;
        InvocationCount = invocationCount;
        ProcessDebugTensorCount = processDebugTensorCount;
        FailureCount = failureCount;
        InFlightCallbackCount = inFlightCallbackCount;
        MaxInFlightCallbackCount = maxInFlightCallbackCount;
        ActiveGateCallCount = activeGateCallCount;
        ReleaseHookCount = releaseHookCount;
        CallbackStatePinned = callbackStatePinned;
        DelegatePinned = delegatePinned;
        DisposeRequested = disposeRequested;
        LastDiagnostic = lastDiagnostic ?? string.Empty;
        ReleaseDiagnostic = releaseDiagnostic ?? string.Empty;
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 获取 readiness 用于识别该门禁的 marker。</summary>
    public string EvidenceKind => "debug-listener-callback-owner-design";

    /// <summary>Gets the callback kind represented by this diagnostic snapshot. 获取该诊断快照代表的 callback 类型。</summary>
    public string CallbackKind => "debug-listener-prototype";

    /// <summary>Gets the runtime evidence kind. 获取 runtime 证据类型。</summary>
    public string RuntimeEvidenceKind => "not-present";

    /// <summary>Gets whether this snapshot proves a real TensorRT callback runtime. 获取该快照是否证明真实 TensorRT callback runtime。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this snapshot as real callback runtime proof. 获取 readiness 是否可将该快照提升为真实 callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the copied owner id. 获取复制出的 owner id。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the copied operation label. 获取复制出的操作标签。</summary>
    public string Operation { get; }

    /// <summary>Gets the TensorRT API line used for this diagnostic. 获取该诊断使用的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the last copied bridge status. 获取最近一次复制出的 bridge status。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the copied debug tensor name. 获取复制出的 debug tensor 名称。</summary>
    public string TensorName { get; }

    /// <summary>Gets the copied debug tensor data type. 获取复制出的 debug tensor 数据类型。</summary>
    public TensorRtDataType DataType { get; }

    /// <summary>Gets the copied debug tensor location. 获取复制出的 debug tensor 位置。</summary>
    public TensorRtTensorLocation Location { get; }

    /// <summary>Gets the copied debug tensor shape rank. 获取复制出的 debug tensor shape rank。</summary>
    public int ShapeRank { get; }

    /// <summary>Gets a compact copied debug tensor shape summary. 获取紧凑的复制 debug tensor shape 摘要。</summary>
    public string ShapeSummary { get; }

    /// <summary>Gets whether the copied metadata describes an input tensor. 获取复制出的元数据是否表示输入 tensor。</summary>
    public bool IsInput { get; }

    /// <summary>Gets whether the copied metadata describes an output tensor. 获取复制出的元数据是否表示输出 tensor。</summary>
    public bool IsOutput { get; }

    /// <summary>Gets whether the copied metadata describes a shape tensor. 获取复制出的元数据是否表示 shape tensor。</summary>
    public bool IsShapeTensor { get; }

    /// <summary>Gets whether the copied metadata describes an execution tensor. 获取复制出的元数据是否表示 execution tensor。</summary>
    public bool IsExecutionTensor { get; }

    /// <summary>Gets the copied invocation count. 获取复制出的调用次数。</summary>
    public long InvocationCount { get; }

    /// <summary>Gets the copied processDebugTensor diagnostic count. 获取复制出的 processDebugTensor 诊断次数。</summary>
    public long ProcessDebugTensorCount { get; }

    /// <summary>Gets the copied failure count. 获取复制出的失败次数。</summary>
    public long FailureCount { get; }

    /// <summary>Gets the copied in-flight callback count. 获取复制出的 in-flight callback 数量。</summary>
    public long InFlightCallbackCount { get; }

    /// <summary>Gets the copied maximum in-flight callback count. 获取复制出的最大 in-flight callback 数量。</summary>
    public long MaxInFlightCallbackCount { get; }

    /// <summary>Gets the copied active gate call count. 获取复制出的 active gate 调用数量。</summary>
    public int ActiveGateCallCount { get; }

    /// <summary>Gets the copied release hook count. 获取复制出的 release hook 次数。</summary>
    public long ReleaseHookCount { get; }

    /// <summary>Gets whether managed callback state remains pinned. 获取托管 callback state 是否仍被 pin 住。</summary>
    public bool CallbackStatePinned { get; }

    /// <summary>Gets whether the managed delegate remains pinned. 获取托管 delegate 是否仍被 pin 住。</summary>
    public bool DelegatePinned { get; }

    /// <summary>Gets whether dispose has been requested. 获取是否已请求释放。</summary>
    public bool DisposeRequested { get; }

    /// <summary>Gets whether this design owner is attached to a TensorRT execution context. 获取该设计 owner 是否已绑定到 TensorRT execution context。</summary>
    public bool IsAttached => false;

    /// <summary>Gets whether debug tensor metadata was copied into this snapshot. 获取 debug tensor metadata 是否已复制到该快照。</summary>
    public bool DebugTensorMetadataCopied => ProcessDebugTensorCount > 0 && !string.IsNullOrEmpty(TensorName);

    /// <summary>Gets whether a debug tensor pointer is exposed by this public API. 获取该 public API 是否暴露 debug tensor pointer。</summary>
    public bool DebugTensorPointerExposed => false;

    /// <summary>Gets whether a debug tensor pointer was produced by this diagnostic gate. 获取该诊断门禁是否产生 debug tensor pointer。</summary>
    public bool DebugTensorPointerProduced => false;

    /// <summary>Gets whether a borrowed debug tensor pointer escaped this public API. 获取 borrowed debug tensor pointer 是否逃逸出该 public API。</summary>
    public bool BorrowedDebugTensorPointerEscaped => false;

    /// <summary>Gets the copied diagnostic message. 获取复制出的诊断消息。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets the copied release diagnostic message. 获取复制出的释放诊断消息。</summary>
    public string ReleaseDiagnostic { get; }

    /// <summary>Gets whether the diagnostic completed without failures. 获取诊断是否未出现失败。</summary>
    public bool Succeeded => LastStatus == BridgeStatusCode.Ok && FailureCount == 0 && InFlightCallbackCount == 0;

    /// <summary>Returns a compact diagnostic representation. 返回紧凑的诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:owner={OwnerId}:operation={Operation}:status={LastStatus}:process={ProcessDebugTensorCount}:failures={FailureCount}:proof={IsRealCallbackRuntimeProof}";
    }
}
