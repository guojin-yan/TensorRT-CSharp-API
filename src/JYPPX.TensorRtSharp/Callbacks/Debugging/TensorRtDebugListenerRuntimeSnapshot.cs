using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied runtime state from a native TensorRT IDebugListener owner.
/// 表示从 native TensorRT IDebugListener owner 复制出的运行时状态。
/// </summary>
public readonly struct TensorRtDebugListenerRuntimeSnapshot
{
    private readonly long[] _shapeDimensions;

    internal TensorRtDebugListenerRuntimeSnapshot(
        TensorRtApiLine line,
        ulong ownerId,
        ulong invocationCount,
        ulong failureCount,
        ulong inFlightCallbackCount,
        ulong maxInFlightCallbackCount,
        ulong attachCount,
        ulong detachCount,
        BridgeStatusCode lastStatus,
        bool isAttached,
        bool lastCallbackSucceeded,
        string tensorName,
        TensorRtDataType dataType,
        TensorRtTensorLocation location,
        long[] shapeDimensions,
        string diagnostic)
    {
        Line = line;
        OwnerId = ownerId;
        InvocationCount = invocationCount;
        FailureCount = failureCount;
        InFlightCallbackCount = inFlightCallbackCount;
        MaxInFlightCallbackCount = maxInFlightCallbackCount;
        AttachCount = attachCount;
        DetachCount = detachCount;
        LastStatus = lastStatus;
        IsAttached = isAttached;
        LastCallbackSucceeded = lastCallbackSucceeded;
        TensorName = tensorName ?? string.Empty;
        DataType = dataType;
        Location = location;
        _shapeDimensions = shapeDimensions == null ? Array.Empty<long>() : (long[])shapeDimensions.Clone();
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>Gets the TensorRT line. 获取 TensorRT 版本线。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied native owner id. 获取复制出的 native owner id。</summary>
    public ulong OwnerId { get; }

    /// <summary>Gets the native vtable invocation count. 获取 native vtable 调用次数。</summary>
    public ulong InvocationCount { get; }

    /// <summary>Gets the callback failure count. 获取 callback 失败次数。</summary>
    public ulong FailureCount { get; }

    /// <summary>Gets the callbacks currently in flight. 获取当前 in-flight callback 数量。</summary>
    public ulong InFlightCallbackCount { get; }

    /// <summary>Gets the maximum concurrent callback count. 获取最大并发 callback 数量。</summary>
    public ulong MaxInFlightCallbackCount { get; }

    /// <summary>Gets the successful attach count. 获取成功 attach 次数。</summary>
    public ulong AttachCount { get; }

    /// <summary>Gets the successful detach count. 获取成功 detach 次数。</summary>
    public ulong DetachCount { get; }

    /// <summary>Gets the last copied bridge status. 获取最近 bridge status。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets whether the native listener is attached. 获取 native listener 是否已绑定。</summary>
    public bool IsAttached { get; }

    /// <summary>Gets whether the last callback completed successfully. 获取最近 callback 是否成功。</summary>
    public bool LastCallbackSucceeded { get; }

    /// <summary>Gets the copied tensor name. 获取复制出的 tensor 名称。</summary>
    public string TensorName { get; }

    /// <summary>Gets the copied tensor data type. 获取复制出的 tensor 数据类型。</summary>
    public TensorRtDataType DataType { get; }

    /// <summary>Gets the copied tensor location. 获取复制出的 tensor 位置。</summary>
    public TensorRtTensorLocation Location { get; }

    /// <summary>Gets copied shape dimensions. 获取复制出的 shape 维度。</summary>
    public ReadOnlyCollection<long> ShapeDimensions => Array.AsReadOnly(_shapeDimensions ?? Array.Empty<long>());

    /// <summary>Gets the copied native diagnostic. 获取复制出的 native 诊断。</summary>
    public string Diagnostic { get; }

    /// <summary>Gets whether a real TensorRT callback was observed successfully. 获取是否已成功观察到真实 TensorRT callback。</summary>
    public bool RealCallbackRuntime =>
        InvocationCount > 0 &&
        FailureCount == 0 &&
        InFlightCallbackCount == 0 &&
        LastCallbackSucceeded;

    /// <summary>Gets whether the copied local evidence proves native callback invocation. 获取本地复制证据是否证明 native callback 已调用。</summary>
    public bool IsRealCallbackRuntimeProof => RealCallbackRuntime;

    /// <summary>Gets whether any borrowed tensor or stream pointer is exposed. 获取是否暴露 borrowed tensor 或 stream pointer。</summary>
    public bool BorrowedPointerExposed => false;

    /// <summary>Gets the evidence classification. 获取证据分类。</summary>
    public string RuntimeEvidenceKind => RealCallbackRuntime ? "local-tensorrt-callback-runtime" : "runtime-attempt";

    /// <summary>Returns a compact diagnostic string. 返回紧凑诊断字符串。</summary>
    public override string ToString()
    {
        return $"{Line}:owner={OwnerId}:attached={IsAttached}:invocations={InvocationCount}:failures={FailureCount}:tensor={TensorName}:proof={IsRealCallbackRuntimeProof}";
    }
}
