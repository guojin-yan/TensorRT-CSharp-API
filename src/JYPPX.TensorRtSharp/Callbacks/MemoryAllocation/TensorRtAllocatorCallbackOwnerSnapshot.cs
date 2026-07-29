using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports a copied, pointer-free lifecycle snapshot from <see cref="TensorRtAllocatorCallbackOwner"/>.
/// 表示从 <see cref="TensorRtAllocatorCallbackOwner"/> 复制出的、无 pointer 的生命周期快照。
/// </summary>
/// <remarks>
/// The snapshot exposes managed keep-alive, in-flight, exception-to-status, and release-hook diagnostics only. It does
/// not expose a native callback owner pointer, device pointer, allocator-owned pointer, or borrowed TensorRT object.
/// 该快照只暴露托管 keep-alive、in-flight、exception-to-status 和 release hook 诊断；不会暴露 native callback owner
/// pointer、device pointer、allocator-owned pointer 或 TensorRT borrowed object。
/// </remarks>
public readonly struct TensorRtAllocatorCallbackOwnerSnapshot
{
    internal TensorRtAllocatorCallbackOwnerSnapshot(TensorRtAllocatorInternalRuntimePrototypeResult prototype)
    {
        OwnerId = prototype.OwnerId;
        Operation = prototype.Operation;
        LastStatus = prototype.LastStatus;
        InvocationCount = prototype.InvocationCount;
        FailureCount = prototype.FailureCount;
        InFlightCallbackCount = prototype.InFlightCallbackCount;
        MaxInFlightCallbackCount = prototype.MaxInFlightCallbackCount;
        ActivePrototypeCallCount = prototype.ActivePrototypeCallCount;
        ReleaseHookCount = prototype.ReleaseHookCount;
        CallbackStatePinned = prototype.CallbackStatePinned;
        DelegatePinned = prototype.DelegatePinned;
        DisposeRequested = prototype.DisposeRequested;
        IsAttached = prototype.IsAttached;
        LastDiagnostic = prototype.LastDiagnostic;
        ReleaseDiagnostic = prototype.ReleaseDiagnostic;
    }

    /// <summary>Gets the marker used by readiness to identify this snapshot. 获取 readiness 用于识别该快照的 marker。</summary>
    public string EvidenceKind => "allocator-owner-internal-runtime-prototype";

    /// <summary>Gets the callback kind represented by this diagnostic snapshot. 获取该诊断快照代表的 callback 类型。</summary>
    public string CallbackKind => "sync-allocator-prototype";

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

    /// <summary>Gets the last copied bridge status. 获取最近一次复制出的 bridge status。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the copied invocation count. 获取复制出的调用次数。</summary>
    public long InvocationCount { get; }

    /// <summary>Gets the copied failure count. 获取复制出的失败次数。</summary>
    public long FailureCount { get; }

    /// <summary>Gets the copied in-flight callback count. 获取复制出的 in-flight callback 数量。</summary>
    public long InFlightCallbackCount { get; }

    /// <summary>Gets the copied maximum in-flight callback count. 获取复制出的最大 in-flight callback 数量。</summary>
    public long MaxInFlightCallbackCount { get; }

    /// <summary>Gets the copied active prototype call count. 获取复制出的 active prototype 调用数量。</summary>
    public int ActivePrototypeCallCount { get; }

    /// <summary>Gets the copied release hook count. 获取复制出的 release hook 次数。</summary>
    public long ReleaseHookCount { get; }

    /// <summary>Gets whether managed callback state remains pinned. 获取托管 callback state 是否仍被 pin 住。</summary>
    public bool CallbackStatePinned { get; }

    /// <summary>Gets whether the managed delegate remains pinned. 获取托管 delegate 是否仍被 pin 住。</summary>
    public bool DelegatePinned { get; }

    /// <summary>Gets whether dispose has been requested. 获取是否已请求释放。</summary>
    public bool DisposeRequested { get; }

    /// <summary>Gets whether this owner is attached to TensorRT. 获取该 owner 是否已绑定到 TensorRT。</summary>
    public bool IsAttached { get; }

    /// <summary>Gets whether the public API exposes a device pointer. 获取 public API 是否暴露 device pointer。</summary>
    public bool DevicePointerExposed => false;

    /// <summary>Gets whether the diagnostic produced a device pointer. 获取该诊断是否产生 device pointer。</summary>
    public bool DevicePointerProduced => false;

    /// <summary>Gets whether a borrowed pointer escaped through this public API. 获取 borrowed pointer 是否通过该 public API 逃逸。</summary>
    public bool BorrowedPointerEscaped => false;

    /// <summary>Gets the copied diagnostic message. 获取复制出的诊断消息。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets the copied release diagnostic message. 获取复制出的释放诊断消息。</summary>
    public string ReleaseDiagnostic { get; }

    /// <summary>Gets whether managed keep-alive handles are still present for a pre-dispose owner. 获取释放前 owner 的托管 keep-alive 句柄是否仍存在。</summary>
    public bool ManagedKeepAliveReady => CallbackStatePinned && DelegatePinned && !DisposeRequested;

    /// <summary>Gets whether dispose released keep-alive handles after callbacks drained. 获取 Dispose 是否在 callback drain 后释放 keep-alive 句柄。</summary>
    public bool DisposeReleaseReady =>
        DisposeRequested &&
        !CallbackStatePinned &&
        !DelegatePinned &&
        InFlightCallbackCount == 0 &&
        ReleaseHookCount > 0;

    /// <summary>Gets whether the copied public surface is pointer-free. 获取复制出的 public surface 是否无 pointer。</summary>
    public bool PointerFreeSurfaceReady =>
        !DevicePointerExposed &&
        !DevicePointerProduced &&
        !BorrowedPointerEscaped;

    /// <summary>Gets whether this copied diagnostic completed without managed failures. 获取该复制诊断是否未出现托管失败。</summary>
    public bool Succeeded => LastStatus == BridgeStatusCode.Ok && FailureCount == 0 && InFlightCallbackCount == 0;

    /// <summary>Returns a compact diagnostic representation. 返回紧凑的诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:owner={OwnerId}:operation={Operation}:status={LastStatus}:invocations={InvocationCount}:failures={FailureCount}:inflight={InFlightCallbackCount}:releaseHooks={ReleaseHookCount}:proof={IsRealCallbackRuntimeProof}";
    }
}
