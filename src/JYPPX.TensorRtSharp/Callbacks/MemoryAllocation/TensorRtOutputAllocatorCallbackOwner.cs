using System;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// High-level diagnostic owner for the future TensorRT output allocator callback bridge.
/// 未来 TensorRT output allocator callback bridge 的高层诊断 owner。
/// </summary>
/// <remarks>
/// This class is a design gate. It composes the managed output allocator runtime gate with the native allocator owner
/// ledger dry-run so callers can validate copied metadata, attach/detach intent, exception-to-status behavior, and
/// pointer-free public API shape. It does not call TensorRT <c>setOutputAllocator</c> and does not unlock the
/// <c>IOutputAllocator::notifyShape</c> or <c>IOutputAllocator::reallocateOutput</c> deferred rows.
/// 该类是设计门禁。它把托管 output allocator runtime gate 与 native allocator owner ledger dry-run 组合起来，使调用方可以
/// 验证复制元数据、attach/detach intent、exception-to-status 行为和不暴露 pointer 的 public API 形状。它不会调用 TensorRT
/// <c>setOutputAllocator</c>，也不会解除 <c>IOutputAllocator::notifyShape</c> 或
/// <c>IOutputAllocator::reallocateOutput</c> 的 deferred rows。
/// </remarks>
public sealed partial class TensorRtOutputAllocatorCallbackOwner : IDisposable
{
    private readonly object _gate = new object();
    private readonly TensorRtOutputAllocatorRuntimeGate _runtimeGate = new TensorRtOutputAllocatorRuntimeGate();
    private readonly TensorRtAllocatorCallbackOwner _nativeLedgerOwner;
    private TensorRtAllocatorOwnerStateDryRunResult? _lastNativeLedgerState;
    private BridgeStatusCode _lastNativeLedgerStatus = BridgeStatusCode.Ok;
    private string _lastNativeLedgerDiagnostic = string.Empty;
    private TensorRtApiLine _lastLine = TensorRtApiLine.TensorRt11;
    private bool _lastNativeLedgerAvailable;
    private bool _disposed;

    /// <summary>
    /// Creates an output allocator callback owner design gate.
    /// 创建 output allocator callback owner 设计门禁。
    /// </summary>
    public TensorRtOutputAllocatorCallbackOwner()
    {
        _nativeLedgerOwner = new TensorRtAllocatorCallbackOwner(static request =>
            TensorRtAllocatorDryRunResult.Success("output-allocator-callback-owner-design managed ledger keep-alive " + request));
    }

    /// <summary>Gets whether dispose has been requested. 获取是否已请求释放。</summary>
    public bool IsDisposed
    {
        get
        {
            lock (_gate)
            {
                return _disposed;
            }
        }
    }

    /// <summary>Gets whether this owner is attached to a TensorRT execution context. 获取该 owner 是否已绑定到 TensorRT execution context。</summary>
    public bool IsAttached => false;

}
