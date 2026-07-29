using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed skeleton and native dry-run diagnostic surface for a future TensorRT allocator callback owner.
/// 未来 TensorRT allocator callback owner 的托管骨架与 native dry-run 诊断表面。
/// </summary>
/// <remarks>
/// This owner is currently dry-run only. It keeps managed callback state alive, records invocation/failure diagnostics, and
/// swallows handler exceptions into <see cref="LastCallbackException"/> and a failed <see cref="TensorRtAllocatorDryRunResult"/>.
/// It can create a short-lived native diagnostic owner through <see cref="RunNativeDryRunDiagnostic"/>, but it is not
/// attached to TensorRT, does not call allocator callbacks, and does not return or own device pointers.
/// 当前 owner 仅用于 dry-run。它会保持托管回调状态存活、记录调用/失败诊断，并把 handler 异常吞吐到
/// <see cref="LastCallbackException"/> 与失败的 <see cref="TensorRtAllocatorDryRunResult"/> 中。它可以通过
/// <see cref="RunNativeDryRunDiagnostic"/> 创建短生命周期 native 诊断 owner，但不会绑定到 TensorRT，不会调用
/// allocator callback，也不会返回或拥有 device pointer。
/// </remarks>
public sealed partial class TensorRtAllocatorCallbackOwner : IDisposable
{
    private static long s_nextOwnerId;

    private readonly object _gate = new object();
    private readonly long _ownerId;
    private readonly CallbackState _callbackState;
    private readonly TensorRtAllocatorInternalRuntimePrototypeCallback _runtimePrototypeCallback;
    private GCHandle _callbackStateHandle;
    private GCHandle _runtimePrototypeCallbackHandle;
    private bool _hasCallbackStateHandle;
    private bool _hasRuntimePrototypeCallbackHandle;
    private bool _disposeRequested;
    private int _activePrototypeCallCount;

    /// <summary>
    /// Creates a dry-run allocator callback owner.
    /// 创建 dry-run allocator callback owner。
    /// </summary>
    /// <param name="handler">The managed dry-run handler. 托管 dry-run 处理器。</param>
    /// <remarks>
    /// The handler is kept alive by this owner until <see cref="Dispose"/>. No TensorRT object borrows this owner in the
    /// current stage, so <see cref="IsAttached"/> always remains <see langword="false"/>.
    /// handler 会由当前 owner 保持存活直到 <see cref="Dispose"/>。当前阶段没有 TensorRT 对象借用该 owner，
    /// 因此 <see cref="IsAttached"/> 始终为 <see langword="false"/>。
    /// </remarks>
    public TensorRtAllocatorCallbackOwner(TensorRtAllocatorDryRunHandler handler)
    {
        if (handler == null)
        {
            throw new ArgumentNullException(nameof(handler));
        }

        _ownerId = Interlocked.Increment(ref s_nextOwnerId);
        _callbackState = new CallbackState(handler);
        _runtimePrototypeCallback = InvokeInternalSyncAllocatorPrototype;
        _callbackStateHandle = GCHandle.Alloc(_callbackState);
        _hasCallbackStateHandle = true;

        try
        {
            _runtimePrototypeCallbackHandle = GCHandle.Alloc(_runtimePrototypeCallback);
            _hasRuntimePrototypeCallbackHandle = true;
        }
        catch
        {
            FreeCallbackState();
            throw;
        }
    }

    /// <summary>
    /// Releases callback owner resources if <see cref="Dispose()"/> was not called.
    /// 如果调用方未调用 <see cref="Dispose()"/>，则释放 callback owner 资源。
    /// </summary>
    ~TensorRtAllocatorCallbackOwner()
    {
        Dispose();
    }

    /// <summary>
    /// Gets whether this owner has been disposed.
    /// 获取当前 owner 是否已释放。
    /// </summary>
    public bool IsDisposed
    {
        get
        {
            lock (_gate)
            {
                return _disposeRequested;
            }
        }
    }

    /// <summary>
    /// Gets whether this owner is currently attached to a TensorRT object.
    /// 获取当前 owner 是否正绑定到 TensorRT 对象。
    /// </summary>
    /// <remarks>
    /// The dry-run skeleton never attaches to TensorRT. This property is present so later native owner work can keep the
    /// same diagnostics shape without exposing borrowed pointers.
    /// dry-run 骨架不会绑定到 TensorRT。保留该属性是为了后续 native owner 工作延续同一诊断形状，同时不暴露 borrowed pointer。
    /// </remarks>
    public bool IsAttached => false;

    /// <summary>
    /// Gets the number of dry-run callback invocations observed by this owner.
    /// 获取当前 owner 观察到的 dry-run 回调调用次数。
    /// </summary>
    public long CallbackInvocationCount => _callbackState.InvocationCount;

    /// <summary>
    /// Gets the number of managed handler exceptions swallowed by this owner.
    /// 获取当前 owner 吞吐的托管 handler 异常次数。
    /// </summary>
    public long CallbackFailureCount => _callbackState.FailureCount;

    /// <summary>
    /// Gets the last exception thrown by the dry-run handler, if any.
    /// 获取 dry-run handler 最近一次抛出的异常。
    /// </summary>
    public Exception? LastCallbackException => _callbackState.LastException;

    /// <summary>
    /// Gets the last diagnostic message observed by this owner.
    /// 获取当前 owner 最近一次观察到的诊断消息。
    /// </summary>
    public string LastDiagnostic => _callbackState.LastDiagnostic;

}
