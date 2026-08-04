using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT logger.
/// TensorRT logger 的托管封装。
/// </summary>
public sealed partial class TensorRtLogger : IDisposable
{
    private readonly object _gate = new object();
    private readonly SafeTensorRtObjectHandle _handle;
    private readonly TensorRtLoggerCallback? _nativeCallback;
    private readonly CallbackState? _callbackState;
    private GCHandle _callbackStateHandle;
    private bool _hasCallbackStateHandle;
    private bool _disposeRequested;
    private bool _handleReleased;
    private int _attachmentCount;

    /// <summary>
    /// Creates a TensorRT logger for one TensorRT API line.
    /// 为一个 TensorRT API line 创建 TensorRT logger。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    public TensorRtLogger(TensorRtApiLine line)
    {
        NativeBridgeLoader.EnsureInitialized();
        Line = line;
        _handle = NativeBridgeApi.CreateLogger(line);
    }

    /// <summary>
    /// Creates a TensorRT logger that forwards native logger messages to a managed handler.
    /// 创建一个会把 native logger 消息转发给托管处理器的 TensorRT logger。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="handler">The managed log handler. The handler is kept alive until this logger is disposed. 托管日志处理器，其生命周期会保持到 logger 释放。</param>
    /// <param name="minimumSeverity">The least severe message that should be forwarded. 数值不高于该级别的消息会被转发。</param>
    /// <remarks>
    /// The native bridge owns the TensorRT logger object while this wrapper owns the managed callback state.
    /// TensorRT owners borrow the native logger pointer. If this logger is disposed while a managed owner still borrows it,
    /// native release is deferred until the owner detaches. Exceptions thrown by <paramref name="handler"/> are swallowed by
    /// the managed trampoline and reported through <see cref="CallbackFailureCount"/> and <see cref="LastCallbackException"/>;
    /// they are never allowed to cross the native ABI boundary.
    /// native bridge 拥有 TensorRT logger 对象；当前托管封装拥有回调状态。释放 logger 前应先释放借用该 logger 的 TensorRT owner。
    /// 如果还有托管 owner 借用 logger，native 释放会延迟到 owner 解除借用之后。<paramref name="handler"/> 抛出的异常会被托管
    /// trampoline 吞吐，并通过 <see cref="CallbackFailureCount"/> 和 <see cref="LastCallbackException"/> 记录，不会跨 native ABI 边界抛出。
    /// </remarks>
    public TensorRtLogger(TensorRtApiLine line, TensorRtLogHandler handler, TensorRtLogSeverity minimumSeverity = TensorRtLogSeverity.Warning)
    {
        if (handler == null)
        {
            throw new ArgumentNullException(nameof(handler));
        }

        NativeBridgeLoader.EnsureInitialized();
        Line = line;
        _callbackState = new CallbackState(handler);
        _nativeCallback = InvokeManagedLogger;
        _callbackStateHandle = GCHandle.Alloc(_callbackState);
        _hasCallbackStateHandle = true;

        try
        {
            _handle = NativeBridgeApi.CreateLogger(line, _nativeCallback, GCHandle.ToIntPtr(_callbackStateHandle), minimumSeverity);
        }
        catch
        {
            FreeCallbackState();
            throw;
        }
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this logger.
    /// 获取当前 logger 使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets whether this logger owns a managed callback state.
    /// 获取当前 logger 是否拥有托管回调状态。
    /// </summary>
    public bool HasManagedCallback => _callbackState != null;

    /// <summary>
    /// Gets the number of managed callback invocations observed by this logger.
    /// 获取当前 logger 观察到的托管回调调用次数。
    /// </summary>
    public long CallbackInvocationCount => _callbackState?.InvocationCount ?? 0;

    /// <summary>
    /// Gets the number of managed callback exceptions swallowed at the ABI boundary.
    /// 获取在 ABI 边界被吞吐的托管回调异常次数。
    /// </summary>
    public long CallbackFailureCount => _callbackState?.FailureCount ?? 0;

    /// <summary>
    /// Gets the last exception thrown by the managed callback, if any.
    /// 获取托管回调最近一次抛出的异常。
    /// </summary>
    public Exception? LastCallbackException => _callbackState?.LastException;

    /// <summary>
    /// Gets whether this logger is currently borrowed by at least one TensorRT owner wrapper.
    /// 获取当前 logger 是否正被至少一个 TensorRT owner 托管封装借用。
    /// </summary>
    public bool IsAttached
    {
        get
        {
            lock (_gate)
            {
                return _attachmentCount > 0;
            }
        }
    }
}
