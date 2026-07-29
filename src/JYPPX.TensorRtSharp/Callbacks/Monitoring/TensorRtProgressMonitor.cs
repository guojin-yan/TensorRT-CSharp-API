using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed owner for a TensorRT progress monitor callback object.
/// TensorRT progress monitor 回调对象的托管 owner。
/// </summary>
public sealed partial class TensorRtProgressMonitor : IDisposable
{
    private readonly object _gate = new object();
    private readonly SafeTensorRtObjectHandle _handle;
    private readonly TensorRtProgressMonitorCallback _nativeCallback;
    private readonly CallbackState _callbackState;
    private GCHandle _callbackStateHandle;
    private bool _hasCallbackStateHandle;
    private bool _disposeRequested;
    private bool _handleReleased;
    private int _attachmentCount;

    /// <summary>
    /// Creates a progress monitor that forwards TensorRT build progress callbacks to managed code.
    /// 创建一个会把 TensorRT 构建进度回调转发给托管代码的 progress monitor。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="handler">The managed progress handler. 托管 progress 处理器。</param>
    /// <remarks>
    /// TensorRT 10 and TensorRT 11 support progress monitors. TensorRT 8 does not expose this callback interface.
    /// Attach this monitor to a builder config with <see cref="TensorRtBuilderConfig.SetProgressMonitor"/> and detach it before
    /// disposing the monitor. If <see cref="Dispose"/> is called while a config still borrows the monitor, the native handle is
    /// released only after the config clears or disposes its attachment. Managed exceptions are swallowed by the trampoline,
    /// counted, and converted to a non-OK status; they never cross the native ABI boundary.
    /// TensorRT 10 和 TensorRT 11 支持 progress monitor；TensorRT 8 不暴露该回调接口。请用
    /// <see cref="TensorRtBuilderConfig.SetProgressMonitor"/> 绑定到 builder config，并在释放 monitor 前先解除绑定。
    /// 如果 config 仍在借用时调用 <see cref="Dispose"/>，native 句柄会延迟到 config clear/dispose 后释放。托管异常会被
    /// trampoline 吞吐、计数并转为非 OK 状态，不会跨 native ABI 边界抛出。
    /// </remarks>
    public TensorRtProgressMonitor(TensorRtApiLine line, TensorRtProgressMonitorHandler handler)
    {
        if (handler == null)
        {
            throw new ArgumentNullException(nameof(handler));
        }

        if (line == TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "TensorRT progress monitor callbacks are available for TensorRT 10 and TensorRT 11 adapters.");
        }

        NativeBridgeLoader.EnsureInitialized();
        Line = line;
        _callbackState = new CallbackState(handler);
        _nativeCallback = InvokeManagedProgressMonitor;
        _callbackStateHandle = GCHandle.Alloc(_callbackState);
        _hasCallbackStateHandle = true;

        try
        {
            _handle = NativeBridgeApi.CreateProgressMonitor(line, _nativeCallback, GCHandle.ToIntPtr(_callbackStateHandle));
        }
        catch
        {
            FreeCallbackState();
            throw;
        }
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this monitor.
    /// 获取当前 monitor 使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets whether this monitor is currently borrowed by at least one TensorRT builder config wrapper.
    /// 获取当前 monitor 是否正被至少一个 TensorRT builder config 托管封装借用。
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

    /// <summary>
    /// Gets the number of managed callback invocations observed by this monitor.
    /// 获取当前 monitor 观察到的托管回调调用次数。
    /// </summary>
    public long CallbackInvocationCount => _callbackState.InvocationCount;

    /// <summary>
    /// Gets the number of managed callback exceptions swallowed at the ABI boundary.
    /// 获取在 ABI 边界被吞吐的托管回调异常次数。
    /// </summary>
    public long CallbackFailureCount => _callbackState.FailureCount;

    /// <summary>
    /// Gets the last exception thrown by the managed callback, if any.
    /// 获取托管回调最近一次抛出的异常。
    /// </summary>
    public Exception? LastCallbackException => _callbackState.LastException;
}
