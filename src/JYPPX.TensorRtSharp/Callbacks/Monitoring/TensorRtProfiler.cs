using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed owner for a TensorRT profiler callback object.
/// TensorRT profiler 回调对象的托管 owner。
/// </summary>
public sealed partial class TensorRtProfiler : IDisposable
{
    private readonly object _gate = new object();
    private readonly SafeTensorRtObjectHandle _handle;
    private readonly TensorRtProfilerCallback _nativeCallback;
    private readonly CallbackState _callbackState;
    private GCHandle _callbackStateHandle;
    private bool _hasCallbackStateHandle;
    private bool _disposeRequested;
    private bool _handleReleased;
    private int _attachmentCount;

    /// <summary>
    /// Creates a profiler that forwards TensorRT layer timing callbacks to managed code.
    /// 创建一个会把 TensorRT layer timing 回调转发给托管代码的 profiler。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="handler">The managed profiling handler. 托管 profiling 处理器。</param>
    /// <remarks>
    /// TensorRT borrows the native profiler pointer when it is attached to an execution context. Keep the profiler alive
    /// until <see cref="TensorRtExecutionContext.ClearProfiler"/> or context disposal detaches it. Managed exceptions are
    /// swallowed by the trampoline, counted, and converted to a non-OK status; they never cross the native ABI boundary.
    /// 当 profiler 绑定到 execution context 时，TensorRT 只借用 native profiler 指针。请让 profiler 至少存活到
    /// <see cref="TensorRtExecutionContext.ClearProfiler"/> 或 context dispose 解除绑定之后。托管异常会被 trampoline
    /// 吞吐、计数并转为非 OK 状态，不会跨 native ABI 边界抛出。
    /// </remarks>
    public TensorRtProfiler(TensorRtApiLine line, TensorRtProfilerHandler handler)
    {
        if (handler == null)
        {
            throw new ArgumentNullException(nameof(handler));
        }

        NativeBridgeLoader.EnsureInitialized();
        Line = line;
        _callbackState = new CallbackState(handler);
        _nativeCallback = InvokeManagedProfiler;
        _callbackStateHandle = GCHandle.Alloc(_callbackState);
        _hasCallbackStateHandle = true;

        try
        {
            _handle = NativeBridgeApi.CreateProfiler(line, _nativeCallback, GCHandle.ToIntPtr(_callbackStateHandle));
        }
        catch
        {
            FreeCallbackState();
            throw;
        }
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this profiler.
    /// 获取当前 profiler 使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets whether this profiler is currently borrowed by at least one execution context wrapper.
    /// 获取当前 profiler 是否正被至少一个 execution context 托管封装借用。
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
    /// Gets the number of managed callback invocations observed by this profiler.
    /// 获取当前 profiler 观察到的托管回调调用次数。
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
