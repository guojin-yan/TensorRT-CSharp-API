using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Severity values used by TensorRT logger callbacks.
/// TensorRT logger 回调使用的严重级别。
/// </summary>
public enum TensorRtLogSeverity
{
    /// <summary>
    /// An internal TensorRT error. TensorRT 内部错误。
    /// </summary>
    InternalError = 0,

    /// <summary>
    /// A TensorRT error. TensorRT 错误。
    /// </summary>
    Error = 1,

    /// <summary>
    /// A TensorRT warning. TensorRT 警告。
    /// </summary>
    Warning = 2,

    /// <summary>
    /// Informational TensorRT output. TensorRT 信息输出。
    /// </summary>
    Info = 3,

    /// <summary>
    /// Verbose TensorRT output. TensorRT 详细输出。
    /// </summary>
    Verbose = 4
}

/// <summary>
/// Receives TensorRT logger messages copied across the native boundary.
/// 接收从 native 边界复制过来的 TensorRT logger 消息。
/// </summary>
/// <param name="severity">The TensorRT log severity. TensorRT 日志严重级别。</param>
/// <param name="message">The copied UTF-8 log message. 复制后的 UTF-8 日志消息。</param>
public delegate void TensorRtLogHandler(TensorRtLogSeverity severity, string message);

/// <summary>
/// Managed wrapper around a TensorRT logger.
/// TensorRT logger 的托管封装。
/// </summary>
public sealed class TensorRtLogger : IDisposable
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
    /// Releases logger callback resources if <see cref="Dispose()"/> was not called.
    /// 如果调用方未调用 <see cref="Dispose()"/>，则释放 logger callback 资源。
    /// </summary>
    ~TensorRtLogger()
    {
        Dispose();
    }

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

    /// <summary>
    /// Gets copied TensorRT versioned-interface metadata for this logger on TensorRT 11.
    /// 获取 TensorRT 11 logger 的 versioned-interface 元数据副本。
    /// </summary>
    /// <remarks>
    /// TensorRT 8 and TensorRT 10 loggers are not versioned interfaces in the NVIDIA headers used by this bridge.
    /// TensorRT 8 和 TensorRT 10 的 logger 在当前 bridge 使用的 NVIDIA 头文件中不是 versioned interface。
    /// </remarks>
    public TensorRtInterfaceInfo InterfaceInfo
    {
        get
        {
            ThrowIfDisposed();
            return NativeBridgeApi.GetLoggerInterfaceInfo(Line, _handle);
        }
    }

    /// <summary>
    /// Gets copied TensorRT versioned-interface API language metadata for this logger on TensorRT 11.
    /// 获取 TensorRT 11 logger 的 versioned-interface API language 元数据副本。
    /// </summary>
    /// <remarks>
    /// The native bridge returns the scalar enum value and does not expose the logger pointer. TensorRT 8 and TensorRT 10 loggers
    /// are not versioned interfaces in the NVIDIA headers used by this bridge.
    /// native bridge 只返回标量 enum 值，不暴露 logger 指针。TensorRT 8 和 TensorRT 10 的 logger 在当前 bridge 使用的
    /// NVIDIA 头文件中不是 versioned interface。
    /// </remarks>
    public TensorRtApiLanguage ApiLanguage
    {
        get
        {
            ThrowIfDisposed();
            return NativeBridgeApi.GetLoggerApiLanguage(Line, _handle);
        }
    }

    /// <summary>
    /// Tries to get copied TensorRT versioned-interface metadata for this logger without exposing the native pointer.
    /// 尝试获取当前 logger 的 TensorRT versioned-interface 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. 查询成功时复制出的 interface 元数据。</param>
    /// <returns><see langword="true"/> when interface metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo)
    {
        return TryGetInterfaceInfo(out interfaceInfo, out _);
    }

    /// <summary>
    /// Tries to get copied TensorRT versioned-interface metadata for this logger without exposing the native pointer.
    /// 尝试获取当前 logger 的 TensorRT versioned-interface 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. 查询成功时复制出的 interface 元数据。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when interface metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// TensorRT 8 and TensorRT 10 loggers are not versioned interfaces in the NVIDIA headers used by this bridge, so this method returns <see langword="false"/> for those lines.
    /// TensorRT 8 和 TensorRT 10 的 logger 在当前 bridge 使用的 NVIDIA 头文件中不是 versioned interface，因此这些版本线会返回 <see langword="false"/>。
    /// </remarks>
    public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)
    {
        ThrowIfDisposed();
        try
        {
            interfaceInfo = NativeBridgeApi.GetLoggerInterfaceInfo(Line, _handle);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            interfaceInfo = new TensorRtInterfaceInfo(string.Empty, 0, 0);
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to get copied TensorRT API language metadata for this logger without exposing the native pointer.
    /// 尝试获取当前 logger 的 TensorRT API language 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="apiLanguage">The copied API language when the query succeeds. 查询成功时复制出的 API language。</param>
    /// <returns><see langword="true"/> when API language metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetApiLanguage(out TensorRtApiLanguage apiLanguage)
    {
        return TryGetApiLanguage(out apiLanguage, out _);
    }

    /// <summary>
    /// Tries to get copied TensorRT API language metadata for this logger without exposing the native pointer.
    /// 尝试获取当前 logger 的 TensorRT API language 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="apiLanguage">The copied API language when the query succeeds. 查询成功时复制出的 API language。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when API language metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetApiLanguage(out TensorRtApiLanguage apiLanguage, out string diagnostic)
    {
        ThrowIfDisposed();
        try
        {
            apiLanguage = NativeBridgeApi.GetLoggerApiLanguage(Line, _handle);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            apiLanguage = TensorRtApiLanguage.Unknown;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Synchronously emits a diagnostic message through the native logger object.
    /// 通过 native logger 对象同步发送一条诊断消息。
    /// </summary>
    /// <param name="severity">The severity to use. 要使用的严重级别。</param>
    /// <param name="message">The message to copy to native UTF-8 memory for the call. 调用时复制到 native UTF-8 内存的消息。</param>
    /// <returns><c>true</c> when the message was accepted without a managed callback exception; otherwise <c>false</c>. 若消息未触发托管回调异常则返回 <c>true</c>，否则返回 <c>false</c>。</returns>
    /// <remarks>
    /// This method is intended for diagnostics and smoke tests. It does not expose the native logger pointer and does not change logger ownership.
    /// 该方法用于诊断和 smoke 测试；它不会暴露 native logger 指针，也不会改变 logger 所有权。
    /// </remarks>
    public bool EmitDiagnostic(TensorRtLogSeverity severity, string message)
    {
        if (message == null)
        {
            throw new ArgumentNullException(nameof(message));
        }

        ThrowIfDisposed();
        return NativeBridgeApi.EmitLoggerDiagnostic(Line, _handle, severity, message);
    }

    /// <summary>
    /// Releases the TensorRT logger handle.
    /// 释放 TensorRT logger 句柄。
    /// </summary>
    public void Dispose()
    {
        bool releaseNow;
        lock (_gate)
        {
            if (_disposeRequested)
            {
                return;
            }

            _disposeRequested = true;
            releaseNow = _attachmentCount == 0;
        }

        if (releaseNow)
        {
            ReleaseHandle();
        }

        GC.SuppressFinalize(this);
    }

    internal void AttachBorrower(TensorRtApiLine expectedLine)
    {
        if (expectedLine != Line)
        {
            throw new ArgumentException("Logger must belong to the same TensorRT API line as the owner.");
        }

        lock (_gate)
        {
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtLogger));
            }

            checked
            {
                _attachmentCount++;
            }
        }
    }

    internal void DetachBorrower()
    {
        bool releaseNow;
        lock (_gate)
        {
            if (_attachmentCount > 0)
            {
                _attachmentCount--;
            }

            releaseNow = _attachmentCount == 0 && _disposeRequested;
        }

        if (releaseNow)
        {
            ReleaseHandle();
        }
    }

    private static BridgeStatusCode InvokeManagedLogger(int severity, IntPtr message, UIntPtr messageLength, IntPtr userState)
    {
        CallbackState? state = null;
        try
        {
            if (userState == IntPtr.Zero)
            {
                return BridgeStatusCode.InvalidArgument;
            }

            state = GCHandle.FromIntPtr(userState).Target as CallbackState;
            if (state == null)
            {
                return BridgeStatusCode.InvalidState;
            }

            state.RecordInvocation();
            state.Handler((TensorRtLogSeverity)severity, DecodeUtf8(message, messageLength));
            return BridgeStatusCode.Ok;
        }
        catch (Exception exception)
        {
            state?.RecordFailure(exception);
            return BridgeStatusCode.InvalidState;
        }
    }

    private static string DecodeUtf8(IntPtr message, UIntPtr messageLength)
    {
        if (message == IntPtr.Zero || messageLength == UIntPtr.Zero)
        {
            return string.Empty;
        }

        ulong length = messageLength.ToUInt64();
        if (length > int.MaxValue)
        {
            throw new InvalidOperationException("TensorRT logger message is too large for a managed string.");
        }

        byte[] buffer = new byte[checked((int)length)];
        Marshal.Copy(message, buffer, 0, buffer.Length);
        return Encoding.UTF8.GetString(buffer, 0, buffer.Length);
    }

    private void FreeCallbackState()
    {
        if (_hasCallbackStateHandle)
        {
            _callbackStateHandle.Free();
            _hasCallbackStateHandle = false;
        }
    }

    private void ReleaseHandle()
    {
        bool shouldRelease;
        lock (_gate)
        {
            shouldRelease = !_handleReleased;
            _handleReleased = true;
        }

        if (!shouldRelease)
        {
            return;
        }

        _handle.Dispose();
        GC.KeepAlive(_nativeCallback);
        FreeCallbackState();
    }

    internal void ThrowIfDisposed()
    {
        lock (_gate)
        {
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtLogger));
            }
        }
    }

    private sealed class CallbackState
    {
        private long _invocationCount;
        private long _failureCount;
        private Exception? _lastException;

        public CallbackState(TensorRtLogHandler handler)
        {
            Handler = handler;
        }

        public TensorRtLogHandler Handler { get; }

        public long InvocationCount => Interlocked.Read(ref _invocationCount);

        public long FailureCount => Interlocked.Read(ref _failureCount);

        public Exception? LastException => Volatile.Read(ref _lastException);

        public void RecordInvocation()
        {
            Interlocked.Increment(ref _invocationCount);
        }

        public void RecordFailure(Exception exception)
        {
            Volatile.Write(ref _lastException, exception);
            Interlocked.Increment(ref _failureCount);
        }
    }
}
