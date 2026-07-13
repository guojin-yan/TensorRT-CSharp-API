using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Receives TensorRT layer profiling records copied across the native boundary.
/// 接收从 native 边界复制过来的 TensorRT layer profiling 记录。
/// </summary>
/// <param name="layerName">The layer name reported by TensorRT. TensorRT 报告的 layer 名称。</param>
/// <param name="milliseconds">The layer execution time in milliseconds. layer 执行耗时，单位毫秒。</param>
public delegate void TensorRtProfilerHandler(string layerName, float milliseconds);

/// <summary>
/// Managed owner for a TensorRT profiler callback object.
/// TensorRT profiler 回调对象的托管 owner。
/// </summary>
public sealed class TensorRtProfiler : IDisposable
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
    /// Releases profiler callback resources if <see cref="Dispose()"/> was not called.
    /// 如果调用方未调用 <see cref="Dispose()"/>，则释放 profiler callback 资源。
    /// </summary>
    ~TensorRtProfiler()
    {
        Dispose();
    }

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

    /// <summary>
    /// Gets copied TensorRT versioned-interface metadata for this profiler on TensorRT 11.
    /// 获取 TensorRT 11 profiler 的 versioned-interface 元数据副本。
    /// </summary>
    /// <remarks>
    /// TensorRT 8 and TensorRT 10 profilers are not versioned interfaces in the NVIDIA headers used by this bridge.
    /// TensorRT 8 和 TensorRT 10 的 profiler 在当前 bridge 使用的 NVIDIA 头文件中不是 versioned interface。
    /// </remarks>
    public TensorRtInterfaceInfo InterfaceInfo
    {
        get
        {
            ThrowIfDisposed();
            return NativeBridgeApi.GetProfilerInterfaceInfo(Line, _handle);
        }
    }

    /// <summary>
    /// Gets copied TensorRT versioned-interface API language metadata for this profiler on TensorRT 11.
    /// 获取 TensorRT 11 profiler 的 versioned-interface API language 元数据副本。
    /// </summary>
    /// <remarks>
    /// The native bridge returns the scalar enum value and does not expose the profiler pointer. TensorRT 8 and TensorRT 10 profilers
    /// are not versioned interfaces in the NVIDIA headers used by this bridge.
    /// native bridge 只返回标量 enum 值，不暴露 profiler 指针。TensorRT 8 和 TensorRT 10 的 profiler 在当前 bridge 使用的
    /// NVIDIA 头文件中不是 versioned interface。
    /// </remarks>
    public TensorRtApiLanguage ApiLanguage
    {
        get
        {
            ThrowIfDisposed();
            return NativeBridgeApi.GetProfilerApiLanguage(Line, _handle);
        }
    }

    /// <summary>
    /// Tries to get copied TensorRT versioned-interface metadata for this profiler without exposing the native pointer.
    /// 尝试获取当前 profiler 的 TensorRT versioned-interface 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. 查询成功时复制出的 interface 元数据。</param>
    /// <returns><see langword="true"/> when interface metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo)
    {
        return TryGetInterfaceInfo(out interfaceInfo, out _);
    }

    /// <summary>
    /// Tries to get copied TensorRT versioned-interface metadata for this profiler without exposing the native pointer.
    /// 尝试获取当前 profiler 的 TensorRT versioned-interface 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. 查询成功时复制出的 interface 元数据。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when interface metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// TensorRT 8 and TensorRT 10 profilers are not versioned interfaces in the NVIDIA headers used by this bridge, so this method returns <see langword="false"/> for those lines.
    /// TensorRT 8 和 TensorRT 10 的 profiler 在当前 bridge 使用的 NVIDIA 头文件中不是 versioned interface，因此这些版本线会返回 <see langword="false"/>。
    /// </remarks>
    public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)
    {
        ThrowIfDisposed();
        try
        {
            interfaceInfo = NativeBridgeApi.GetProfilerInterfaceInfo(Line, _handle);
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
    /// Tries to get copied TensorRT API language metadata for this profiler without exposing the native pointer.
    /// 尝试获取当前 profiler 的 TensorRT API language 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="apiLanguage">The copied API language when the query succeeds. 查询成功时复制出的 API language。</param>
    /// <returns><see langword="true"/> when API language metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetApiLanguage(out TensorRtApiLanguage apiLanguage)
    {
        return TryGetApiLanguage(out apiLanguage, out _);
    }

    /// <summary>
    /// Tries to get copied TensorRT API language metadata for this profiler without exposing the native pointer.
    /// 尝试获取当前 profiler 的 TensorRT API language 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="apiLanguage">The copied API language when the query succeeds. 查询成功时复制出的 API language。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when API language metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetApiLanguage(out TensorRtApiLanguage apiLanguage, out string diagnostic)
    {
        ThrowIfDisposed();
        try
        {
            apiLanguage = NativeBridgeApi.GetProfilerApiLanguage(Line, _handle);
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
    /// Synchronously emits a diagnostic profiler record through the native profiler object.
    /// 通过 native profiler 对象同步发送一条诊断 profiler 记录。
    /// </summary>
    /// <param name="layerName">The layer name copied to native UTF-8 memory. 复制到 native UTF-8 内存的 layer 名称。</param>
    /// <param name="milliseconds">The diagnostic elapsed time in milliseconds. 诊断耗时，单位毫秒。</param>
    /// <returns><see langword="true"/> when the callback completed without a managed exception or non-OK status. 回调未发生托管异常或非 OK 状态时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// This method is intended for diagnostics and smoke tests. It does not expose the native profiler pointer and does not attach the profiler to an execution context.
    /// 该方法用于诊断和 smoke 测试；它不会暴露 native profiler 指针，也不会把 profiler 绑定到 execution context。
    /// </remarks>
    public bool EmitDiagnostic(string layerName, float milliseconds)
    {
        if (layerName == null)
        {
            throw new ArgumentNullException(nameof(layerName));
        }

        ThrowIfDisposed();
        return NativeBridgeApi.EmitProfilerDiagnostic(Line, _handle, layerName, milliseconds);
    }

    /// <summary>
    /// Releases this profiler after all execution context attachments have been cleared.
    /// 在所有 execution context 绑定解除后释放当前 profiler。
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
            throw new ArgumentException("Profiler must belong to the same TensorRT API line as the execution context.");
        }

        lock (_gate)
        {
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtProfiler));
            }

            checked
            {
                _attachmentCount++;
            }
        }
    }

    internal void DetachBorrower()
    {
        bool releaseNow = false;
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

    internal void ThrowIfDisposed()
    {
        lock (_gate)
        {
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtProfiler));
            }
        }
    }

    private static BridgeStatusCode InvokeManagedProfiler(IntPtr layerName, UIntPtr layerNameLength, float milliseconds, IntPtr userState)
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
            state.Handler(DecodeUtf8(layerName, layerNameLength), milliseconds);
            return BridgeStatusCode.Ok;
        }
        catch (Exception exception)
        {
            state?.RecordFailure(exception);
            return BridgeStatusCode.InvalidState;
        }
    }

    private static string DecodeUtf8(IntPtr value, UIntPtr length)
    {
        if (value == IntPtr.Zero || length == UIntPtr.Zero)
        {
            return string.Empty;
        }

        ulong byteLength = length.ToUInt64();
        if (byteLength > int.MaxValue)
        {
            throw new InvalidOperationException("TensorRT profiler layer name is too large for a managed string.");
        }

        byte[] buffer = new byte[checked((int)byteLength)];
        Marshal.Copy(value, buffer, 0, buffer.Length);
        return Encoding.UTF8.GetString(buffer, 0, buffer.Length);
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

    private void FreeCallbackState()
    {
        if (_hasCallbackStateHandle)
        {
            _callbackStateHandle.Free();
            _hasCallbackStateHandle = false;
        }
    }

    private sealed class CallbackState
    {
        private long _invocationCount;
        private long _failureCount;
        private Exception? _lastException;

        public CallbackState(TensorRtProfilerHandler handler)
        {
            Handler = handler;
        }

        public TensorRtProfilerHandler Handler { get; }

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
