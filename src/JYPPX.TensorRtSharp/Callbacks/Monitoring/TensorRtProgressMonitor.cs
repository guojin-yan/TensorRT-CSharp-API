using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// TensorRT progress monitor event kinds.
/// TensorRT progress monitor 事件类型。
/// </summary>
public enum TensorRtProgressMonitorEventKind
{
    /// <summary>
    /// Unknown event kind. 未知事件。
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// A build phase has started. 构建阶段开始。
    /// </summary>
    PhaseStart = 1,

    /// <summary>
    /// A build phase step has completed. 构建阶段中的一个步骤完成。
    /// </summary>
    StepComplete = 2,

    /// <summary>
    /// A build phase has finished. 构建阶段结束。
    /// </summary>
    PhaseFinish = 3
}

/// <summary>
/// A copied TensorRT progress monitor event.
/// 从 TensorRT 复制出的 progress monitor 事件。
/// </summary>
public readonly struct TensorRtProgressMonitorEvent
{
    internal TensorRtProgressMonitorEvent(TensorRtProgressMonitorEventKind kind, string phaseName, string? parentPhase, int step, int stepCount)
    {
        Kind = kind;
        PhaseName = phaseName;
        ParentPhase = parentPhase;
        Step = step;
        StepCount = stepCount;
    }

    /// <summary>
    /// Gets the event kind. 获取事件类型。
    /// </summary>
    public TensorRtProgressMonitorEventKind Kind { get; }

    /// <summary>
    /// Gets the TensorRT phase name copied from native memory.
    /// 获取从 native 内存复制出的 TensorRT 阶段名称。
    /// </summary>
    public string PhaseName { get; }

    /// <summary>
    /// Gets the parent phase name, when TensorRT provided one.
    /// 获取父阶段名称；当 TensorRT 未提供父阶段时为 <see langword="null"/>。
    /// </summary>
    public string? ParentPhase { get; }

    /// <summary>
    /// Gets the completed step index for <see cref="TensorRtProgressMonitorEventKind.StepComplete"/> events.
    /// 获取 <see cref="TensorRtProgressMonitorEventKind.StepComplete"/> 事件的完成步骤索引。
    /// </summary>
    public int Step { get; }

    /// <summary>
    /// Gets the total step count reported for <see cref="TensorRtProgressMonitorEventKind.PhaseStart"/> events.
    /// 获取 <see cref="TensorRtProgressMonitorEventKind.PhaseStart"/> 事件中 TensorRT 报告的总步骤数。
    /// </summary>
    public int StepCount { get; }
}

/// <summary>
/// Result returned by a native progress monitor diagnostic emission.
/// native progress monitor 诊断触发返回的结果。
/// </summary>
public readonly struct TensorRtProgressMonitorDiagnosticResult
{
    internal TensorRtProgressMonitorDiagnosticResult(bool shouldContinue, bool callbackAccepted)
    {
        ShouldContinue = shouldContinue;
        CallbackAccepted = callbackAccepted;
    }

    /// <summary>
    /// Gets whether TensorRT should continue after a step-complete callback.
    /// 获取 step-complete 回调后 TensorRT 是否应继续。
    /// </summary>
    public bool ShouldContinue { get; }

    /// <summary>
    /// Gets whether the callback completed without a managed exception or non-OK status.
    /// 获取回调是否未发生托管异常或非 OK 状态。
    /// </summary>
    public bool CallbackAccepted { get; }
}

/// <summary>
/// Receives TensorRT progress monitor events copied across the native boundary.
/// 接收从 native 边界复制过来的 TensorRT progress monitor 事件。
/// </summary>
/// <param name="progressEvent">The copied progress event. 复制后的 progress 事件。</param>
/// <returns>
/// <see langword="true"/> to continue a build after <see cref="TensorRtProgressMonitorEventKind.StepComplete"/>;
/// <see langword="false"/> to request cancellation. Phase-start and phase-finish return values are ignored.
/// 对 <see cref="TensorRtProgressMonitorEventKind.StepComplete"/> 返回 <see langword="true"/> 表示继续构建，返回
/// <see langword="false"/> 表示请求取消。phase-start 和 phase-finish 的返回值会被忽略。
/// </returns>
public delegate bool TensorRtProgressMonitorHandler(TensorRtProgressMonitorEvent progressEvent);

/// <summary>
/// Managed owner for a TensorRT progress monitor callback object.
/// TensorRT progress monitor 回调对象的托管 owner。
/// </summary>
public sealed class TensorRtProgressMonitor : IDisposable
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
    /// Releases progress monitor callback resources if <see cref="Dispose()"/> was not called.
    /// 如果调用方未调用 <see cref="Dispose()"/>，则释放 progress monitor callback 资源。
    /// </summary>
    ~TensorRtProgressMonitor()
    {
        Dispose();
    }

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

    /// <summary>
    /// Gets copied TensorRT versioned-interface metadata for this progress monitor.
    /// 获取当前 progress monitor 的 TensorRT versioned-interface 元数据副本。
    /// </summary>
    /// <remarks>
    /// TensorRT 10 and TensorRT 11 expose progress monitors as versioned interfaces.
    /// TensorRT 10 和 TensorRT 11 将 progress monitor 暴露为 versioned interface。
    /// </remarks>
    public TensorRtInterfaceInfo InterfaceInfo
    {
        get
        {
            ThrowIfDisposed();
            return NativeBridgeApi.GetProgressMonitorInterfaceInfo(Line, _handle);
        }
    }

    /// <summary>
    /// Gets copied TensorRT versioned-interface API language metadata for this progress monitor.
    /// 获取当前 progress monitor 的 TensorRT versioned-interface API language 元数据副本。
    /// </summary>
    /// <remarks>
    /// TensorRT 10 and TensorRT 11 expose progress monitors as versioned interfaces. The native bridge returns the scalar enum
    /// value and does not expose the monitor pointer.
    /// TensorRT 10 和 TensorRT 11 将 progress monitor 暴露为 versioned interface。native bridge 只返回标量 enum 值，
    /// 不暴露 monitor 指针。
    /// </remarks>
    public TensorRtApiLanguage ApiLanguage
    {
        get
        {
            ThrowIfDisposed();
            return NativeBridgeApi.GetProgressMonitorApiLanguage(Line, _handle);
        }
    }

    /// <summary>
    /// Tries to get copied TensorRT versioned-interface metadata for this progress monitor without exposing the native pointer.
    /// 尝试获取当前 progress monitor 的 TensorRT versioned-interface 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. 查询成功时复制出的 interface 元数据。</param>
    /// <returns><see langword="true"/> when interface metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo)
    {
        return TryGetInterfaceInfo(out interfaceInfo, out _);
    }

    /// <summary>
    /// Tries to get copied TensorRT versioned-interface metadata for this progress monitor without exposing the native pointer.
    /// 尝试获取当前 progress monitor 的 TensorRT versioned-interface 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. 查询成功时复制出的 interface 元数据。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when interface metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// TensorRT 10 and TensorRT 11 expose progress monitors as versioned interfaces. Unsupported or unavailable adapters return <see langword="false"/> with a diagnostic.
    /// TensorRT 10 和 TensorRT 11 将 progress monitor 暴露为 versioned interface；不支持或不可用的 adapter 会返回 <see langword="false"/> 并给出诊断。
    /// </remarks>
    public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)
    {
        ThrowIfDisposed();
        try
        {
            interfaceInfo = NativeBridgeApi.GetProgressMonitorInterfaceInfo(Line, _handle);
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
    /// Tries to get copied TensorRT API language metadata for this progress monitor without exposing the native pointer.
    /// 尝试获取当前 progress monitor 的 TensorRT API language 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="apiLanguage">The copied API language when the query succeeds. 查询成功时复制出的 API language。</param>
    /// <returns><see langword="true"/> when API language metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetApiLanguage(out TensorRtApiLanguage apiLanguage)
    {
        return TryGetApiLanguage(out apiLanguage, out _);
    }

    /// <summary>
    /// Tries to get copied TensorRT API language metadata for this progress monitor without exposing the native pointer.
    /// 尝试获取当前 progress monitor 的 TensorRT API language 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="apiLanguage">The copied API language when the query succeeds. 查询成功时复制出的 API language。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when API language metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetApiLanguage(out TensorRtApiLanguage apiLanguage, out string diagnostic)
    {
        ThrowIfDisposed();
        try
        {
            apiLanguage = NativeBridgeApi.GetProgressMonitorApiLanguage(Line, _handle);
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
    /// Synchronously emits a diagnostic progress monitor event through the native monitor object.
    /// 通过 native monitor 对象同步发送一条诊断 progress 事件。
    /// </summary>
    /// <param name="kind">The event kind to emit. 要触发的事件类型。</param>
    /// <param name="phaseName">The phase name copied to native UTF-8 memory. 复制到 native UTF-8 内存的阶段名称。</param>
    /// <param name="parentPhase">The optional parent phase. 可选父阶段。</param>
    /// <param name="step">The completed step index for step-complete events. step-complete 事件的完成步骤索引。</param>
    /// <param name="stepCount">The phase step count for phase-start events. phase-start 事件的总步骤数。</param>
    /// <returns>The diagnostic result reported by the native bridge. native bridge 返回的诊断结果。</returns>
    /// <remarks>
    /// This method is intended for diagnostics and smoke tests. It does not expose the native monitor pointer and does not attach the monitor to a builder config.
    /// 该方法用于诊断和 smoke 测试；它不会暴露 native monitor 指针，也不会把 monitor 绑定到 builder config。
    /// </remarks>
    public TensorRtProgressMonitorDiagnosticResult EmitDiagnostic(
        TensorRtProgressMonitorEventKind kind,
        string phaseName,
        string? parentPhase = null,
        int step = -1,
        int stepCount = 0)
    {
        if (phaseName == null)
        {
            throw new ArgumentNullException(nameof(phaseName));
        }

        ThrowIfDisposed();
        return NativeBridgeApi.EmitProgressMonitorDiagnostic(Line, _handle, kind, phaseName, parentPhase, step, stepCount);
    }

    /// <summary>
    /// Releases this progress monitor after all builder config attachments have been cleared.
    /// 在所有 builder config 绑定解除后释放当前 progress monitor。
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
            throw new ArgumentException("Progress monitor must belong to the same TensorRT API line as the builder config.");
        }

        lock (_gate)
        {
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtProgressMonitor));
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
                throw new ObjectDisposedException(nameof(TensorRtProgressMonitor));
            }
        }
    }

    private static BridgeStatusCode InvokeManagedProgressMonitor(
        int eventKind,
        IntPtr phaseName,
        UIntPtr phaseNameLength,
        IntPtr parentPhase,
        UIntPtr parentPhaseLength,
        int step,
        int stepCount,
        out int shouldContinue,
        IntPtr userState)
    {
        shouldContinue = 1;
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
            bool continueBuild = state.Handler(new TensorRtProgressMonitorEvent(
                (TensorRtProgressMonitorEventKind)eventKind,
                DecodeUtf8(phaseName, phaseNameLength),
                DecodeUtf8Nullable(parentPhase, parentPhaseLength),
                step,
                stepCount));
            shouldContinue = continueBuild ? 1 : 0;
            return BridgeStatusCode.Ok;
        }
        catch (Exception exception)
        {
            shouldContinue = 1;
            state?.RecordFailure(exception);
            return BridgeStatusCode.InvalidState;
        }
    }

    private static string DecodeUtf8(IntPtr value, UIntPtr length)
    {
        return DecodeUtf8Nullable(value, length) ?? string.Empty;
    }

    private static string? DecodeUtf8Nullable(IntPtr value, UIntPtr length)
    {
        if (value == IntPtr.Zero || length == UIntPtr.Zero)
        {
            return null;
        }

        ulong byteLength = length.ToUInt64();
        if (byteLength > int.MaxValue)
        {
            throw new InvalidOperationException("TensorRT progress monitor text is too large for a managed string.");
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

        public CallbackState(TensorRtProgressMonitorHandler handler)
        {
            Handler = handler;
        }

        public TensorRtProgressMonitorHandler Handler { get; }

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
