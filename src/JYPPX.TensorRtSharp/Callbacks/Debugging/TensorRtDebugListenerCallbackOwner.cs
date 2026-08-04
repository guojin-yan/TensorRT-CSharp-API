using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Owns a TensorRT 10/11 debug-listener callback and its managed lifetime.
/// 管理 TensorRT 10/11 debug-listener callback 及其托管生命周期。
/// </summary>
/// <remarks>
/// The line-and-handler constructor creates a native <c>IDebugListener</c> implementation. TensorRT-owned tensor data and
/// the CUDA stream remain borrowed inside the native callback; only copied name/type/location/shape metadata reaches
/// managed code. The parameterless constructor remains available for the repository's historical design diagnostics.
/// 带版本线与 handler 的构造函数会创建真实 native <c>IDebugListener</c> 实现。TensorRT 拥有的 tensor 数据与 CUDA
/// stream 始终留在 native callback 内，只把复制后的名称、类型、位置和 shape 元数据传到托管侧。无参构造函数继续供
/// 仓库既有设计诊断使用。
/// </remarks>
public sealed partial class TensorRtDebugListenerCallbackOwner : IDisposable
{
    private const int MaxShapeRank = 8;
    private static long s_nextOwnerId;

    private readonly object _gate = new object();
    private readonly long _ownerId;
    private readonly CallbackState _callbackState;
    private readonly TensorRtDebugListenerDesignGateCallback _callback;
    private readonly TensorRtDebugListenerNativeCallback? _nativeCallback;
    private readonly SafeTensorRtObjectHandle? _nativeHandle;
    private readonly TensorRtApiLine? _runtimeLine;
    private GCHandle _callbackStateHandle;
    private GCHandle _callbackHandle;
    private bool _hasCallbackStateHandle;
    private bool _hasCallbackHandle;
    private bool _disposeRequested;
    private bool _nativeHandleReleased;
    private int _activeGateCallCount;
    private int _attachmentCount;

    /// <summary>
    /// Creates a debug-listener callback owner design gate.
    /// 创建 debug-listener callback owner 设计门禁。
    /// </summary>
    public TensorRtDebugListenerCallbackOwner()
    {
        _ownerId = Interlocked.Increment(ref s_nextOwnerId);
        _callbackState = new CallbackState(handler: null);
        _callback = InvokeDebugListenerDesignGate;
        _nativeCallback = null;
        _nativeHandle = null;
        _runtimeLine = null;
        _callbackStateHandle = GCHandle.Alloc(_callbackState);
        _callbackHandle = GCHandle.Alloc(_callback);
        _hasCallbackStateHandle = true;
        _hasCallbackHandle = true;
    }

    /// <summary>
    /// Creates a real TensorRT debug-listener callback owner.
    /// 创建真实 TensorRT debug-listener callback owner。
    /// </summary>
    /// <param name="line">TensorRT 10 or TensorRT 11. TensorRT 10 或 TensorRT 11。</param>
    /// <param name="handler">The managed copied-metadata handler. 托管复制元数据处理器。</param>
    public TensorRtDebugListenerCallbackOwner(TensorRtApiLine line, TensorRtDebugListenerHandler handler)
    {
        if (line != TensorRtApiLine.TensorRt10 && line != TensorRtApiLine.TensorRt11)
        {
            throw new NotSupportedException("TensorRT debug listeners require TensorRT 10 or TensorRT 11.");
        }

        if (handler == null)
        {
            throw new ArgumentNullException(nameof(handler));
        }

        NativeBridgeLoader.EnsureInitialized();
        _ownerId = Interlocked.Increment(ref s_nextOwnerId);
        _runtimeLine = line;
        _callbackState = new CallbackState(handler);
        _callback = InvokeDebugListenerDesignGate;
        _nativeCallback = InvokeManagedDebugListener;
        _callbackStateHandle = GCHandle.Alloc(_callbackState);
        _callbackHandle = GCHandle.Alloc(_nativeCallback);
        _hasCallbackStateHandle = true;
        _hasCallbackHandle = true;

        try
        {
            _nativeHandle = NativeBridgeApi.CreateDebugListenerOwner(
                line,
                _nativeCallback,
                GCHandle.ToIntPtr(_callbackStateHandle));
        }
        catch
        {
            FreeCallbackState();
            throw;
        }
    }

    /// <summary>Gets whether dispose has been requested. 获取是否已请求释放。</summary>
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

    /// <summary>Gets whether this owner is attached to a TensorRT execution context. 获取该 owner 是否已绑定到 TensorRT execution context。</summary>
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

    /// <summary>Gets whether this instance owns a real native IDebugListener. 获取当前实例是否拥有真实 native IDebugListener。</summary>
    public bool HasNativeListener => _nativeHandle != null;

    /// <summary>Gets the runtime TensorRT line, or null for a design-only owner. 获取 runtime TensorRT 版本线；纯设计 owner 返回 null。</summary>
    public TensorRtApiLine? RuntimeLine => _runtimeLine;

    internal SafeTensorRtObjectHandle NativeHandle => _nativeHandle
        ?? throw new InvalidOperationException("This debug-listener owner was created for design diagnostics only.");

}
