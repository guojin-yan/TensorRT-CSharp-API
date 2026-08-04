using System;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Owns a TensorRT output allocator callback and its managed lifetime.
/// 管理 TensorRT output allocator callback 及其托管生命周期。
/// </summary>
/// <remarks>
/// The line-and-handler constructor creates a real native <c>IOutputAllocator</c>. Native code owns every CUDA
/// allocation and exposes only copied request metadata to the managed handler. The parameterless constructor remains
/// available for the repository's historical design diagnostics.
/// 带版本线与 handler 的构造函数会创建真实 native <c>IOutputAllocator</c>。所有 CUDA 分配均由 native 代码持有，托管
/// handler 只接收复制后的请求元数据。无参构造函数继续供仓库既有设计诊断使用。
/// </remarks>
public sealed partial class TensorRtOutputAllocatorCallbackOwner : IDisposable
{
    private readonly object _gate = new object();
    private readonly TensorRtOutputAllocatorRuntimeGate _runtimeGate = new TensorRtOutputAllocatorRuntimeGate();
    private readonly TensorRtAllocatorCallbackOwner _nativeLedgerOwner;
    private readonly RuntimeCallbackState? _runtimeCallbackState;
    private readonly TensorRtOutputAllocatorNativeCallback? _nativeCallback;
    private readonly SafeTensorRtObjectHandle? _nativeHandle;
    private readonly TensorRtApiLine? _runtimeLine;
    private GCHandle _runtimeCallbackStateHandle;
    private GCHandle _nativeCallbackHandle;
    private bool _hasRuntimeCallbackStateHandle;
    private bool _hasNativeCallbackHandle;
    private TensorRtAllocatorOwnerStateDryRunResult? _lastNativeLedgerState;
    private BridgeStatusCode _lastNativeLedgerStatus = BridgeStatusCode.Ok;
    private string _lastNativeLedgerDiagnostic = string.Empty;
    private TensorRtApiLine _lastLine = TensorRtApiLine.TensorRt11;
    private bool _lastNativeLedgerAvailable;
    private bool _disposed;
    private bool _resourcesReleased;
    private int _attachmentCount;

    /// <summary>
    /// Creates an output allocator callback owner design gate.
    /// 创建 output allocator callback owner 设计门禁。
    /// </summary>
    public TensorRtOutputAllocatorCallbackOwner()
    {
        _nativeLedgerOwner = new TensorRtAllocatorCallbackOwner(static request =>
            TensorRtAllocatorDryRunResult.Success("output-allocator-callback-owner-design managed ledger keep-alive " + request));
    }

    /// <summary>
    /// Creates a real TensorRT output allocator callback owner.
    /// 创建真实 TensorRT output allocator callback owner。
    /// </summary>
    /// <param name="line">TensorRT 8, 10, or 11. TensorRT 8、10 或 11。</param>
    /// <param name="handler">The pointer-free managed request handler. 无指针托管请求处理器。</param>
    public TensorRtOutputAllocatorCallbackOwner(TensorRtApiLine line, TensorRtOutputAllocatorHandler handler)
    {
        if (line != TensorRtApiLine.TensorRt8 &&
            line != TensorRtApiLine.TensorRt10 &&
            line != TensorRtApiLine.TensorRt11)
        {
            throw new NotSupportedException("Output allocator callback owners require TensorRT 8, 10, or 11.");
        }

        if (handler == null)
        {
            throw new ArgumentNullException(nameof(handler));
        }

        _nativeLedgerOwner = new TensorRtAllocatorCallbackOwner(static request =>
            TensorRtAllocatorDryRunResult.Success("output-allocator-callback-owner-runtime managed ledger keep-alive " + request));
        NativeBridgeLoader.EnsureInitialized();
        _runtimeLine = line;
        _runtimeCallbackState = new RuntimeCallbackState(line, handler);
        _nativeCallback = InvokeManagedOutputAllocator;
        _runtimeCallbackStateHandle = GCHandle.Alloc(_runtimeCallbackState);
        _nativeCallbackHandle = GCHandle.Alloc(_nativeCallback);
        _hasRuntimeCallbackStateHandle = true;
        _hasNativeCallbackHandle = true;

        try
        {
            _nativeHandle = NativeBridgeApi.CreateOutputAllocatorOwner(
                line,
                _nativeCallback,
                GCHandle.ToIntPtr(_runtimeCallbackStateHandle));
        }
        catch
        {
            FreeRuntimeCallbackHandles();
            _runtimeGate.Dispose();
            _nativeLedgerOwner.Dispose();
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
                return _disposed;
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

    /// <summary>Gets whether this instance owns a real native IOutputAllocator. 获取当前实例是否拥有真实 native IOutputAllocator。</summary>
    public bool HasNativeAllocator => _nativeHandle != null;

    /// <summary>Gets the runtime TensorRT line, or null for a design-only owner. 获取 runtime TensorRT 版本线；纯设计 owner 返回 null。</summary>
    public TensorRtApiLine? RuntimeLine => _runtimeLine;

    internal SafeTensorRtObjectHandle NativeHandle => _nativeHandle
        ?? throw new InvalidOperationException("This output allocator owner was created for design diagnostics only.");

}
