using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Owns a native TensorRT <c>IGpuAllocator</c> and its pointer-free managed policy callback.
/// 管理 native TensorRT IGpuAllocator 及其无指针托管策略回调。
/// </summary>
/// <remarks>
/// Native code owns all CUDA allocations. Runtime and builder wrappers retain this owner, and every engine they create
/// inherits an additional lease because TensorRT may keep using the allocator after its creator is detached or disposed.
/// 所有 CUDA 分配均由 native 代码持有。Runtime、builder 及其创建的每个 engine 都会保留 owner 租约。
/// </remarks>
public sealed class TensorRtGpuAllocatorCallbackOwner : IDisposable
{
    private readonly object _gate = new object();
    private readonly TensorRtGpuAllocatorNativeCallback _nativeCallback;
    private readonly SafeTensorRtObjectHandle _nativeHandle;
    private GCHandle _callbackStateHandle;
    private GCHandle _nativeCallbackHandle;
    private bool _hasCallbackStateHandle;
    private bool _hasNativeCallbackHandle;
    private bool _disposeRequested;
    private bool _resourcesReleased;
    private int _borrowerCount;
    private TensorRtGpuAllocatorAttachmentTarget _attachmentTarget;

    [ThreadStatic]
    private static int s_runtimeCallbackDepth;

    /// <summary>Creates a real native GPU allocator owner. 创建真实 native GPU allocator owner。</summary>
    /// <param name="line">TensorRT 8, 10, or 11. TensorRT 8、10 或 11。</param>
    /// <param name="handler">The pointer-free managed policy handler. 无指针托管策略处理器。</param>
    public TensorRtGpuAllocatorCallbackOwner(TensorRtApiLine line, TensorRtGpuAllocatorHandler handler)
    {
        if (line != TensorRtApiLine.TensorRt8 && line != TensorRtApiLine.TensorRt10 && line != TensorRtApiLine.TensorRt11)
        {
            throw new NotSupportedException("GPU allocator callback owners require TensorRT 8, 10, or 11.");
        }

        if (handler == null)
        {
            throw new ArgumentNullException(nameof(handler));
        }

        NativeBridgeLoader.EnsureInitialized();
        Line = line;
        CallbackState state = new CallbackState(line, handler);
        _nativeCallback = InvokeManagedCallback;
        _callbackStateHandle = GCHandle.Alloc(state);
        _nativeCallbackHandle = GCHandle.Alloc(_nativeCallback);
        _hasCallbackStateHandle = true;
        _hasNativeCallbackHandle = true;
        try
        {
            _nativeHandle = NativeBridgeApi.CreateGpuAllocatorOwner(line, _nativeCallback, GCHandle.ToIntPtr(_callbackStateHandle));
        }
        catch
        {
            FreeCallbackHandles();
            throw;
        }
    }

    /// <summary>
    /// Releases native resources after all TensorRT borrowers are gone when the caller did not dispose the owner explicitly.
    /// 调用方未显式释放 owner 时，在所有 TensorRT 借用者结束后释放 native 资源。
    /// </summary>
    ~TensorRtGpuAllocatorCallbackOwner()
    {
        Dispose();
    }

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API 版本线。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether dispose was requested. 获取是否已请求释放。</summary>
    public bool IsDisposed
    {
        get { lock (_gate) { return _disposeRequested; } }
    }

    /// <summary>Gets whether the owner is attached to a runtime or builder. 获取 owner 是否挂载到 runtime 或 builder。</summary>
    public bool IsAttached
    {
        get { lock (_gate) { return _attachmentTarget != TensorRtGpuAllocatorAttachmentTarget.None; } }
    }

    /// <summary>Gets the current managed attachment target. 获取当前托管挂载目标。</summary>
    public TensorRtGpuAllocatorAttachmentTarget AttachmentTarget
    {
        get { lock (_gate) { return _attachmentTarget; } }
    }

    /// <summary>Gets a copied, pointer-free native runtime snapshot. 获取复制后的无指针 native 运行快照。</summary>
    public TensorRtGpuAllocatorRuntimeSnapshot GetRuntimeSnapshot()
    {
        lock (_gate)
        {
            if (_resourcesReleased)
            {
                throw new ObjectDisposedException(nameof(TensorRtGpuAllocatorCallbackOwner));
            }
        }

        NativeTensorRtGpuAllocatorOwnerInfo info = NativeBridgeApi.GetGpuAllocatorOwnerInfo(Line, _nativeHandle);
        TensorRtGpuAllocatorCallbackKind kind = Enum.IsDefined(typeof(TensorRtGpuAllocatorCallbackKind), info.LastCallbackKind)
            ? (TensorRtGpuAllocatorCallbackKind)info.LastCallbackKind
            : TensorRtGpuAllocatorCallbackKind.Unknown;
        TensorRtGpuAllocatorAttachmentTarget target = Enum.IsDefined(typeof(TensorRtGpuAllocatorAttachmentTarget), info.AttachmentTarget)
            ? (TensorRtGpuAllocatorAttachmentTarget)info.AttachmentTarget
            : TensorRtGpuAllocatorAttachmentTarget.None;
        return new TensorRtGpuAllocatorRuntimeSnapshot(
            (TensorRtApiLine)info.Line,
            info.OwnerId,
            info.InvocationCount,
            info.AllocateCount,
            info.ReallocateCount,
            info.DeallocateCount,
            info.AllocateAsyncCount,
            info.DeallocateAsyncCount,
            info.RejectedCount,
            info.CallbackFailureCount,
            info.CudaFailureCount,
            info.InFlightCallbackCount,
            info.MaxInFlightCallbackCount,
            info.AttachCount,
            info.DetachCount,
            info.LiveAllocationCount,
            info.LiveAllocationBytes,
            info.PeakLiveAllocationBytes,
            info.LastRequestedSize,
            info.LastAlignment,
            info.LastAllocatorFlags,
            (BridgeStatusCode)info.LastStatus,
            target,
            kind,
            info.IsAttached != 0,
            info.LastCallbackSucceeded != 0,
            info.LastOperationSucceeded != 0,
            info.LastHadCurrentMemory != 0,
            info.LastHadStream != 0,
            BridgeInfoMapper.ReadFixedUtf8(info.LastDiagnostic));
    }

    /// <summary>Requests release; actual release is deferred until every TensorRT borrower is gone. 请求释放；实际释放延迟到所有 TensorRT 借用者结束。</summary>
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
            releaseNow = _borrowerCount == 0;
        }

        if (releaseNow)
        {
            ReleaseResources();
        }
        GC.SuppressFinalize(this);
    }

    internal SafeTensorRtObjectHandle NativeHandle => _nativeHandle;
    internal static bool IsExecutingRuntimeCallbackOnCurrentThread => s_runtimeCallbackDepth > 0;

    internal void AttachTargetBorrower(TensorRtApiLine expectedLine, TensorRtGpuAllocatorAttachmentTarget target)
    {
        if (target == TensorRtGpuAllocatorAttachmentTarget.None)
        {
            throw new ArgumentOutOfRangeException(nameof(target));
        }

        lock (_gate)
        {
            ThrowIfUnavailableLocked(expectedLine);
            if (_attachmentTarget != TensorRtGpuAllocatorAttachmentTarget.None)
            {
                throw new InvalidOperationException("A GPU allocator owner can be attached to only one runtime or builder at a time.");
            }

            _attachmentTarget = target;
            _borrowerCount++;
        }
    }

    internal void DetachTargetBorrower(TensorRtGpuAllocatorAttachmentTarget target)
    {
        bool releaseNow;
        lock (_gate)
        {
            if (_attachmentTarget != target)
            {
                throw new InvalidOperationException("GPU allocator attachment target does not match the borrower being detached.");
            }

            _attachmentTarget = TensorRtGpuAllocatorAttachmentTarget.None;
            _borrowerCount--;
            releaseNow = _disposeRequested && _borrowerCount == 0;
        }

        if (releaseNow)
        {
            ReleaseResources();
        }
    }

    internal void AttachEngineBorrower(TensorRtApiLine expectedLine)
    {
        lock (_gate)
        {
            if (_resourcesReleased || expectedLine != Line)
            {
                throw new InvalidOperationException("GPU allocator and engine must remain available on the same TensorRT API line.");
            }

            _borrowerCount++;
        }
    }

    internal void DetachEngineBorrower()
    {
        bool releaseNow;
        lock (_gate)
        {
            if (_borrowerCount <= 0)
            {
                throw new InvalidOperationException("GPU allocator engine borrower ledger underflow.");
            }

            _borrowerCount--;
            releaseNow = _disposeRequested && _borrowerCount == 0;
        }

        if (releaseNow)
        {
            ReleaseResources();
        }
    }

    internal void ThrowIfDisposed()
    {
        lock (_gate)
        {
            ThrowIfUnavailableLocked(Line);
        }
    }

    private static BridgeStatusCode InvokeManagedCallback(
        uint line,
        int callbackKind,
        ulong requestedSize,
        ulong alignment,
        uint allocatorFlags,
        int hasCurrentMemory,
        int hasStream,
        out int shouldProceed,
        IntPtr userState)
    {
        shouldProceed = 0;
        s_runtimeCallbackDepth++;
        try
        {
            if (userState == IntPtr.Zero)
            {
                return BridgeStatusCode.InvalidArgument;
            }

            CallbackState? state = GCHandle.FromIntPtr(userState).Target as CallbackState;
            if (state == null || line != (uint)state.Line || !Enum.IsDefined(typeof(TensorRtGpuAllocatorCallbackKind), callbackKind))
            {
                return BridgeStatusCode.InvalidState;
            }

            TensorRtGpuAllocatorCallbackKind kind = (TensorRtGpuAllocatorCallbackKind)callbackKind;
            if (kind == TensorRtGpuAllocatorCallbackKind.Unknown)
            {
                return BridgeStatusCode.InvalidArgument;
            }

            TensorRtGpuAllocatorCallbackRequest request = new TensorRtGpuAllocatorCallbackRequest(
                kind,
                requestedSize,
                alignment,
                allocatorFlags,
                hasCurrentMemory != 0,
                hasStream != 0);
            bool accepted = state.Handler(request);
            shouldProceed = request.IsRelease || accepted ? 1 : 0;
            return BridgeStatusCode.Ok;
        }
        catch
        {
            shouldProceed = 0;
            return BridgeStatusCode.InvalidState;
        }
        finally
        {
            s_runtimeCallbackDepth--;
        }
    }

    private void ThrowIfUnavailableLocked(TensorRtApiLine expectedLine)
    {
        if (_disposeRequested || _resourcesReleased)
        {
            throw new ObjectDisposedException(nameof(TensorRtGpuAllocatorCallbackOwner));
        }
        if (expectedLine != Line)
        {
            throw new ArgumentException("GPU allocator and TensorRT target must use the same API line.");
        }
    }

    private void ReleaseResources()
    {
        lock (_gate)
        {
            if (_resourcesReleased)
            {
                return;
            }
            _resourcesReleased = true;
        }

        _nativeHandle.Dispose();
        GC.KeepAlive(_nativeCallback);
        FreeCallbackHandles();
    }

    private void FreeCallbackHandles()
    {
        if (_hasNativeCallbackHandle)
        {
            _nativeCallbackHandle.Free();
            _hasNativeCallbackHandle = false;
        }
        if (_hasCallbackStateHandle)
        {
            _callbackStateHandle.Free();
            _hasCallbackStateHandle = false;
        }
    }

    private sealed class CallbackState
    {
        public CallbackState(TensorRtApiLine line, TensorRtGpuAllocatorHandler handler)
        {
            Line = line;
            Handler = handler;
        }
        public TensorRtApiLine Line { get; }
        public TensorRtGpuAllocatorHandler Handler { get; }
    }
}
