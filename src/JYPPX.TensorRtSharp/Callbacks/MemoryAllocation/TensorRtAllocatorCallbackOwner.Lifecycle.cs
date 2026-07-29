using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtAllocatorCallbackOwner
{
    /// <summary>
    /// Runs a pointer-free lifecycle diagnostic through the managed allocator callback owner prototype.
    /// 通过托管 allocator callback owner prototype 执行无 pointer 的生命周期诊断。
    /// </summary>
    /// <param name="request">The copied dry-run request. 复制出的 dry-run 请求。</param>
    /// <returns>A copied lifecycle snapshot with no native handle or device pointer. 不包含 native handle 或 device pointer 的生命周期快照。</returns>
    /// <remarks>
    /// This method exercises the managed keep-alive and exception-to-status path used by the allocator owner prototype.
    /// It does not register the owner with TensorRT, does not call <c>setGpuAllocator</c>, and does not implement
    /// <c>IGpuAllocator::allocate/free/deallocate/reallocate</c>.
    /// 该方法会触发 allocator owner prototype 使用的托管 keep-alive 与 exception-to-status 路径。它不会把 owner 注册到
    /// TensorRT，不会调用 <c>setGpuAllocator</c>，也不会实现 <c>IGpuAllocator::allocate/free/deallocate/reallocate</c>。
    /// </remarks>
    public TensorRtAllocatorCallbackOwnerSnapshot RunLifecycleDiagnostic(TensorRtAllocatorDryRunRequest request)
    {
        return new TensorRtAllocatorCallbackOwnerSnapshot(RunInternalSyncAllocatorRuntimePrototype(request));
    }

    /// <summary>
    /// Gets a copied lifecycle snapshot of the current allocator callback owner state.
    /// 获取当前 allocator callback owner 状态的复制生命周期快照。
    /// </summary>
    /// <param name="operation">A copied operation label. 复制出的操作标签。</param>
    /// <returns>A copied lifecycle snapshot with no native handle or device pointer. 不包含 native handle 或 device pointer 的生命周期快照。</returns>
    /// <remarks>
    /// This method can be called after <see cref="Dispose"/> to verify release-hook and keep-alive cleanup diagnostics.
    /// It is still not proof that TensorRT invoked an allocator callback.
    /// 该方法可在 <see cref="Dispose"/> 后调用，用于验证 release hook 与 keep-alive 清理诊断；它仍然不证明 TensorRT
    /// 已调用 allocator callback。
    /// </remarks>
    public TensorRtAllocatorCallbackOwnerSnapshot GetSnapshot(string operation = "snapshot")
    {
        return new TensorRtAllocatorCallbackOwnerSnapshot(GetInternalRuntimePrototypeSnapshot(operation));
    }

    /// <summary>
    /// Releases the managed callback state keep-alive handle.
    /// 释放托管回调状态的 keep-alive 句柄。
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
            releaseNow = _activePrototypeCallCount == 0;
        }

        if (releaseNow)
        {
            FreeCallbackState();
        }

        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        lock (_gate)
        {
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtAllocatorCallbackOwner));
            }
        }
    }

    private void FreeCallbackState()
    {
        bool released = false;
        if (_hasRuntimePrototypeCallbackHandle)
        {
            _runtimePrototypeCallbackHandle.Free();
            _hasRuntimePrototypeCallbackHandle = false;
            released = true;
        }

        if (_hasCallbackStateHandle)
        {
            _callbackStateHandle.Free();
            _hasCallbackStateHandle = false;
            released = true;
        }

        if (released)
        {
            _callbackState.RecordReleaseHook("allocator-owner-internal-runtime-prototype release hook released managed GCHandle and delegate keep-alive handles after callbacks drained.");
            GC.KeepAlive(_runtimePrototypeCallback);
        }
    }

}
