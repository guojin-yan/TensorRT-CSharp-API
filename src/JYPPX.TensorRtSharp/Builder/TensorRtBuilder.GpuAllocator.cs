using System;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtBuilder
{
    private readonly object _gpuAllocatorLeaseLock = new object();
    private TensorRtGpuAllocatorCallbackOwner? _gpuAllocatorKeepAlive;

    /// <summary>
    /// Attaches an owner-safe native <c>IGpuAllocator</c> to this builder.
    /// 将 owner-safe native IGpuAllocator 挂载到当前 builder。
    /// </summary>
    /// <param name="owner">The owner retained by this builder and every engine it creates. 由 builder 及其创建的每个 engine 保留的 owner。</param>
    public void SetGpuAllocator(TensorRtGpuAllocatorCallbackOwner owner)
    {
        if (owner == null)
        {
            throw new ArgumentNullException(nameof(owner));
        }
        ThrowIfGpuAllocatorCallbackReentry();

        lock (_gpuAllocatorLeaseLock)
        {
            ThrowIfBuilderDisposedForGpuAllocator();
            owner.ThrowIfDisposed();
            if (owner.Line != Line)
            {
                throw new ArgumentException("GPU allocator and builder must use the same TensorRT API line.", nameof(owner));
            }
            if (ReferenceEquals(_gpuAllocatorKeepAlive, owner))
            {
                return;
            }

            DetachManagedGpuAllocatorLocked(clearExternalAllocator: false);
            owner.AttachTargetBorrower(Line, TensorRtGpuAllocatorAttachmentTarget.Builder);
            try
            {
                if (!NativeBridgeApi.AttachGpuAllocatorOwnerToBuilder(Line, owner.NativeHandle, _handle))
                {
                    throw new InvalidOperationException("TensorRT did not accept the GPU allocator owner on the builder.");
                }
                _gpuAllocatorKeepAlive = owner;
            }
            catch
            {
                owner.DetachTargetBorrower(TensorRtGpuAllocatorAttachmentTarget.Builder);
                throw;
            }
        }
    }

    /// <summary>Gets whether this wrapper retains a managed GPU allocator owner. 获取该 wrapper 是否保留托管 GPU allocator owner。</summary>
    public bool HasManagedGpuAllocator
    {
        get { lock (_gpuAllocatorLeaseLock) { return _gpuAllocatorKeepAlive != null; } }
    }

    private T ExecuteWithGpuAllocatorLease<T>(Func<T> operation)
    {
        lock (_gpuAllocatorLeaseLock)
        {
            ThrowIfBuilderDisposedForGpuAllocator();
            return operation();
        }
    }

    private TensorRtEngine BuildEngineWithGpuAllocatorLease(Func<SafeTensorRtObjectHandle> build)
    {
        lock (_gpuAllocatorLeaseLock)
        {
            ThrowIfBuilderDisposedForGpuAllocator();
            SafeTensorRtObjectHandle engineHandle = build();
            try
            {
                return new TensorRtEngine(Line, engineHandle, _gpuAllocatorKeepAlive);
            }
            catch
            {
                engineHandle.Dispose();
                throw;
            }
        }
    }

    private void ClearManagedGpuAllocator()
    {
        ThrowIfGpuAllocatorCallbackReentry();
        lock (_gpuAllocatorLeaseLock)
        {
            ThrowIfBuilderDisposedForGpuAllocator();
            DetachManagedGpuAllocatorLocked(clearExternalAllocator: true);
        }
    }

    private void ReleaseManagedGpuAllocatorForDispose()
    {
        ThrowIfGpuAllocatorCallbackReentry();
        lock (_gpuAllocatorLeaseLock)
        {
            DetachManagedGpuAllocatorLocked(clearExternalAllocator: false);
        }
    }

    private void DetachManagedGpuAllocatorLocked(bool clearExternalAllocator)
    {
        TensorRtGpuAllocatorCallbackOwner? owner = _gpuAllocatorKeepAlive;
        if (owner == null)
        {
            if (clearExternalAllocator)
            {
                NativeBridgeApi.ClearBuilderGpuAllocator(Line, _handle);
            }
            return;
        }

        if (!NativeBridgeApi.DetachGpuAllocatorOwner(Line, owner.NativeHandle))
        {
            throw new InvalidOperationException("TensorRT did not detach the GPU allocator owner from the builder.");
        }
        _gpuAllocatorKeepAlive = null;
        owner.DetachTargetBorrower(TensorRtGpuAllocatorAttachmentTarget.Builder);
    }

    private void ThrowIfBuilderDisposedForGpuAllocator()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(TensorRtBuilder));
        }
    }

    private static void ThrowIfGpuAllocatorCallbackReentry()
    {
        if (TensorRtGpuAllocatorCallbackOwner.IsExecutingRuntimeCallbackOnCurrentThread)
        {
            throw new InvalidOperationException("A TensorRT GPU allocator cannot be replaced, cleared, or disposed from inside its own callback.");
        }
    }
}
