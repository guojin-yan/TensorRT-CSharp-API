using System;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtRuntime
{
    private readonly object _gpuAllocatorLeaseLock = new object();
    private TensorRtGpuAllocatorCallbackOwner? _gpuAllocatorKeepAlive;

    /// <summary>
    /// Attaches an owner-safe native <c>IGpuAllocator</c> to this runtime.
    /// 将 owner-safe native IGpuAllocator 挂载到当前 runtime。
    /// </summary>
    /// <param name="owner">The owner retained by this runtime and every engine it creates. 由 runtime 及其创建的每个 engine 保留的 owner。</param>
    public void SetGpuAllocator(TensorRtGpuAllocatorCallbackOwner owner)
    {
        if (owner == null)
        {
            throw new ArgumentNullException(nameof(owner));
        }
        ThrowIfGpuAllocatorCallbackReentry();

        lock (_gpuAllocatorLeaseLock)
        {
            ThrowIfRuntimeDisposedForGpuAllocator();
            owner.ThrowIfDisposed();
            if (owner.Line != Line)
            {
                throw new ArgumentException("GPU allocator and runtime must use the same TensorRT API line.", nameof(owner));
            }
            if (ReferenceEquals(_gpuAllocatorKeepAlive, owner))
            {
                return;
            }

            DetachManagedGpuAllocatorLocked(clearExternalAllocator: false);
            owner.AttachTargetBorrower(Line, TensorRtGpuAllocatorAttachmentTarget.Runtime);
            try
            {
                if (!NativeBridgeApi.AttachGpuAllocatorOwnerToRuntime(Line, owner.NativeHandle, _handle))
                {
                    throw new InvalidOperationException("TensorRT did not accept the GPU allocator owner on the runtime.");
                }
                _gpuAllocatorKeepAlive = owner;
            }
            catch
            {
                owner.DetachTargetBorrower(TensorRtGpuAllocatorAttachmentTarget.Runtime);
                throw;
            }
        }
    }

    /// <summary>Gets whether this wrapper retains a managed GPU allocator owner. 获取该 wrapper 是否保留托管 GPU allocator owner。</summary>
    public bool HasManagedGpuAllocator
    {
        get { lock (_gpuAllocatorLeaseLock) { return _gpuAllocatorKeepAlive != null; } }
    }

    private TensorRtEngine DeserializeWithGpuAllocatorLease(Func<SafeTensorRtObjectHandle> deserialize)
    {
        return DeserializeWithGpuAllocatorLease(deserialize, null);
    }

    private TensorRtEngine DeserializeWithGpuAllocatorLease(
        Func<SafeTensorRtObjectHandle> deserialize,
        TensorRtStreamReader? streamReaderKeepAlive)
    {
        lock (_gpuAllocatorLeaseLock)
        {
            ThrowIfRuntimeDisposedForGpuAllocator();
            SafeTensorRtObjectHandle engineHandle = deserialize();
            try
            {
                return new TensorRtEngine(Line, engineHandle, _gpuAllocatorKeepAlive, streamReaderKeepAlive);
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
            ThrowIfRuntimeDisposedForGpuAllocator();
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
                NativeBridgeApi.ClearRuntimeGpuAllocator(Line, _handle);
            }
            return;
        }

        if (!NativeBridgeApi.DetachGpuAllocatorOwner(Line, owner.NativeHandle))
        {
            throw new InvalidOperationException("TensorRT did not detach the GPU allocator owner from the runtime.");
        }
        _gpuAllocatorKeepAlive = null;
        owner.DetachTargetBorrower(TensorRtGpuAllocatorAttachmentTarget.Runtime);
    }

    private void ThrowIfRuntimeDisposedForGpuAllocator()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(TensorRtRuntime));
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
