using System;
using System.Threading;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Handles;

/// <summary>Keeps caller-owned CUDA memory alive while TensorRT may borrow it. 在 TensorRT 借用期间保持调用方 CUDA 内存有效。</summary>
internal sealed class TensorRtDeviceMemoryHandleLease : IDisposable
{
    private SafeCudaMemoryHandle? _handle;

    private TensorRtDeviceMemoryHandleLease(SafeCudaMemoryHandle handle, int sizeInBytes)
    {
        _handle = handle;
        SizeInBytes = sizeInBytes;
    }

    public SafeCudaMemoryHandle Handle =>
        Volatile.Read(ref _handle)
        ?? throw new ObjectDisposedException(nameof(TensorRtDeviceMemoryHandleLease));

    public int SizeInBytes { get; }

    public static TensorRtDeviceMemoryHandleLease Create(SafeCudaMemoryHandle handle, int sizeInBytes)
    {
        if (handle == null)
        {
            throw new ArgumentNullException(nameof(handle));
        }
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }
        if (handle.IsClosed || handle.IsInvalid)
        {
            throw new ObjectDisposedException(nameof(handle), "CUDA memory must be live when TensorRT acquires its lease.");
        }

        bool addedRef = false;
        try
        {
            handle.DangerousAddRef(ref addedRef);
            if (!addedRef || handle.DangerousGetHandle() == IntPtr.Zero)
            {
                throw new ObjectDisposedException(nameof(handle), "CUDA memory was disposed while TensorRT acquired its lease.");
            }

            return new TensorRtDeviceMemoryHandleLease(handle, sizeInBytes);
        }
        catch
        {
            if (addedRef)
            {
                handle.DangerousRelease();
            }

            throw;
        }
    }

    public void Dispose()
    {
        ReleaseHandle();
        GC.SuppressFinalize(this);
    }

    ~TensorRtDeviceMemoryHandleLease()
    {
        ReleaseHandle();
    }

    private void ReleaseHandle()
    {
        SafeCudaMemoryHandle? handle = Interlocked.Exchange(ref _handle, null);
        handle?.DangerousRelease();
    }
}
