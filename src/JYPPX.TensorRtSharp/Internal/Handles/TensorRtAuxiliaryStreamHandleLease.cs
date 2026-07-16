using System;
using System.Collections.Generic;
using System.Threading;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Handles;

/// <summary>
/// Keeps caller-owned CUDA streams alive for as long as TensorRT may borrow them.
/// 在 TensorRT 可能借用调用方 CUDA stream 的期间保持其 native handle 有效。
/// </summary>
internal sealed class TensorRtAuxiliaryStreamHandleLease : IDisposable
{
    private SafeCudaStreamHandle[]? _handles;

    private TensorRtAuxiliaryStreamHandleLease(SafeCudaStreamHandle[] handles)
    {
        _handles = handles;
    }

    public IReadOnlyList<SafeCudaStreamHandle> Handles =>
        Volatile.Read(ref _handles)
        ?? throw new ObjectDisposedException(nameof(TensorRtAuxiliaryStreamHandleLease));

    public static TensorRtAuxiliaryStreamHandleLease Create(IReadOnlyList<SafeCudaStreamHandle> handles)
    {
        if (handles == null)
        {
            throw new ArgumentNullException(nameof(handles));
        }
        if (handles.Count == 0)
        {
            throw new ArgumentException("At least one auxiliary CUDA stream is required for a handle lease.", nameof(handles));
        }

        SafeCudaStreamHandle[] leasedHandles = new SafeCudaStreamHandle[handles.Count];
        HashSet<IntPtr> nativeHandles = new HashSet<IntPtr>();
        int acquiredCount = 0;
        try
        {
            for (int i = 0; i < handles.Count; i++)
            {
                SafeCudaStreamHandle handle = handles[i];
                if (handle == null || handle.IsClosed || handle.IsInvalid)
                {
                    throw new ArgumentException(
                        "Auxiliary streams must contain live, non-default CUDA stream handles.",
                        nameof(handles));
                }

                bool addedRef = false;
                handle.DangerousAddRef(ref addedRef);
                if (!addedRef)
                {
                    throw new ObjectDisposedException(nameof(handles), "An auxiliary CUDA stream was disposed while the lease was being acquired.");
                }

                leasedHandles[i] = handle;
                acquiredCount++;

                IntPtr nativeHandle = handle.DangerousGetHandle();
                if (nativeHandle == IntPtr.Zero || nativeHandle == new IntPtr(-1))
                {
                    throw new ArgumentException(
                        "Auxiliary streams must contain live, non-default CUDA stream handles.",
                        nameof(handles));
                }
                if (!nativeHandles.Add(nativeHandle))
                {
                    throw new ArgumentException(
                        "Auxiliary CUDA streams must be unique to avoid TensorRT cross-stream deadlocks.",
                        nameof(handles));
                }
            }

            return new TensorRtAuxiliaryStreamHandleLease(leasedHandles);
        }
        catch
        {
            for (int i = acquiredCount - 1; i >= 0; i--)
            {
                leasedHandles[i].DangerousRelease();
            }

            throw;
        }
    }

    public void Dispose()
    {
        SafeCudaStreamHandle[]? handles = Interlocked.Exchange(ref _handles, null);
        if (handles != null)
        {
            for (int i = handles.Length - 1; i >= 0; i--)
            {
                handles[i].DangerousRelease();
            }
        }

        GC.SuppressFinalize(this);
    }
}
