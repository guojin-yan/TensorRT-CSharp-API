using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaHandleLease : IDisposable
{
    private SafeHandle[]? _handles;
    private readonly IntPtr[] _values;

    private SafeCudaHandleLease(SafeHandle[] handles, IntPtr[] values)
    {
        _handles = handles;
        _values = values;
    }

    public static SafeCudaHandleLease Create(IReadOnlyList<SafeHandle> handles)
    {
        if (handles == null)
        {
            throw new ArgumentNullException(nameof(handles));
        }

        SafeHandle[] leasedHandles = new SafeHandle[handles.Count];
        IntPtr[] values = new IntPtr[handles.Count];
        int leasedCount = 0;
        try
        {
            for (int index = 0; index < handles.Count; ++index)
            {
                SafeHandle handle = handles[index] ?? throw new ArgumentException("CUDA handle leases must not contain null handles.", nameof(handles));
                bool addedRef = false;
                handle.DangerousAddRef(ref addedRef);
                if (!addedRef)
                {
                    throw new ObjectDisposedException(handle.GetType().Name);
                }

                leasedHandles[index] = handle;
                values[index] = handle.DangerousGetHandle();
                leasedCount++;
            }

            return new SafeCudaHandleLease(leasedHandles, values);
        }
        catch
        {
            for (int index = leasedCount - 1; index >= 0; --index)
            {
                leasedHandles[index].DangerousRelease();
            }
            throw;
        }
    }

    public IntPtr GetHandle(int index)
    {
        if (_handles == null)
        {
            throw new ObjectDisposedException(nameof(SafeCudaHandleLease));
        }
        return _values[index];
    }

    public void Dispose()
    {
        ReleaseHandles();
        GC.SuppressFinalize(this);
    }

    ~SafeCudaHandleLease()
    {
        ReleaseHandles();
    }

    private void ReleaseHandles()
    {
        SafeHandle[]? handles = _handles;
        if (handles == null)
        {
            return;
        }

        _handles = null;
        for (int index = handles.Length - 1; index >= 0; --index)
        {
            handles[index].DangerousRelease();
        }
    }
}
