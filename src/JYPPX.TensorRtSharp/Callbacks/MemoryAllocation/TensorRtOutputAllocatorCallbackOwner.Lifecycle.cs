using System;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOutputAllocatorCallbackOwner
{
    /// <summary>Releases an undisposed runtime callback owner. 释放未显式 Dispose 的 runtime callback owner。</summary>
    ~TensorRtOutputAllocatorCallbackOwner()
    {
        Dispose();
    }

    /// <summary>
    /// Releases managed keep-alive handles owned by the design gate.
    /// 释放该设计门禁持有的托管 keep-alive 句柄。
    /// </summary>
    public void Dispose()
    {
        bool releaseNow;
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            releaseNow = _attachmentCount == 0;
        }

        if (releaseNow)
        {
            ReleaseResources();
        }

        GC.SuppressFinalize(this);
    }

    internal void AttachBorrower(TensorRtApiLine expectedLine)
    {
        if (_runtimeLine == null || _nativeHandle == null)
        {
            throw new InvalidOperationException("The parameterless output allocator owner supports design diagnostics only.");
        }

        if (_runtimeLine.Value != expectedLine)
        {
            throw new ArgumentException("Output allocator and execution context must use the same TensorRT API line.");
        }

        lock (_gate)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(TensorRtOutputAllocatorCallbackOwner));
            }

            if (_attachmentCount != 0)
            {
                throw new InvalidOperationException("An output allocator owner can be attached to only one execution context tensor at a time.");
            }

            _attachmentCount = 1;
        }
    }

    internal void DetachBorrower()
    {
        bool releaseNow;
        lock (_gate)
        {
            _attachmentCount = 0;
            releaseNow = _disposed;
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
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(TensorRtOutputAllocatorCallbackOwner));
            }
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

        _nativeHandle?.Dispose();
        GC.KeepAlive(_nativeCallback);
        FreeRuntimeCallbackHandles();
        _runtimeGate.Dispose();
        _nativeLedgerOwner.Dispose();
    }

    private void FreeRuntimeCallbackHandles()
    {
        if (_hasNativeCallbackHandle)
        {
            _nativeCallbackHandle.Free();
            _hasNativeCallbackHandle = false;
        }

        if (_hasRuntimeCallbackStateHandle)
        {
            _runtimeCallbackStateHandle.Free();
            _hasRuntimeCallbackStateHandle = false;
        }
    }
}
