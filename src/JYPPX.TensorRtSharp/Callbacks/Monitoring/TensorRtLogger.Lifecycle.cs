using System;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLogger
{
    /// <summary>
    /// Releases logger callback resources if <see cref="Dispose()"/> was not called.
    /// 如果调用方未调用 <see cref="Dispose()"/>，则释放 logger callback 资源。
    /// </summary>
    ~TensorRtLogger()
    {
        Dispose();
    }

    /// <summary>
    /// Releases the TensorRT logger handle.
    /// 释放 TensorRT logger 句柄。
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
            releaseNow = _attachmentCount == 0;
        }

        if (releaseNow)
        {
            ReleaseHandle();
        }

        GC.SuppressFinalize(this);
    }

    internal void AttachBorrower(TensorRtApiLine expectedLine)
    {
        if (expectedLine != Line)
        {
            throw new ArgumentException("Logger must belong to the same TensorRT API line as the owner.");
        }

        lock (_gate)
        {
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtLogger));
            }

            checked
            {
                _attachmentCount++;
            }
        }
    }

    internal void DetachBorrower()
    {
        bool releaseNow;
        lock (_gate)
        {
            if (_attachmentCount > 0)
            {
                _attachmentCount--;
            }

            releaseNow = _attachmentCount == 0 && _disposeRequested;
        }

        if (releaseNow)
        {
            ReleaseHandle();
        }
    }

    private void FreeCallbackState()
    {
        if (_hasCallbackStateHandle)
        {
            _callbackStateHandle.Free();
            _hasCallbackStateHandle = false;
        }
    }

    private void ReleaseHandle()
    {
        bool shouldRelease;
        lock (_gate)
        {
            shouldRelease = !_handleReleased;
            _handleReleased = true;
        }

        if (!shouldRelease)
        {
            return;
        }

        _handle.Dispose();
        GC.KeepAlive(_nativeCallback);
        FreeCallbackState();
    }

    internal void ThrowIfDisposed()
    {
        lock (_gate)
        {
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtLogger));
            }
        }
    }
}
