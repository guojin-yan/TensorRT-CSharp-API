using System;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtProgressMonitor
{
    /// <summary>
    /// Releases progress monitor callback resources if <see cref="Dispose()"/> was not called.
    /// 如果调用方未调用 <see cref="Dispose()"/>，则释放 progress monitor callback 资源。
    /// </summary>
    ~TensorRtProgressMonitor()
    {
        Dispose();
    }

    /// <summary>
    /// Releases this progress monitor after all builder config attachments have been cleared.
    /// 在所有 builder config 绑定解除后释放当前 progress monitor。
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
            throw new ArgumentException("Progress monitor must belong to the same TensorRT API line as the builder config.");
        }

        lock (_gate)
        {
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtProgressMonitor));
            }

            checked
            {
                _attachmentCount++;
            }
        }
    }

    internal void DetachBorrower()
    {
        bool releaseNow = false;
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

    internal void ThrowIfDisposed()
    {
        lock (_gate)
        {
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtProgressMonitor));
            }
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

    private void FreeCallbackState()
    {
        if (_hasCallbackStateHandle)
        {
            _callbackStateHandle.Free();
            _hasCallbackStateHandle = false;
        }
    }
}
