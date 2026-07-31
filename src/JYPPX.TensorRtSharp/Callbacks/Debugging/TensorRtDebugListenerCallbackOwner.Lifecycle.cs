using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtDebugListenerCallbackOwner
{
    /// <summary>Releases an undisposed runtime callback owner. 释放未显式 Dispose 的 runtime callback owner。</summary>
    ~TensorRtDebugListenerCallbackOwner()
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
            if (_disposeRequested)
            {
                return;
            }

            _disposeRequested = true;
            releaseNow = _activeGateCallCount == 0 && _attachmentCount == 0;
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
            throw new InvalidOperationException("The parameterless debug-listener owner supports design diagnostics only.");
        }

        if (_runtimeLine.Value != expectedLine)
        {
            throw new ArgumentException("Debug listener and execution context must use the same TensorRT API line.");
        }

        lock (_gate)
        {
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtDebugListenerCallbackOwner));
            }

            if (_attachmentCount != 0)
            {
                throw new InvalidOperationException("A debug listener owner can be attached to only one execution context at a time.");
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
            releaseNow = _disposeRequested && _activeGateCallCount == 0;
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
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtDebugListenerCallbackOwner));
            }
        }
    }

    private void ReleaseResources()
    {
        lock (_gate)
        {
            if (_nativeHandleReleased)
            {
                return;
            }

            _nativeHandleReleased = true;
        }

        _nativeHandle?.Dispose();
        GC.KeepAlive(_nativeCallback);
        FreeCallbackState();
    }

    private void FreeCallbackState()
    {
        bool released = false;
        if (_hasCallbackHandle)
        {
            _callbackHandle.Free();
            _hasCallbackHandle = false;
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
            _callbackState.RecordReleaseHook(
                _nativeHandle == null
                    ? "debug-listener-callback-owner-design release hook released managed GCHandle and delegate keep-alive handles after callbacks drained."
                    : "debug-listener callback owner released its native vtable before unpinning managed callback state.");
            GC.KeepAlive(_callback);
        }
    }

}
