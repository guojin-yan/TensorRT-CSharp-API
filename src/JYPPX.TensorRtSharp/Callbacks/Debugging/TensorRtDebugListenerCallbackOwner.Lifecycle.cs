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
            releaseNow = _activeGateCallCount == 0;
        }

        if (releaseNow)
        {
            FreeCallbackState();
        }

        GC.SuppressFinalize(this);
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
            _callbackState.RecordReleaseHook("debug-listener-callback-owner-design release hook released managed GCHandle and delegate keep-alive handles after callbacks drained.");
            GC.KeepAlive(_callback);
        }
    }

}
