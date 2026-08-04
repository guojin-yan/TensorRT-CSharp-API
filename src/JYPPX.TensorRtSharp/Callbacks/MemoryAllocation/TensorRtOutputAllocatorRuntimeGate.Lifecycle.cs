using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

internal sealed partial class TensorRtOutputAllocatorRuntimeGate
{
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
        if (_hasRuntimeGateCallbackHandle)
        {
            _runtimeGateCallbackHandle.Free();
            _hasRuntimeGateCallbackHandle = false;
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
            _callbackState.RecordReleaseHook("output-allocator-internal-runtime-gate release hook released managed GCHandle and delegate keep-alive handles after callbacks drained.");
            GC.KeepAlive(_runtimeGateCallback);
        }
    }

}
