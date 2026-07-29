using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

internal sealed partial class TensorRtOutputAllocatorRuntimeGate : IDisposable
{
    private const int MaxShapeRank = 8;
    private const int NotifyShapeOperation = 1;
    private const int ReallocateOutputOperation = 2;
    private static long s_nextOwnerId;

    private readonly object _gate = new object();
    private readonly long _ownerId;
    private readonly CallbackState _callbackState = new CallbackState();
    private readonly TensorRtOutputAllocatorInternalRuntimeGateCallback _runtimeGateCallback;
    private GCHandle _callbackStateHandle;
    private GCHandle _runtimeGateCallbackHandle;
    private bool _hasCallbackStateHandle;
    private bool _hasRuntimeGateCallbackHandle;
    private bool _disposeRequested;
    private int _activeGateCallCount;

    internal TensorRtOutputAllocatorRuntimeGate()
    {
        _ownerId = Interlocked.Increment(ref s_nextOwnerId);
        _runtimeGateCallback = InvokeOutputAllocatorRuntimeGate;
        _callbackStateHandle = GCHandle.Alloc(_callbackState);
        _runtimeGateCallbackHandle = GCHandle.Alloc(_runtimeGateCallback);
        _hasCallbackStateHandle = true;
        _hasRuntimeGateCallbackHandle = true;
    }

}
