using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
internal delegate BridgeStatusCode TensorRtProfilerCallback(
    IntPtr layerName,
    UIntPtr layerNameLength,
    float milliseconds,
    IntPtr userState);
