using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
internal delegate BridgeStatusCode TensorRtProfilerCallback(
    IntPtr layerName,
    UIntPtr layerNameLength,
    float milliseconds,
    IntPtr userState);
