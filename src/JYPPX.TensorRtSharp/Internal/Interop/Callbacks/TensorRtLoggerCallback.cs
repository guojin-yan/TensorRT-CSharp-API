using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
internal delegate BridgeStatusCode TensorRtLoggerCallback(int severity, IntPtr message, UIntPtr messageLength, IntPtr userState);
