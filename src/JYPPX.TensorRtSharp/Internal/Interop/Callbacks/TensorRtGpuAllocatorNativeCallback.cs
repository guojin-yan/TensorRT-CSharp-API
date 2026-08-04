using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
internal delegate BridgeStatusCode TensorRtGpuAllocatorNativeCallback(
    uint line,
    int callbackKind,
    ulong requestedSize,
    ulong alignment,
    uint allocatorFlags,
    int hasCurrentMemory,
    int hasStream,
    out int shouldProceed,
    IntPtr userState);
