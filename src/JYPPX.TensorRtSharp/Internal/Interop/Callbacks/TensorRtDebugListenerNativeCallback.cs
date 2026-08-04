using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
internal delegate BridgeStatusCode TensorRtDebugListenerNativeCallback(
    uint line,
    IntPtr tensorName,
    UIntPtr tensorNameLength,
    int dataType,
    int location,
    int shapeRank,
    long dim0,
    long dim1,
    long dim2,
    long dim3,
    long dim4,
    long dim5,
    long dim6,
    long dim7,
    IntPtr userState);
