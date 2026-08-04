using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
internal delegate BridgeStatusCode TensorRtProgressMonitorCallback(
    int eventKind,
    IntPtr phaseName,
    UIntPtr phaseNameLength,
    IntPtr parentPhase,
    UIntPtr parentPhaseLength,
    int step,
    int stepCount,
    out int shouldContinue,
    IntPtr userState);
