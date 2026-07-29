using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static void ClearEngineInspectorErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle inspector)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_inspector_clear_error_recorder(inspector),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_clear_error_recorder(inspector),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_clear_error_recorder(inspector),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

}
