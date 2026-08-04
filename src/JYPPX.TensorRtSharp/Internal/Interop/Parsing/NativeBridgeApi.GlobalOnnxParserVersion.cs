using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static int GetGlobalOnnxParserVersion(TensorRtApiLine line)
    {
        int value;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_global_get_onnx_parser_version(out value);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_get_onnx_parser_version(out value);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_get_onnx_parser_version(out value);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

}
