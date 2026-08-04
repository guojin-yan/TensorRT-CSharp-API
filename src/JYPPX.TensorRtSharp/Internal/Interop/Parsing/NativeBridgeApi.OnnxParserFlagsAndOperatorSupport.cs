using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static uint GetOnnxParserFlags(TensorRtApiLine line, SafeTensorRtObjectHandle parser)
    {
        uint flags;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_get_flags(parser, out flags),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_get_flags(parser, out flags),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_get_flags(parser, out flags),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return flags;
    }

    public static void SetOnnxParserFlags(TensorRtApiLine line, SafeTensorRtObjectHandle parser, uint flags)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_set_flags(parser, flags),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_set_flags(parser, flags),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_set_flags(parser, flags),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool GetOnnxParserFlag(TensorRtApiLine line, SafeTensorRtObjectHandle parser, TensorRtOnnxParserFlag flag)
    {
        int enabled;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_get_flag(parser, (int)flag, out enabled),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_get_flag(parser, (int)flag, out enabled),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_get_flag(parser, (int)flag, out enabled),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    public static void SetOnnxParserFlag(TensorRtApiLine line, SafeTensorRtObjectHandle parser, TensorRtOnnxParserFlag flag)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_set_flag(parser, (int)flag),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_set_flag(parser, (int)flag),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_set_flag(parser, (int)flag),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void ClearOnnxParserFlag(TensorRtApiLine line, SafeTensorRtObjectHandle parser, TensorRtOnnxParserFlag flag)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_clear_flag(parser, (int)flag),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_clear_flag(parser, (int)flag),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_clear_flag(parser, (int)flag),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool OnnxParserSupportsOperator(TensorRtApiLine line, SafeTensorRtObjectHandle parser, string operatorName)
    {
        if (string.IsNullOrWhiteSpace(operatorName))
        {
            throw new ArgumentException("ONNX operator name must not be null or empty.", nameof(operatorName));
        }

        using Utf8Interop.Utf8StringScope operatorNameUtf8 = Utf8Interop.ToNativeString(operatorName);
        int supported;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_supports_operator(parser, operatorNameUtf8.Pointer, out supported),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_supports_operator(parser, operatorNameUtf8.Pointer, out supported),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_supports_operator(parser, operatorNameUtf8.Pointer, out supported),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return supported != 0;
    }

}
