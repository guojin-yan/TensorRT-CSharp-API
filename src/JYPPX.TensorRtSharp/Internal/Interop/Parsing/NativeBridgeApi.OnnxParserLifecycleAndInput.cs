using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateOnnxParser(TensorRtApiLine line, SafeTensorRtObjectHandle logger, SafeTensorRtObjectHandle network)
    {
        SafeTensorRtObjectHandle parser;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_onnx_parser_create(logger, network, out parser);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_onnx_parser_create(logger, network, out parser);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_onnx_parser_create(logger, network, out parser);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return parser;
    }

    public static bool ParseOnnxFromFile(TensorRtApiLine line, SafeTensorRtObjectHandle parser, string filePath, int verbosity)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("ONNX model path must not be null or empty.", nameof(filePath));
        }

        using Utf8Interop.Utf8StringScope pathUtf8 = Utf8Interop.ToNativeString(filePath);
        int parsed;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_onnx_parser_parse_from_file(parser, pathUtf8.Pointer, verbosity, out parsed);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_onnx_parser_parse_from_file(parser, pathUtf8.Pointer, verbosity, out parsed);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_onnx_parser_parse_from_file(parser, pathUtf8.Pointer, verbosity, out parsed);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return parsed != 0;
    }

    public static bool ParseOnnxFromMemory(TensorRtApiLine line, SafeTensorRtObjectHandle parser, byte[] modelData, string? modelPath)
    {
        if (modelData == null)
        {
            throw new ArgumentNullException(nameof(modelData));
        }

        if (modelData.Length == 0)
        {
            throw new ArgumentException("ONNX model data must not be empty.", nameof(modelData));
        }

        using Utf8Interop.Utf8StringScope modelPathUtf8 = Utf8Interop.ToNativeString(modelPath);
        GCHandle pinned = GCHandle.Alloc(modelData, GCHandleType.Pinned);
        try
        {
            IntPtr modelDataPointer = pinned.AddrOfPinnedObject();
            int parsed;
            BridgeStatusCode status;
            switch (line)
            {
                case TensorRtApiLine.TensorRt8:
                    status = NativeMethodsTensorRt.jyppx_trt8_onnx_parser_parse_from_memory(parser, modelDataPointer, new UIntPtr((ulong)modelData.Length), modelPathUtf8.Pointer, out parsed);
                    break;
                case TensorRtApiLine.TensorRt10:
                    status = NativeMethodsTensorRt.jyppx_trt10_onnx_parser_parse_from_memory(parser, modelDataPointer, new UIntPtr((ulong)modelData.Length), modelPathUtf8.Pointer, out parsed);
                    break;
                case TensorRtApiLine.TensorRt11:
                    status = NativeMethodsTensorRt.jyppx_trt11_onnx_parser_parse_from_memory(parser, modelDataPointer, new UIntPtr((ulong)modelData.Length), modelPathUtf8.Pointer, out parsed);
                    break;
                default:
                    throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
            }

            NativeStatus.ThrowIfFailed(status);
            return parsed != 0;
        }
        finally
        {
            pinned.Free();
        }
    }

}
