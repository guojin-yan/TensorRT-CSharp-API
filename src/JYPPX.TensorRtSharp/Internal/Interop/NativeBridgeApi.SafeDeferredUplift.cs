using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool InitializeLibNvInferPlugins(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle logger,
        string? libNamespace)
    {
        if (logger == null || logger.IsInvalid)
        {
            throw new ArgumentException("A valid TensorRT logger is required for plugin initialization.", nameof(logger));
        }

        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(libNamespace);
        int initialized;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_global_init_lib_nvinfer_plugins(logger, namespaceUtf8.Pointer, out initialized),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_global_init_lib_nvinfer_plugins(logger, namespaceUtf8.Pointer, out initialized),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_global_init_lib_nvinfer_plugins(logger, namespaceUtf8.Pointer, out initialized),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return initialized != 0;
    }

    public static bool ParseOnnxWithWeightDescriptors(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle parser,
        byte[] modelData)
    {
        if (parser == null || parser.IsInvalid)
        {
            throw new ArgumentException("A valid TensorRT ONNX parser is required.", nameof(parser));
        }

        if (modelData == null)
        {
            throw new ArgumentNullException(nameof(modelData));
        }

        if (modelData.Length == 0)
        {
            throw new ArgumentException("ONNX model data must not be empty.", nameof(modelData));
        }

        GCHandle pinned = GCHandle.Alloc(modelData, GCHandleType.Pinned);
        try
        {
            int parsed;
            IntPtr modelPointer = pinned.AddrOfPinnedObject();
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_parse_with_weight_descriptors(parser, modelPointer, new UIntPtr((ulong)modelData.Length), out parsed),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_parse_with_weight_descriptors(parser, modelPointer, new UIntPtr((ulong)modelData.Length), out parsed),
                TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parseWithWeightDescriptors was removed from the TensorRT 11 vendor interface."),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
            return parsed != 0;
        }
        finally
        {
            pinned.Free();
        }
    }
}
