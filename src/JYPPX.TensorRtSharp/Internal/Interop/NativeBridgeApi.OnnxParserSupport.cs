using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static IReadOnlyList<string> GetOnnxParserUsedVCPluginLibraries(TensorRtApiLine line, SafeTensorRtObjectHandle parser)
    {
        long count = GetOnnxParserUsedVCPluginLibraryCount(line, parser);
        if (count <= 0)
        {
            return Array.Empty<string>();
        }

        if (count > int.MaxValue)
        {
            throw new BridgeProbeException(BridgeStatusCode.BufferTooSmall, BridgeErrorCategory.TensorRt, "ONNX parser VC plugin library count is too large for a managed list.");
        }

        string[] libraries = new string[(int)count];
        for (int index = 0; index < libraries.Length; index++)
        {
            long currentIndex = index;
            libraries[index] = ReadParserErrorString(
                (IntPtr buffer, UIntPtr size, out UIntPtr required) => line switch
                {
                    TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_get_used_vc_plugin_library(parser, currentIndex, buffer, size, out required),
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_get_used_vc_plugin_library(parser, currentIndex, buffer, size, out required),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_get_used_vc_plugin_library(parser, currentIndex, buffer, size, out required),
                    _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
                });
        }

        return libraries;
    }

    public static bool OnnxParserSupportsModelV2(TensorRtApiLine line, SafeTensorRtObjectHandle parser, byte[] modelData, string? modelPath)
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
            int supported;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_supports_model(parser, modelDataPointer, new UIntPtr((ulong)modelData.Length), modelPathUtf8.Pointer, out supported),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_supports_model_v2(parser, modelDataPointer, new UIntPtr((ulong)modelData.Length), modelPathUtf8.Pointer, out supported),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_supports_model_v2(parser, modelDataPointer, new UIntPtr((ulong)modelData.Length), modelPathUtf8.Pointer, out supported),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
            return supported != 0;
        }
        finally
        {
            pinned.Free();
        }
    }

    public static long GetOnnxParserSubgraphCount(TensorRtApiLine line, SafeTensorRtObjectHandle parser)
    {
        long count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_get_subgraph_count(parser, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_get_subgraph_count(parser, out count),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser subgraph support query is exposed for TensorRT 10 and 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static bool IsOnnxParserSubgraphSupported(TensorRtApiLine line, SafeTensorRtObjectHandle parser, long index)
    {
        int supported;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_is_subgraph_supported(parser, index, out supported),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_is_subgraph_supported(parser, index, out supported),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser subgraph support query is exposed for TensorRT 10 and 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return supported != 0;
    }

    public static long GetOnnxParserSubgraphNodeCount(TensorRtApiLine line, SafeTensorRtObjectHandle parser, long index)
    {
        long count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_get_subgraph_node_count(parser, index, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_get_subgraph_node_count(parser, index, out count),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser subgraph node query is exposed for TensorRT 10 and 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static long GetOnnxParserSubgraphNode(TensorRtApiLine line, SafeTensorRtObjectHandle parser, long subgraphIndex, long nodeIndex)
    {
        long node;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_get_subgraph_node(parser, subgraphIndex, nodeIndex, out node),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_get_subgraph_node(parser, subgraphIndex, nodeIndex, out node),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser subgraph node query is exposed for TensorRT 10 and 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return node;
    }

    public static long GetOnnxParserSupportedSubgraphCount(TensorRtApiLine line, SafeTensorRtObjectHandle parser)
    {
        long count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_get_supported_subgraph_count(parser, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_get_supported_subgraph_count(parser, out count),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser subgraph support query is exposed for TensorRT 10 and 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static long GetOnnxParserUnsupportedSubgraphCount(TensorRtApiLine line, SafeTensorRtObjectHandle parser)
    {
        long count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_get_unsupported_subgraph_count(parser, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_get_unsupported_subgraph_count(parser, out count),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser subgraph support query is exposed for TensorRT 10 and 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static bool OnnxParserLayerOutputTensorExists(TensorRtApiLine line, SafeTensorRtObjectHandle parser, string layerName, long outputIndex)
    {
        if (string.IsNullOrWhiteSpace(layerName))
        {
            throw new ArgumentException("ONNX parser layer name must not be null or empty.", nameof(layerName));
        }

        using Utf8Interop.Utf8StringScope layerNameUtf8 = Utf8Interop.ToNativeString(layerName);
        int exists;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_layer_output_tensor_exists(parser, layerNameUtf8.Pointer, outputIndex, out exists),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_layer_output_tensor_exists(parser, layerNameUtf8.Pointer, outputIndex, out exists),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser layer output tensor query is exposed for TensorRT 10 and 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return exists != 0;
    }

    private static long GetOnnxParserUsedVCPluginLibraryCount(TensorRtApiLine line, SafeTensorRtObjectHandle parser)
    {
        long count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_get_used_vc_plugin_library_count(parser, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_get_used_vc_plugin_library_count(parser, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_get_used_vc_plugin_library_count(parser, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }
}
