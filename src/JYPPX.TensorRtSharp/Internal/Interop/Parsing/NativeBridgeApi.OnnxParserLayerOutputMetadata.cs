using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    private const uint OnnxLayerOutputHasDynamicDimension = 1U;
    private const uint OnnxLayerOutputIsShapeTensor = 2U;
    private const uint OnnxLayerOutputIsExecutionTensor = 4U;
    private const uint OnnxLayerOutputIsNetworkInput = 8U;
    private const uint OnnxLayerOutputIsNetworkOutput = 16U;

    public static bool TryGetOnnxParserLayerOutputTensorMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle parser,
        string layerName,
        long outputIndex,
        out TensorRtOnnxLayerOutputTensorMetadata? metadata)
    {
        if (string.IsNullOrWhiteSpace(layerName))
        {
            throw new ArgumentException("ONNX parser layer name must not be null or empty.", nameof(layerName));
        }

        if (outputIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputIndex));
        }

        if (line == TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(
                BridgeStatusCode.NotSupported,
                BridgeErrorCategory.TensorRt,
                "ONNX parser layer output tensor metadata is available only for TensorRT 10 and 11 adapters.");
        }

        using Utf8Interop.Utf8StringScope layerNameUtf8 = Utf8Interop.ToNativeString(layerName);
        int exists = 0;
        NativeTensorRtDims64 nativeShape = default;
        int dataType = 0;
        int location = 0;
        uint allowedFormats = 0;
        uint flags = 0;

        string tensorName = ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_get_layer_output_tensor_metadata(
                    parser, layerNameUtf8.Pointer, outputIndex, buffer, size, out required, out exists,
                    out nativeShape, out dataType, out location, out allowedFormats, out flags),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_get_layer_output_tensor_metadata(
                    parser, layerNameUtf8.Pointer, outputIndex, buffer, size, out required, out exists,
                    out nativeShape, out dataType, out location, out allowedFormats, out flags),
                _ => throw new BridgeProbeException(
                    BridgeStatusCode.InvalidArgument,
                    BridgeErrorCategory.Common,
                    "Unsupported TensorRT API line.")
            },
            "ONNX parser layer output tensor name is too large for the managed buffer.");

        if (exists == 0)
        {
            metadata = null;
            return false;
        }

        metadata = new TensorRtOnnxLayerOutputTensorMetadata(
            line,
            layerName,
            outputIndex,
            tensorName,
            TensorRtDims64.FromNative(nativeShape),
            (TensorRtDataType)dataType,
            (TensorRtTensorLocation)location,
            (TensorRtTensorFormats)allowedFormats,
            (flags & OnnxLayerOutputHasDynamicDimension) != 0,
            (flags & OnnxLayerOutputIsShapeTensor) != 0,
            (flags & OnnxLayerOutputIsExecutionTensor) != 0,
            (flags & OnnxLayerOutputIsNetworkInput) != 0,
            (flags & OnnxLayerOutputIsNetworkOutput) != 0);
        return true;
    }
}
