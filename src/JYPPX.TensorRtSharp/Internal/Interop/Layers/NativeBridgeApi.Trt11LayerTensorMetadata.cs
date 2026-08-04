using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    private delegate BridgeStatusCode LayerSlotBoolGetter(SafeTensorRtObjectHandle layer, int index, out int value);
    private delegate BridgeStatusCode LayerSlotIntGetter(SafeTensorRtObjectHandle layer, int index, out int value);
    private delegate BridgeStatusCode LayerSlotUIntGetter(SafeTensorRtObjectHandle layer, int index, out uint value);
    private delegate BridgeStatusCode LayerSlotDimsGetter(SafeTensorRtObjectHandle layer, int index, out NativeTensorRtDims value);
    private delegate BridgeStatusCode LayerSlotStringGetter(SafeTensorRtObjectHandle layer, int index, byte[] outputBuffer, UIntPtr outputBufferSize, out UIntPtr requiredSize);
    private delegate BridgeStatusCode LayerSlotDimensionBoolGetter(SafeTensorRtObjectHandle layer, int index, int dimensionIndex, out int value);
    private delegate BridgeStatusCode LayerSlotDimensionIntGetter(SafeTensorRtObjectHandle layer, int index, int dimensionIndex, out int value);
    private delegate BridgeStatusCode LayerSlotDimensionStringGetter(SafeTensorRtObjectHandle layer, int index, int dimensionIndex, byte[] outputBuffer, UIntPtr outputBufferSize, out UIntPtr requiredSize);

    public static bool IsTensorNetworkInput(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_is_network_input(tensor, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_is_network_input(tensor, out value),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_is_network_input(tensor, out value),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }

    public static bool IsTensorNetworkOutput(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_is_network_output(tensor, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_is_network_output(tensor, out value),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_is_network_output(tensor, out value),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }

    public static bool HasLayerInputTensor(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotBool(line, layer, index, nameof(HasLayerInputTensor), NativeMethodsTensorRt.jyppx_trt11_layer_input_has_tensor);
    }

    public static bool HasLayerOutputTensor(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotBool(line, layer, index, nameof(HasLayerOutputTensor), NativeMethodsTensorRt.jyppx_trt11_layer_output_has_tensor);
    }

    public static string GetLayerInputTensorName(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotString(line, layer, index, nameof(GetLayerInputTensorName), NativeMethodsTensorRt.jyppx_trt11_layer_input_get_tensor_name, "Layer input tensor name is too large for the managed buffer.");
    }

    public static string GetLayerOutputTensorName(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotString(line, layer, index, nameof(GetLayerOutputTensorName), NativeMethodsTensorRt.jyppx_trt11_layer_output_get_tensor_name, "Layer output tensor name is too large for the managed buffer.");
    }

    public static TensorRtDataType GetLayerInputTensorDataType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return (TensorRtDataType)GetLayerSlotInt(line, layer, index, nameof(GetLayerInputTensorDataType), NativeMethodsTensorRt.jyppx_trt11_layer_input_get_tensor_data_type);
    }

    public static TensorRtDataType GetLayerOutputTensorDataType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return (TensorRtDataType)GetLayerSlotInt(line, layer, index, nameof(GetLayerOutputTensorDataType), NativeMethodsTensorRt.jyppx_trt11_layer_output_get_tensor_data_type);
    }

    public static TensorRtDims GetLayerInputTensorShape(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotDims(line, layer, index, nameof(GetLayerInputTensorShape), NativeMethodsTensorRt.jyppx_trt11_layer_input_get_tensor_shape);
    }

    public static TensorRtDims GetLayerOutputTensorShape(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotDims(line, layer, index, nameof(GetLayerOutputTensorShape), NativeMethodsTensorRt.jyppx_trt11_layer_output_get_tensor_shape);
    }

    public static int GetLayerInputTensorDimensionCount(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotInt(line, layer, index, nameof(GetLayerInputTensorDimensionCount), NativeMethodsTensorRt.jyppx_trt11_layer_input_get_tensor_dimension_count);
    }

    public static int GetLayerOutputTensorDimensionCount(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotInt(line, layer, index, nameof(GetLayerOutputTensorDimensionCount), NativeMethodsTensorRt.jyppx_trt11_layer_output_get_tensor_dimension_count);
    }

    public static int GetLayerInputTensorDimensionExtent(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, int dimensionIndex)
    {
        return GetLayerSlotDimensionInt(line, layer, index, dimensionIndex, nameof(GetLayerInputTensorDimensionExtent), NativeMethodsTensorRt.jyppx_trt11_layer_input_get_tensor_dimension_extent);
    }

    public static int GetLayerOutputTensorDimensionExtent(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, int dimensionIndex)
    {
        return GetLayerSlotDimensionInt(line, layer, index, dimensionIndex, nameof(GetLayerOutputTensorDimensionExtent), NativeMethodsTensorRt.jyppx_trt11_layer_output_get_tensor_dimension_extent);
    }

    public static bool LayerInputTensorHasDynamicDimension(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotBool(line, layer, index, nameof(LayerInputTensorHasDynamicDimension), NativeMethodsTensorRt.jyppx_trt11_layer_input_tensor_has_dynamic_dimension);
    }

    public static bool LayerOutputTensorHasDynamicDimension(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotBool(line, layer, index, nameof(LayerOutputTensorHasDynamicDimension), NativeMethodsTensorRt.jyppx_trt11_layer_output_tensor_has_dynamic_dimension);
    }

    public static TensorRtTensorLocation GetLayerInputTensorLocation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return (TensorRtTensorLocation)GetLayerSlotInt(line, layer, index, nameof(GetLayerInputTensorLocation), NativeMethodsTensorRt.jyppx_trt11_layer_input_get_tensor_location);
    }

    public static TensorRtTensorLocation GetLayerOutputTensorLocation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return (TensorRtTensorLocation)GetLayerSlotInt(line, layer, index, nameof(GetLayerOutputTensorLocation), NativeMethodsTensorRt.jyppx_trt11_layer_output_get_tensor_location);
    }

    public static TensorRtTensorFormats GetLayerInputTensorAllowedFormats(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return (TensorRtTensorFormats)GetLayerSlotUInt(line, layer, index, nameof(GetLayerInputTensorAllowedFormats), NativeMethodsTensorRt.jyppx_trt11_layer_input_get_tensor_allowed_formats);
    }

    public static TensorRtTensorFormats GetLayerOutputTensorAllowedFormats(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return (TensorRtTensorFormats)GetLayerSlotUInt(line, layer, index, nameof(GetLayerOutputTensorAllowedFormats), NativeMethodsTensorRt.jyppx_trt11_layer_output_get_tensor_allowed_formats);
    }

    public static bool IsLayerInputShapeTensor(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotBool(line, layer, index, nameof(IsLayerInputShapeTensor), NativeMethodsTensorRt.jyppx_trt11_layer_input_tensor_is_shape_tensor);
    }

    public static bool IsLayerOutputShapeTensor(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotBool(line, layer, index, nameof(IsLayerOutputShapeTensor), NativeMethodsTensorRt.jyppx_trt11_layer_output_tensor_is_shape_tensor);
    }

    public static bool IsLayerInputExecutionTensor(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotBool(line, layer, index, nameof(IsLayerInputExecutionTensor), NativeMethodsTensorRt.jyppx_trt11_layer_input_tensor_is_execution_tensor);
    }

    public static bool IsLayerOutputExecutionTensor(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotBool(line, layer, index, nameof(IsLayerOutputExecutionTensor), NativeMethodsTensorRt.jyppx_trt11_layer_output_tensor_is_execution_tensor);
    }

    public static bool IsLayerInputNetworkInput(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotBool(line, layer, index, nameof(IsLayerInputNetworkInput), NativeMethodsTensorRt.jyppx_trt11_layer_input_tensor_is_network_input);
    }

    public static bool IsLayerOutputNetworkInput(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotBool(line, layer, index, nameof(IsLayerOutputNetworkInput), NativeMethodsTensorRt.jyppx_trt11_layer_output_tensor_is_network_input);
    }

    public static bool IsLayerInputNetworkOutput(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotBool(line, layer, index, nameof(IsLayerInputNetworkOutput), NativeMethodsTensorRt.jyppx_trt11_layer_input_tensor_is_network_output);
    }

    public static bool IsLayerOutputNetworkOutput(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotBool(line, layer, index, nameof(IsLayerOutputNetworkOutput), NativeMethodsTensorRt.jyppx_trt11_layer_output_tensor_is_network_output);
    }

    public static bool LayerInputTensorDimensionHasName(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, int dimensionIndex)
    {
        return GetLayerSlotDimensionBool(line, layer, index, dimensionIndex, nameof(LayerInputTensorDimensionHasName), NativeMethodsTensorRt.jyppx_trt11_layer_input_tensor_dimension_has_name);
    }

    public static bool LayerOutputTensorDimensionHasName(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, int dimensionIndex)
    {
        return GetLayerSlotDimensionBool(line, layer, index, dimensionIndex, nameof(LayerOutputTensorDimensionHasName), NativeMethodsTensorRt.jyppx_trt11_layer_output_tensor_dimension_has_name);
    }

    public static string GetLayerInputTensorDimensionName(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, int dimensionIndex)
    {
        return GetLayerSlotDimensionString(line, layer, index, dimensionIndex, nameof(GetLayerInputTensorDimensionName), NativeMethodsTensorRt.jyppx_trt11_layer_input_get_tensor_dimension_name, "Layer input tensor dimension name is too large for the managed buffer.");
    }

    public static string GetLayerOutputTensorDimensionName(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, int dimensionIndex)
    {
        return GetLayerSlotDimensionString(line, layer, index, dimensionIndex, nameof(GetLayerOutputTensorDimensionName), NativeMethodsTensorRt.jyppx_trt11_layer_output_get_tensor_dimension_name, "Layer output tensor dimension name is too large for the managed buffer.");
    }

    public static string GetLayerInputTensorSummary(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotString(line, layer, index, nameof(GetLayerInputTensorSummary), NativeMethodsTensorRt.jyppx_trt11_layer_input_get_tensor_summary, "Layer input tensor summary is too large for the managed buffer.");
    }

    public static string GetLayerOutputTensorSummary(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotString(line, layer, index, nameof(GetLayerOutputTensorSummary), NativeMethodsTensorRt.jyppx_trt11_layer_output_get_tensor_summary, "Layer output tensor summary is too large for the managed buffer.");
    }

    private static bool GetLayerSlotBool(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, string apiName, LayerSlotBoolGetter getter)
    {
        EnsureTensorRt11LayerSlotApi(line, index, apiName);
        BridgeStatusCode status = getter(layer, index, out int value);
        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }

    private static int GetLayerSlotInt(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, string apiName, LayerSlotIntGetter getter)
    {
        EnsureTensorRt11LayerSlotApi(line, index, apiName);
        BridgeStatusCode status = getter(layer, index, out int value);
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static uint GetLayerSlotUInt(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, string apiName, LayerSlotUIntGetter getter)
    {
        EnsureTensorRt11LayerSlotApi(line, index, apiName);
        BridgeStatusCode status = getter(layer, index, out uint value);
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static TensorRtDims GetLayerSlotDims(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, string apiName, LayerSlotDimsGetter getter)
    {
        EnsureTensorRt11LayerSlotApi(line, index, apiName);
        BridgeStatusCode status = getter(layer, index, out NativeTensorRtDims value);
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(value);
    }

    private static string GetLayerSlotString(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, string apiName, LayerSlotStringGetter getter, string tooLargeMessage)
    {
        EnsureTensorRt11LayerSlotApi(line, index, apiName);
        return ReadUtf8Buffer((byte[] buffer, UIntPtr size, out UIntPtr required) => getter(layer, index, buffer, size, out required), tooLargeMessage);
    }

    private static bool GetLayerSlotDimensionBool(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, int dimensionIndex, string apiName, LayerSlotDimensionBoolGetter getter)
    {
        EnsureTensorRt11LayerSlotDimensionApi(line, index, dimensionIndex, apiName);
        BridgeStatusCode status = getter(layer, index, dimensionIndex, out int value);
        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }

    private static int GetLayerSlotDimensionInt(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, int dimensionIndex, string apiName, LayerSlotDimensionIntGetter getter)
    {
        EnsureTensorRt11LayerSlotDimensionApi(line, index, dimensionIndex, apiName);
        BridgeStatusCode status = getter(layer, index, dimensionIndex, out int value);
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static string GetLayerSlotDimensionString(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, int dimensionIndex, string apiName, LayerSlotDimensionStringGetter getter, string tooLargeMessage)
    {
        EnsureTensorRt11LayerSlotDimensionApi(line, index, dimensionIndex, apiName);
        return ReadUtf8Buffer((byte[] buffer, UIntPtr size, out UIntPtr required) => getter(layer, index, dimensionIndex, buffer, size, out required), tooLargeMessage);
    }

    private static void EnsureTensorRt11LayerSlotApi(TensorRtApiLine line, int index, string apiName)
    {
        EnsureTensorRt11DeploymentApi(line, apiName);
        if (index < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Layer tensor slot index must be greater than or equal to zero.");
        }
    }

    private static void EnsureTensorRt11LayerSlotDimensionApi(TensorRtApiLine line, int index, int dimensionIndex, string apiName)
    {
        EnsureTensorRt11LayerSlotApi(line, index, apiName);
        if (dimensionIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensionIndex), "Tensor dimension index must be greater than or equal to zero.");
        }
    }
}
