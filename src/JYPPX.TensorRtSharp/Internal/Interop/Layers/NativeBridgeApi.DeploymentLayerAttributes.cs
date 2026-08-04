using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtGatherMode GetGatherMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtGatherMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_gather_layer_get_mode, NativeMethodsTensorRt.jyppx_trt10_gather_layer_get_mode, NativeMethodsTensorRt.jyppx_trt11_gather_layer_get_mode);
    }

    public static void SetGatherMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtGatherMode mode)
    {
        SetLayerInt(line, layer, (int)mode, NativeMethodsTensorRt.jyppx_trt8_gather_layer_set_mode, NativeMethodsTensorRt.jyppx_trt10_gather_layer_set_mode, NativeMethodsTensorRt.jyppx_trt11_gather_layer_set_mode);
    }

    public static int GetGatherElementWiseDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_gather_layer_get_nb_elementwise_dims, NativeMethodsTensorRt.jyppx_trt10_gather_layer_get_nb_elementwise_dims, NativeMethodsTensorRt.jyppx_trt11_gather_layer_get_nb_elementwise_dims);
    }

    public static void SetGatherElementWiseDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int dimensions)
    {
        SetLayerInt(line, layer, dimensions, NativeMethodsTensorRt.jyppx_trt8_gather_layer_set_nb_elementwise_dims, NativeMethodsTensorRt.jyppx_trt10_gather_layer_set_nb_elementwise_dims, NativeMethodsTensorRt.jyppx_trt11_gather_layer_set_nb_elementwise_dims);
    }

    public static TensorRtScatterMode GetScatterMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtScatterMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_scatter_layer_get_mode, NativeMethodsTensorRt.jyppx_trt10_scatter_layer_get_mode, NativeMethodsTensorRt.jyppx_trt11_scatter_layer_get_mode);
    }

    public static void SetScatterMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtScatterMode mode)
    {
        SetLayerInt(line, layer, (int)mode, NativeMethodsTensorRt.jyppx_trt8_scatter_layer_set_mode, NativeMethodsTensorRt.jyppx_trt10_scatter_layer_set_mode, NativeMethodsTensorRt.jyppx_trt11_scatter_layer_set_mode);
    }

    public static int GetScatterAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_scatter_layer_get_axis, NativeMethodsTensorRt.jyppx_trt10_scatter_layer_get_axis, NativeMethodsTensorRt.jyppx_trt11_scatter_layer_get_axis);
    }

    public static void SetScatterAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int axis)
    {
        SetLayerInt(line, layer, axis, NativeMethodsTensorRt.jyppx_trt8_scatter_layer_set_axis, NativeMethodsTensorRt.jyppx_trt10_scatter_layer_set_axis, NativeMethodsTensorRt.jyppx_trt11_scatter_layer_set_axis);
    }

    public static int GetOneHotAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_one_hot_layer_get_axis, NativeMethodsTensorRt.jyppx_trt10_one_hot_layer_get_axis, NativeMethodsTensorRt.jyppx_trt11_one_hot_layer_get_axis);
    }

    public static void SetOneHotAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int axis)
    {
        SetLayerInt(line, layer, axis, NativeMethodsTensorRt.jyppx_trt8_one_hot_layer_set_axis, NativeMethodsTensorRt.jyppx_trt10_one_hot_layer_set_axis, NativeMethodsTensorRt.jyppx_trt11_one_hot_layer_set_axis);
    }

    public static TensorRtCumulativeOperation GetCumulativeOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtCumulativeOperation)GetLayerIntTensorRt10OrNewer(line, layer, NativeMethodsTensorRt.jyppx_trt10_cumulative_layer_get_operation, NativeMethodsTensorRt.jyppx_trt11_cumulative_layer_get_operation, nameof(GetCumulativeOperation));
    }

    public static void SetCumulativeOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtCumulativeOperation operation)
    {
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_cumulative_layer_set_operation(layer, (int)operation, out int success);
                NativeStatus.ThrowIfFailed(status);
                if (success == 0)
                {
                    throw new BridgeProbeException(BridgeStatusCode.RuntimeError, BridgeErrorCategory.TensorRt, "TensorRT rejected the cumulative operation.");
                }

                return;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_cumulative_layer_set_operation(layer, (int)operation);
                NativeStatus.ThrowIfFailed(status);
                return;
            case TensorRtApiLine.TensorRt8:
                throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "SetCumulativeOperation is available for TensorRT 10 and TensorRT 11 adapters.");
            default:
                throw UnsupportedLine();
        }
    }

    public static bool GetCumulativeExclusive(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerIntTensorRt10OrNewer(line, layer, NativeMethodsTensorRt.jyppx_trt10_cumulative_layer_get_exclusive, NativeMethodsTensorRt.jyppx_trt11_cumulative_layer_get_exclusive, nameof(GetCumulativeExclusive)) != 0;
    }

    public static void SetCumulativeExclusive(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool exclusive)
    {
        SetLayerIntTensorRt10OrNewer(line, layer, exclusive ? 1 : 0, NativeMethodsTensorRt.jyppx_trt10_cumulative_layer_set_exclusive, NativeMethodsTensorRt.jyppx_trt11_cumulative_layer_set_exclusive, nameof(SetCumulativeExclusive));
    }

    public static bool GetCumulativeReverse(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerIntTensorRt10OrNewer(line, layer, NativeMethodsTensorRt.jyppx_trt10_cumulative_layer_get_reverse, NativeMethodsTensorRt.jyppx_trt11_cumulative_layer_get_reverse, nameof(GetCumulativeReverse)) != 0;
    }

    public static void SetCumulativeReverse(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool reverse)
    {
        SetLayerIntTensorRt10OrNewer(line, layer, reverse ? 1 : 0, NativeMethodsTensorRt.jyppx_trt10_cumulative_layer_set_reverse, NativeMethodsTensorRt.jyppx_trt11_cumulative_layer_set_reverse, nameof(SetCumulativeReverse));
    }

    public static string GetAssertionMessage(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return ReadUtf8Buffer(line switch
        {
            TensorRtApiLine.TensorRt8 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt8_assertion_layer_get_message(layer, buffer, size, out required),
            TensorRtApiLine.TensorRt10 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt10_assertion_layer_get_message(layer, buffer, size, out required),
            TensorRtApiLine.TensorRt11 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_assertion_layer_get_message(layer, buffer, size, out required),
            _ => throw UnsupportedLine()
        }, "Assertion message is too large for the managed buffer.");
    }

    public static void SetAssertionMessage(TensorRtApiLine line, SafeTensorRtObjectHandle layer, string message)
    {
        using Utf8Interop.Utf8StringScope messageUtf8 = Utf8Interop.ToNativeString(message ?? string.Empty);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_assertion_layer_set_message(layer, messageUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_assertion_layer_set_message(layer, messageUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_assertion_layer_set_message(layer, messageUtf8.Pointer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtInterpolationMode GetGridSampleInterpolationMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtInterpolationMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_grid_sample_layer_get_interpolation_mode, NativeMethodsTensorRt.jyppx_trt10_grid_sample_layer_get_interpolation_mode, NativeMethodsTensorRt.jyppx_trt11_grid_sample_layer_get_interpolation_mode);
    }

    public static void SetGridSampleInterpolationMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtInterpolationMode mode)
    {
        SetLayerInt(line, layer, (int)mode, NativeMethodsTensorRt.jyppx_trt8_grid_sample_layer_set_interpolation_mode, NativeMethodsTensorRt.jyppx_trt10_grid_sample_layer_set_interpolation_mode, NativeMethodsTensorRt.jyppx_trt11_grid_sample_layer_set_interpolation_mode);
    }

    public static bool GetGridSampleAlignCorners(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_grid_sample_layer_get_align_corners, NativeMethodsTensorRt.jyppx_trt10_grid_sample_layer_get_align_corners, NativeMethodsTensorRt.jyppx_trt11_grid_sample_layer_get_align_corners) != 0;
    }

    public static void SetGridSampleAlignCorners(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool alignCorners)
    {
        SetLayerInt(line, layer, alignCorners ? 1 : 0, NativeMethodsTensorRt.jyppx_trt8_grid_sample_layer_set_align_corners, NativeMethodsTensorRt.jyppx_trt10_grid_sample_layer_set_align_corners, NativeMethodsTensorRt.jyppx_trt11_grid_sample_layer_set_align_corners);
    }

    public static TensorRtSampleMode GetGridSampleMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtSampleMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_grid_sample_layer_get_sample_mode, NativeMethodsTensorRt.jyppx_trt10_grid_sample_layer_get_sample_mode, NativeMethodsTensorRt.jyppx_trt11_grid_sample_layer_get_sample_mode);
    }

    public static void SetGridSampleMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtSampleMode mode)
    {
        SetLayerInt(line, layer, (int)mode, NativeMethodsTensorRt.jyppx_trt8_grid_sample_layer_set_sample_mode, NativeMethodsTensorRt.jyppx_trt10_grid_sample_layer_set_sample_mode, NativeMethodsTensorRt.jyppx_trt11_grid_sample_layer_set_sample_mode);
    }

    public static double GetNormalizationEpsilon(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDouble(line, layer, NativeMethodsTensorRt.jyppx_trt8_normalization_layer_get_epsilon, NativeMethodsTensorRt.jyppx_trt10_normalization_layer_get_epsilon, NativeMethodsTensorRt.jyppx_trt11_normalization_layer_get_epsilon);
    }

    public static void SetNormalizationEpsilon(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double epsilon)
    {
        SetLayerDouble(line, layer, epsilon, NativeMethodsTensorRt.jyppx_trt8_normalization_layer_set_epsilon, NativeMethodsTensorRt.jyppx_trt10_normalization_layer_set_epsilon, NativeMethodsTensorRt.jyppx_trt11_normalization_layer_set_epsilon);
    }

    public static uint GetNormalizationAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerUInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_normalization_layer_get_axes, NativeMethodsTensorRt.jyppx_trt10_normalization_layer_get_axes, NativeMethodsTensorRt.jyppx_trt11_normalization_layer_get_axes);
    }

    public static void SetNormalizationAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer, uint axes)
    {
        SetLayerUInt(line, layer, axes, NativeMethodsTensorRt.jyppx_trt8_normalization_layer_set_axes, NativeMethodsTensorRt.jyppx_trt10_normalization_layer_set_axes, NativeMethodsTensorRt.jyppx_trt11_normalization_layer_set_axes);
    }

    public static long GetNormalizationGroupCount(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt64(line, layer, NativeMethodsTensorRt.jyppx_trt8_normalization_layer_get_nb_groups, NativeMethodsTensorRt.jyppx_trt10_normalization_layer_get_nb_groups, NativeMethodsTensorRt.jyppx_trt11_normalization_layer_get_nb_groups);
    }

    public static void SetNormalizationGroupCount(TensorRtApiLine line, SafeTensorRtObjectHandle layer, long groupCount)
    {
        SetLayerInt64(line, layer, groupCount, NativeMethodsTensorRt.jyppx_trt8_normalization_layer_set_nb_groups, NativeMethodsTensorRt.jyppx_trt10_normalization_layer_set_nb_groups, NativeMethodsTensorRt.jyppx_trt11_normalization_layer_set_nb_groups);
    }

    public static bool IsNormalizationV2(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(IsNormalizationV2));
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt11_normalization_layer_is_v2, NativeMethodsTensorRt.jyppx_trt11_normalization_layer_is_v2, NativeMethodsTensorRt.jyppx_trt11_normalization_layer_is_v2) != 0;
    }

    public static TensorRtDataType GetDynamicQuantizeToType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtDataType)GetLayerIntTensorRt10OrNewer(line, layer, NativeMethodsTensorRt.jyppx_trt10_dynamic_quantize_layer_get_to_type, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_get_to_type, nameof(GetDynamicQuantizeToType));
    }

    public static void SetDynamicQuantizeToType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDataType dataType)
    {
        SetLayerIntTensorRt10OrNewer(line, layer, (int)dataType, NativeMethodsTensorRt.jyppx_trt10_dynamic_quantize_layer_set_to_type, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_set_to_type, nameof(SetDynamicQuantizeToType));
    }

    public static TensorRtDataType GetDynamicQuantizeScaleType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtDataType)GetLayerIntTensorRt10OrNewer(line, layer, NativeMethodsTensorRt.jyppx_trt10_dynamic_quantize_layer_get_scale_type, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_get_scale_type, nameof(GetDynamicQuantizeScaleType));
    }

    public static void SetDynamicQuantizeScaleType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDataType dataType)
    {
        SetLayerIntTensorRt10OrNewer(line, layer, (int)dataType, NativeMethodsTensorRt.jyppx_trt10_dynamic_quantize_layer_set_scale_type, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_set_scale_type, nameof(SetDynamicQuantizeScaleType));
    }

    public static int GetDynamicQuantizeAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerIntTensorRt10OrNewer(line, layer, NativeMethodsTensorRt.jyppx_trt10_dynamic_quantize_layer_get_axis, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_get_axis, nameof(GetDynamicQuantizeAxis));
    }

    public static void SetDynamicQuantizeAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int axis)
    {
        SetLayerIntTensorRt10OrNewer(line, layer, axis, NativeMethodsTensorRt.jyppx_trt10_dynamic_quantize_layer_set_axis, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_set_axis, nameof(SetDynamicQuantizeAxis));
    }

    public static int GetDynamicQuantizeBlockSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerIntTensorRt10OrNewer(line, layer, NativeMethodsTensorRt.jyppx_trt10_dynamic_quantize_layer_get_block_size, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_get_block_size, nameof(GetDynamicQuantizeBlockSize));
    }

    public static void SetDynamicQuantizeBlockSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int blockSize)
    {
        if (blockSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(blockSize), "Dynamic quantize block size must be positive.");
        }

        SetLayerIntTensorRt10OrNewer(line, layer, blockSize, NativeMethodsTensorRt.jyppx_trt10_dynamic_quantize_layer_set_block_size, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_set_block_size, nameof(SetDynamicQuantizeBlockSize));
    }

    public static TensorRtDims GetDynamicQuantizeBlockShape(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetDynamicQuantizeBlockShape));
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_get_block_shape, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_get_block_shape, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_get_block_shape);
    }

    public static void SetDynamicQuantizeBlockShape(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims blockShape)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetDynamicQuantizeBlockShape));
        SetLayerDims(line, layer, blockShape, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_set_block_shape, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_set_block_shape, NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_set_block_shape);
    }

    public static void SetLayerInput(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, SafeTensorRtObjectHandle tensor)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetLayerInput));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_layer_set_input(layer, index, tensor);
        NativeStatus.ThrowIfFailed(status);
    }

    public static string GetLayerMetadata(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => ReadUtf8Buffer((byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt8_layer_get_metadata(layer, buffer, size, out required), "Layer metadata is too large for the managed buffer."),
            TensorRtApiLine.TensorRt10 => ReadUtf8Buffer((byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt10_layer_get_metadata(layer, buffer, size, out required), "Layer metadata is too large for the managed buffer."),
            TensorRtApiLine.TensorRt11 => ReadUtf8Buffer((byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_layer_get_metadata(layer, buffer, size, out required), "Layer metadata is too large for the managed buffer."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
    }

    public static void SetLayerMetadata(TensorRtApiLine line, SafeTensorRtObjectHandle layer, string metadata)
    {
        using Utf8Interop.Utf8StringScope metadataUtf8 = Utf8Interop.ToNativeString(metadata ?? string.Empty);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_set_metadata(layer, metadataUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_set_metadata(layer, metadataUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_set_metadata(layer, metadataUtf8.Pointer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool SetLayerRankCount(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int rankCount)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetLayerRankCount));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_layer_set_nb_ranks(layer, rankCount, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static int GetLayerRankCount(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetLayerRankCount));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_layer_get_nb_ranks(layer, out int rankCount);
        NativeStatus.ThrowIfFailed(status);
        return rankCount;
    }

    public static TensorRtDataType GetCastToType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtDataType)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_cast_layer_get_to_type, NativeMethodsTensorRt.jyppx_trt10_cast_layer_get_to_type, NativeMethodsTensorRt.jyppx_trt11_cast_layer_get_to_type);
    }

    public static void SetCastToType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDataType dataType)
    {
        SetLayerInt(line, layer, (int)dataType, NativeMethodsTensorRt.jyppx_trt8_cast_layer_set_to_type, NativeMethodsTensorRt.jyppx_trt10_cast_layer_set_to_type, NativeMethodsTensorRt.jyppx_trt11_cast_layer_set_to_type);
    }

    public static TensorRtDataType GetNonZeroIndicesType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetNonZeroIndicesType));
        return (TensorRtDataType)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt11_non_zero_layer_get_indices_type, NativeMethodsTensorRt.jyppx_trt11_non_zero_layer_get_indices_type, NativeMethodsTensorRt.jyppx_trt11_non_zero_layer_get_indices_type);
    }

    public static bool SetNonZeroIndicesType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDataType dataType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetNonZeroIndicesType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_non_zero_layer_set_indices_type(layer, (int)dataType, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static TensorRtBoundingBoxFormat GetNmsBoundingBoxFormat(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtBoundingBoxFormat)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_nms_layer_get_bounding_box_format, NativeMethodsTensorRt.jyppx_trt10_nms_layer_get_bounding_box_format, NativeMethodsTensorRt.jyppx_trt11_nms_layer_get_bounding_box_format);
    }

    public static void SetNmsBoundingBoxFormat(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtBoundingBoxFormat format)
    {
        SetLayerInt(line, layer, (int)format, NativeMethodsTensorRt.jyppx_trt8_nms_layer_set_bounding_box_format, NativeMethodsTensorRt.jyppx_trt10_nms_layer_set_bounding_box_format, NativeMethodsTensorRt.jyppx_trt11_nms_layer_set_bounding_box_format);
    }

    public static int GetNmsTopKBoxLimit(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_nms_layer_get_topk_box_limit, NativeMethodsTensorRt.jyppx_trt10_nms_layer_get_topk_box_limit, NativeMethodsTensorRt.jyppx_trt11_nms_layer_get_topk_box_limit);
    }

    public static void SetNmsTopKBoxLimit(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int limit)
    {
        SetLayerInt(line, layer, limit, NativeMethodsTensorRt.jyppx_trt8_nms_layer_set_topk_box_limit, NativeMethodsTensorRt.jyppx_trt10_nms_layer_set_topk_box_limit, NativeMethodsTensorRt.jyppx_trt11_nms_layer_set_topk_box_limit);
    }

    public static void SetNmsIouThresholdTensor(TensorRtApiLine line, SafeTensorRtObjectHandle layer, SafeTensorRtObjectHandle tensor)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetNmsIouThresholdTensor));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_nms_layer_set_iou_threshold_tensor(layer, tensor);
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetNmsScoreThresholdTensor(TensorRtApiLine line, SafeTensorRtObjectHandle layer, SafeTensorRtObjectHandle tensor)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetNmsScoreThresholdTensor));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_nms_layer_set_score_threshold_tensor(layer, tensor);
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDataType GetNmsIndicesType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetNmsIndicesType));
        return (TensorRtDataType)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt11_nms_layer_get_indices_type, NativeMethodsTensorRt.jyppx_trt11_nms_layer_get_indices_type, NativeMethodsTensorRt.jyppx_trt11_nms_layer_get_indices_type);
    }

    public static bool SetNmsIndicesType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDataType dataType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetNmsIndicesType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_nms_layer_set_indices_type(layer, (int)dataType, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static TensorRtDims GetConstantLayerDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_constant_layer_get_dimensions, NativeMethodsTensorRt.jyppx_trt10_constant_layer_get_dimensions, NativeMethodsTensorRt.jyppx_trt11_constant_layer_get_dimensions);
    }

    public static void SetConstantLayerDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims dimensions)
    {
        SetLayerDims(line, layer, dimensions, NativeMethodsTensorRt.jyppx_trt8_constant_layer_set_dimensions, NativeMethodsTensorRt.jyppx_trt10_constant_layer_set_dimensions, NativeMethodsTensorRt.jyppx_trt11_constant_layer_set_dimensions);
    }

    public static string GetEinsumEquation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return ReadUtf8Buffer(line switch
        {
            TensorRtApiLine.TensorRt8 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt8_einsum_layer_get_equation(layer, buffer, size, out required),
            TensorRtApiLine.TensorRt10 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt10_einsum_layer_get_equation(layer, buffer, size, out required),
            TensorRtApiLine.TensorRt11 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_einsum_layer_get_equation(layer, buffer, size, out required),
            _ => throw UnsupportedLine()
        }, "Einsum equation is too large for the managed buffer.");
    }

    public static bool SetEinsumEquation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, string equation)
    {
        using Utf8Interop.Utf8StringScope equationUtf8 = Utf8Interop.ToNativeString(RequireNonEmpty(equation, nameof(equation)));
        int success;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_einsum_layer_set_equation(layer, equationUtf8.Pointer, out success),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_einsum_layer_set_equation(layer, equationUtf8.Pointer, out success),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_einsum_layer_set_equation(layer, equationUtf8.Pointer, out success),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static int GetReverseSequenceBatchAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_reverse_sequence_layer_get_batch_axis, NativeMethodsTensorRt.jyppx_trt10_reverse_sequence_layer_get_batch_axis, NativeMethodsTensorRt.jyppx_trt11_reverse_sequence_layer_get_batch_axis);
    }

    public static void SetReverseSequenceBatchAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int axis)
    {
        SetLayerInt(line, layer, axis, NativeMethodsTensorRt.jyppx_trt8_reverse_sequence_layer_set_batch_axis, NativeMethodsTensorRt.jyppx_trt10_reverse_sequence_layer_set_batch_axis, NativeMethodsTensorRt.jyppx_trt11_reverse_sequence_layer_set_batch_axis);
    }

    public static int GetReverseSequenceSequenceAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_reverse_sequence_layer_get_sequence_axis, NativeMethodsTensorRt.jyppx_trt10_reverse_sequence_layer_get_sequence_axis, NativeMethodsTensorRt.jyppx_trt11_reverse_sequence_layer_get_sequence_axis);
    }

    public static void SetReverseSequenceSequenceAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int axis)
    {
        SetLayerInt(line, layer, axis, NativeMethodsTensorRt.jyppx_trt8_reverse_sequence_layer_set_sequence_axis, NativeMethodsTensorRt.jyppx_trt10_reverse_sequence_layer_set_sequence_axis, NativeMethodsTensorRt.jyppx_trt11_reverse_sequence_layer_set_sequence_axis);
    }

    private static uint GetLayerUIntTrt11(SafeTensorRtObjectHandle layer, LayerUIntGetter getter)
    {
        BridgeStatusCode status = getter(layer, out uint value);
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static int GetLayerIntTensorRt10OrNewer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        LayerIntGetter trt10,
        LayerIntGetter trt11,
        string apiName)
    {
        BridgeStatusCode status;
        int value;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = trt10(layer, out value);
                break;
            case TensorRtApiLine.TensorRt11:
                status = trt11(layer, out value);
                break;
            case TensorRtApiLine.TensorRt8:
                throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{apiName} is available for TensorRT 10 and TensorRT 11 adapters.");
            default:
                throw UnsupportedLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static void SetLayerIntTensorRt10OrNewer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        int value,
        LayerIntSetter trt10,
        LayerIntSetter trt11,
        string apiName)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => trt10(layer, value),
            TensorRtApiLine.TensorRt11 => trt11(layer, value),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{apiName} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
    }

    private static void EnsureTensorRt11DeploymentApi(TensorRtApiLine line, string apiName)
    {
        if (line != TensorRtApiLine.TensorRt11)
        {
            throw new BridgeProbeException(
                BridgeStatusCode.NotSupported,
                BridgeErrorCategory.TensorRt,
                $"{apiName} is currently exposed only for the TensorRT 11 adapter.");
        }
    }

    private static string RequireNonEmpty(string value, string parameterName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException("Value must not be null, empty, or whitespace.", parameterName);
        }

        return value;
    }
}
