using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    private const int MaxPluginIoElementCount = 1_000_000;

    public static TensorRtPluginFormatSupportSnapshot GetPluginV2CurrentFormatSupport(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        TensorRtPluginFormatCapabilityKind capability,
        int inputCount,
        int outputCount)
    {
        ValidatePluginIoCounts(inputCount, outputCount, "PluginV2 current format support");
        int requiredCount = checked(inputCount + outputCount);
        byte[] values = new byte[requiredCount];
        BridgeStatusCode status;
        switch (capability)
        {
            case TensorRtPluginFormatCapabilityKind.PluginV2DynamicExt:
                switch (line)
                {
                    case TensorRtApiLine.TensorRt8:
                        status = NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_copy_dynamic_format_support(layer, values, (UIntPtr)values.Length, out UIntPtr _);
                        break;
                    case TensorRtApiLine.TensorRt10:
                        status = NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_copy_dynamic_format_support(layer, values, (UIntPtr)values.Length, out UIntPtr _);
                        break;
                    case TensorRtApiLine.TensorRt11:
                        status = NativeMethodsTensorRt.jyppx_trt11_plugin_v2_layer_copy_dynamic_format_support(layer, values, (UIntPtr)values.Length, out UIntPtr _);
                        break;
                    default:
                        throw UnsupportedPluginFormatCapability(capability);
                }
                break;
            case TensorRtPluginFormatCapabilityKind.PluginV2IoExt:
                switch (line)
                {
                    case TensorRtApiLine.TensorRt8:
                        status = NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_copy_io_ext_format_support(layer, values, (UIntPtr)values.Length, out UIntPtr _);
                        break;
                    case TensorRtApiLine.TensorRt10:
                        status = NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_copy_io_ext_format_support(layer, values, (UIntPtr)values.Length, out UIntPtr _);
                        break;
                    case TensorRtApiLine.TensorRt11:
                        status = NativeMethodsTensorRt.jyppx_trt11_plugin_v2_layer_copy_io_ext_format_support(layer, values, (UIntPtr)values.Length, out UIntPtr _);
                        break;
                    default:
                        throw UnsupportedPluginFormatCapability(capability);
                }
                break;
            default:
                throw UnsupportedPluginFormatCapability(capability);
        }
        NativeStatus.ThrowIfFailed(status);
        return new TensorRtPluginFormatSupportSnapshot(line, capability, inputCount, outputCount, ToBooleanList(values));
    }

    public static TensorRtPluginV3BuildIoSnapshot GetPluginV3BuildIoSnapshot(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer)
    {
        EnsurePluginV3QueryLine(line);
        BridgeStatusCode status = line == TensorRtApiLine.TensorRt10
            ? NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_build_io_counts(layer, out int inputCount, out int outputCount)
            : NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_build_io_counts(layer, out inputCount, out outputCount);
        NativeStatus.ThrowIfFailed(status);
        ValidatePluginIoCounts(inputCount, outputCount, "PluginV3 build IO");

        int[] outputTypes = new int[outputCount];
        status = line == TensorRtApiLine.TensorRt10
            ? NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_copy_build_output_data_types(
                layer, outputTypes, (UIntPtr)outputTypes.Length, out UIntPtr outputTypeCount)
            : NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_copy_build_output_data_types(
                layer, outputTypes, (UIntPtr)outputTypes.Length, out outputTypeCount);
        NativeStatus.ThrowIfFailed(status);
        ValidateCopiedCount(outputTypeCount, outputTypes.Length, "PluginV3 output data types");

        int[] aliasedInputs = new int[outputCount];
        bool aliasMetadataAvailable;
        status = line == TensorRtApiLine.TensorRt10
            ? NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_copy_build_aliased_inputs(
                layer, aliasedInputs, (UIntPtr)aliasedInputs.Length, out UIntPtr aliasedInputCount)
            : NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_copy_build_aliased_inputs(
                layer, aliasedInputs, (UIntPtr)aliasedInputs.Length, out aliasedInputCount);
        if (status == BridgeStatusCode.NotFound)
        {
            for (int index = 0; index < aliasedInputs.Length; ++index)
            {
                aliasedInputs[index] = -1;
            }
            aliasMetadataAvailable = false;
        }
        else
        {
            NativeStatus.ThrowIfFailed(status);
            ValidateCopiedCount(aliasedInputCount, aliasedInputs.Length, "PluginV3 aliased inputs");
            aliasMetadataAvailable = true;
        }

        int formatCount = checked(inputCount + outputCount);
        byte[] formatSupport = new byte[formatCount];
        status = line == TensorRtApiLine.TensorRt10
            ? NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_copy_build_current_format_support(
                layer, formatSupport, (UIntPtr)formatSupport.Length, out UIntPtr requiredFormatCount)
            : NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_copy_build_current_format_support(
                layer, formatSupport, (UIntPtr)formatSupport.Length, out requiredFormatCount);
        NativeStatus.ThrowIfFailed(status);
        ValidateCopiedCount(requiredFormatCount, formatSupport.Length, "PluginV3 current format support");

        TensorRtDataType[] managedOutputTypes = new TensorRtDataType[outputTypes.Length];
        for (int index = 0; index < outputTypes.Length; ++index)
        {
            managedOutputTypes[index] = (TensorRtDataType)outputTypes[index];
        }

        TensorRtPluginFormatSupportSnapshot supportSnapshot = new TensorRtPluginFormatSupportSnapshot(
            line,
            TensorRtPluginFormatCapabilityKind.PluginV3OneBuild,
            inputCount,
            outputCount,
            ToBooleanList(formatSupport));
        return new TensorRtPluginV3BuildIoSnapshot(
            line,
            inputCount,
            outputCount,
            managedOutputTypes,
            aliasedInputs,
            aliasMetadataAvailable,
            supportSnapshot);
    }

    public static TensorRtPluginV3SerializationFieldInventory GetPluginV3RuntimeSerializationFields(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer)
    {
        EnsurePluginV3QueryLine(line);
        BridgeStatusCode status = line == TensorRtApiLine.TensorRt10
            ? NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_runtime_serialization_field_count(layer, out int fieldCount)
            : NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_runtime_serialization_field_count(layer, out fieldCount);
        NativeStatus.ThrowIfFailed(status);
        if (fieldCount < 0 || fieldCount > 1_000_000)
        {
            throw new BridgeProbeException(
                BridgeStatusCode.InvalidState,
                BridgeErrorCategory.TensorRt,
                "PluginV3 serialization field count is outside the managed safety bound.");
        }

        List<TensorRtPluginFieldInfo> fields = new List<TensorRtPluginFieldInfo>(fieldCount);
        for (int fieldIndex = 0; fieldIndex < fieldCount; ++fieldIndex)
        {
            int capturedIndex = fieldIndex;
            string name = ReadUtf8Buffer(
                (byte[] buffer, UIntPtr size, out UIntPtr required) => line == TensorRtApiLine.TensorRt10
                    ? NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_runtime_serialization_field_name(
                        layer, capturedIndex, buffer, size, out required)
                    : NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_runtime_serialization_field_name(
                        layer, capturedIndex, buffer, size, out required),
                "PluginV3 serialization field name is too large for the managed buffer.");

            status = line == TensorRtApiLine.TensorRt10
                ? NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_runtime_serialization_field_metadata(
                    layer, fieldIndex, out int fieldType, out int length, out int hasData)
                : NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_runtime_serialization_field_metadata(
                    layer, fieldIndex, out fieldType, out length, out hasData);
            NativeStatus.ThrowIfFailed(status);
            fields.Add(new TensorRtPluginFieldInfo(name, (TensorRtPluginFieldType)fieldType, length, hasData != 0));
        }

        return new TensorRtPluginV3SerializationFieldInventory(line, fields);
    }

    private static IReadOnlyList<bool> ToBooleanList(byte[] values)
    {
        bool[] result = new bool[values.Length];
        for (int index = 0; index < values.Length; ++index)
        {
            result[index] = values[index] != 0;
        }
        return result;
    }

    private static void ValidateCopiedCount(UIntPtr requiredCount, int actualCount, string feature)
    {
        if (requiredCount.ToUInt64() != (ulong)actualCount)
        {
            throw new BridgeProbeException(
                BridgeStatusCode.InvalidState,
                BridgeErrorCategory.TensorRt,
                $"{feature} count changed during the owner-scoped copy.");
        }
    }

    private static void ValidatePluginIoCounts(int inputCount, int outputCount, string feature)
    {
        long totalCount = (long)inputCount + outputCount;
        if (inputCount < 0 || outputCount <= 0 || totalCount > MaxPluginIoElementCount)
        {
            throw new BridgeProbeException(
                BridgeStatusCode.InvalidState,
                BridgeErrorCategory.TensorRt,
                $"{feature} counts are outside the managed safety bound.");
        }
    }

    private static void EnsurePluginV3QueryLine(TensorRtApiLine line)
    {
        if (line != TensorRtApiLine.TensorRt10 && line != TensorRtApiLine.TensorRt11)
        {
            throw UnsupportedPluginFormatCapability(TensorRtPluginFormatCapabilityKind.PluginV3OneBuild);
        }
    }

    private static BridgeProbeException UnsupportedPluginFormatCapability(TensorRtPluginFormatCapabilityKind capability)
    {
        return new BridgeProbeException(
            BridgeStatusCode.NotSupported,
            BridgeErrorCategory.TensorRt,
            $"{capability} owner-scoped query is not supported by this TensorRT adapter.");
    }
}
