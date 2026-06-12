using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static int GetBuilderPluginRegistryCreatorCount(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_plugin_registry_get_creator_count(builder, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_plugin_registry_get_creator_count(builder, out count);
                break;
            default:
                throw UnsupportedPluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static bool HasBuilderPluginRegistryErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int hasRecorder;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_plugin_registry_has_error_recorder(builder, out hasRecorder);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_plugin_registry_has_error_recorder(builder, out hasRecorder);
                break;
            default:
                throw UnsupportedPluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    public static bool IsBuilderPluginRegistryParentSearchEnabled(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int enabled;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_plugin_registry_is_parent_search_enabled(builder, out enabled);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_plugin_registry_is_parent_search_enabled(builder, out enabled);
                break;
            default:
                throw UnsupportedPluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    public static string GetBuilderPluginCreatorName(TensorRtApiLine line, SafeTensorRtObjectHandle builder, int creatorIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_get_name(builder, creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_get_name(builder, creatorIndex, buffer, size, out required),
                _ => throw UnsupportedPluginRegistryInventoryLine()
            },
            "Plugin creator name is too large for the managed buffer.");
    }

    public static string GetBuilderPluginCreatorVersion(TensorRtApiLine line, SafeTensorRtObjectHandle builder, int creatorIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_get_version(builder, creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_get_version(builder, creatorIndex, buffer, size, out required),
                _ => throw UnsupportedPluginRegistryInventoryLine()
            },
            "Plugin creator version is too large for the managed buffer.");
    }

    public static string GetBuilderPluginCreatorNamespace(TensorRtApiLine line, SafeTensorRtObjectHandle builder, int creatorIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_get_namespace(builder, creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_get_namespace(builder, creatorIndex, buffer, size, out required),
                _ => throw UnsupportedPluginRegistryInventoryLine()
            },
            "Plugin creator namespace is too large for the managed buffer.");
    }

    public static string GetBuilderPluginCreatorInterfaceKind(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        int creatorIndex,
        out int interfaceMajor,
        out int interfaceMinor)
    {
        int major = 0;
        int minor = 0;
        string result = ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) =>
            {
                BridgeStatusCode status = line switch
                {
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_get_interface_info(builder, creatorIndex, buffer, size, out required, out major, out minor),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_get_interface_info(builder, creatorIndex, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedPluginRegistryInventoryLine()
                };
                return status;
            },
            "Plugin creator interface kind is too large for the managed buffer.");

        interfaceMajor = major;
        interfaceMinor = minor;
        return result;
    }

    public static int GetBuilderPluginCreatorFieldCount(TensorRtApiLine line, SafeTensorRtObjectHandle builder, int creatorIndex)
    {
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_get_field_count(builder, creatorIndex, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_get_field_count(builder, creatorIndex, out count);
                break;
            default:
                throw UnsupportedPluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static string GetBuilderPluginCreatorFieldName(TensorRtApiLine line, SafeTensorRtObjectHandle builder, int creatorIndex, int fieldIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_get_field_name(builder, creatorIndex, fieldIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_get_field_name(builder, creatorIndex, fieldIndex, buffer, size, out required),
                _ => throw UnsupportedPluginRegistryInventoryLine()
            },
            "Plugin creator field name is too large for the managed buffer.");
    }

    public static void GetBuilderPluginCreatorFieldMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        int creatorIndex,
        int fieldIndex,
        out TensorRtPluginFieldType fieldType,
        out int length,
        out bool hasData)
    {
        int typeValue;
        int lengthValue;
        int hasDataValue;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_get_field_metadata(builder, creatorIndex, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_get_field_metadata(builder, creatorIndex, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            default:
                throw UnsupportedPluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        fieldType = (TensorRtPluginFieldType)typeValue;
        length = lengthValue;
        hasData = hasDataValue != 0;
    }

    private static BridgeProbeException UnsupportedPluginRegistryInventoryLine()
    {
        return new BridgeProbeException(
            BridgeStatusCode.NotSupported,
            BridgeErrorCategory.TensorRt,
            "Plugin registry inventory is exposed by this bridge for TensorRT 10 and 11.");
    }
}
