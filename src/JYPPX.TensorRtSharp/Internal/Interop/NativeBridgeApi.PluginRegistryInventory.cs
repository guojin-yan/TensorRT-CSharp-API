using System;
using System.Collections.Generic;
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
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_builder_plugin_registry_get_creator_count(builder, out count);
                break;
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

    public static bool IsBuilderPluginRegistryAvailable(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int exists;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_builder_plugin_registry_exists(builder, out exists);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_plugin_registry_exists(builder, out exists);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_plugin_registry_exists(builder, out exists);
                break;
            default:
                throw UnsupportedPluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return exists != 0;
    }

    public static int GetBuilderPluginRegistryRecursiveCreatorCount(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                return 0;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_plugin_registry_get_recursive_creator_count(builder, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_plugin_registry_get_recursive_creator_count(builder, out count);
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
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_builder_plugin_registry_has_error_recorder(builder, out hasRecorder);
                break;
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
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_builder_plugin_registry_is_parent_search_enabled(builder, out enabled);
                break;
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
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_plugin_creator_get_name(builder, creatorIndex, buffer, size, out required),
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
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_plugin_creator_get_version(builder, creatorIndex, buffer, size, out required),
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
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_plugin_creator_get_namespace(builder, creatorIndex, buffer, size, out required),
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
        if (line == TensorRtApiLine.TensorRt8)
        {
            interfaceMajor = 8;
            interfaceMinor = 0;
            return "IPluginCreator";
        }

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
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_builder_plugin_creator_get_field_count(builder, creatorIndex, out count);
                break;
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

    public static TensorRtApiLanguage GetBuilderPluginCreatorApiLanguage(TensorRtApiLine line, SafeTensorRtObjectHandle builder, int creatorIndex)
    {
        if (line == TensorRtApiLine.TensorRt8)
        {
            return TensorRtApiLanguage.Unknown;
        }

        int apiLanguage = (int)TensorRtApiLanguage.Unknown;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_get_api_language(builder, creatorIndex, out apiLanguage),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_get_api_language(builder, creatorIndex, out apiLanguage),
            _ => throw UnsupportedPluginRegistryInventoryLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return ToTensorRtApiLanguage(apiLanguage);
    }

    public static string GetBuilderPluginCreatorFieldName(TensorRtApiLine line, SafeTensorRtObjectHandle builder, int creatorIndex, int fieldIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_plugin_creator_get_field_name(builder, creatorIndex, fieldIndex, buffer, size, out required),
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
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_builder_plugin_creator_get_field_metadata(builder, creatorIndex, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
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

    public static bool IsBuilderPluginCreatorRegistered(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        string pluginName,
        string pluginVersion,
        string pluginNamespace)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(pluginName ?? string.Empty);
        using Utf8Interop.Utf8StringScope versionUtf8 = Utf8Interop.ToNativeString(pluginVersion ?? string.Empty);
        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(pluginNamespace ?? string.Empty);

        int found;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_builder_plugin_creator_lookup(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out found);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_lookup(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out found);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_lookup(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out found);
                break;
            default:
                throw UnsupportedPluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return found != 0;
    }

    public static bool TryGetBuilderPluginCreator(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        out TensorRtPluginCreatorInfo? creator)
    {
        if (!IsBuilderPluginCreatorRegistered(line, builder, pluginName, pluginVersion, pluginNamespace))
        {
            creator = null;
            return false;
        }

        string interfaceKind = GetBuilderLookupPluginCreatorInterfaceKind(
            line,
            builder,
            pluginName,
            pluginVersion,
            pluginNamespace,
            out int interfaceMajor,
            out int interfaceMinor);
        TensorRtApiLanguage apiLanguage = GetBuilderLookupPluginCreatorApiLanguage(line, builder, pluginName, pluginVersion, pluginNamespace);
        int fieldCount = GetBuilderLookupPluginCreatorFieldCount(line, builder, pluginName, pluginVersion, pluginNamespace);
        List<TensorRtPluginFieldInfo> fields = new List<TensorRtPluginFieldInfo>(fieldCount);

        for (int fieldIndex = 0; fieldIndex < fieldCount; fieldIndex++)
        {
            string fieldName = GetBuilderLookupPluginCreatorFieldName(line, builder, pluginName, pluginVersion, pluginNamespace, fieldIndex);
            GetBuilderLookupPluginCreatorFieldMetadata(line, builder, pluginName, pluginVersion, pluginNamespace, fieldIndex, out TensorRtPluginFieldType fieldType, out int length, out bool hasData);
            fields.Add(new TensorRtPluginFieldInfo(fieldName, fieldType, length, hasData));
        }

        creator = new TensorRtPluginCreatorInfo(
            index: -1,
            pluginName,
            pluginVersion,
            pluginNamespace,
            interfaceKind,
            interfaceMajor,
            interfaceMinor,
            apiLanguage,
            fields);
        return true;
    }

    private static string GetBuilderLookupPluginCreatorInterfaceKind(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        out int interfaceMajor,
        out int interfaceMinor)
    {
        if (line == TensorRtApiLine.TensorRt8)
        {
            interfaceMajor = 8;
            interfaceMinor = 0;
            return "IPluginCreator";
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(pluginName ?? string.Empty);
        using Utf8Interop.Utf8StringScope versionUtf8 = Utf8Interop.ToNativeString(pluginVersion ?? string.Empty);
        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(pluginNamespace ?? string.Empty);

        int major = 0;
        int minor = 0;
        string result = ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) =>
            {
                BridgeStatusCode status = line switch
                {
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_lookup_get_interface_info(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, buffer, size, out required, out major, out minor),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_lookup_get_interface_info(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedPluginRegistryInventoryLine()
                };
                return status;
            },
            "Builder plugin creator lookup interface kind is too large for the managed buffer.");

        interfaceMajor = major;
        interfaceMinor = minor;
        return result;
    }

    private static int GetBuilderLookupPluginCreatorFieldCount(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        string pluginName,
        string pluginVersion,
        string pluginNamespace)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(pluginName ?? string.Empty);
        using Utf8Interop.Utf8StringScope versionUtf8 = Utf8Interop.ToNativeString(pluginVersion ?? string.Empty);
        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(pluginNamespace ?? string.Empty);

        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_builder_plugin_creator_lookup_get_field_count(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out count);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_lookup_get_field_count(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_lookup_get_field_count(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out count);
                break;
            default:
                throw UnsupportedPluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static TensorRtApiLanguage GetBuilderLookupPluginCreatorApiLanguage(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        string pluginName,
        string pluginVersion,
        string pluginNamespace)
    {
        if (line == TensorRtApiLine.TensorRt8)
        {
            return TensorRtApiLanguage.Unknown;
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(pluginName ?? string.Empty);
        using Utf8Interop.Utf8StringScope versionUtf8 = Utf8Interop.ToNativeString(pluginVersion ?? string.Empty);
        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(pluginNamespace ?? string.Empty);

        int apiLanguage = (int)TensorRtApiLanguage.Unknown;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_lookup_get_api_language(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out apiLanguage),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_lookup_get_api_language(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out apiLanguage),
            _ => throw UnsupportedPluginRegistryInventoryLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return ToTensorRtApiLanguage(apiLanguage);
    }

    private static string GetBuilderLookupPluginCreatorFieldName(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        int fieldIndex)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(pluginName ?? string.Empty);
        using Utf8Interop.Utf8StringScope versionUtf8 = Utf8Interop.ToNativeString(pluginVersion ?? string.Empty);
        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(pluginNamespace ?? string.Empty);

        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_plugin_creator_lookup_get_field_name(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_lookup_get_field_name(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_lookup_get_field_name(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, buffer, size, out required),
                _ => throw UnsupportedPluginRegistryInventoryLine()
            },
            "Builder plugin creator lookup field name is too large for the managed buffer.");
    }

    private static void GetBuilderLookupPluginCreatorFieldMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        int fieldIndex,
        out TensorRtPluginFieldType fieldType,
        out int length,
        out bool hasData)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(pluginName ?? string.Empty);
        using Utf8Interop.Utf8StringScope versionUtf8 = Utf8Interop.ToNativeString(pluginVersion ?? string.Empty);
        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(pluginNamespace ?? string.Empty);

        int typeValue;
        int lengthValue;
        int hasDataValue;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_builder_plugin_creator_lookup_get_field_metadata(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_plugin_creator_lookup_get_field_metadata(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_plugin_creator_lookup_get_field_metadata(builder, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
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
            "Plugin registry inventory is exposed by this bridge for TensorRT 8, 10, and 11.");
    }
}
