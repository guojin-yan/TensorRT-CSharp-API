using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool BuilderCapabilityPluginRegistryExists(TensorRtApiLine line, TensorRtEngineCapability capability)
    {
        int exists;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_registry_exists((int)capability, out exists);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_registry_exists((int)capability, out exists);
                break;
            default:
                throw UnsupportedBuilderCapabilityPluginRegistryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return exists != 0;
    }

    public static bool BuilderSafePluginRegistryExists(TensorRtApiLine line, TensorRtEngineCapability capability)
    {
        int exists;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_safe_plugin_registry_exists((int)capability, out exists);
                break;
            default:
                throw UnsupportedBuilderSafePluginRegistryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return exists != 0;
    }

    public static TensorRtPluginRegistryInventory GetBuilderCapabilityPluginRegistryInventory(TensorRtApiLine line, TensorRtEngineCapability capability)
    {
        int creatorCount = GetBuilderCapabilityPluginRegistryCreatorCount(line, capability);
        int recursiveCreatorCount = GetBuilderCapabilityPluginRegistryRecursiveCreatorCount(line, capability);
        bool hasErrorRecorder = HasBuilderCapabilityPluginRegistryErrorRecorder(line, capability);
        bool parentSearchEnabled = IsBuilderCapabilityPluginRegistryParentSearchEnabled(line, capability);
        List<TensorRtPluginCreatorInfo> creators = new List<TensorRtPluginCreatorInfo>(creatorCount);

        for (int creatorIndex = 0; creatorIndex < creatorCount; creatorIndex++)
        {
            string name = GetBuilderCapabilityPluginCreatorName(line, capability, creatorIndex);
            string version = GetBuilderCapabilityPluginCreatorVersion(line, capability, creatorIndex);
            string pluginNamespace = GetBuilderCapabilityPluginCreatorNamespace(line, capability, creatorIndex);
            string interfaceKind = GetBuilderCapabilityPluginCreatorInterfaceKind(line, capability, creatorIndex, out int interfaceMajor, out int interfaceMinor);
            TensorRtApiLanguage apiLanguage = GetBuilderCapabilityPluginCreatorApiLanguage(line, capability, creatorIndex);
            int fieldCount = GetBuilderCapabilityPluginCreatorFieldCount(line, capability, creatorIndex);
            List<TensorRtPluginFieldInfo> fields = new List<TensorRtPluginFieldInfo>(fieldCount);

            for (int fieldIndex = 0; fieldIndex < fieldCount; fieldIndex++)
            {
                string fieldName = GetBuilderCapabilityPluginCreatorFieldName(line, capability, creatorIndex, fieldIndex);
                GetBuilderCapabilityPluginCreatorFieldMetadata(line, capability, creatorIndex, fieldIndex, out TensorRtPluginFieldType fieldType, out int length, out bool hasData);
                fields.Add(new TensorRtPluginFieldInfo(fieldName, fieldType, length, hasData));
            }

            creators.Add(new TensorRtPluginCreatorInfo(
                creatorIndex,
                name,
                version,
                pluginNamespace,
                interfaceKind,
                interfaceMajor,
                interfaceMinor,
                apiLanguage,
                fields));
        }

        return new TensorRtPluginRegistryInventory(line, TensorRtPluginRegistrySource.BuilderCapability, hasErrorRecorder, parentSearchEnabled, recursiveCreatorCount, creators);
    }

    public static bool IsBuilderCapabilityPluginCreatorRegistered(TensorRtApiLine line, TensorRtEngineCapability capability, string pluginName, string pluginVersion, string pluginNamespace)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(pluginName ?? string.Empty);
        using Utf8Interop.Utf8StringScope versionUtf8 = Utf8Interop.ToNativeString(pluginVersion ?? string.Empty);
        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(pluginNamespace ?? string.Empty);

        int found;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_lookup((int)capability, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out found);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_lookup((int)capability, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out found);
                break;
            default:
                throw UnsupportedBuilderCapabilityPluginRegistryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return found != 0;
    }

    public static bool TryGetBuilderCapabilityPluginCreator(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        out TensorRtPluginCreatorInfo? creator)
    {
        if (!IsBuilderCapabilityPluginCreatorRegistered(line, capability, pluginName, pluginVersion, pluginNamespace))
        {
            creator = null;
            return false;
        }

        string interfaceKind = GetBuilderCapabilityLookupPluginCreatorInterfaceKind(
            line,
            capability,
            pluginName,
            pluginVersion,
            pluginNamespace,
            out int interfaceMajor,
            out int interfaceMinor);
        TensorRtApiLanguage apiLanguage = GetBuilderCapabilityLookupPluginCreatorApiLanguage(line, capability, pluginName, pluginVersion, pluginNamespace);
        int fieldCount = GetBuilderCapabilityLookupPluginCreatorFieldCount(line, capability, pluginName, pluginVersion, pluginNamespace);
        List<TensorRtPluginFieldInfo> fields = new List<TensorRtPluginFieldInfo>(fieldCount);

        for (int fieldIndex = 0; fieldIndex < fieldCount; fieldIndex++)
        {
            string fieldName = GetBuilderCapabilityLookupPluginCreatorFieldName(line, capability, pluginName, pluginVersion, pluginNamespace, fieldIndex);
            GetBuilderCapabilityLookupPluginCreatorFieldMetadata(line, capability, pluginName, pluginVersion, pluginNamespace, fieldIndex, out TensorRtPluginFieldType fieldType, out int length, out bool hasData);
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

    private static int GetBuilderCapabilityPluginRegistryCreatorCount(TensorRtApiLine line, TensorRtEngineCapability capability)
    {
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_registry_get_creator_count((int)capability, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_registry_get_creator_count((int)capability, out count);
                break;
            default:
                throw UnsupportedBuilderCapabilityPluginRegistryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static int GetBuilderCapabilityPluginRegistryRecursiveCreatorCount(TensorRtApiLine line, TensorRtEngineCapability capability)
    {
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_registry_get_recursive_creator_count((int)capability, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_registry_get_recursive_creator_count((int)capability, out count);
                break;
            default:
                throw UnsupportedBuilderCapabilityPluginRegistryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static bool HasBuilderCapabilityPluginRegistryErrorRecorder(TensorRtApiLine line, TensorRtEngineCapability capability)
    {
        int hasRecorder;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_registry_has_error_recorder((int)capability, out hasRecorder);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_registry_has_error_recorder((int)capability, out hasRecorder);
                break;
            default:
                throw UnsupportedBuilderCapabilityPluginRegistryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    private static bool IsBuilderCapabilityPluginRegistryParentSearchEnabled(TensorRtApiLine line, TensorRtEngineCapability capability)
    {
        int enabled;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_registry_is_parent_search_enabled((int)capability, out enabled);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_registry_is_parent_search_enabled((int)capability, out enabled);
                break;
            default:
                throw UnsupportedBuilderCapabilityPluginRegistryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    private static string GetBuilderCapabilityPluginCreatorName(TensorRtApiLine line, TensorRtEngineCapability capability, int creatorIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_get_name((int)capability, creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_get_name((int)capability, creatorIndex, buffer, size, out required),
                _ => throw UnsupportedBuilderCapabilityPluginRegistryLine()
            },
            "Builder capability plugin creator name is too large for the managed buffer.");
    }

    private static string GetBuilderCapabilityPluginCreatorVersion(TensorRtApiLine line, TensorRtEngineCapability capability, int creatorIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_get_version((int)capability, creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_get_version((int)capability, creatorIndex, buffer, size, out required),
                _ => throw UnsupportedBuilderCapabilityPluginRegistryLine()
            },
            "Builder capability plugin creator version is too large for the managed buffer.");
    }

    private static string GetBuilderCapabilityPluginCreatorNamespace(TensorRtApiLine line, TensorRtEngineCapability capability, int creatorIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_get_namespace((int)capability, creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_get_namespace((int)capability, creatorIndex, buffer, size, out required),
                _ => throw UnsupportedBuilderCapabilityPluginRegistryLine()
            },
            "Builder capability plugin creator namespace is too large for the managed buffer.");
    }

    private static string GetBuilderCapabilityPluginCreatorInterfaceKind(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
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
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_get_interface_info((int)capability, creatorIndex, buffer, size, out required, out major, out minor),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_get_interface_info((int)capability, creatorIndex, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedBuilderCapabilityPluginRegistryLine()
                };
                return status;
            },
            "Builder capability plugin creator interface kind is too large for the managed buffer.");

        interfaceMajor = major;
        interfaceMinor = minor;
        return result;
    }

    private static int GetBuilderCapabilityPluginCreatorFieldCount(TensorRtApiLine line, TensorRtEngineCapability capability, int creatorIndex)
    {
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_get_field_count((int)capability, creatorIndex, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_get_field_count((int)capability, creatorIndex, out count);
                break;
            default:
                throw UnsupportedBuilderCapabilityPluginRegistryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static TensorRtApiLanguage GetBuilderCapabilityPluginCreatorApiLanguage(TensorRtApiLine line, TensorRtEngineCapability capability, int creatorIndex)
    {
        int apiLanguage = (int)TensorRtApiLanguage.Unknown;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_get_api_language((int)capability, creatorIndex, out apiLanguage),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_get_api_language((int)capability, creatorIndex, out apiLanguage),
            _ => throw UnsupportedBuilderCapabilityPluginRegistryLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return ToTensorRtApiLanguage(apiLanguage);
    }

    private static string GetBuilderCapabilityPluginCreatorFieldName(TensorRtApiLine line, TensorRtEngineCapability capability, int creatorIndex, int fieldIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_get_field_name((int)capability, creatorIndex, fieldIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_get_field_name((int)capability, creatorIndex, fieldIndex, buffer, size, out required),
                _ => throw UnsupportedBuilderCapabilityPluginRegistryLine()
            },
            "Builder capability plugin creator field name is too large for the managed buffer.");
    }

    private static void GetBuilderCapabilityPluginCreatorFieldMetadata(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
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
                status = NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_get_field_metadata((int)capability, creatorIndex, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_get_field_metadata((int)capability, creatorIndex, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            default:
                throw UnsupportedBuilderCapabilityPluginRegistryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        fieldType = (TensorRtPluginFieldType)typeValue;
        length = lengthValue;
        hasData = hasDataValue != 0;
    }

    private static string GetBuilderCapabilityLookupPluginCreatorInterfaceKind(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        out int interfaceMajor,
        out int interfaceMinor)
    {
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
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_lookup_get_interface_info((int)capability, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, buffer, size, out required, out major, out minor),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_lookup_get_interface_info((int)capability, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedBuilderCapabilityPluginRegistryLine()
                };
                return status;
            },
            "Builder capability lookup plugin creator interface kind is too large for the managed buffer.");

        interfaceMajor = major;
        interfaceMinor = minor;
        return result;
    }

    private static int GetBuilderCapabilityLookupPluginCreatorFieldCount(TensorRtApiLine line, TensorRtEngineCapability capability, string pluginName, string pluginVersion, string pluginNamespace)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(pluginName ?? string.Empty);
        using Utf8Interop.Utf8StringScope versionUtf8 = Utf8Interop.ToNativeString(pluginVersion ?? string.Empty);
        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(pluginNamespace ?? string.Empty);

        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_lookup_get_field_count((int)capability, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_lookup_get_field_count((int)capability, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out count);
                break;
            default:
                throw UnsupportedBuilderCapabilityPluginRegistryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static TensorRtApiLanguage GetBuilderCapabilityLookupPluginCreatorApiLanguage(TensorRtApiLine line, TensorRtEngineCapability capability, string pluginName, string pluginVersion, string pluginNamespace)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(pluginName ?? string.Empty);
        using Utf8Interop.Utf8StringScope versionUtf8 = Utf8Interop.ToNativeString(pluginVersion ?? string.Empty);
        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(pluginNamespace ?? string.Empty);

        int apiLanguage = (int)TensorRtApiLanguage.Unknown;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_lookup_get_api_language((int)capability, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out apiLanguage),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_lookup_get_api_language((int)capability, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out apiLanguage),
            _ => throw UnsupportedBuilderCapabilityPluginRegistryLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return ToTensorRtApiLanguage(apiLanguage);
    }

    private static string GetBuilderCapabilityLookupPluginCreatorFieldName(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
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
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_lookup_get_field_name((int)capability, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_lookup_get_field_name((int)capability, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, buffer, size, out required),
                _ => throw UnsupportedBuilderCapabilityPluginRegistryLine()
            },
            "Builder capability lookup plugin creator field name is too large for the managed buffer.");
    }

    private static void GetBuilderCapabilityLookupPluginCreatorFieldMetadata(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
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
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_capability_plugin_creator_lookup_get_field_metadata((int)capability, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_capability_plugin_creator_lookup_get_field_metadata((int)capability, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            default:
                throw UnsupportedBuilderCapabilityPluginRegistryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        fieldType = (TensorRtPluginFieldType)typeValue;
        length = lengthValue;
        hasData = hasDataValue != 0;
    }

    private static BridgeProbeException UnsupportedBuilderCapabilityPluginRegistryLine()
    {
        return new BridgeProbeException(
            BridgeStatusCode.NotSupported,
            BridgeErrorCategory.TensorRt,
            "Builder capability plugin registry probes are exposed by this bridge for TensorRT 10 and 11.");
    }

    private static BridgeProbeException UnsupportedBuilderSafePluginRegistryLine()
    {
        return new BridgeProbeException(
            BridgeStatusCode.NotSupported,
            BridgeErrorCategory.TensorRt,
            "Builder safe plugin registry probes are exposed by this bridge for TensorRT 11.");
    }
}
