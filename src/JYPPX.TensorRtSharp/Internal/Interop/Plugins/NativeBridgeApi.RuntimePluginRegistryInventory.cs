using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool RuntimePluginRegistryExists(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        int exists;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_registry_exists(runtime, out exists);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_registry_exists(runtime, out exists);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_registry_exists(runtime, out exists);
                break;
            default:
                throw UnsupportedRuntimePluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return exists != 0;
    }

    public static TensorRtPluginRegistryInventory GetRuntimePluginRegistryInventory(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        int creatorCount = GetRuntimePluginRegistryCreatorCount(line, runtime);
        int? recursiveCreatorCount = line == TensorRtApiLine.TensorRt8
            ? null
            : GetRuntimePluginRegistryRecursiveCreatorCount(line, runtime);
        bool hasErrorRecorder = HasRuntimePluginRegistryErrorRecorder(line, runtime);
        bool parentSearchEnabled = IsRuntimePluginRegistryParentSearchEnabled(line, runtime);
        List<TensorRtPluginCreatorInfo> creators = new List<TensorRtPluginCreatorInfo>(creatorCount);

        for (int creatorIndex = 0; creatorIndex < creatorCount; creatorIndex++)
        {
            string name = GetRuntimePluginCreatorName(line, runtime, creatorIndex);
            string version = GetRuntimePluginCreatorVersion(line, runtime, creatorIndex);
            string pluginNamespace = GetRuntimePluginCreatorNamespace(line, runtime, creatorIndex);
            string interfaceKind = GetRuntimePluginCreatorInterfaceKind(line, runtime, creatorIndex, out int interfaceMajor, out int interfaceMinor);
            TensorRtApiLanguage apiLanguage = GetRuntimePluginCreatorApiLanguage(line, runtime, creatorIndex);
            int? tensorRtVersion = line == TensorRtApiLine.TensorRt8
                ? GetRuntimePluginCreatorTensorRtVersion(line, runtime, creatorIndex)
                : null;
            int fieldCount = GetRuntimePluginCreatorFieldCount(line, runtime, creatorIndex);
            List<TensorRtPluginFieldInfo> fields = new List<TensorRtPluginFieldInfo>(fieldCount);

            for (int fieldIndex = 0; fieldIndex < fieldCount; fieldIndex++)
            {
                string fieldName = GetRuntimePluginCreatorFieldName(line, runtime, creatorIndex, fieldIndex);
                GetRuntimePluginCreatorFieldMetadata(line, runtime, creatorIndex, fieldIndex, out TensorRtPluginFieldType fieldType, out int length, out bool hasData);
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
                fields,
                tensorRtVersion));
        }

        return new TensorRtPluginRegistryInventory(line, TensorRtPluginRegistrySource.Runtime, hasErrorRecorder, parentSearchEnabled, recursiveCreatorCount: recursiveCreatorCount, creators);
    }

    public static bool IsRuntimePluginCreatorRegistered(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
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
                status = NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_creator_lookup(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out found);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_lookup(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out found);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_lookup(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out found);
                break;
            default:
                throw UnsupportedRuntimePluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return found != 0;
    }

    public static bool TryGetRuntimePluginCreator(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        out TensorRtPluginCreatorInfo? creator)
    {
        string normalizedName = pluginName ?? string.Empty;
        string normalizedVersion = pluginVersion ?? string.Empty;
        string normalizedNamespace = pluginNamespace ?? string.Empty;

        if (!IsRuntimePluginCreatorRegistered(line, runtime, normalizedName, normalizedVersion, normalizedNamespace))
        {
            creator = null;
            return false;
        }

        string interfaceKind = GetRuntimeLookupPluginCreatorInterfaceKind(
            line,
            runtime,
            normalizedName,
            normalizedVersion,
            normalizedNamespace,
            out int interfaceMajor,
            out int interfaceMinor);
        TensorRtApiLanguage apiLanguage = GetRuntimeLookupPluginCreatorApiLanguage(line, runtime, normalizedName, normalizedVersion, normalizedNamespace);
        int? tensorRtVersion = line == TensorRtApiLine.TensorRt8
            ? GetRuntimeLookupPluginCreatorTensorRtVersion(line, runtime, normalizedName, normalizedVersion, normalizedNamespace)
            : null;
        int fieldCount = GetRuntimeLookupPluginCreatorFieldCount(line, runtime, normalizedName, normalizedVersion, normalizedNamespace);
        List<TensorRtPluginFieldInfo> fields = new List<TensorRtPluginFieldInfo>(fieldCount);

        for (int fieldIndex = 0; fieldIndex < fieldCount; fieldIndex++)
        {
            string fieldName = GetRuntimeLookupPluginCreatorFieldName(line, runtime, normalizedName, normalizedVersion, normalizedNamespace, fieldIndex);
            GetRuntimeLookupPluginCreatorFieldMetadata(line, runtime, normalizedName, normalizedVersion, normalizedNamespace, fieldIndex, out TensorRtPluginFieldType fieldType, out int length, out bool hasData);
            fields.Add(new TensorRtPluginFieldInfo(fieldName, fieldType, length, hasData));
        }

        creator = new TensorRtPluginCreatorInfo(
            index: -1,
            normalizedName,
            normalizedVersion,
            normalizedNamespace,
            interfaceKind,
            interfaceMajor,
            interfaceMinor,
            apiLanguage,
            fields,
            tensorRtVersion);
        return true;
    }

    private static int GetRuntimePluginCreatorTensorRtVersion(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
        int creatorIndex)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw UnsupportedRuntimePluginRegistryInventoryLine();
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_creator_get_tensor_rt_version(
            runtime,
            creatorIndex,
            out int tensorRtVersion);
        NativeStatus.ThrowIfFailed(status);
        return tensorRtVersion;
    }

    private static int GetRuntimeLookupPluginCreatorTensorRtVersion(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
        string pluginName,
        string pluginVersion,
        string pluginNamespace)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw UnsupportedRuntimePluginRegistryInventoryLine();
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(pluginName ?? string.Empty);
        using Utf8Interop.Utf8StringScope versionUtf8 = Utf8Interop.ToNativeString(pluginVersion ?? string.Empty);
        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(pluginNamespace ?? string.Empty);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_creator_lookup_get_tensor_rt_version(
            runtime,
            nameUtf8.Pointer,
            versionUtf8.Pointer,
            namespaceUtf8.Pointer,
            out int tensorRtVersion);
        NativeStatus.ThrowIfFailed(status);
        return tensorRtVersion;
    }

    private static int GetRuntimePluginRegistryCreatorCount(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_registry_get_creator_count(runtime, out count);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_registry_get_creator_count(runtime, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_registry_get_creator_count(runtime, out count);
                break;
            default:
                throw UnsupportedRuntimePluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static int GetRuntimePluginRegistryRecursiveCreatorCount(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_registry_get_recursive_creator_count(runtime, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_registry_get_recursive_creator_count(runtime, out count);
                break;
            default:
                throw UnsupportedRuntimePluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static bool HasRuntimePluginRegistryErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        int hasRecorder;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_registry_has_error_recorder(runtime, out hasRecorder);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_registry_has_error_recorder(runtime, out hasRecorder);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_registry_has_error_recorder(runtime, out hasRecorder);
                break;
            default:
                throw UnsupportedRuntimePluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    private static bool IsRuntimePluginRegistryParentSearchEnabled(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        int enabled;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_registry_is_parent_search_enabled(runtime, out enabled);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_registry_is_parent_search_enabled(runtime, out enabled);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_registry_is_parent_search_enabled(runtime, out enabled);
                break;
            default:
                throw UnsupportedRuntimePluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    private static string GetRuntimePluginCreatorName(TensorRtApiLine line, SafeTensorRtObjectHandle runtime, int creatorIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_creator_get_name(runtime, creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_get_name(runtime, creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_get_name(runtime, creatorIndex, buffer, size, out required),
                _ => throw UnsupportedRuntimePluginRegistryInventoryLine()
            },
            "Runtime plugin creator name is too large for the managed buffer.");
    }

    private static string GetRuntimePluginCreatorVersion(TensorRtApiLine line, SafeTensorRtObjectHandle runtime, int creatorIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_creator_get_version(runtime, creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_get_version(runtime, creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_get_version(runtime, creatorIndex, buffer, size, out required),
                _ => throw UnsupportedRuntimePluginRegistryInventoryLine()
            },
            "Runtime plugin creator version is too large for the managed buffer.");
    }

    private static string GetRuntimePluginCreatorNamespace(TensorRtApiLine line, SafeTensorRtObjectHandle runtime, int creatorIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_creator_get_namespace(runtime, creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_get_namespace(runtime, creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_get_namespace(runtime, creatorIndex, buffer, size, out required),
                _ => throw UnsupportedRuntimePluginRegistryInventoryLine()
            },
            "Runtime plugin creator namespace is too large for the managed buffer.");
    }

    private static string GetRuntimePluginCreatorInterfaceKind(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
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
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_get_interface_info(runtime, creatorIndex, buffer, size, out required, out major, out minor),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_get_interface_info(runtime, creatorIndex, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedRuntimePluginRegistryInventoryLine()
                };
                return status;
            },
            "Runtime plugin creator interface kind is too large for the managed buffer.");

        interfaceMajor = major;
        interfaceMinor = minor;
        return result;
    }

    private static int GetRuntimePluginCreatorFieldCount(TensorRtApiLine line, SafeTensorRtObjectHandle runtime, int creatorIndex)
    {
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_creator_get_field_count(runtime, creatorIndex, out count);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_get_field_count(runtime, creatorIndex, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_get_field_count(runtime, creatorIndex, out count);
                break;
            default:
                throw UnsupportedRuntimePluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static TensorRtApiLanguage GetRuntimePluginCreatorApiLanguage(TensorRtApiLine line, SafeTensorRtObjectHandle runtime, int creatorIndex)
    {
        if (line == TensorRtApiLine.TensorRt8)
        {
            return TensorRtApiLanguage.Unknown;
        }

        int apiLanguage = (int)TensorRtApiLanguage.Unknown;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_get_api_language(runtime, creatorIndex, out apiLanguage),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_get_api_language(runtime, creatorIndex, out apiLanguage),
            _ => throw UnsupportedRuntimePluginRegistryInventoryLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return ToTensorRtApiLanguage(apiLanguage);
    }

    private static string GetRuntimePluginCreatorFieldName(TensorRtApiLine line, SafeTensorRtObjectHandle runtime, int creatorIndex, int fieldIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_creator_get_field_name(runtime, creatorIndex, fieldIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_get_field_name(runtime, creatorIndex, fieldIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_get_field_name(runtime, creatorIndex, fieldIndex, buffer, size, out required),
                _ => throw UnsupportedRuntimePluginRegistryInventoryLine()
            },
            "Runtime plugin creator field name is too large for the managed buffer.");
    }

    private static void GetRuntimePluginCreatorFieldMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
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
                status = NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_creator_get_field_metadata(runtime, creatorIndex, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_get_field_metadata(runtime, creatorIndex, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_get_field_metadata(runtime, creatorIndex, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            default:
                throw UnsupportedRuntimePluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        fieldType = (TensorRtPluginFieldType)typeValue;
        length = lengthValue;
        hasData = hasDataValue != 0;
    }

    private static TensorRtApiLanguage GetRuntimeLookupPluginCreatorApiLanguage(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
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
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_lookup_get_api_language(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out apiLanguage),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_lookup_get_api_language(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out apiLanguage),
            _ => throw UnsupportedRuntimePluginRegistryInventoryLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return ToTensorRtApiLanguage(apiLanguage);
    }

    private static string GetRuntimeLookupPluginCreatorInterfaceKind(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
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
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_lookup_get_interface_info(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, buffer, size, out required, out major, out minor),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_lookup_get_interface_info(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedRuntimePluginRegistryInventoryLine()
                };
                return status;
            },
            "Runtime plugin creator lookup interface kind is too large for the managed buffer.");

        interfaceMajor = major;
        interfaceMinor = minor;
        return result;
    }

    private static int GetRuntimeLookupPluginCreatorFieldCount(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
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
                status = NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_creator_lookup_get_field_count(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out count);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_lookup_get_field_count(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_lookup_get_field_count(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out count);
                break;
            default:
                throw UnsupportedRuntimePluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static string GetRuntimeLookupPluginCreatorFieldName(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
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
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_creator_lookup_get_field_name(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_lookup_get_field_name(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_lookup_get_field_name(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, buffer, size, out required),
                _ => throw UnsupportedRuntimePluginRegistryInventoryLine()
            },
            "Runtime plugin creator lookup field name is too large for the managed buffer.");
    }

    private static void GetRuntimeLookupPluginCreatorFieldMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
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
                status = NativeMethodsTensorRt.jyppx_trt8_runtime_plugin_creator_lookup_get_field_metadata(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_runtime_plugin_creator_lookup_get_field_metadata(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_runtime_plugin_creator_lookup_get_field_metadata(runtime, nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            default:
                throw UnsupportedRuntimePluginRegistryInventoryLine();
        }

        NativeStatus.ThrowIfFailed(status);
        fieldType = (TensorRtPluginFieldType)typeValue;
        length = lengthValue;
        hasData = hasDataValue != 0;
    }

    private static BridgeProbeException UnsupportedRuntimePluginRegistryInventoryLine()
    {
        return new BridgeProbeException(
            BridgeStatusCode.NotSupported,
            BridgeErrorCategory.TensorRt,
            "Runtime-local plugin registry inventory is exposed by this bridge for TensorRT 8, 10, and 11.");
    }
}
