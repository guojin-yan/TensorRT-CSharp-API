using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtGlobalRuntimeVersion GetGlobalRuntimeVersion(TensorRtApiLine line)
    {
        return new TensorRtGlobalRuntimeVersion(
            line,
            GetGlobalInferLibVersion(line),
            GetGlobalInferLibMajorVersion(line),
            GetGlobalInferLibMinorVersion(line),
            GetGlobalInferLibPatchVersion(line),
            GetGlobalInferLibBuildVersion(line),
            GetGlobalOnnxParserVersion(line),
            GlobalHasLogger(line));
    }

    public static TensorRtPluginRegistryInventory GetGlobalPluginRegistryInventory(TensorRtApiLine line)
    {
        return GetGlobalPluginRegistryInventory(line, includeCreatorFields: true);
    }

    public static TensorRtPluginRegistryInventory GetGlobalPluginRegistryInventory(TensorRtApiLine line, bool includeCreatorFields)
    {
        int creatorCount = GetGlobalPluginRegistryCreatorCount(line);
        int recursiveCreatorCount = GetGlobalPluginRegistryRecursiveCreatorCount(line);
        bool hasErrorRecorder = HasGlobalPluginRegistryErrorRecorder(line);
        bool parentSearchEnabled = IsGlobalPluginRegistryParentSearchEnabled(line);
        List<TensorRtPluginCreatorInfo> creators = new List<TensorRtPluginCreatorInfo>(creatorCount);

        for (int creatorIndex = 0; creatorIndex < creatorCount; creatorIndex++)
        {
            string name = GetGlobalPluginCreatorName(line, creatorIndex);
            string version = GetGlobalPluginCreatorVersion(line, creatorIndex);
            string pluginNamespace = GetGlobalPluginCreatorNamespace(line, creatorIndex);
            string interfaceKind = GetGlobalPluginCreatorInterfaceKind(line, creatorIndex, out int interfaceMajor, out int interfaceMinor);
            TensorRtApiLanguage apiLanguage = GetGlobalPluginCreatorApiLanguage(line, creatorIndex);
            IReadOnlyList<TensorRtPluginFieldInfo> fields = Array.Empty<TensorRtPluginFieldInfo>();

            if (includeCreatorFields)
            {
                int fieldCount = GetGlobalPluginCreatorFieldCount(line, creatorIndex);
                List<TensorRtPluginFieldInfo> fieldList = new List<TensorRtPluginFieldInfo>(fieldCount);

                for (int fieldIndex = 0; fieldIndex < fieldCount; fieldIndex++)
                {
                    string fieldName = GetGlobalPluginCreatorFieldName(line, creatorIndex, fieldIndex);
                    GetGlobalPluginCreatorFieldMetadata(line, creatorIndex, fieldIndex, out TensorRtPluginFieldType fieldType, out int length, out bool hasData);
                    fieldList.Add(new TensorRtPluginFieldInfo(fieldName, fieldType, length, hasData));
                }

                fields = fieldList;
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

        return new TensorRtPluginRegistryInventory(line, TensorRtPluginRegistrySource.Global, hasErrorRecorder, parentSearchEnabled, recursiveCreatorCount, creators);
    }

    public static bool IsGlobalPluginCreatorRegistered(TensorRtApiLine line, string pluginName, string pluginVersion, string pluginNamespace)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(pluginName ?? string.Empty);
        using Utf8Interop.Utf8StringScope versionUtf8 = Utf8Interop.ToNativeString(pluginVersion ?? string.Empty);
        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(pluginNamespace ?? string.Empty);

        int found;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_lookup(nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out found);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_lookup(nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out found);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return found != 0;
    }

    public static bool TryGetGlobalPluginCreator(
        TensorRtApiLine line,
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        out TensorRtPluginCreatorInfo? creator)
    {
        if (!IsGlobalPluginCreatorRegistered(line, pluginName, pluginVersion, pluginNamespace))
        {
            creator = null;
            return false;
        }

        string interfaceKind = GetGlobalLookupPluginCreatorInterfaceKind(
            line,
            pluginName,
            pluginVersion,
            pluginNamespace,
            out int interfaceMajor,
            out int interfaceMinor);
        TensorRtApiLanguage apiLanguage = GetGlobalLookupPluginCreatorApiLanguage(line, pluginName, pluginVersion, pluginNamespace);
        int fieldCount = GetGlobalLookupPluginCreatorFieldCount(line, pluginName, pluginVersion, pluginNamespace);
        List<TensorRtPluginFieldInfo> fields = new List<TensorRtPluginFieldInfo>(fieldCount);

        for (int fieldIndex = 0; fieldIndex < fieldCount; fieldIndex++)
        {
            string fieldName = GetGlobalLookupPluginCreatorFieldName(line, pluginName, pluginVersion, pluginNamespace, fieldIndex);
            GetGlobalLookupPluginCreatorFieldMetadata(line, pluginName, pluginVersion, pluginNamespace, fieldIndex, out TensorRtPluginFieldType fieldType, out int length, out bool hasData);
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

    public static int GetGlobalInferLibVersion(TensorRtApiLine line)
    {
        int value;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_global_get_infer_lib_version(out value);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_get_infer_lib_version(out value);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_get_infer_lib_version(out value);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static int GetGlobalInferLibMajorVersion(TensorRtApiLine line)
    {
        int value;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_get_infer_lib_major_version(out value);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_get_infer_lib_major_version(out value);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static int GetGlobalInferLibMinorVersion(TensorRtApiLine line)
    {
        int value;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_get_infer_lib_minor_version(out value);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_get_infer_lib_minor_version(out value);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static int GetGlobalInferLibPatchVersion(TensorRtApiLine line)
    {
        int value;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_get_infer_lib_patch_version(out value);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_get_infer_lib_patch_version(out value);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static int GetGlobalInferLibBuildVersion(TensorRtApiLine line)
    {
        int value;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_get_infer_lib_build_version(out value);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_get_infer_lib_build_version(out value);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static int GetGlobalOnnxParserVersion(TensorRtApiLine line)
    {
        int value;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_global_get_onnx_parser_version(out value);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_get_onnx_parser_version(out value);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_get_onnx_parser_version(out value);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static bool GlobalHasLogger(TensorRtApiLine line)
    {
        int hasLogger;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_global_has_logger(out hasLogger);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_has_logger(out hasLogger);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_has_logger(out hasLogger);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return hasLogger != 0;
    }

    public static bool GlobalPluginRegistryExists(TensorRtApiLine line)
    {
        int exists;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_plugin_registry_exists(out exists);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_plugin_registry_exists(out exists);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return exists != 0;
    }

    private static int GetGlobalPluginRegistryCreatorCount(TensorRtApiLine line)
    {
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_plugin_registry_get_creator_count(out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_plugin_registry_get_creator_count(out count);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static int GetGlobalPluginRegistryRecursiveCreatorCount(TensorRtApiLine line)
    {
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_plugin_registry_get_recursive_creator_count(out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_plugin_registry_get_recursive_creator_count(out count);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static bool HasGlobalPluginRegistryErrorRecorder(TensorRtApiLine line)
    {
        int hasRecorder;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_plugin_registry_has_error_recorder(out hasRecorder);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_plugin_registry_has_error_recorder(out hasRecorder);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    private static bool IsGlobalPluginRegistryParentSearchEnabled(TensorRtApiLine line)
    {
        int enabled;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_plugin_registry_is_parent_search_enabled(out enabled);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_plugin_registry_is_parent_search_enabled(out enabled);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    private static string GetGlobalPluginCreatorName(TensorRtApiLine line, int creatorIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_get_name(creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_get_name(creatorIndex, buffer, size, out required),
                _ => throw UnsupportedGlobalRuntimeProbeLine()
            },
            "Global plugin creator name is too large for the managed buffer.");
    }

    private static string GetGlobalPluginCreatorVersion(TensorRtApiLine line, int creatorIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_get_version(creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_get_version(creatorIndex, buffer, size, out required),
                _ => throw UnsupportedGlobalRuntimeProbeLine()
            },
            "Global plugin creator version is too large for the managed buffer.");
    }

    private static string GetGlobalPluginCreatorNamespace(TensorRtApiLine line, int creatorIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_get_namespace(creatorIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_get_namespace(creatorIndex, buffer, size, out required),
                _ => throw UnsupportedGlobalRuntimeProbeLine()
            },
            "Global plugin creator namespace is too large for the managed buffer.");
    }

    private static string GetGlobalPluginCreatorInterfaceKind(
        TensorRtApiLine line,
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
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_get_interface_info(creatorIndex, buffer, size, out required, out major, out minor),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_get_interface_info(creatorIndex, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedGlobalRuntimeProbeLine()
                };
                return status;
            },
            "Global plugin creator interface kind is too large for the managed buffer.");

        interfaceMajor = major;
        interfaceMinor = minor;
        return result;
    }

    private static int GetGlobalPluginCreatorFieldCount(TensorRtApiLine line, int creatorIndex)
    {
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_get_field_count(creatorIndex, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_get_field_count(creatorIndex, out count);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static TensorRtApiLanguage GetGlobalPluginCreatorApiLanguage(TensorRtApiLine line, int creatorIndex)
    {
        int apiLanguage = (int)TensorRtApiLanguage.Unknown;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_get_api_language(creatorIndex, out apiLanguage),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_get_api_language(creatorIndex, out apiLanguage),
            _ => throw UnsupportedGlobalRuntimeProbeLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return ToTensorRtApiLanguage(apiLanguage);
    }

    private static string GetGlobalPluginCreatorFieldName(TensorRtApiLine line, int creatorIndex, int fieldIndex)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_get_field_name(creatorIndex, fieldIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_get_field_name(creatorIndex, fieldIndex, buffer, size, out required),
                _ => throw UnsupportedGlobalRuntimeProbeLine()
            },
            "Global plugin creator field name is too large for the managed buffer.");
    }

    private static void GetGlobalPluginCreatorFieldMetadata(
        TensorRtApiLine line,
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
                status = NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_get_field_metadata(creatorIndex, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_get_field_metadata(creatorIndex, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        fieldType = (TensorRtPluginFieldType)typeValue;
        length = lengthValue;
        hasData = hasDataValue != 0;
    }

    private static string GetGlobalLookupPluginCreatorInterfaceKind(
        TensorRtApiLine line,
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
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_lookup_get_interface_info(nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, buffer, size, out required, out major, out minor),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_lookup_get_interface_info(nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedGlobalRuntimeProbeLine()
                };
                return status;
            },
            "Global lookup plugin creator interface kind is too large for the managed buffer.");

        interfaceMajor = major;
        interfaceMinor = minor;
        return result;
    }

    private static int GetGlobalLookupPluginCreatorFieldCount(TensorRtApiLine line, string pluginName, string pluginVersion, string pluginNamespace)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(pluginName ?? string.Empty);
        using Utf8Interop.Utf8StringScope versionUtf8 = Utf8Interop.ToNativeString(pluginVersion ?? string.Empty);
        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(pluginNamespace ?? string.Empty);

        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_lookup_get_field_count(nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_lookup_get_field_count(nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out count);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static TensorRtApiLanguage GetGlobalLookupPluginCreatorApiLanguage(TensorRtApiLine line, string pluginName, string pluginVersion, string pluginNamespace)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(pluginName ?? string.Empty);
        using Utf8Interop.Utf8StringScope versionUtf8 = Utf8Interop.ToNativeString(pluginVersion ?? string.Empty);
        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(pluginNamespace ?? string.Empty);

        int apiLanguage = (int)TensorRtApiLanguage.Unknown;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_lookup_get_api_language(nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out apiLanguage),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_lookup_get_api_language(nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, out apiLanguage),
            _ => throw UnsupportedGlobalRuntimeProbeLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return ToTensorRtApiLanguage(apiLanguage);
    }

    private static string GetGlobalLookupPluginCreatorFieldName(
        TensorRtApiLine line,
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
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_lookup_get_field_name(nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_lookup_get_field_name(nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, buffer, size, out required),
                _ => throw UnsupportedGlobalRuntimeProbeLine()
            },
            "Global lookup plugin creator field name is too large for the managed buffer.");
    }

    private static void GetGlobalLookupPluginCreatorFieldMetadata(
        TensorRtApiLine line,
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
                status = NativeMethodsTensorRt.jyppx_trt10_global_plugin_creator_lookup_get_field_metadata(nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_plugin_creator_lookup_get_field_metadata(nameUtf8.Pointer, versionUtf8.Pointer, namespaceUtf8.Pointer, fieldIndex, out typeValue, out lengthValue, out hasDataValue);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        fieldType = (TensorRtPluginFieldType)typeValue;
        length = lengthValue;
        hasData = hasDataValue != 0;
    }

    private static BridgeProbeException UnsupportedGlobalRuntimeProbeLine()
    {
        return new BridgeProbeException(
            BridgeStatusCode.NotSupported,
            BridgeErrorCategory.TensorRt,
            "Global TensorRT runtime and plugin registry probes are exposed by this bridge for TensorRT 10 and 11.");
    }
}
