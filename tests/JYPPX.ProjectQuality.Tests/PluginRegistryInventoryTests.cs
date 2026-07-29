using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PluginRegistryInventoryTests
{
    [Fact]
    public void BuilderCapabilityInventoryCopiesCreatorFieldsByDefaultAndSupportsIdentityOnlySnapshots()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Plugins", "NativeBridgeApi.BuilderCapabilityPluginRegistry.cs");
        string inventoryMethod = ExtractBetween(
            source,
            "public static TensorRtPluginRegistryInventory GetBuilderCapabilityPluginRegistryInventory",
            "public static bool IsBuilderCapabilityPluginCreatorRegistered");

        Assert.Contains("return GetBuilderCapabilityPluginRegistryInventory(line, capability, includeCreatorFields: true);", inventoryMethod);
        Assert.Contains("if (includeCreatorFields)", inventoryMethod);
        Assert.Contains("int fieldCount = GetBuilderCapabilityPluginCreatorFieldCount(line, capability, creatorIndex);", inventoryMethod);
        Assert.Contains("GetBuilderCapabilityPluginCreatorFieldName(line, capability, creatorIndex, fieldIndex)", inventoryMethod);
        Assert.Contains("GetBuilderCapabilityPluginCreatorFieldMetadata(line, capability, creatorIndex, fieldIndex", inventoryMethod);
        Assert.Contains("fieldList.Add(new TensorRtPluginFieldInfo(fieldName, fieldType, length, hasData));", inventoryMethod);
        Assert.Contains("fields,", inventoryMethod);
        Assert.Contains("tensorRtVersion));", inventoryMethod);
        Assert.Contains("Array.Empty<TensorRtPluginFieldInfo>()", inventoryMethod);
    }

    [Fact]
    public void GlobalLookupCopiesCreatorMetadataIntoManagedSnapshot()
    {
        string interopSource = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Plugins", "NativeBridgeApi.GlobalPluginRegistry.cs");
        string environmentProbe = ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.GlobalPluginRegistry.cs");
        string smokeProgram = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "global_runtime_plugin_probe.inc");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string header11 = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string manifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-seventh-batch-global-runtime-plugin-probe.manifest.json");
        string manifest11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-twenty-seventh-batch-global-runtime-plugin-probe.manifest.json");

        string lookupMethod = ExtractBetween(
            interopSource,
            "public static bool TryGetGlobalPluginCreator",
            "public static bool GlobalPluginRegistryExists");

        Assert.Contains("GetGlobalLookupPluginCreatorInterfaceKind", lookupMethod);
        Assert.Contains("GetGlobalLookupPluginCreatorFieldCount", lookupMethod);
        Assert.Contains("GetGlobalLookupPluginCreatorFieldName", lookupMethod);
        Assert.Contains("GetGlobalLookupPluginCreatorFieldMetadata", lookupMethod);
        Assert.Contains("new TensorRtPluginCreatorInfo(", lookupMethod);
        Assert.Contains("public static bool TryGetGlobalPluginCreator", environmentProbe);
        Assert.Contains("NativeBridgeApi.TryGetGlobalPluginCreator", environmentProbe);
        Assert.Contains("TryGetGlobalPluginCreator(", smokeProgram);
        Assert.Contains("public static bool IsGlobalPluginRegistryAvailable", environmentProbe);
        Assert.Contains("public static bool TryIsGlobalPluginRegistryAvailable", environmentProbe);
        Assert.Contains("NativeBridgeApi.GlobalPluginRegistryExists", environmentProbe);
        Assert.Contains("GlobalPluginRegistryExists", interopSource);
        Assert.Contains("TryIsGlobalPluginRegistryAvailable", smokeProgram);
        Assert.Contains("GlobalPluginRegistry Exists=", smokeProgram);
        Assert.Contains("global_plugin_registry_exists", nativeSource);
        Assert.Contains("global_plugin_creator_lookup_get_interface_info", nativeSource);
        Assert.Contains("global_plugin_creator_lookup_get_field_metadata", nativeSource);
        Assert.Contains("jyppx_trt10_global_plugin_registry_exists", header10);
        Assert.Contains("jyppx_trt10_global_plugin_creator_lookup_get_interface_info", header10);
        Assert.Contains("jyppx_trt10_global_plugin_creator_lookup_get_field_metadata", header10);
        Assert.Contains("jyppx_trt11_global_plugin_registry_exists", header11);
        Assert.Contains("jyppx_trt11_global_plugin_creator_lookup_get_interface_info", header11);
        Assert.Contains("jyppx_trt11_global_plugin_creator_lookup_get_field_metadata", header11);
        Assert.Contains("trt10-global-plugin-registry-exists", manifest10);
        Assert.Contains("trt11-global-plugin-registry-exists", manifest11);
        Assert.DoesNotContain("out_registry", environmentProbe);
    }

    [Fact]
    public void BuilderOwnedLookupCopiesCreatorMetadataIntoManagedSnapshot()
    {
        string interopSource = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Plugins", "NativeBridgeApi.PluginRegistryInventory.cs");
        string builderSource = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.PluginRegistryInventory.cs");
        string smokeProgram = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "plugin_registry_inventory.inc");
        string manifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-sixth-batch-plugin-registry-inventory.manifest.json");
        string manifest11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-twenty-sixth-batch-plugin-registry-inventory.manifest.json");

        string lookupMethod = ExtractBetween(
            interopSource,
            "public static bool TryGetBuilderPluginCreator",
            "private static string GetBuilderLookupPluginCreatorInterfaceKind");

        Assert.Contains("GetBuilderLookupPluginCreatorInterfaceKind", lookupMethod);
        Assert.Contains("GetBuilderLookupPluginCreatorFieldCount", lookupMethod);
        Assert.Contains("GetBuilderLookupPluginCreatorFieldName", lookupMethod);
        Assert.Contains("GetBuilderLookupPluginCreatorFieldMetadata", lookupMethod);
        Assert.Contains("new TensorRtPluginCreatorInfo(", lookupMethod);
        Assert.Contains("GetBuilderPluginRegistryRecursiveCreatorCount", interopSource);
        Assert.Contains("jyppx_trt10_builder_plugin_registry_exists", interopSource);
        Assert.Contains("jyppx_trt11_builder_plugin_registry_exists", interopSource);
        Assert.DoesNotContain("exists = 1;", interopSource);
        Assert.Contains("public bool IsPluginCreatorRegistered", builderSource);
        Assert.Contains("public bool TryGetPluginCreator", builderSource);
        Assert.Contains("NativeBridgeApi.TryGetBuilderPluginCreator", builderSource);
        Assert.Contains("Line == TensorRtApiLine.TensorRt8", builderSource);
        Assert.Contains("NativeBridgeApi.GetBuilderPluginRegistryRecursiveCreatorCount(Line, _handle)", builderSource);
        Assert.Contains("recursiveCreatorCount: recursiveCreatorCount", builderSource);
        Assert.Contains("ValidateBuilderOwnedLookup", smokeProgram);
        Assert.Contains("TryGetPluginCreator(", smokeProgram);
        Assert.Contains("builder_plugin_creator_lookup_get_interface_info", nativeSource);
        Assert.Contains("builder_plugin_creator_lookup_get_field_metadata", nativeSource);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(builder_plugin_registry_exists)", nativeSource);
        Assert.Contains("*out_exists = registry != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", nativeSource);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(builder_plugin_registry_get_recursive_creator_count)", nativeSource);
        Assert.Contains("trt10-builder-plugin-registry-exists", manifest10);
        Assert.Contains("trt11-builder-plugin-registry-exists", manifest11);
        Assert.Contains("trt10-builder-plugin-creator-lookup-get-interface-info", manifest10);
        Assert.Contains("trt11-builder-plugin-creator-lookup-get-field-metadata", manifest11);
        Assert.Contains("trt10-builder-plugin-registry-get-recursive-creator-count", manifest10);
        Assert.Contains("trt11-builder-plugin-registry-get-recursive-creator-count", manifest11);
    }

    [Fact]
    public void TensorRt8BuilderPluginRegistryInventoryPromotesSafeReadOnlyMetadata()
    {
        string nativeSource = ReadSource("native", "src", "tensorrt", "v8", "modules", "plugin", "trt8_plugin_registry_inventory.inc");
        string apiSource = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string manifest8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-plugin-registry-inventory.manifest.json");
        string deferred8 = ReadTensorRtManifest("v8", "trt8-cross-version-eleventh-batch-plugin-deferred.manifest.json");
        string interopSource = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Plugins", "NativeBridgeApi.PluginRegistryInventory.cs");
        string builderSource = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.PluginRegistryInventory.cs");
        string smokeProgram = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");

        Assert.Contains("modules/plugin/trt8_plugin_registry_inventory.inc", apiSource);
        Assert.Contains("jyppx_trt8_builder_plugin_registry_exists", manifest8);
        Assert.Contains("jyppx_trt8_builder_plugin_registry_get_creator_count", manifest8);
        Assert.Contains("jyppx_trt8_builder_plugin_registry_has_error_recorder", manifest8);
        Assert.Contains("jyppx_trt8_builder_plugin_registry_is_parent_search_enabled", manifest8);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_get_name", manifest8);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_get_version", manifest8);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_get_namespace", manifest8);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_get_field_count", manifest8);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_get_field_name", manifest8);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_get_field_metadata", manifest8);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_lookup", manifest8);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_lookup_get_field_count", manifest8);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_lookup_get_field_name", manifest8);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_lookup_get_field_metadata", manifest8);

        Assert.Contains("builder_payload->getPluginRegistry()", nativeSource);
        Assert.Contains("registry->getPluginCreatorList(out_count)", nativeSource);
        Assert.Contains("registry->getPluginCreator(plugin_name, plugin_version, plugin_namespace)", nativeSource);
        Assert.Contains("jyppx_trt8_get_plugin_registry_has_error_recorder_with_seh_guard", nativeSource);
        Assert.Contains("registry->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", nativeSource);
        Assert.Contains("creator->getPluginName()", nativeSource);
        Assert.Contains("creator->getPluginVersion()", nativeSource);
        Assert.Contains("creator->getPluginNamespace()", nativeSource);
        Assert.Contains("creator->getFieldNames()", nativeSource);
        Assert.Contains("copy_string_to_buffer", nativeSource);
        Assert.Contains("jyppx_trt8_get_builder_plugin_registry_with_seh_guard", nativeSource);

        Assert.Contains("jyppx_trt8_builder_plugin_registry_exists", header8);
        Assert.Contains("jyppx_trt8_builder_plugin_registry_has_error_recorder", header8);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_lookup_get_field_metadata", header8);

        Assert.Contains("TensorRtApiLine.TensorRt8", interopSource);
        Assert.Contains("jyppx_trt8_builder_plugin_registry_has_error_recorder", interopSource);
        Assert.Contains("jyppx_trt8_builder_plugin_registry_get_creator_count", interopSource);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_get_name", interopSource);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_lookup_get_field_metadata", interopSource);
        Assert.Contains("return \"IPluginCreator\";", interopSource);
        Assert.Contains("public bool IsPluginRegistryAvailable", builderSource);
        Assert.Contains("TryIsPluginRegistryAvailable", builderSource);
        Assert.Contains("ValidateGlobalParentSearchRoundTrip", smokeProgram);
        Assert.DoesNotContain("TensorRt8GlobalPluginRegistrySkipped=True", smokeProgram);
        Assert.DoesNotContain("TensorRt8GlobalAndCapabilityPluginRegistriesSkipped", smokeProgram);
        Assert.DoesNotContain("PluginRegistryInventoryRequiresTensorRt10Or11", smokeProgram);

        Assert.Contains("trt8-plugin-creator-get-plugin-name-deferred", deferred8);
        Assert.Contains("trt8-plugin-registry-get-error-recorder-deferred", deferred8);
        Assert.Contains("trt8-plugin-registry-get-plugin-creator-list-deferred", deferred8);
        Assert.DoesNotContain("public IntPtr", builderSource);
        Assert.DoesNotContain("public nint", builderSource);
    }

    [Fact]
    public void RuntimeLocalInventoryCopiesCreatorMetadataWithoutExposingNativePointers()
    {
        string interopSource = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Plugins", "NativeBridgeApi.RuntimePluginRegistryInventory.cs");
        string runtimeSource = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.PluginRegistryInventory.cs");
        string inventoryModels = string.Join(
            '\n',
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginRegistryTypes.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginRegistryInventory.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginCreatorInfo.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginFieldInfo.cs"));
        string smokeProgram = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "plugin_registry_inventory.inc");
        string native8 = ReadSource("native", "src", "tensorrt", "v8", "modules", "plugin", "trt8_plugin_registry_inventory.inc");
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string manifest8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-plugin-registry-inventory.manifest.json");
        string manifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-runtime-local-plugin-registry-inventory.manifest.json");
        string manifest11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-runtime-local-plugin-registry-inventory.manifest.json");
        string deferred8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json");
        string deferred10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-cross-version-fifth-batch-runtime-serialization-deferred.manifest.json");
        string deferred11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json");

        string inventoryMethod = ExtractBetween(
            interopSource,
            "public static TensorRtPluginRegistryInventory GetRuntimePluginRegistryInventory",
            "public static bool IsRuntimePluginCreatorRegistered");

        Assert.Contains("RuntimePluginRegistryExists", interopSource);
        Assert.Contains("GetRuntimePluginCreatorName", inventoryMethod);
        Assert.Contains("GetRuntimePluginCreatorVersion", inventoryMethod);
        Assert.Contains("GetRuntimePluginCreatorNamespace", inventoryMethod);
        Assert.Contains("GetRuntimePluginCreatorFieldCount", inventoryMethod);
        Assert.Contains("GetRuntimePluginCreatorFieldName", inventoryMethod);
        Assert.Contains("GetRuntimePluginCreatorFieldMetadata", inventoryMethod);
        Assert.Contains("new TensorRtPluginCreatorInfo(", inventoryMethod);
        Assert.Contains("GetRuntimePluginRegistryRecursiveCreatorCount", interopSource);
        Assert.Contains("GetRuntimeLookupPluginCreatorInterfaceKind", interopSource);
        Assert.Contains("GetRuntimeLookupPluginCreatorFieldCount", interopSource);
        Assert.Contains("GetRuntimeLookupPluginCreatorFieldName", interopSource);
        Assert.Contains("GetRuntimeLookupPluginCreatorFieldMetadata", interopSource);
        Assert.Contains("line == TensorRtApiLine.TensorRt8", inventoryMethod);
        Assert.Contains("? null", inventoryMethod);
        Assert.Contains(": GetRuntimePluginRegistryRecursiveCreatorCount(line, runtime)", inventoryMethod);
        Assert.Contains("recursiveCreatorCount: recursiveCreatorCount", inventoryMethod);
        Assert.Contains("Runtime = 3", inventoryModels);

        Assert.Contains("public bool IsPluginRegistryAvailable", runtimeSource);
        Assert.Contains("public TensorRtPluginRegistryInventory GetPluginRegistryInventory", runtimeSource);
        Assert.Contains("public bool TryGetPluginCreator", runtimeSource);
        Assert.Contains("The returned metadata is copied into managed objects.", runtimeSource);
        Assert.Contains("TensorRT 8, 10, and 11 are queried through copied metadata probes.", runtimeSource);
        Assert.DoesNotContain("IntPtr", runtimeSource);
        Assert.DoesNotContain("nint", runtimeSource);

        Assert.Contains("TensorRtApiLine.TensorRt8", interopSource);
        Assert.Contains("jyppx_trt8_runtime_plugin_registry_exists", interopSource);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_get_name", interopSource);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_get_field_metadata", interopSource);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_lookup", interopSource);
        Assert.Contains("return \"IPluginCreator\";", interopSource);

        Assert.Contains("runtime_payload->getPluginRegistry()", native8);
        Assert.Contains("jyppx_trt8_get_runtime_plugin_registry_with_seh_guard", native8);
        Assert.Contains("jyppx_trt8_runtime_plugin_registry_exists", native8);
        Assert.Contains("jyppx_trt8_runtime_plugin_registry_get_creator_count", native8);
        Assert.Contains("jyppx_trt8_runtime_plugin_registry_is_parent_search_enabled", native8);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_get_name", native8);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_get_version", native8);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_get_namespace", native8);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_get_field_count", native8);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_get_field_name", native8);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_get_field_metadata", native8);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_lookup", native8);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_lookup_get_field_count", native8);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_lookup_get_field_name", native8);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_lookup_get_field_metadata", native8);
        Assert.Contains("copy_string_to_buffer", native8);

        Assert.Contains("JYPPX_StatusCode jyppx_trt_get_runtime_plugin_registry_with_seh_guard", nativeSource);
        Assert.Contains("runtime_payload->getPluginRegistry()", nativeSource);
        Assert.Contains("JYPPX_TENSORRT_OBJECT_KIND_RUNTIME", nativeSource);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(runtime_plugin_registry_get_creator_count)", nativeSource);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(runtime_plugin_registry_get_recursive_creator_count)", nativeSource);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(runtime_plugin_creator_lookup)", nativeSource);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(runtime_plugin_creator_lookup_get_interface_info)", nativeSource);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(runtime_plugin_creator_lookup_get_field_count)", nativeSource);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(runtime_plugin_creator_lookup_get_field_name)", nativeSource);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(runtime_plugin_creator_lookup_get_field_metadata)", nativeSource);

        Assert.Contains("jyppx_trt8_runtime_plugin_registry_exists", header8);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_get_field_metadata", header8);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_lookup", header8);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_lookup_get_field_metadata", header8);
        Assert.Contains("trt8-runtime-plugin-registry-exists", manifest8);
        Assert.Contains("trt8-runtime-plugin-creator-get-name", manifest8);
        Assert.Contains("trt8-runtime-plugin-creator-get-field-metadata", manifest8);
        Assert.Contains("trt8-runtime-plugin-creator-lookup", manifest8);
        Assert.Contains("trt8-runtime-plugin-creator-lookup-get-field-count", manifest8);
        Assert.Contains("trt8-runtime-plugin-creator-lookup-get-field-name", manifest8);
        Assert.Contains("trt8-runtime-plugin-creator-lookup-get-field-metadata", manifest8);
        Assert.Contains("trt10-runtime-plugin-registry-exists", manifest10);
        Assert.Contains("trt10-runtime-plugin-creator-get-field-metadata", manifest10);
        Assert.Contains("trt10-runtime-plugin-creator-lookup-get-interface-info", manifest10);
        Assert.Contains("trt10-runtime-plugin-creator-lookup-get-field-count", manifest10);
        Assert.Contains("trt10-runtime-plugin-creator-lookup-get-field-name", manifest10);
        Assert.Contains("trt10-runtime-plugin-creator-lookup-get-field-metadata", manifest10);
        Assert.Contains("trt11-runtime-plugin-registry-get-creator-count", manifest11);
        Assert.Contains("trt11-runtime-plugin-creator-lookup", manifest11);
        Assert.Contains("trt11-runtime-plugin-creator-lookup-get-interface-info", manifest11);
        Assert.Contains("trt11-runtime-plugin-creator-lookup-get-field-count", manifest11);
        Assert.Contains("trt11-runtime-plugin-creator-lookup-get-field-name", manifest11);
        Assert.Contains("trt11-runtime-plugin-creator-lookup-get-field-metadata", manifest11);
        Assert.Contains("trt10-runtime-plugin-registry-get-recursive-creator-count", manifest10);
        Assert.Contains("trt11-runtime-plugin-registry-get-recursive-creator-count", manifest11);
        Assert.Contains("trt8-runtime-get-plugin-registry-deferred", deferred8);
        Assert.Contains("trt10-runtime-get-plugin-registry-deferred", deferred10);
        Assert.Contains("trt11-runtime-get-plugin-registry-deferred", deferred11);

        Assert.Contains("RuntimePluginRegistry", smokeProgram);
        Assert.Contains("ValidateGlobalParentSearchRoundTrip", smokeProgram);
        Assert.DoesNotContain("TensorRt8GlobalPluginRegistrySkipped=True", smokeProgram);
        Assert.DoesNotContain("TensorRt8GlobalAndCapabilityPluginRegistriesSkipped", smokeProgram);
        Assert.Contains("ValidateRuntimeLocalLookup", smokeProgram);
        Assert.Contains("TryIsPluginRegistryAvailable", smokeProgram);
        Assert.Contains("TryGetPluginRegistryInventory", smokeProgram);
        Assert.Contains("TryGetPluginCreator(", smokeProgram);
        Assert.DoesNotContain("LoadLibrary", runtimeSource);
        Assert.DoesNotContain("RegisterCreator", runtimeSource);
    }

    [Fact]
    public void PublicPluginInventoryApiUsesCopiedValueObjectsInsteadOfNativeCreatorPointers()
    {
        string environmentProbe = string.Join(
            '\n',
            ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.GlobalPluginRegistry.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.BuilderPluginRegistry.cs"));
        string inventoryModels = string.Join(
            '\n',
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginRegistryInventory.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginCreatorInfo.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginCreatorSummary.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginFieldInfo.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginFieldSummary.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginRegistryInventoryDiagnostics.cs"));
        string builderInventory = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.PluginRegistryInventory.cs");
        string runtimeInventory = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.PluginRegistryInventory.cs");
        string bridgePackageConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        Assert.Contains("public static bool IsBuilderCapabilityPluginRegistryAvailable", environmentProbe);
        Assert.Contains("public static bool TryIsBuilderCapabilityPluginRegistryAvailable", environmentProbe);
        Assert.Contains("public static bool IsBuilderSafePluginRegistryAvailable", environmentProbe);
        Assert.Contains("public static bool TryIsBuilderSafePluginRegistryAvailable", environmentProbe);
        Assert.Contains("public static bool IsGlobalPluginRegistryAvailable", environmentProbe);
        Assert.Contains("public static bool TryIsGlobalPluginRegistryAvailable", environmentProbe);
        Assert.Contains("public static bool TryIsGlobalPluginCreatorRegistered", environmentProbe);
        Assert.Contains("public static bool TryGetGlobalPluginCreator", environmentProbe);
        Assert.Contains("public static bool TryIsBuilderCapabilityPluginCreatorRegistered", environmentProbe);
        Assert.Contains("public static TensorRtPluginRegistryInventory GetBuilderCapabilityPluginRegistryInventory", environmentProbe);
        Assert.Contains("public static bool TryGetBuilderCapabilityPluginCreator", environmentProbe);
        Assert.Contains("The returned metadata is copied into managed objects.", environmentProbe);
        Assert.DoesNotContain("IntPtr", environmentProbe);
        Assert.DoesNotContain("nint", environmentProbe);

        Assert.Contains("public bool TryGetPluginCreator", builderInventory);
        Assert.Contains("The returned metadata is copied into managed objects.", builderInventory);
        Assert.DoesNotContain("IntPtr", builderInventory);
        Assert.DoesNotContain("nint", builderInventory);

        Assert.Contains("public bool TryGetPluginCreator", runtimeInventory);
        Assert.Contains("The returned metadata is copied into managed objects.", runtimeInventory);
        Assert.DoesNotContain("IntPtr", runtimeInventory);
        Assert.DoesNotContain("nint", runtimeInventory);

        Assert.Contains("public IReadOnlyList<TensorRtPluginCreatorInfo> Creators { get; }", inventoryModels);
        Assert.Contains("public IReadOnlyList<TensorRtPluginFieldInfo> Fields { get; }", inventoryModels);
        Assert.Contains("public sealed class TensorRtPluginCreatorSummary", inventoryModels);
        Assert.Contains("public int FieldCount { get; }", inventoryModels);
        Assert.Contains("public IReadOnlyList<TensorRtPluginCreatorSummary> GetCreatorSummaries(int maxCreators = int.MaxValue)", inventoryModels);
        Assert.Contains("This method does not call TensorRT, expose native creator pointers, or copy plugin field data payloads.", inventoryModels);
        Assert.Contains("public sealed class TensorRtPluginRegistryInventoryDiagnostics", inventoryModels);
        Assert.Contains("public TensorRtPluginRegistryInventoryDiagnostics GetDiagnostics()", inventoryModels);
        Assert.Contains("public bool IsConsistent", inventoryModels);
        Assert.Contains("public bool CanPromoteRuntimeProof => false", inventoryModels);
        Assert.Contains("public bool CanDeleteDeferredRecord => false", inventoryModels);
        Assert.Contains("public int TotalFieldCount { get; }", inventoryModels);
        Assert.Contains("public int EmptyNameCount { get; }", inventoryModels);
        Assert.Contains("public int NegativeFieldLengthCount { get; }", inventoryModels);
        Assert.Contains("This method only inspects already-copied metadata.", inventoryModels);
        Assert.Contains("public TensorRtPluginCreatorInfo? FindCreator(string pluginName, string pluginVersion, string pluginNamespace)", inventoryModels);
        Assert.Contains("public bool TryFindCreator(string pluginName, string pluginVersion, string pluginNamespace, out TensorRtPluginCreatorInfo? creator)", inventoryModels);
        Assert.Contains("This method only scans already-copied metadata.", inventoryModels);
        Assert.Contains("This is a managed snapshot lookup.", inventoryModels);
        Assert.DoesNotContain("IntPtr", inventoryModels);
        Assert.DoesNotContain("nint", inventoryModels);

        Assert.Contains("TensorRtPluginRegistryInventory.FindCreator", bridgePackageConsumer);
        Assert.Contains("TensorRtPluginRegistryInventory.TryFindCreator", bridgePackageConsumer);
        Assert.Contains("TensorRtPluginRegistryInventory.GetCreatorSummaries", bridgePackageConsumer);
        Assert.Contains("TensorRtPluginRegistryInventory.GetDiagnostics", bridgePackageConsumer);
        Assert.Contains("TensorRtPluginRegistryInventoryDiagnostics", bridgePackageConsumer);
        Assert.Contains("pluginInventoryDiagnostics", bridgePackageConsumer);
        Assert.Contains("pluginInventoryDiagnosticsText", bridgePackageConsumer);
        Assert.Contains("TensorRtPluginCreatorSummary", bridgePackageConsumer);
        Assert.Contains("snapshotFindCreator", bridgePackageConsumer);
        Assert.Contains("snapshotTryFindCreator", bridgePackageConsumer);
        Assert.Contains("creatorSummaries", bridgePackageConsumer);
        Assert.Contains("limitedCreatorSummaries", bridgePackageConsumer);
        Assert.Contains("creatorSummaryIdentity", bridgePackageConsumer);
    }

    [Fact]
    public void PluginInventorySmokeRunnerIsRegistered()
    {
        string smokeReadme = ReadSource("smoke", "README.md");
        string solution = ReadSource("TensorRtSharp.sln");
        string program = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");

        Assert.Contains("PluginRegistryInventorySmokeRunner", smokeReadme);
        Assert.Contains("PluginRegistryInventorySmokeRunner.csproj", solution);
        Assert.Contains("TryIsGlobalPluginRegistryAvailable", program);
        Assert.Contains("TryIsBuilderCapabilityPluginRegistryAvailable", program);
        Assert.Contains("TryIsBuilderSafePluginRegistryAvailable", program);
        Assert.Contains("TryIsGlobalPluginCreatorRegistered", program);
        Assert.Contains("TryGetGlobalPluginCreator", program);
        Assert.Contains("TryIsBuilderCapabilityPluginCreatorRegistered", program);
        Assert.Contains("TryGetBuilderCapabilityPluginRegistryInventory", program);
        Assert.Contains("TryGetBuilderCapabilityPluginCreator", program);
        Assert.Contains("TryGetPluginRegistryInventory", program);
        Assert.Contains("TryIsPluginCreatorRegistered", program);
        Assert.Contains("TryGetPluginCreator", program);
        Assert.Contains("ValidateBuilderOwnedLookup", program);
        Assert.Contains("ValidateRuntimeLocalLookup", program);
        Assert.Contains("PluginCreatorSnapshot", program);
        Assert.Contains("FindCreator(candidate.Name, candidate.Version, candidate.Namespace)", program);
        Assert.Contains("TryFindCreator(candidate.Name, candidate.Version, candidate.Namespace", program);
        Assert.Contains("RuntimePluginRegistry", program);
        Assert.Contains("RuntimePluginCreatorLookup", program);
        Assert.Contains("CreatorSummary Name=", program);
        Assert.Contains("GetDiagnostics()", program);
        Assert.Contains("PluginRegistryInventoryDiagnostics", program);
        Assert.Contains("IsConsistent", program);
        Assert.Contains("TotalFieldCount", program);
        Assert.Contains("FirstSummary=", program);
        Assert.Contains("GetCreatorSummaries(maxCreators: 1)", program);
        Assert.DoesNotContain("CreatePlugin", program);
        Assert.DoesNotContain("LoadLibrary", program);
        Assert.DoesNotContain("RegisterCreator", program);
    }

    [Fact]
    public void PluginInventorySmokeOnlySkipsTheExplicitCompatibleHostNullRuntimeFailure()
    {
        string program = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");

        Assert.Contains("IsCompatibleHostNullTensorRtObjectException(bridgeProbe)", program, StringComparison.Ordinal);
        Assert.Contains("exception.StatusCode == BridgeStatusCode.RuntimeError", program, StringComparison.Ordinal);
        Assert.Contains("returned a null TensorRT object.", program, StringComparison.Ordinal);
        Assert.Contains("StringComparison.Ordinal", program, StringComparison.Ordinal);
        Assert.DoesNotContain("exception is EntryPointNotFoundException", program, StringComparison.Ordinal);
        Assert.DoesNotContain("bridgeProbe.InnerException is EntryPointNotFoundException", program, StringComparison.Ordinal);
    }

    [Fact]
    public void PluginInstanceAndResourceBoundariesRemainDeferred()
    {
        string pluginOwnershipBoundary = ReadSource("docs", "articles", "zh-cn", "plugin-ownership-boundary.md");
        string trt8PluginDeferred = ReadTensorRtManifest("v8", "trt8-cross-version-eleventh-batch-plugin-deferred.manifest.json");
        string trt10PluginDeferred = ReadTensorRtManifest("v10", "trt10-cross-version-second-batch-plugin-deferred.manifest.json");
        string trt11PluginDeferred = ReadTensorRtManifest("v11", "trt11-forty-sixth-batch-plugin-deferred.manifest.json");
        string trt11CoverageDeferred = ReadTensorRtManifest("v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json");
        string native10 = ReadSource("native", "src", "tensorrt", "v10", "modules", "deferred", "cross_version_plugin_deferred.inc");
        string native11 = ReadSource("native", "src", "tensorrt", "v11", "modules", "deferred", "twenty_third_batch_deferred.inc");
        string native8 = ReadSource("native", "src", "tensorrt", "v8", "modules", "deferred", "cross_version_eleventh_batch_plugin_deferred.inc");

        Assert.Contains("builder-owned plugin registry inventory", pluginOwnershipBoundary);
        Assert.Contains("global runtime plugin registry probe", pluginOwnershipBoundary);
        Assert.Contains("runtime-local plugin registry inventory", pluginOwnershipBoundary);
        Assert.Contains("registry register / deregister", pluginOwnershipBoundary);
        Assert.Contains("load library / deregister library", pluginOwnershipBoundary);
        Assert.Contains("plugin resource acquire / release", pluginOwnershipBoundary);
        Assert.Contains("plugin instance create / clone / destroy", pluginOwnershipBoundary);
        Assert.Contains("Plugin V2/V3 callback trampoline", pluginOwnershipBoundary);
        Assert.Contains("不暴露 `public IntPtr` / `public nint` plugin creator", pluginOwnershipBoundary);
        Assert.Contains("只能作为边界规划输入，不能作为 release proof 或低风险提升清单", pluginOwnershipBoundary);

        foreach (string manifest in new[] { trt10PluginDeferred, trt11PluginDeferred })
        {
            Assert.Contains("acquire-plugin-resource-deferred", manifest);
            Assert.Contains("release-plugin-resource-deferred", manifest);
            Assert.Contains("plugin-resource-clone-deferred", manifest);
            Assert.Contains("plugin-resource-get-interface-info-deferred", manifest);
            Assert.Contains("plugin-resource-context-get-error-recorder-deferred", manifest);
            Assert.Contains("plugin-resource-context-get-gpu-allocator-deferred", manifest);
            Assert.Contains("plugin-v2-clone-deferred", manifest);
            Assert.Contains("plugin-v2-enqueue-deferred", manifest);
        }

        Assert.Contains("plugin-creator-create-plugin-deferred", trt8PluginDeferred);
        Assert.Contains("plugin-registry-register-creator-deferred", trt8PluginDeferred);
        Assert.Contains("plugin-registry-load-library-deferred", trt8PluginDeferred);
        Assert.Contains("plugin-v2-clone-deferred", trt8PluginDeferred);
        Assert.Contains("plugin-v2-enqueue-deferred", trt8PluginDeferred);
        Assert.Contains("plugin-creator-v3-one-create-plugin-deferred", trt10PluginDeferred);
        Assert.Contains("plugin-creator-create-plugin-deferred", trt11CoverageDeferred);
        Assert.Contains("plugin-creator-v3-one-create-plugin-deferred", trt11CoverageDeferred);

        Assert.Contains("IPluginRegistry::acquirePluginResource", native10);
        Assert.Contains("IPluginRegistry::releasePluginResource", native10);
        Assert.Contains("IPluginResource::clone", native10);
        Assert.Contains("IPluginResourceContext::getErrorRecorder", native10);
        Assert.Contains("IPluginResourceContext::getGpuAllocator", native10);
        Assert.Contains("IPluginCreator::createPlugin", native11);
        Assert.Contains("IPluginV2::clone", native10);
        Assert.Contains("IPluginV2::enqueue", native10);

        Assert.Contains("IPluginResourceContext::getErrorRecorder deferred: resource context returns a borrowed callback object owned by TensorRT/plugin execution", native10);
        Assert.Contains("IPluginResourceContext::getGpuAllocator deferred: resource context returns a borrowed allocator callback with device-memory ownership", native10);
        Assert.Contains("IPluginRegistry::acquirePluginResource deferred: plugin resources cross registry ownership", native11);
        Assert.Contains("IPluginRegistry::releasePluginResource deferred: plugin resource release must be paired", native11);
        Assert.Contains("IPluginResourceContext::getErrorRecorder deferred: resource context returns a borrowed callback object owned by TensorRT/plugin execution", native11);
        Assert.Contains("IPluginResourceContext::getGpuAllocator deferred: resource context returns a borrowed allocator callback", native11);
        Assert.Contains("IPluginCreator::createPlugin deferred: plugin factory, callback, or opaque plugin ownership", native11);
        Assert.Contains("IPluginV2::clone deferred: plugin clone returns a new opaque plugin instance", native11);
        Assert.Contains("IPluginV2::enqueue deferred: plugin enqueue is a TensorRT-to-plugin callback", native11);

        Assert.Contains("IPluginRegistry::registerCreator", native8);
        Assert.Contains("IPluginRegistry::loadLibrary", native8);
        Assert.Contains("IPluginV2DynamicExt::clone", native8);
        Assert.Contains("IPluginV2DynamicExt::enqueue", native8);
    }

    [Fact]
    public void PluginCreatorListUsesCopiedInventoryWhileBorrowedPointerExportRemainsDeferred()
    {
        string trt8PluginDeferred = ReadTensorRtManifest("v8", "trt8-cross-version-eleventh-batch-plugin-deferred.manifest.json");
        string trt10PluginDeferred = ReadTensorRtManifest("v10", "trt10-cross-version-second-batch-plugin-deferred.manifest.json");
        string native10 = ReadSource("native", "src", "tensorrt", "v10", "modules", "deferred", "cross_version_plugin_deferred.inc");
        string native8 = ReadSource("native", "src", "tensorrt", "v8", "modules", "deferred", "cross_version_eleventh_batch_plugin_deferred.inc");
        string coverageExport = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        string inventoryModels = string.Join(
            '\n',
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginRegistryInventory.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginCreatorInfo.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginCreatorSummary.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginFieldInfo.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginFieldSummary.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginRegistryInventoryDiagnostics.cs"));
        string builderInventory = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.PluginRegistryInventory.cs");
        string runtimeInventory = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.PluginRegistryInventory.cs");
        string environmentProbe = string.Join(
            '\n',
            ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.GlobalPluginRegistry.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.BuilderPluginRegistry.cs"));

        Assert.Contains("trt8-plugin-registry-get-plugin-creator-list-deferred", trt8PluginDeferred);
        Assert.Contains("trt10-plugin-registry-get-plugin-creator-list-deferred", trt10PluginDeferred);
        Assert.Contains("IPluginRegistry::getPluginCreatorList deferred: returns borrowed plugin creator pointer lists", native8);
        Assert.Contains("IPluginRegistry::getPluginCreatorList deferred: returns borrowed plugin creator pointer lists", native10);
        Assert.Contains("use copied plugin registry inventory metadata", native8);
        Assert.Contains("use copied plugin registry inventory metadata", native10);

        Assert.Contains("\"IPluginRegistry::getPluginCreatorList\" = @(\"id:*plugin-registry-get-creator-count\"", coverageExport);
        Assert.Contains("\"IPluginRegistry::getErrorRecorder\" = @(\"id:*plugin-registry-has-error-recorder\", \"id:*plugin-registry-get-error-recorder-deferred\")", coverageExport);
        Assert.Contains("\"IPluginRegistry::getPluginCreatorList\",", coverageExport);
        Assert.Contains("\"IPluginRegistry\",\"getPluginCreatorList\",\"IPluginRegistry::getPluginCreatorList\",\"plugin\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IPluginRegistry\",\"getErrorRecorder\",\"IPluginRegistry::getErrorRecorder\",\"plugin\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("trt8-builder-plugin-registry-has-error-recorder", comparison);
        Assert.Contains("trt8-runtime-plugin-registry-has-error-recorder", comparison);
        Assert.Contains("trt8-plugin-registry-get-error-recorder-deferred", comparison);
        Assert.Contains("trt8-plugin-registry-get-plugin-creator-list-deferred", comparison);
        Assert.Contains("trt10-plugin-registry-get-plugin-creator-list-deferred", comparison);
        Assert.Contains("trt8-builder-plugin-registry-get-creator-count", comparison);
        Assert.Contains("trt10-builder-plugin-registry-get-creator-count", comparison);
        Assert.Contains("trt10-runtime-plugin-registry-get-creator-count", comparison);

        Assert.Contains("public IReadOnlyList<TensorRtPluginCreatorInfo> Creators { get; }", inventoryModels);
        Assert.Contains("public TensorRtPluginCreatorInfo? FindCreator", inventoryModels);
        Assert.Contains("public bool TryFindCreator", inventoryModels);
        Assert.Contains("public sealed class TensorRtPluginFieldSummary", inventoryModels);
        Assert.Contains("public IReadOnlyList<TensorRtPluginFieldSummary> GetFieldSummaries", inventoryModels);
        Assert.Contains("The pointer value itself is never exposed", inventoryModels);
        Assert.Contains("This method scans the managed snapshot only. It does not call TensorRT", inventoryModels);
        Assert.Contains("summaries.Add(new TensorRtPluginFieldSummary(", inventoryModels);
        Assert.Contains("GetFieldSummaries(maxCreators: 1, maxFieldsPerCreator: 1)", ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs"));
        Assert.Contains("FieldSummary Creator=", ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs"));
        Assert.Contains("TensorRtPluginFieldSummary", ReadSource("docs", "articles", "zh-cn", "plugin-inventory-readonly-api.md"));
        Assert.Contains("不会暴露字段 data pointer", ReadSource("docs", "articles", "zh-cn", "plugin-inventory-readonly-api.md"));
        Assert.DoesNotContain("GetPluginCreatorList", inventoryModels + builderInventory + runtimeInventory + environmentProbe);
        Assert.DoesNotContain("PluginCreatorList", inventoryModels + builderInventory + runtimeInventory + environmentProbe);
        Assert.DoesNotContain("public IntPtr", inventoryModels + builderInventory + runtimeInventory + environmentProbe);
        Assert.DoesNotContain("public nint", inventoryModels + builderInventory + runtimeInventory + environmentProbe);
    }

    private static string ReadTensorRtManifest(string lineDirectory, string manifestName)
    {
        return ReadSource("native", "manifests", "tensorrt", lineDirectory, manifestName);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }

    private static string ExtractBetween(string text, string startToken, string endToken)
    {
        int start = text.IndexOf(startToken, StringComparison.Ordinal);
        Assert.True(start >= 0, $"Unable to find start token: {startToken}");

        int end = text.IndexOf(endToken, start, StringComparison.Ordinal);
        Assert.True(end > start, $"Unable to find end token: {endToken}");

        return text[start..end];
    }
}
