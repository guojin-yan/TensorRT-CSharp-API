using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PluginCreatorApiLanguageReadonlyTests
{
    [Fact]
    public void ManifestsPromotePluginCreatorApiLanguageAsScalarCopyOnly()
    {
        string builder10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-sixth-batch-plugin-registry-inventory.manifest.json");
        string builder11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-twenty-sixth-batch-plugin-registry-inventory.manifest.json");
        string global10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-seventh-batch-global-runtime-plugin-probe.manifest.json");
        string global11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-twenty-seventh-batch-global-runtime-plugin-probe.manifest.json");
        string capability10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-eighth-batch-builder-capability-plugin-registry.manifest.json");
        string capability11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-twenty-eighth-batch-builder-capability-plugin-registry.manifest.json");
        string runtime10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-runtime-local-plugin-registry-inventory.manifest.json");
        string runtime11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-runtime-local-plugin-registry-inventory.manifest.json");

        foreach (string manifest in new[] { builder10, builder11, global10, global11, capability10, capability11, runtime10, runtime11 })
        {
            Assert.Contains("\"out_api_language\", \"type\": \"int32_t*\", \"direction\": \"out\"", manifest);
            Assert.DoesNotContain("IPluginCreatorInterface*", manifest);
            Assert.DoesNotContain("IntPtr", manifest);
            Assert.DoesNotContain("nint", manifest);
        }

        Assert.Contains("trt10-builder-plugin-creator-get-api-language", builder10);
        Assert.Contains("trt10-builder-plugin-creator-lookup-get-api-language", builder10);
        Assert.Contains("trt11-builder-plugin-creator-get-api-language", builder11);
        Assert.Contains("trt11-builder-plugin-creator-lookup-get-api-language", builder11);
        Assert.Contains("trt10-global-plugin-creator-get-api-language", global10);
        Assert.Contains("trt10-global-plugin-creator-lookup-get-api-language", global10);
        Assert.Contains("trt11-global-plugin-creator-get-api-language", global11);
        Assert.Contains("trt11-global-plugin-creator-lookup-get-api-language", global11);
        Assert.Contains("trt10-builder-capability-plugin-creator-get-api-language", capability10);
        Assert.Contains("trt10-builder-capability-plugin-creator-lookup-get-api-language", capability10);
        Assert.Contains("trt11-builder-capability-plugin-creator-get-api-language", capability11);
        Assert.Contains("trt11-builder-capability-plugin-creator-lookup-get-api-language", capability11);
        Assert.Contains("trt10-runtime-plugin-creator-get-api-language", runtime10);
        Assert.Contains("trt10-runtime-plugin-creator-lookup-get-api-language", runtime10);
        Assert.Contains("trt11-runtime-plugin-creator-get-api-language", runtime11);
        Assert.Contains("trt11-runtime-plugin-creator-lookup-get-api-language", runtime11);
    }

    [Fact]
    public void NativeImplementationCopiesApiLanguageWithoutExportingBorrowedCreators()
    {
        string inventoryNative = ReadSource("native", "src", "tensorrt", "common", "plugin_registry_inventory.inc");
        string globalNative = ReadSource("native", "src", "tensorrt", "common", "global_runtime_plugin_probe.inc");
        string trt10Header = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string trt11Header = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");

        Assert.Contains("jyppx_trt_get_creator_api_language_with_seh_guard", inventoryNative);
        Assert.Contains("static_cast<int32_t>(creator->getAPILanguage())", inventoryNative);
        Assert.Contains("*out_api_language = -1", inventoryNative);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(builder_plugin_creator_get_api_language)", inventoryNative);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(builder_plugin_creator_lookup_get_api_language)", inventoryNative);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(runtime_plugin_creator_get_api_language)", inventoryNative);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(runtime_plugin_creator_lookup_get_api_language)", inventoryNative);
        Assert.Contains("JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_creator_get_api_language)", globalNative);
        Assert.Contains("JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_creator_lookup_get_api_language)", globalNative);
        Assert.Contains("JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_get_api_language)", globalNative);
        Assert.Contains("JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_lookup_get_api_language)", globalNative);

        foreach (string header in new[] { trt10Header, trt11Header })
        {
            Assert.Contains("plugin_creator_get_api_language", header);
            Assert.Contains("plugin_creator_lookup_get_api_language", header);
            Assert.Contains("int32_t* out_api_language", header);
        }
    }

    [Fact]
    public void ManagedSnapshotsExposeEnumAndKeepRawPointersOutOfPublicApi()
    {
        string inventoryModels = ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginRegistryInventory.cs");
        string builderInventory = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.PluginRegistryInventory.cs");
        string builderInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.PluginRegistryInventory.cs");
        string globalInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.GlobalRuntimePluginProbe.cs");
        string capabilityInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.BuilderCapabilityPluginRegistry.cs");
        string runtimeInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.RuntimePluginRegistryInventory.cs");
        string smoke = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");

        Assert.Contains("public TensorRtApiLanguage ApiLanguage { get; }", inventoryModels);
        Assert.Contains("creator.ApiLanguage", inventoryModels);
        Assert.Contains("GetBuilderPluginCreatorApiLanguage(Line, _handle, creatorIndex)", builderInventory);
        Assert.Contains("GetBuilderLookupPluginCreatorApiLanguage", builderInterop);
        Assert.Contains("GetGlobalLookupPluginCreatorApiLanguage", globalInterop);
        Assert.Contains("GetBuilderCapabilityLookupPluginCreatorApiLanguage", capabilityInterop);
        Assert.Contains("GetRuntimeLookupPluginCreatorApiLanguage", runtimeInterop);
        Assert.Contains("return TensorRtApiLanguage.Unknown;", builderInterop);
        Assert.Contains("return TensorRtApiLanguage.Unknown;", runtimeInterop);
        Assert.Contains("ApiLanguage=", smoke);
        Assert.Contains("first.ApiLanguage != firstSummary.ApiLanguage", smoke);

        string publicSources = string.Join(
            Environment.NewLine,
            inventoryModels,
            builderInventory,
            ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.PluginRegistryInventory.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.cs"));
        Assert.DoesNotContain("public IntPtr", publicSources);
        Assert.DoesNotContain("public nint", publicSources);
    }

    [Fact]
    public void GeneratedBindingsCarryPluginCreatorApiLanguageEntrypoints()
    {
        string nativeMethods = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeMethodsTensorRt.Generated.g.cs");
        string entryPointNames = ReadSource("src", "JYPPX.Shared", "Generated", "GeneratedEntryPointNames.g.cs");

        Assert.Contains("jyppx_trt10_builder_plugin_creator_get_api_language", nativeMethods);
        Assert.Contains("jyppx_trt11_builder_plugin_creator_lookup_get_api_language", nativeMethods);
        Assert.Contains("jyppx_trt10_global_plugin_creator_lookup_get_api_language", nativeMethods);
        Assert.Contains("jyppx_trt11_builder_capability_plugin_creator_get_api_language", nativeMethods);
        Assert.Contains("jyppx_trt10_runtime_plugin_creator_lookup_get_api_language", nativeMethods);
        Assert.Contains("jyppx_trt10_runtime_plugin_creator_lookup_get_interface_info", nativeMethods);
        Assert.Contains("jyppx_trt11_runtime_plugin_creator_lookup_get_field_name", nativeMethods);
        Assert.Contains("jyppx_trt8_runtime_plugin_creator_lookup_get_field_metadata", nativeMethods);
        Assert.Contains("out int out_api_language", nativeMethods);

        Assert.Contains("Trt10BuilderPluginCreatorGetApiLanguage", entryPointNames);
        Assert.Contains("Trt11BuilderPluginCreatorLookupGetApiLanguage", entryPointNames);
        Assert.Contains("Trt10GlobalPluginCreatorLookupGetApiLanguage", entryPointNames);
        Assert.Contains("Trt11BuilderCapabilityPluginCreatorGetApiLanguage", entryPointNames);
        Assert.Contains("Trt10RuntimePluginCreatorLookupGetApiLanguage", entryPointNames);
        Assert.Contains("Trt10RuntimePluginCreatorLookupGetInterfaceInfo", entryPointNames);
        Assert.Contains("Trt11RuntimePluginCreatorLookupGetFieldName", entryPointNames);
        Assert.Contains("Trt8RuntimePluginCreatorLookupGetFieldMetadata", entryPointNames);
    }

    [Fact]
    public void StreamReaderAndWriterCallbackOwnershipRemainsDeferred()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        string streamGate = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtStreamIoInterfaceInfoDesignGate.cs");
        string streamGateTests = ReadSource("tests", "JYPPX.ProjectQuality.Tests", "StreamIoInterfaceInfoDesignGateTests.cs");

        Assert.Contains("IStreamReader::getInterfaceInfo", comparison);
        Assert.Contains("IStreamReaderV2::getInterfaceInfo", comparison);
        Assert.Contains("IStreamWriter::getInterfaceInfo", comparison);
        Assert.Contains("trt10-stream-reader-get-interface-info-deferred", comparison);
        Assert.Contains("trt11-stream-writer-get-interface-info-deferred", comparison);
        Assert.Contains("direct IStreamReader, IStreamReaderV2, and IStreamWriter pointer access remains deferred by design.", streamGate);
        Assert.Contains("StreamReaderPointerExposed => false", streamGate);
        Assert.Contains("StreamWriterPointerExposed => false", streamGate);
        Assert.Contains("Assert.False(gate.StreamReaderPointerExposed)", streamGateTests);
        Assert.Contains("Assert.False(gate.StreamWriterPointerExposed)", streamGateTests);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
