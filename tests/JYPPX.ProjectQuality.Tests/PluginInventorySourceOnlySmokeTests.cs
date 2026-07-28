using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PluginInventorySourceOnlySmokeTests
{
    [Fact]
    public void PublicInventorySurfaceCopiesMetadataAndKeepsPluginCreatorPointersPrivate()
    {
        string inventoryModels = ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginRegistryInventory.cs");
        string environmentProbe = ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.cs");
        string builderInventory = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.PluginRegistryInventory.cs");
        string runtimeInventory = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.PluginRegistryInventory.cs");
        string smokeProgram = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");

        Assert.Contains("public IReadOnlyList<TensorRtPluginCreatorInfo> Creators { get; }", inventoryModels);
        Assert.Contains("public IReadOnlyList<TensorRtPluginFieldInfo> Fields { get; }", inventoryModels);
        Assert.Contains("public TensorRtApiLanguage ApiLanguage { get; }", inventoryModels);
        Assert.Contains("public int FieldCount { get; }", inventoryModels);
        Assert.Contains("public TensorRtPluginCreatorInfo? FindCreator", inventoryModels);
        Assert.Contains("public bool TryFindCreator", inventoryModels);
        Assert.Contains("public IReadOnlyList<TensorRtPluginCreatorSummary> GetCreatorSummaries", inventoryModels);
        Assert.Contains("public TensorRtPluginRegistryInventoryDiagnostics GetDiagnostics()", inventoryModels);
        Assert.Contains("This method does not call TensorRT, expose native creator pointers, or copy plugin field data payloads.", inventoryModels);
        Assert.Contains("CanPromoteRuntimeProof => false", inventoryModels);
        Assert.Contains("CanDeleteDeferredRecord => false", inventoryModels);

        Assert.Contains("public static bool IsGlobalPluginRegistryAvailable", environmentProbe);
        Assert.Contains("public static bool TryIsGlobalPluginRegistryAvailable", environmentProbe);
        Assert.Contains("public static bool TryGetGlobalPluginCreator", environmentProbe);
        Assert.Contains("public bool IsPluginRegistryAvailable", builderInventory);
        Assert.Contains("public TensorRtPluginRegistryInventory GetPluginRegistryInventory", builderInventory);
        Assert.Contains("public bool TryGetPluginCreator", builderInventory);
        Assert.Contains("public bool IsPluginRegistryAvailable", runtimeInventory);
        Assert.Contains("public TensorRtPluginRegistryInventory GetPluginRegistryInventory", runtimeInventory);
        Assert.Contains("public bool TryGetPluginCreator", runtimeInventory);

        Assert.Contains("TryIsGlobalPluginRegistryAvailable", smokeProgram);
        Assert.Contains("TryGetGlobalPluginCreator", smokeProgram);
        Assert.Contains("TryGetPluginRegistryInventory", smokeProgram);
        Assert.Contains("CreatorSummary Name=", smokeProgram);
        Assert.Contains("GetDiagnostics()", smokeProgram);
        Assert.Contains("FieldSummary Creator=", smokeProgram);

        string publicSources = string.Join(Environment.NewLine, inventoryModels, environmentProbe, builderInventory, runtimeInventory);
        Assert.DoesNotContain("public IntPtr", publicSources);
        Assert.DoesNotContain("public nint", publicSources);
        Assert.DoesNotContain("GetPluginCreatorList", publicSources);
        Assert.DoesNotContain("PluginCreatorList", publicSources);
    }

    [Fact]
    public void NativeAndManifestInventoryEntrypointsUseScalarAndCallerBufferCopyOut()
    {
        string manifest8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-plugin-registry-inventory.manifest.json");
        string manifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-sixth-batch-plugin-registry-inventory.manifest.json");
        string manifest11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-twenty-sixth-batch-plugin-registry-inventory.manifest.json");
        string runtime10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-runtime-local-plugin-registry-inventory.manifest.json");
        string runtime11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-runtime-local-plugin-registry-inventory.manifest.json");
        string nativeCommon = ReadSource("native", "src", "tensorrt", "common", "plugin_registry_inventory.inc");
        string native8 = ReadSource("native", "src", "tensorrt", "v8", "modules", "plugin", "trt8_plugin_registry_inventory.inc");
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string header11 = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");

        foreach (string manifest in new[] { manifest8, manifest10, manifest11, runtime10, runtime11 })
        {
            Assert.Contains("plugin-registry-exists", manifest);
            Assert.Contains("plugin-registry-get-creator-count", manifest);
            Assert.Contains("plugin-creator-get-name", manifest);
            Assert.Contains("plugin-creator-get-version", manifest);
            Assert.Contains("plugin-creator-get-namespace", manifest);
            Assert.Contains("plugin-creator-get-field-count", manifest);
            Assert.Contains("plugin-creator-get-field-name", manifest);
            Assert.Contains("plugin-creator-get-field-metadata", manifest);
            Assert.Contains("plugin-creator-lookup", manifest);
            Assert.Contains("plugin-creator-lookup-get-field-count", manifest);
            Assert.Contains("plugin-creator-lookup-get-field-name", manifest);
            Assert.Contains("plugin-creator-lookup-get-field-metadata", manifest);
            Assert.DoesNotContain("IPluginCreatorInterface*", manifest);
            Assert.DoesNotContain("IntPtr", manifest);
            Assert.DoesNotContain("nint", manifest);
        }

        foreach (string manifest in new[] { manifest10, manifest11, runtime10, runtime11 })
        {
            Assert.Contains("\"out_api_language\", \"type\": \"int32_t*\", \"direction\": \"out\"", manifest);
        }

        Assert.Contains("JYPPX_TRT_PLUGIN_FN(builder_plugin_registry_exists)", nativeCommon);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(builder_plugin_registry_get_creator_count)", nativeCommon);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(runtime_plugin_registry_get_creator_count)", nativeCommon);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(builder_plugin_creator_lookup_get_interface_info)", nativeCommon);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(runtime_plugin_creator_lookup_get_field_metadata)", nativeCommon);
        Assert.Contains("jyppx_trt_get_creator_api_language_with_seh_guard", nativeCommon);
        Assert.Contains("static_cast<int32_t>(creator->getAPILanguage())", nativeCommon);

        Assert.Contains("builder_payload->getPluginRegistry()", native8);
        Assert.Contains("runtime_payload->getPluginRegistry()", native8);
        Assert.Contains("registry->getPluginCreatorList(out_count)", native8);
        Assert.Contains("registry->getPluginCreator(plugin_name, plugin_version, plugin_namespace)", native8);
        Assert.Contains("creator->getPluginName()", native8);
        Assert.Contains("creator->getPluginVersion()", native8);
        Assert.Contains("creator->getPluginNamespace()", native8);
        Assert.Contains("creator->getFieldNames()", native8);
        Assert.Contains("copy_string_to_buffer", native8);

        foreach (string header in new[] { header8, header10, header11 })
        {
            Assert.Contains("plugin_registry_exists", header);
            Assert.Contains("plugin_registry_get_creator_count", header);
            Assert.Contains("plugin_creator_get_name", header);
            Assert.Contains("plugin_creator_get_version", header);
            Assert.Contains("plugin_creator_get_namespace", header);
            Assert.Contains("plugin_creator_get_field_count", header);
            Assert.Contains("plugin_creator_get_field_name", header);
            Assert.Contains("plugin_creator_get_field_metadata", header);
            Assert.Contains("plugin_creator_lookup", header);
            Assert.Contains("plugin_creator_lookup_get_field_metadata", header);
        }

        foreach (string header in new[] { header10, header11 })
        {
            Assert.Contains("int32_t* out_api_language", header);
        }
    }

    [Fact]
    public void HighRiskPluginOwnershipOperationsRemainDeferred()
    {
        string pluginOwnershipBoundary = ReadSource("docs", "articles", "zh-cn", "plugin-ownership-boundary.md");
        string trt8PluginDeferred = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-eleventh-batch-plugin-deferred.manifest.json");
        string trt10PluginDeferred = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-cross-version-second-batch-plugin-deferred.manifest.json");
        string trt11PluginDeferred = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-forty-sixth-batch-plugin-deferred.manifest.json");
        string trt11CoverageDeferred = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json");
        string builderInventory = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.PluginRegistryInventory.cs");
        string runtimeInventory = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.PluginRegistryInventory.cs");
        string environmentProbe = ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.cs");
        string smokeProgram = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");
        string allDeferred = string.Join(Environment.NewLine, trt8PluginDeferred, trt10PluginDeferred, trt11PluginDeferred, trt11CoverageDeferred);

        Assert.Contains("registry register / deregister", pluginOwnershipBoundary);
        Assert.Contains("load library / deregister library", pluginOwnershipBoundary);
        Assert.Contains("plugin resource acquire / release", pluginOwnershipBoundary);
        Assert.Contains("plugin instance create / clone / destroy", pluginOwnershipBoundary);
        Assert.Contains("Plugin V2/V3 callback trampoline", pluginOwnershipBoundary);
        Assert.Contains("不暴露 `public IntPtr` / `public nint` plugin creator", pluginOwnershipBoundary);

        Assert.Contains("plugin-creator-create-plugin-deferred", allDeferred);
        Assert.Contains("plugin-v2-clone-deferred", allDeferred);
        Assert.Contains("plugin-v2-enqueue-deferred", allDeferred);
        Assert.Contains("plugin-registry-register-creator-deferred", trt8PluginDeferred);
        Assert.Contains("plugin-registry-load-library-deferred", trt8PluginDeferred);
        Assert.Contains("acquire-plugin-resource-deferred", allDeferred);
        Assert.Contains("release-plugin-resource-deferred", allDeferred);

        string publicSources = string.Join(Environment.NewLine, builderInventory, runtimeInventory, environmentProbe, smokeProgram);
        Assert.DoesNotContain("LoadLibrary", publicSources);
        Assert.DoesNotContain("RegisterCreator", publicSources);
        Assert.DoesNotContain("CreatePlugin", publicSources);
        Assert.DoesNotContain("AcquirePluginResource", publicSources);
        Assert.DoesNotContain("ReleasePluginResource", publicSources);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
