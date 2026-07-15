using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PluginV3LayerMetadataReadonlyUpliftTests
{
    private static readonly string[] ExpectedEntryPointSuffixes =
    {
        "plugin_v3_layer_get_plugin_interface_info",
        "plugin_v3_layer_get_capability_presence",
        "plugin_v3_layer_get_core_plugin_name",
        "plugin_v3_layer_get_core_plugin_version",
        "plugin_v3_layer_get_core_plugin_namespace",
        "plugin_v3_layer_get_core_interface_info",
        "plugin_v3_layer_get_build_interface_info",
        "plugin_v3_layer_get_build_nb_outputs",
        "plugin_v3_layer_get_build_nb_tactics",
        "plugin_v3_layer_get_build_format_combination_limit",
        "plugin_v3_layer_get_build_timing_cache_id",
        "plugin_v3_layer_get_build_metadata_string",
        "plugin_v3_layer_get_runtime_interface_info"
    };

    [Fact]
    public void ManifestsExposeThirteenOwnerScopedQueriesForTensorRt10And11()
    {
        foreach (string line in new[] { "10", "11" })
        {
            string manifestPath = SourcePath(
                "native",
                "manifests",
                "tensorrt",
                $"v{line}",
                $"trt{line}-plugin-v3-layer-metadata-snapshot.manifest.json");
            string manifest = File.ReadAllText(manifestPath);
            using JsonDocument document = JsonDocument.Parse(manifest);

            JsonElement apis = document.RootElement.GetProperty("apis");
            Assert.Equal(ExpectedEntryPointSuffixes.Length, apis.GetArrayLength());
            foreach (string suffix in ExpectedEntryPointSuffixes)
            {
                Assert.Contains($"jyppx_trt{line}_{suffix}", manifest);
            }

            Assert.Contains("JYPPX_TensorRtLayer*", manifest);
            Assert.Contains("caller-owned", manifest);
            Assert.Contains($"JYPPX_TENSORRT_VERSION_MAJOR_NUM == {line}", manifest);
            Assert.DoesNotContain("JYPPX_TensorRtPlugin*", manifest);
            Assert.DoesNotContain("IPluginCapability*", manifest);
        }
    }

    [Fact]
    public void NativeBoundaryUsesLayerValidationCopyBuffersAndExceptionGuards()
    {
        string native = ReadSource(
            "native",
            "src",
            "tensorrt",
            "common",
            "plugin_v3_layer_metadata_snapshot.inc");

        Assert.Contains("JYPPX_TENSORRT_OBJECT_KIND_LAYER", native);
        Assert.Contains("nvinfer1::LayerType::kPLUGIN_V3", native);
        Assert.Contains("plugin_layer->getPlugin()", native);
        Assert.Contains("plugin.getCapabilityInterface", native);
        Assert.Contains("copy_interface_info_to_buffer", native);
        Assert.Contains("copy_string_to_buffer", native);
        Assert.Contains("capture_vendor_seh_exception_code", native);
        Assert.Contains("report_vendor_exception", native);
        Assert.Contains("JYPPX_STATUS_NOT_FOUND", native);
        Assert.DoesNotContain("JYPPX_TensorRtPlugin**", native);
        Assert.DoesNotContain("IPluginCapability**", native);

        foreach (string line in new[] { "10", "11" })
        {
            string api = ReadSource("native", "src", "tensorrt", $"v{line}", "api.cpp");
            Assert.Contains("plugin_v3_layer_metadata_snapshot.inc", api);

            string header = ReadSource("native", "include", "jyppx", "tensorrt", $"trt{line}.h");
            foreach (string suffix in ExpectedEntryPointSuffixes)
            {
                Assert.Contains($"jyppx_trt{line}_{suffix}", header);
            }
        }
    }

    [Fact]
    public void OriginalBorrowedPointerAndDescriptorDependentApisRemainDeferred()
    {
        string trt10PluginDeferred = ReadSource(
            "native", "manifests", "tensorrt", "v10", "trt10-cross-version-second-batch-plugin-deferred.manifest.json");
        string trt10RuntimeDeferred = ReadSource(
            "native", "manifests", "tensorrt", "v10", "trt10-cross-version-fifth-batch-runtime-serialization-deferred.manifest.json");
        string trt11PluginDeferred = ReadSource(
            "native", "manifests", "tensorrt", "v11", "trt11-forty-sixth-batch-plugin-deferred.manifest.json");
        string trt11RuntimeDeferred = ReadSource(
            "native", "manifests", "tensorrt", "v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json");

        foreach (string deferred in new[] { trt10PluginDeferred, trt11PluginDeferred })
        {
            Assert.Contains("plugin-v3-get-capability-interface-deferred", deferred);
            Assert.Contains("plugin-v3-one-core-get-plugin-name-deferred", deferred);
            Assert.Contains("plugin-v3-one-build-get-output-shapes-deferred", deferred);
            Assert.Contains("plugin-v3-one-build-configure-plugin-deferred", deferred);
        }

        foreach (string deferred in new[] { trt10RuntimeDeferred, trt11RuntimeDeferred })
        {
            Assert.Contains("plugin-v3-one-runtime-enqueue-deferred", deferred);
            Assert.Contains("plugin-v3-one-runtime-on-shape-change-deferred", deferred);
        }
    }

    [Fact]
    public void ManagedApiRequiresNetworkOwnerLeaseAndExposesPointerFreeSnapshots()
    {
        string metadata = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtPluginV3LayerMetadata.cs");
        string interop = ReadSource(
            "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.PluginV3LayerMetadata.cs");

        Assert.Contains("public sealed class TensorRtPluginV3LayerMetadata", metadata);
        Assert.Contains("public TensorRtPluginV3LayerMetadata GetPluginV3Metadata()", metadata);
        Assert.Contains("public bool TryGetPluginV3Metadata", metadata);
        Assert.Contains("if (_ownerLease == null)", metadata);
        Assert.Contains("TensorRtNetworkDefinition.GetLayer", metadata);
        Assert.Contains("public TensorRtPluginV3BuildMetadata? Build", metadata);
        Assert.Contains("public TensorRtPluginV3RuntimeMetadata? Runtime", metadata);
        Assert.Contains("EnsurePluginV3LayerMetadataLine", interop);
        Assert.Contains("TensorRtApiLine.TensorRt10", interop);
        Assert.Contains("TensorRtApiLine.TensorRt11", interop);
        Assert.Contains("ReadUtf8Buffer", interop);
        Assert.Contains("NativeStatus.ThrowIfFailed", interop);

        string publicSurface = metadata + interop;
        Assert.DoesNotContain("public IntPtr", publicSurface);
        Assert.DoesNotContain("public nint", publicSurface);
        Assert.DoesNotContain("public SafeTensorRtObjectHandle", publicSurface);
    }

    [Fact]
    public void GeneratedBindingsSmokePackageConsumerAndCoverageCarryTheUplift()
    {
        string generated = ReadSource(
            "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeMethodsTensorRt.Generated.g.cs");
        string networkSmoke = ReadSource("smoke", "NetworkLayersSmokeRunner", "Program.cs");
        string packageConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        foreach (string line in new[] { "10", "11" })
        {
            foreach (string suffix in ExpectedEntryPointSuffixes)
            {
                Assert.Contains($"jyppx_trt{line}_{suffix}", generated);
            }
        }

        Assert.Contains("PluginV3LayerMetadataRejected=True", networkSmoke);
        Assert.Contains("Func<TensorRtLayer, TensorRtPluginV3LayerMetadata> pluginV3LayerMetadata", packageConsumer);
        Assert.Contains("layer.TryGetPluginV3Metadata", packageConsumer);
        Assert.Contains("nameof(TensorRtPluginV3LayerMetadata)", packageConsumer);
        Assert.Contains("nameof(TensorRtLayer.GetPluginV3Metadata)", packageConsumer);
        Assert.Contains("\"IPluginV3Layer\",\"getPlugin\",\"IPluginV3Layer::getPlugin\",\"network-layer\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IPluginV3OneCore\",\"getPluginName\",\"IPluginV3OneCore::getPluginName\",\"plugin\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IPluginV3OneBuild\",\"getNbOutputs\",\"IPluginV3OneBuild::getNbOutputs\",\"plugin\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IPluginV3OneRuntime\",\"getInterfaceInfo\",\"IPluginV3OneRuntime::getInterfaceInfo\",\"runtime-serialization\",\"implemented-with-deferred-history\"", comparison);
    }

    private static string ReadSource(params string[] pathParts) => File.ReadAllText(SourcePath(pathParts));

    private static string SourcePath(params string[] pathParts) =>
        Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
}
