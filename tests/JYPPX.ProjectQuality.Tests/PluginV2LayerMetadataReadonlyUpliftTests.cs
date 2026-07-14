using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PluginV2LayerMetadataReadonlyUpliftTests
{
    [Fact]
    public void ManifestsExposeFiveOwnerScopedCopyQueriesPerTensorRtLine()
    {
        foreach (string line in new[] { "8", "10", "11" })
        {
            string manifest = ReadSource(
                "native",
                "manifests",
                "tensorrt",
                $"v{line}",
                $"trt{line}-plugin-v2-layer-metadata-snapshot.manifest.json");

            Assert.Contains($"trt{line}-plugin-v2-layer-get-plugin-type", manifest);
            Assert.Contains($"trt{line}-plugin-v2-layer-get-plugin-version", manifest);
            Assert.Contains($"trt{line}-plugin-v2-layer-get-plugin-namespace", manifest);
            Assert.Contains($"trt{line}-plugin-v2-layer-get-serialization-size", manifest);
            Assert.Contains($"trt{line}-plugin-v2-layer-get-tensor-rt-version", manifest);
            Assert.Contains("JYPPX_TensorRtLayer*", manifest);
            Assert.Contains("byte[]", manifest);
            Assert.Contains("out UIntPtr", manifest);
            Assert.Contains("out int", manifest);
            Assert.DoesNotContain("JYPPX_TensorRtPlugin*", manifest);
        }
    }

    [Fact]
    public void NativeBoundaryCopiesMetadataWithoutReturningBorrowedPluginPointers()
    {
        string native = ReadSource(
            "native",
            "src",
            "tensorrt",
            "common",
            "plugin_v2_layer_metadata_snapshot.inc");

        Assert.Contains("JYPPX_TENSORRT_OBJECT_KIND_LAYER", native);
        Assert.Contains("nvinfer1::LayerType::kPLUGIN_V2", native);
        Assert.Contains("plugin_layer->getPlugin()", native);
        Assert.Contains("plugin.getPluginType()", native);
        Assert.Contains("plugin.getPluginVersion()", native);
        Assert.Contains("plugin.getPluginNamespace()", native);
        Assert.Contains("plugin.getSerializationSize()", native);
        Assert.Contains("plugin.getTensorRTVersion()", native);
        Assert.Contains("copy_string_to_buffer", native);
        Assert.Contains("capture_vendor_seh_exception_code", native);
        Assert.Contains("report_vendor_exception", native);
        Assert.DoesNotContain("JYPPX_TensorRtPlugin**", native);
    }

    [Fact]
    public void RawPluginPointerDeferredHistoryRemainsTracked()
    {
        string trt8 = ReadSource(
            "native",
            "manifests",
            "tensorrt",
            "v8",
            "trt8-cross-version-ninth-batch-network-layer-deferred.manifest.json");
        string trt10 = ReadSource(
            "native",
            "manifests",
            "tensorrt",
            "v10",
            "trt10-cross-version-first-batch-network-compat.manifest.json");
        string trt11 = ReadSource(
            "native",
            "manifests",
            "tensorrt",
            "v11",
            "trt11-forty-fifth-batch-callback-deferred.manifest.json");

        Assert.Contains("trt8-plugin-v2-layer-get-plugin-deferred", trt8);
        Assert.Contains("trt10-plugin-v2-layer-get-plugin-deferred", trt10);
        Assert.Contains("trt11-plugin-v2-layer-get-plugin-deferred", trt11);
    }

    [Fact]
    public void ManagedApiRequiresNetworkOwnerLeaseAndExposesPointerFreeSnapshot()
    {
        string metadata = ReadSource(
            "src",
            "JYPPX.TensorRtSharp",
            "TensorRtPluginV2LayerMetadata.cs");
        string interop = ReadSource(
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop",
            "NativeBridgeApi.PluginV2LayerMetadata.cs");

        Assert.Contains("public sealed class TensorRtPluginV2LayerMetadata", metadata);
        Assert.Contains("public TensorRtPluginV2LayerMetadata GetPluginV2Metadata()", metadata);
        Assert.Contains("public bool TryGetPluginV2Metadata", metadata);
        Assert.Contains("if (_ownerLease == null)", metadata);
        Assert.Contains("TensorRtNetworkDefinition.GetLayer", metadata);
        Assert.Contains("public ulong SerializationSize", metadata);
        Assert.Contains("public int PackedTensorRtVersion", metadata);
        Assert.Contains("public byte PluginApiVersionTag", metadata);
        Assert.Contains("public int TensorRtVersion", metadata);
        Assert.Contains("GetPluginV2LayerString", interop);
        Assert.Contains("ReadUtf8Buffer", interop);
        Assert.Contains("NativeStatus.ThrowIfFailed", interop);

        string publicSurface = metadata + interop;
        Assert.DoesNotContain("public IntPtr", publicSurface);
        Assert.DoesNotContain("public nint", publicSurface);
    }

    [Fact]
    public void TensorRt8CreatorVersionFlowsThroughInventoryAndLookup()
    {
        string manifest = ReadSource(
            "native",
            "manifests",
            "tensorrt",
            "v8",
            "trt8-plugin-registry-inventory.manifest.json");
        string native = ReadSource(
            "native",
            "src",
            "tensorrt",
            "v8",
            "modules",
            "plugin",
            "trt8_plugin_registry_inventory.inc");
        string model = ReadSource(
            "src",
            "JYPPX.TensorRtSharp",
            "TensorRtPluginRegistryInventory.cs");
        string builder = ReadSource(
            "src",
            "JYPPX.TensorRtSharp",
            "TensorRtBuilder.PluginRegistryInventory.cs");
        string runtimeInterop = ReadSource(
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop",
            "NativeBridgeApi.RuntimePluginRegistryInventory.cs");

        Assert.Contains("trt8-builder-plugin-creator-get-tensor-rt-version", manifest);
        Assert.Contains("trt8-builder-plugin-creator-lookup-get-tensor-rt-version", manifest);
        Assert.Contains("trt8-runtime-plugin-creator-get-tensor-rt-version", manifest);
        Assert.Contains("trt8-runtime-plugin-creator-lookup-get-tensor-rt-version", manifest);
        Assert.Contains("creator->getTensorRTVersion()", native);
        Assert.Contains("jyppx_trt8_get_creator_tensor_rt_version_with_seh_guard", native);
        Assert.Contains("public int? TensorRtVersion", model);
        Assert.Contains("GetBuilderPluginCreatorTensorRtVersion", builder);
        Assert.Contains("GetRuntimeLookupPluginCreatorTensorRtVersion", runtimeInterop);
    }

    [Fact]
    public void GeneratedBindingsSmokeAndCoverageCarryTheUplift()
    {
        string generated = ReadSource(
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop",
            "Generated",
            "NativeMethodsTensorRt.Generated.g.cs");
        string networkSmoke = ReadSource("smoke", "NetworkLayersSmokeRunner", "Program.cs");
        string inventorySmoke = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");
        string packageConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string comparison = ReadSource(
            "artifacts",
            "interface-coverage",
            "tensorrt-interface-comparison.csv");

        Assert.Contains("jyppx_trt8_plugin_v2_layer_get_plugin_type", generated);
        Assert.Contains("jyppx_trt10_plugin_v2_layer_get_serialization_size", generated);
        Assert.Contains("jyppx_trt11_plugin_v2_layer_get_tensor_rt_version", generated);
        Assert.Contains("jyppx_trt8_builder_plugin_creator_get_tensor_rt_version", generated);
        Assert.Contains("PluginV2LayerMetadataRejected=True", networkSmoke);
        Assert.Contains("TensorRtVersion", inventorySmoke);
        Assert.Contains("Func<TensorRtLayer, TensorRtPluginV2LayerMetadata> pluginV2LayerMetadata", packageConsumer);
        Assert.Contains("layer.TryGetPluginV2Metadata", packageConsumer);
        Assert.Contains("nameof(TensorRtPluginCreatorInfo.TensorRtVersion)", packageConsumer);
        Assert.Contains("nameof(TensorRtPluginCreatorSummary.TensorRtVersion)", packageConsumer);
        Assert.Contains("\"IPluginV2\",\"getPluginType\",\"IPluginV2::getPluginType\",\"plugin\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IPluginV2Layer\",\"getPlugin\",\"IPluginV2Layer::getPlugin\",\"network-layer\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IPluginCreator\",\"getTensorRTVersion\",\"IPluginCreator::getTensorRTVersion\",\"plugin\",\"implemented-with-deferred-history\"", comparison);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
