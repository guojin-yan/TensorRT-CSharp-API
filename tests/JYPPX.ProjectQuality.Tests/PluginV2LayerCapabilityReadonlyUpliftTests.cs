using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PluginV2LayerCapabilityReadonlyUpliftTests
{
    private static readonly string[] CommonEntryPointSuffixes =
    {
        "plugin_v2_layer_get_plugin_output_count",
        "plugin_v2_layer_get_capability_presence",
        "plugin_v2_layer_get_legacy_output_dimensions",
        "plugin_v2_layer_get_legacy_workspace_size",
        "plugin_v2_layer_supports_legacy_format",
        "plugin_v2_layer_get_output_data_type"
    };

    private static readonly string[] BroadcastEntryPointSuffixes =
    {
        "plugin_v2_layer_can_broadcast_input_across_batch",
        "plugin_v2_layer_is_output_broadcast_across_batch"
    };

    [Fact]
    public void ManifestsExposeEightQueriesForTensorRt8And10AndSixForTensorRt11()
    {
        foreach (string line in new[] { "8", "10" })
        {
            string manifest = ReadSource(
                "native", "manifests", "tensorrt", $"v{line}",
                $"trt{line}-plugin-v2-layer-capability-queries.manifest.json");
            using JsonDocument document = JsonDocument.Parse(manifest);

            Assert.Equal(8, document.RootElement.GetProperty("apis").GetArrayLength());
            foreach (string suffix in CommonEntryPointSuffixes.Concat(BroadcastEntryPointSuffixes))
            {
                Assert.Contains($"jyppx_trt{line}_{suffix}", manifest);
            }

            Assert.Contains("JYPPX_TensorRtLayer*", manifest);
            Assert.Contains("caller-owned", manifest);
            Assert.Contains($"JYPPX_TENSORRT_VERSION_MAJOR_NUM == {line}", manifest);
            Assert.Contains("\"manualOverride\": true", manifest);
            Assert.DoesNotContain("JYPPX_TensorRtPlugin*", manifest);
        }

        string trt11Manifest = ReadSource(
            "native", "manifests", "tensorrt", "v11",
            "trt11-plugin-v2-layer-capability-queries.manifest.json");
        using (JsonDocument document = JsonDocument.Parse(trt11Manifest))
        {
            Assert.Equal(6, document.RootElement.GetProperty("apis").GetArrayLength());
        }

        foreach (string suffix in CommonEntryPointSuffixes)
        {
            Assert.Contains($"jyppx_trt11_{suffix}", trt11Manifest);
        }
        foreach (string suffix in BroadcastEntryPointSuffixes)
        {
            Assert.DoesNotContain($"jyppx_trt11_{suffix}", trt11Manifest);
        }
        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11", trt11Manifest);
        Assert.Contains("\"manualOverride\": true", trt11Manifest);
    }

    [Fact]
    public void NativeBoundaryUsesLayerOwnerCopyBuffersAndVersionGuards()
    {
        string native = ReadSource(
            "native", "src", "tensorrt", "common", "plugin_v2_layer_metadata_snapshot.inc");

        Assert.Contains("JYPPX_TENSORRT_OBJECT_KIND_LAYER", native);
        Assert.Contains("nvinfer1::LayerType::kPLUGIN_V2", native);
        Assert.Contains("plugin_layer->getPlugin()", native);
        Assert.Contains("dynamic_cast<nvinfer1::IPluginV2Ext*>", native);
        Assert.Contains("new (std::nothrow)", native);
        Assert.Contains("capture_vendor_seh_exception_code", native);
        Assert.Contains("report_vendor_exception", native);
        Assert.Contains("input_is_broadcasted count must match", native);
        Assert.DoesNotContain("JYPPX_TensorRtPlugin**", native);

        foreach (string line in new[] { "8", "10", "11" })
        {
            string api = ReadSource("native", "src", "tensorrt", $"v{line}", "api.cpp");
            string header = ReadSource("native", "include", "jyppx", "tensorrt", $"trt{line}.h");
            Assert.Contains("plugin_v2_layer_metadata_snapshot.inc", api);
            foreach (string suffix in CommonEntryPointSuffixes)
            {
                Assert.Contains($"jyppx_trt{line}_{suffix}", header);
            }

            if (line == "11")
            {
                Assert.DoesNotContain("plugin_v2_layer_can_broadcast_input_across_batch", header);
                Assert.DoesNotContain("plugin_v2_layer_is_output_broadcast_across_batch", header);
            }
            else
            {
                foreach (string suffix in BroadcastEntryPointSuffixes)
                {
                    Assert.Contains($"jyppx_trt{line}_{suffix}", header);
                }
            }
        }
    }

    [Fact]
    public void DeferredPluginV2HistoryRemainsPresentAcrossVersionLines()
    {
        foreach ((string line, string fileName) in new[]
        {
            ("8", "trt8-cross-version-eleventh-batch-plugin-deferred.manifest.json"),
            ("10", "trt10-cross-version-second-batch-plugin-deferred.manifest.json"),
            ("11", "trt11-forty-sixth-batch-plugin-deferred.manifest.json")
        })
        {
            string deferred = ReadSource("native", "manifests", "tensorrt", $"v{line}", fileName);
            Assert.Contains($"trt{line}-plugin-v2-get-nb-outputs-deferred", deferred);
            Assert.Contains($"trt{line}-plugin-v2-get-output-dimensions-deferred", deferred);
            Assert.Contains($"trt{line}-plugin-v2-get-workspace-size-deferred", deferred);
            Assert.Contains($"trt{line}-plugin-v2-supports-format-deferred", deferred);
            Assert.Contains($"trt{line}-plugin-v2-ext-get-output-data-type-deferred", deferred);

            if (line is "8" or "10")
            {
                Assert.Contains($"trt{line}-plugin-v2-ext-can-broadcast-input-across-batch-deferred", deferred);
                Assert.Contains($"trt{line}-plugin-v2-ext-is-output-broadcast-across-batch-deferred", deferred);
            }
        }
    }

    [Fact]
    public void ManagedApiIsOwnerBoundAndPointerFree()
    {
        string metadata = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtPluginV2LayerMetadata.cs");
        string interop = ReadSource(
            "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.PluginV2LayerMetadata.cs");
        string publicSurface = metadata + interop;

        Assert.Contains("public int OutputCount", metadata);
        Assert.Contains("public bool HasExtCapability", metadata);
        Assert.Contains("public bool HasIoExtCapability", metadata);
        Assert.Contains("public bool HasDynamicExtCapability", metadata);
        Assert.Contains("public TensorRtDims GetPluginV2LegacyOutputDimensions", metadata);
        Assert.Contains("public ulong GetPluginV2LegacyWorkspaceSize", metadata);
        Assert.Contains("public bool SupportsPluginV2LegacyFormat", metadata);
        Assert.Contains("public TensorRtDataType GetPluginV2OutputDataType", metadata);
        Assert.Contains("public bool CanPluginV2BroadcastInputAcrossBatch", metadata);
        Assert.Contains("public bool IsPluginV2OutputBroadcastAcrossBatch", metadata);
        Assert.Contains("if (_ownerLease == null)", metadata);
        Assert.Contains("ReadUtf8Buffer", interop);
        Assert.Contains("EnsurePluginV2BroadcastLine", interop);
        Assert.Contains("NativeStatus.ThrowIfFailed", interop);
        Assert.DoesNotContain("public IntPtr", publicSurface);
        Assert.DoesNotContain("public nint", publicSurface);
        Assert.DoesNotContain("public SafeTensorRtObjectHandle", publicSurface);
    }

    [Fact]
    public void GeneratedSmokePackageConsumerAndCoverageCarryVersionBoundaries()
    {
        string generated = ReadSource(
            "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeMethodsTensorRt.Generated.g.cs");
        string smoke = ReadSource("smoke", "NetworkLayersSmokeRunner", "Program.cs");
        string packageConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        foreach (string line in new[] { "8", "10", "11" })
        {
            foreach (string suffix in CommonEntryPointSuffixes)
            {
                Assert.Contains($"jyppx_trt{line}_{suffix}", generated);
            }
        }
        foreach (string suffix in BroadcastEntryPointSuffixes)
        {
            Assert.Contains($"jyppx_trt8_{suffix}", generated);
            Assert.Contains($"jyppx_trt10_{suffix}", generated);
            Assert.DoesNotContain($"jyppx_trt11_{suffix}", generated);
        }

        Assert.Contains("PluginV2CapabilityQueryRejected=True", smoke);
        Assert.Contains("PluginV2BroadcastQueryUnsupported=True", smoke);
        Assert.Contains("GetPluginV2LegacyOutputDimensions", packageConsumer);
        Assert.Contains("CanPluginV2BroadcastInputAcrossBatch", packageConsumer);
        Assert.Contains("nameof(TensorRtPluginV2LayerMetadata.OutputCount)", packageConsumer);
        Assert.Contains("nameof(TensorRtLayer.IsPluginV2OutputBroadcastAcrossBatch)", packageConsumer);
        Assert.Contains("\"IPluginV2\",\"getNbOutputs\",\"IPluginV2::getNbOutputs\",\"plugin\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IPluginV2\",\"getOutputDimensions\",\"IPluginV2::getOutputDimensions\",\"plugin\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IPluginV2Ext\",\"getOutputDataType\",\"IPluginV2Ext::getOutputDataType\",\"plugin\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IPluginV2Ext\",\"canBroadcastInputAcrossBatch\",\"IPluginV2Ext::canBroadcastInputAcrossBatch\",\"plugin\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IPluginV2Ext\",\"isOutputBroadcastAcrossBatch\",\"IPluginV2Ext::isOutputBroadcastAcrossBatch\",\"plugin\",\"implemented-with-deferred-history\"", comparison);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
