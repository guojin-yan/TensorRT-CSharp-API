using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PluginLayerOwnerScopedFormatSerializationUpliftTests
{
    private static readonly string[] V2EntryPointSuffixes =
    {
        "plugin_v2_layer_copy_dynamic_format_support",
        "plugin_v2_layer_copy_io_ext_format_support"
    };

    private static readonly string[] V3EntryPointSuffixes =
    {
        "plugin_v3_layer_get_build_io_counts",
        "plugin_v3_layer_copy_build_output_data_types",
        "plugin_v3_layer_copy_build_aliased_inputs",
        "plugin_v3_layer_copy_build_current_format_support",
        "plugin_v3_layer_get_runtime_serialization_field_count",
        "plugin_v3_layer_get_runtime_serialization_field_name",
        "plugin_v3_layer_get_runtime_serialization_field_metadata"
    };

    [Fact]
    public void ManifestsDeclareTwentyOwnerScopedEntriesAcrossThreeVersionLines()
    {
        int total = 0;
        foreach (string line in new[] { "8", "10", "11" })
        {
            string manifest = ReadSource(
                "native", "manifests", "tensorrt", $"v{line}",
                $"trt{line}-plugin-layer-owner-scoped-format-serialization-queries.manifest.json");
            using JsonDocument document = JsonDocument.Parse(manifest);
            JsonElement apis = document.RootElement.GetProperty("apis");
            int expectedCount = line == "8" ? 2 : 9;
            Assert.Equal(expectedCount, apis.GetArrayLength());
            total += expectedCount;

            Assert.All(apis.EnumerateArray(), entry =>
            {
                Assert.Equal("caller-owned", entry.GetProperty("ownership").GetString());
                Assert.Equal($"JYPPX_TENSORRT_VERSION_MAJOR_NUM == {line}", entry.GetProperty("versionGuard").GetString());
                Assert.True(entry.GetProperty("manualOverride").GetBoolean());
            });
            Assert.Contains("JYPPX_TensorRtLayer*", manifest);
            Assert.Contains("out_required_count", manifest);
            Assert.DoesNotContain("void*", manifest);
            Assert.DoesNotContain("IntPtr", manifest);
            Assert.DoesNotContain("nint", manifest);
            Assert.DoesNotContain("SafeHandle", manifest);
        }
        Assert.Equal(20, total);
    }

    [Fact]
    public void NativeImplementationCopiesInsideOwnerScopeAndNeverReadsFieldData()
    {
        string native = ReadSource(
            "native", "src", "tensorrt", "common", "plugin_layer_owner_scoped_query_snapshots.inc");

        Assert.Contains("get_plugin_v2_layer", native);
        Assert.Contains("get_plugin_v3_layer", native);
        Assert.Contains("resolve_plugin_v3_target", native);
        Assert.Contains("supportsFormatCombination", native);
        Assert.Contains("getOutputDataTypes", native);
        Assert.Contains("getAliasedInput", native);
        Assert.Contains("getFieldsToSerialize", native);
        Assert.Contains("copy_string_to_buffer", native);
        Assert.Contains("capture_vendor_seh_exception_code", native);
        Assert.Contains("report_vendor_exception", native);
        Assert.Contains("field.data == nullptr", native);
        Assert.DoesNotContain("memcpy(field.data", native);
        Assert.DoesNotContain("return field.data", native);

        string publicNativeHeaders = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h") +
            ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        Assert.DoesNotContain("IPluginV3OneRuntime", publicNativeHeaders);

        foreach (string line in new[] { "8", "10", "11" })
        {
            string api = ReadSource("native", "src", "tensorrt", $"v{line}", "api.cpp");
            Assert.Contains("plugin_layer_owner_scoped_query_snapshots.inc", api);
        }
    }

    [Fact]
    public void ManagedSurfaceIsTypedPointerFreeAndOwnerBound()
    {
        string model = ReadSource(
            "src", "JYPPX.TensorRtSharp", "TensorRtPluginLayerOwnerScopedQuerySnapshots.cs");
        string interop = ReadSource(
            "src", "JYPPX.TensorRtSharp", "Internal", "Interop",
            "NativeBridgeApi.PluginLayerOwnerScopedQuerySnapshots.cs");
        string publicSurface = model + interop;

        Assert.Contains("public sealed class TensorRtPluginFormatSupportSnapshot", model);
        Assert.Contains("public sealed class TensorRtPluginV3BuildIoSnapshot", model);
        Assert.Contains("public sealed class TensorRtPluginV3SerializationFieldInventory", model);
        Assert.Contains("public IReadOnlyList<TensorRtDataType> OutputDataTypes", model);
        Assert.Contains("public IReadOnlyList<int> AliasedInputIndices", model);
        Assert.Contains("public IReadOnlyList<TensorRtPluginFieldInfo> Fields", model);
        Assert.Contains("PointerFreeCopiedInventory => true", model);
        Assert.Contains("if (_ownerLease == null)", model);
        Assert.Contains("TryGetPluginV2DynamicFormatSupportSnapshot", model);
        Assert.Contains("TryGetPluginV3BuildIoSnapshot", model);
        Assert.Contains("TryGetPluginV3RuntimeSerializationFields", model);
        Assert.Contains("ValidatePluginIoCounts", interop);
        Assert.Contains("MaxPluginIoElementCount = 1_000_000", interop);
        Assert.DoesNotContain("public IntPtr", publicSurface);
        Assert.DoesNotContain("public nint", publicSurface);
        Assert.DoesNotContain("public SafeHandle", publicSurface);
    }

    [Fact]
    public void CoverageAliasesPreferSafeEntriesAndPreserveDeferredHistory()
    {
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        foreach (string key in new[]
        {
            "IPluginV2DynamicExt::supportsFormatCombination",
            "IPluginV2IOExt::supportsFormatCombination",
            "IPluginV3OneBuild::getAliasedInput",
            "IPluginV3OneBuild::getOutputDataTypes",
            "IPluginV3OneBuild::supportsFormatCombination",
            "IPluginV3OneRuntime::getFieldsToSerialize"
        })
        {
            Assert.Contains($"\"{key}\" = @(", script);
        }

        int matcherStart = script.IndexOf("function Find-MatchedManifestApis", StringComparison.Ordinal);
        int priorityStart = script.IndexOf("if ($interfaceKey -in @(", matcherStart, StringComparison.Ordinal);
        int heuristicStart = script.IndexOf("foreach ($candidate in $methodCandidates)", priorityStart, StringComparison.Ordinal);
        Assert.True(matcherStart >= 0 && priorityStart > matcherStart && heuristicStart > priorityStart);
        string priorityBlock = script.Substring(priorityStart, heuristicStart - priorityStart);
        Assert.Contains("IPluginV2DynamicExt::supportsFormatCombination", priorityBlock);
        Assert.Contains("IPluginV3OneRuntime::getFieldsToSerialize", priorityBlock);

        foreach (string expected in new[]
        {
            "\"IPluginV2DynamicExt\",\"supportsFormatCombination\",\"IPluginV2DynamicExt::supportsFormatCombination\",\"plugin\",\"implemented-with-deferred-history\"",
            "\"IPluginV2IOExt\",\"supportsFormatCombination\",\"IPluginV2IOExt::supportsFormatCombination\",\"plugin\",\"implemented-with-deferred-history\"",
            "\"IPluginV3OneBuild\",\"getOutputDataTypes\",\"IPluginV3OneBuild::getOutputDataTypes\",\"plugin\",\"implemented-with-deferred-history\"",
            "\"IPluginV3OneRuntime\",\"getFieldsToSerialize\",\"IPluginV3OneRuntime::getFieldsToSerialize\",\"runtime-serialization\",\"implemented-with-deferred-history\""
        })
        {
            Assert.Contains(expected, comparison);
        }
    }

    [Fact]
    public void UnsafePluginCallbacksAndDescriptorLifetimesRemainDeferred()
    {
        foreach ((string line, string file) in new[]
        {
            ("8", "trt8-cross-version-eleventh-batch-plugin-deferred.manifest.json"),
            ("10", "trt10-cross-version-second-batch-plugin-deferred.manifest.json"),
            ("11", "trt11-forty-sixth-batch-plugin-deferred.manifest.json")
        })
        {
            string deferred = ReadSource("native", "manifests", "tensorrt", $"v{line}", file);
            Assert.Contains($"trt{line}-plugin-v2-dynamic-ext-get-output-dimensions-deferred", deferred);
            Assert.Contains($"trt{line}-plugin-v2-dynamic-ext-get-workspace-size-deferred", deferred);
            Assert.Contains($"trt{line}-plugin-v2-dynamic-ext-enqueue-deferred", deferred);
        }

        foreach ((string line, string buildFile, string runtimeFile) in new[]
        {
            (
                "10",
                "trt10-cross-version-second-batch-plugin-deferred.manifest.json",
                "trt10-cross-version-fifth-batch-runtime-serialization-deferred.manifest.json"),
            (
                "11",
                "trt11-forty-sixth-batch-plugin-deferred.manifest.json",
                "trt11-twenty-third-batch-deferred-coverage.manifest.json")
        })
        {
            string buildDeferred = ReadSource("native", "manifests", "tensorrt", $"v{line}", buildFile);
            string runtimeDeferred = ReadSource("native", "manifests", "tensorrt", $"v{line}", runtimeFile);
            Assert.Contains($"trt{line}-plugin-v3-one-build-get-output-shapes-deferred", buildDeferred);
            Assert.Contains($"trt{line}-plugin-v3-one-build-get-valid-tactics-deferred", buildDeferred);
            Assert.Contains($"trt{line}-plugin-v3-one-build-get-workspace-size-deferred", buildDeferred);
            Assert.Contains($"trt{line}-plugin-v3-one-runtime-enqueue-deferred", runtimeDeferred);
            Assert.Contains($"trt{line}-plugin-v3-one-runtime-attach-to-context-deferred", runtimeDeferred);
        }
    }

    [Fact]
    public void GeneratedBindingsSmokeAndPackageConsumerCoverAllNewSurfaces()
    {
        string generated = ReadSource(
            "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated",
            "NativeMethodsTensorRt.Generated.g.cs");
        string smoke = ReadSource("smoke", "NetworkLayersSmokeRunner", "Program.cs");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        foreach (string line in new[] { "8", "10", "11" })
        {
            foreach (string suffix in V2EntryPointSuffixes)
            {
                Assert.Contains($"jyppx_trt{line}_{suffix}", generated);
            }
        }
        foreach (string line in new[] { "10", "11" })
        {
            foreach (string suffix in V3EntryPointSuffixes)
            {
                Assert.Contains($"jyppx_trt{line}_{suffix}", generated);
            }
        }

        Assert.Contains("PluginV2OwnerScopedFormatQueriesRejected=True", smoke);
        Assert.Contains("PluginV3OwnerScopedQuerySnapshotsRejected=True", smoke);
        Assert.Contains("GetPluginV2DynamicFormatSupportSnapshot", consumer);
        Assert.Contains("GetPluginV3BuildIoSnapshot", consumer);
        Assert.Contains("GetPluginV3RuntimeSerializationFields", consumer);
        Assert.Contains("nameof(TensorRtPluginV3SerializationFieldInventory.PointerFreeCopiedInventory)", consumer);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
