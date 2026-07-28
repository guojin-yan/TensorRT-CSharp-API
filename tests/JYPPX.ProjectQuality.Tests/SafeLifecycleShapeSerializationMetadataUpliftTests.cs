using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class SafeLifecycleShapeSerializationMetadataUpliftTests
{
    [Fact]
    public void ManifestsDeclareNineVersionGuardedEntries()
    {
        int total = 0;
        foreach ((string line, string fileName, int expectedCount) in new[]
        {
            ("8", "trt8-safe-lifecycle-shape-serialization-metadata.manifest.json", 5),
            ("10", "trt10-safe-lifecycle-error-code-metadata.manifest.json", 2),
            ("11", "trt11-refitter-logger-error-code-metadata.manifest.json", 2)
        })
        {
            string manifest = ReadSource("native", "manifests", "tensorrt", $"v{line}", fileName);
            using JsonDocument document = JsonDocument.Parse(manifest);
            JsonElement apis = document.RootElement.GetProperty("apis");
            Assert.Equal(expectedCount, apis.GetArrayLength());
            total += expectedCount;
            Assert.All(apis.EnumerateArray(), api =>
            {
                Assert.Equal($"JYPPX_TENSORRT_VERSION_MAJOR_NUM == {line}", api.GetProperty("versionGuard").GetString());
                Assert.False(api.GetProperty("manualOverride").GetBoolean());
            });
        }

        Assert.Equal(9, total);
    }

    [Fact]
    public void NativeBoundariesOwnResultsCopyInputsAndContainVendorFailures()
    {
        string directBuild = ReadSource("native", "src", "tensorrt", "common", "direct_engine_build.inc");
        string errorMetadata = ReadSource("native", "src", "tensorrt", "common", "error_code_metadata.inc");
        string pluginPaths = ReadSource("native", "src", "tensorrt", "v8", "modules", "builder", "safe_plugin_serialization_paths.inc");
        string shapeBinding = ReadSource("native", "src", "tensorrt", "v8", "modules", "context", "legacy_shape_binding_setter.inc");
        string refitterLogger = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "refitter_logger_presence.inc");

        Assert.Contains("buildEngineWithConfig", directBuild);
        Assert.Contains("JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE", directBuild);
        Assert.Contains("create_handle_with_payload", directBuild);
        Assert.Contains("capture_vendor_seh_exception_code", directBuild);
        Assert.Contains("report_vendor_exception", directBuild);
        Assert.Contains("getPluginToSerialize", pluginPaths);
        Assert.Contains("copy_string_to_buffer", pluginPaths);
        Assert.Contains("setPluginsToSerialize", pluginPaths);
        Assert.Contains("path_count > 1000000", pluginPaths);
        Assert.Contains("setInputShapeBinding", shapeBinding);
        Assert.Contains("bindingIsInput", shapeBinding);
        Assert.Contains("isShapeBinding", shapeBinding);
        Assert.Contains("value_count > 1000000", shapeBinding);
        Assert.Contains("capture_vendor_seh_exception_code", pluginPaths + shapeBinding + refitterLogger);
        Assert.Contains("refitter_payload->getLogger() != nullptr", refitterLogger);
        Assert.Contains("EnumMax<nvinfer1::ErrorCode>()", errorMetadata);
    }

    [Fact]
    public void ManagedSurfaceIsOwnerBoundAndPointerFree()
    {
        string builder = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.Trt11BuildOutputs.cs");
        string config = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11PluginSerialization.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11Diagnostics.cs");
        string context = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.cs");
        string metadata = ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtErrorCodeMetadata.cs");
        string publicSurface = builder + config + context + metadata;

        Assert.Contains("public TensorRtEngine BuildEngineWithConfig", builder);
        Assert.Contains("TensorRT 8, 10, or 11", builder);
        Assert.Contains("TensorRT 8/10/11", config);
        Assert.Contains("public bool SetInputShapeBinding(int bindingIndex, IReadOnlyList<int> values)", context);
        Assert.Contains("public static class TensorRtErrorCodeMetadata", metadata);
        Assert.Contains("GetExclusiveUpperBound", metadata);
        Assert.Contains("IsDefinedRangeValue", metadata);
        Assert.DoesNotContain("public IntPtr", publicSurface);
        Assert.DoesNotContain("public nint", publicSurface);
        Assert.DoesNotContain("public SafeHandle", publicSurface);
    }

    [Fact]
    public void CoverageAliasesAreExplicitPriorityAndPreserveDeferredHistory()
    {
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        foreach (string key in new[]
        {
            "IBuilder::buildEngineWithConfig",
            "IBuilder::destroy",
            "IBuilderConfig::destroy",
            "ICudaEngine::destroy",
            "IExecutionContext::destroy",
            "IBuilderConfig::getPluginToSerialize",
            "IBuilderConfig::setPluginsToSerialize",
            "IExecutionContext::setInputShapeBinding",
            "IErrorRecorder::EnumMax",
            "IRefitter::getLogger",
            "IPluginV2DynamicExt::canBroadcastInputAcrossBatch",
            "IPluginV2DynamicExt::isOutputBroadcastAcrossBatch"
        })
        {
            Assert.Contains($"\"{key}\" = @(", script);
        }

        int matcherStart = script.IndexOf("function Find-MatchedManifestApis", StringComparison.Ordinal);
        int priorityStart = script.IndexOf("if ($interfaceKey -in @(", matcherStart, StringComparison.Ordinal);
        int heuristicStart = script.IndexOf("foreach ($candidate in $methodCandidates)", priorityStart, StringComparison.Ordinal);
        string priorityBlock = script.Substring(priorityStart, heuristicStart - priorityStart);
        Assert.Contains("IBuilder::buildEngineWithConfig", priorityBlock);
        Assert.Contains("IErrorRecorder::EnumMax", priorityBlock);
        Assert.Contains("IPluginV2DynamicExt::canBroadcastInputAcrossBatch", priorityBlock);

        foreach (string expected in new[]
        {
            "\"IBuilder\",\"buildEngineWithConfig\",\"IBuilder::buildEngineWithConfig\",\"builder\",\"implemented-with-deferred-history\"",
            "\"IBuilderConfig\",\"getPluginToSerialize\",\"IBuilderConfig::getPluginToSerialize\",\"builder\",\"implemented-with-deferred-history\"",
            "\"IExecutionContext\",\"setInputShapeBinding\",\"IExecutionContext::setInputShapeBinding\",\"engine-context\",\"implemented-with-deferred-history\"",
            "\"IErrorRecorder\",\"EnumMax\",\"IErrorRecorder::EnumMax\",\"diagnostics\",\"implemented-with-deferred-history\"",
            "\"IRefitter\",\"getLogger\",\"IRefitter::getLogger\",\"refitter\",\"implemented-with-deferred-history\"",
            "\"IPluginV2DynamicExt\",\"canBroadcastInputAcrossBatch\",\"IPluginV2DynamicExt::canBroadcastInputAcrossBatch\",\"plugin\",\"implemented-with-deferred-history\""
        })
        {
            Assert.Contains(expected, comparison);
        }
    }

    [Fact]
    public void UnsafeOwnershipAndCallbackRowsRemainDeferred()
    {
        string trt8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-twenty-third-batch-deferred-coverage.manifest.json");
        string trt8Network = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-ninth-batch-network-layer-deferred.manifest.json");
        string trt10Runtime = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-cross-version-fifth-batch-runtime-serialization-deferred.manifest.json");
        string trt11Plugin = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-forty-sixth-batch-plugin-deferred.manifest.json");

        Assert.Contains("trt8-builder-build-engine-with-config-deferred", trt8);
        Assert.Contains("trt8-execution-context-set-input-shape-binding-deferred", trt8);
        Assert.Contains("trt8-rnnv2-layer-set-weights-for-gate-deferred", trt8Network);
        Assert.Contains("trt10-runtime-deserialize-cuda-engine-v2-deferred", trt10Runtime);
        Assert.Contains("trt10-plugin-v3-one-runtime-set-tactic-deferred", trt10Runtime);
        Assert.Contains("trt11-plugin-v3-one-build-get-valid-tactics-deferred", trt11Plugin);
        Assert.Contains("trt11-plugin-v3-one-build-configure-plugin-deferred", trt11Plugin);
    }

    [Fact]
    public void GeneratedBindingsSmokeAndPackageConsumerCoverTheBatch()
    {
        string generated = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeMethodsTensorRt.Generated.g.cs");
        string networkSmoke = ReadSource("smoke", "NetworkBuilderSmokeRunner", "Program.cs");
        string pluginSmoke = ReadSource("smoke", "PluginSerializationPathsSmokeRunner", "Program.cs");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        foreach (string line in new[] { "8", "10", "11" })
        {
            Assert.Contains($"jyppx_trt{line}_error_code_get_exclusive_upper_bound", generated);
        }
        Assert.Contains("jyppx_trt8_builder_build_engine_with_config", generated);
        Assert.Contains("jyppx_trt10_builder_build_engine_with_config", generated);
        Assert.Contains("jyppx_trt8_execution_context_set_input_shape_binding", generated);
        Assert.Contains("jyppx_trt11_refitter_has_logger", generated);
        Assert.Contains("DirectEngineBuild=True", networkSmoke);
        Assert.Contains("ErrorCodeUpperBound=", networkSmoke);
        Assert.Contains("RefitterHasLogger=", networkSmoke);
        Assert.DoesNotContain("PluginSerializationPathsRequireTensorRt10Or11", pluginSmoke);
        Assert.Contains("TensorRtErrorCodeMetadata.GetExclusiveUpperBound", consumer);
        Assert.Contains("builder.BuildEngineWithConfig", consumer);
        Assert.Contains("context.SetInputShapeBinding", consumer);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
