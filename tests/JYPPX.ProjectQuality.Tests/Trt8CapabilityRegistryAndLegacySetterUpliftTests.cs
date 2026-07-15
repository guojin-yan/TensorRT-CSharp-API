using System;
using System.IO;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class Trt8CapabilityRegistryAndLegacySetterUpliftTests
{
    [Fact]
    public void ManifestsDeclareTwentySevenPointerFreeEntriesAndKeepDeferredHistory()
    {
        string capability = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-builder-capability-plugin-registry.manifest.json");
        string setters = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-legacy-scalar-and-rnnv2-setters.manifest.json");
        string deferred = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json") +
            ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-ninth-batch-network-layer-deferred.manifest.json") +
            ReadSource("native", "manifests", "tensorrt", "v8", "trt8-twenty-third-batch-deferred-coverage.manifest.json") +
            ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-eleventh-batch-plugin-deferred.manifest.json");

        Assert.Equal(17, CountOccurrences(capability, "\"entryPoint\""));
        Assert.Equal(10, CountOccurrences(setters, "\"entryPoint\""));
        Assert.DoesNotContain("IntPtr", capability + setters, StringComparison.Ordinal);
        Assert.DoesNotContain("void*", capability + setters, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt8_global_get_builder_plugin_registry_deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt8_rnn_v2_layer_set_weights_for_gate_deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt8_rnn_v2_layer_set_bias_for_gate_deferred", deferred, StringComparison.Ordinal);
    }

    [Fact]
    public void NativeSourceUsesCopyOutValidationAndExceptionGuards()
    {
        string capability = ReadSource("native", "src", "tensorrt", "v8", "modules", "plugin", "trt8_builder_capability_plugin_registry.inc");
        string scalar = ReadSource("native", "src", "tensorrt", "v8", "modules", "builder", "legacy_scalar_setters.inc");
        string rnn = ReadSource("native", "src", "tensorrt", "v8", "modules", "layers", "rnn_v2_setters.inc");
        string api = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");

        Assert.Contains("copy_string_to_buffer", capability, StringComparison.Ordinal);
        Assert.Contains("validate_index", capability, StringComparison.Ordinal);
        Assert.Contains("getBuilderPluginRegistry", capability, StringComparison.Ordinal);
        Assert.Contains("capture_vendor_seh_exception_code", capability + scalar + rnn, StringComparison.Ordinal);
        Assert.Contains("report_vendor_exception", capability + scalar + rnn, StringComparison.Ordinal);
        Assert.Contains("get_payload<nvinfer1::ITensor>", rnn, StringComparison.Ordinal);
        Assert.Contains("trt8_builder_capability_plugin_registry.inc", api, StringComparison.Ordinal);
        Assert.Contains("legacy_scalar_setters.inc", api, StringComparison.Ordinal);
        Assert.Contains("rnn_v2_setters.inc", api, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfacePreservesVersionDifferencesAndOwnerBoundTensorSetters()
    {
        string capability = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.BuilderCapabilityPluginRegistry.cs");
        string environment = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtEnvironmentProbe.cs");
        string builder = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtBuilder.Trt11BoundaryControls.cs");
        string config = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtBuilderConfig.cs");
        string rnn = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtLayer.Trt8RnnV2Diagnostics.cs");

        Assert.Contains("TensorRtApiLine.TensorRt8", capability, StringComparison.Ordinal);
        Assert.Contains("return \"IPluginCreator\"", capability, StringComparison.Ordinal);
        Assert.Contains("return TensorRtApiLanguage.Unknown", capability, StringComparison.Ordinal);
        Assert.Contains("GetBuilderCapabilityPluginCreatorTensorRtVersion", capability, StringComparison.Ordinal);
        Assert.Contains("GetBuilderCapabilityPluginRegistryInventory", environment, StringComparison.Ordinal);
        Assert.Contains("public void SetMaxBatchSizeCompatibility", builder, StringComparison.Ordinal);
        Assert.Contains("public void SetMaxWorkspaceSizeCompatibility", config, StringComparison.Ordinal);
        Assert.Contains("public void SetMinTimingIterationsCompatibility", config, StringComparison.Ordinal);
        Assert.Contains("public void SetRnnV2Operation", rnn, StringComparison.Ordinal);
        Assert.Contains("public void SetRnnV2CellState", rnn, StringComparison.Ordinal);
        Assert.Contains("tensor.Line != Line", rnn, StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", capability + environment + builder + config + rnn, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", capability + environment + builder + config + rnn, StringComparison.Ordinal);
    }

    [Fact]
    public void SmokeAndPackageConsumerExerciseTheNewSurface()
    {
        string builderSmoke = ReadSource("smoke", "NetworkBuilderSmokeRunner", "Program.cs");
        string registrySmoke = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");
        string packageConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        Assert.Contains("SetMaxBatchSizeCompatibility", builderSmoke, StringComparison.Ordinal);
        Assert.Contains("SetMaxWorkspaceSizeCompatibility", builderSmoke, StringComparison.Ordinal);
        Assert.Contains("SetMinTimingIterationsCompatibility", builderSmoke, StringComparison.Ordinal);
        Assert.Contains("TryGetBuilderCapabilityPluginRegistryInventory", registrySmoke, StringComparison.Ordinal);
        Assert.Contains("TensorRt8GlobalPluginRegistrySkipped=True", registrySmoke, StringComparison.Ordinal);
        Assert.DoesNotContain("TensorRt8GlobalAndCapabilityPluginRegistriesSkipped", registrySmoke, StringComparison.Ordinal);
        Assert.Contains("SetRnnV2Operation", packageConsumer, StringComparison.Ordinal);
        Assert.Contains("SetMaxBatchSizeCompatibility", packageConsumer, StringComparison.Ordinal);
    }

    [Fact]
    public void CoveragePromotesSafeRowsAndLeavesWeightLifetimeRowsDeferred()
    {
        string comparison = ReadArtifact("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        Assert.Contains("\"Global\",\"getBuilderPluginRegistry\",\"Global::getBuilderPluginRegistry\",\"global\",\"implemented-with-deferred-history\"", comparison, StringComparison.Ordinal);
        Assert.Contains("\"IPluginRegistry\",\"getBuilderSafePluginRegistry\",\"IPluginRegistry::getBuilderSafePluginRegistry\",\"plugin\",\"implemented-with-deferred-history\"", comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilder\",\"setMaxBatchSize\",\"IBuilder::setMaxBatchSize\",\"builder\",\"implemented-with-deferred-history\"", comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"setMaxWorkspaceSize\",\"IBuilderConfig::setMaxWorkspaceSize\",\"builder\",\"implemented-with-deferred-history\"", comparison, StringComparison.Ordinal);
        Assert.Contains("\"IRNNv2Layer\",\"setOperation\",\"IRNNv2Layer::setOperation\",\"network-layer\",\"implemented-with-deferred-history\"", comparison, StringComparison.Ordinal);
        Assert.Contains("\"IRNNv2Layer\",\"setWeightsForGate\",\"IRNNv2Layer::setWeightsForGate\",\"network-layer\",\"deferred-only\"", comparison, StringComparison.Ordinal);
        Assert.Contains("\"IRNNv2Layer\",\"setBiasForGate\",\"IRNNv2Layer::setBiasForGate\",\"network-layer\",\"deferred-only\"", comparison, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageAliasesPreferCapabilityRegistryEntriesOverDeferredHistory()
    {
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("\"Global::getBuilderPluginRegistry\" = @(\"id:*builder-capability-plugin-registry-exists\")", script, StringComparison.Ordinal);
        Assert.Contains("\"IPluginRegistry::getBuilderSafePluginRegistry\" = @(\"id:*builder-safe-plugin-registry-exists\", \"id:*builder-capability-plugin-registry-exists\")", script, StringComparison.Ordinal);

        int matcherStart = script.IndexOf("function Find-MatchedManifestApis", StringComparison.Ordinal);
        int explicitPriorityStart = script.IndexOf("if ($interfaceKey -in @(", matcherStart, StringComparison.Ordinal);
        int heuristicStart = script.IndexOf("foreach ($candidate in $methodCandidates)", explicitPriorityStart, StringComparison.Ordinal);
        Assert.True(matcherStart >= 0 && explicitPriorityStart > matcherStart && heuristicStart > explicitPriorityStart);

        string explicitPriorityBlock = script.Substring(explicitPriorityStart, heuristicStart - explicitPriorityStart);
        Assert.Contains("\"Global::getBuilderPluginRegistry\"", explicitPriorityBlock, StringComparison.Ordinal);
        Assert.Contains("\"IPluginRegistry::getBuilderSafePluginRegistry\"", explicitPriorityBlock, StringComparison.Ordinal);
        Assert.Contains("Find-ExplicitTensorRtInterfaceAliasApis", explicitPriorityBlock, StringComparison.Ordinal);
    }

    private static int CountOccurrences(string value, string marker)
    {
        int count = 0;
        int index = 0;
        while ((index = value.IndexOf(marker, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += marker.Length;
        }
        return count;
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(RepositoryPaths.Root, Path.Combine(parts)));
    }

    private static string ReadArtifact(params string[] parts)
    {
        return ReadSource(parts);
    }
}
