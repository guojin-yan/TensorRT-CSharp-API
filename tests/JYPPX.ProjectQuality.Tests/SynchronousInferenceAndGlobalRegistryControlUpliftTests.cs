using System;
using System.IO;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed partial class Trt8CapabilityRegistryAndLegacySetterUpliftTests
{
    [Fact]
    public void ManifestsDeclareEighteenSafeEntriesAndKeepDeferredHistory()
    {
        string trt8Execution = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-synchronous-inference.manifest.json");
        string trt8Registry = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-global-plugin-registry-copied-inventory.manifest.json");
        string trt10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-synchronous-inference-and-global-registry-control.manifest.json");
        string trt11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-synchronous-inference-and-global-registry-control.manifest.json");
        string manifests = trt8Execution + trt8Registry + trt10 + trt11;
        string deferred = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json") +
            ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-eleventh-batch-plugin-deferred.manifest.json") +
            ReadSource("native", "manifests", "tensorrt", "v8", "trt8-twenty-third-batch-deferred-coverage.manifest.json") +
            ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-third-batch-deferred-coverage.manifest.json") +
            ReadSource("native", "manifests", "tensorrt", "v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json");

        Assert.Equal(18, CountOccurrences(manifests, "\"entryPoint\""));
        Assert.Equal(3, CountOccurrences(trt8Execution, "\"entryPoint\""));
        Assert.Equal(11, CountOccurrences(trt8Registry, "\"entryPoint\""));
        Assert.Equal(2, CountOccurrences(trt10, "\"entryPoint\""));
        Assert.Equal(2, CountOccurrences(trt11, "\"entryPoint\""));
        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8", trt8Execution + trt8Registry, StringComparison.Ordinal);
        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10", trt10, StringComparison.Ordinal);
        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11", trt11, StringComparison.Ordinal);
        Assert.DoesNotContain("IntPtr", manifests, StringComparison.Ordinal);
        Assert.DoesNotContain("nint", manifests, StringComparison.Ordinal);
        Assert.DoesNotContain("SafeHandle", manifests, StringComparison.Ordinal);
        Assert.Contains("trt8-global-get-plugin-registry-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("trt8-execution-context-execute-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("trt8-execution-context-execute-v2-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("trt8-execution-context-enqueue-v2-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("trt10-execution-context-execute-v2-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("trt11-execution-context-execute-v2-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("trt8-plugin-registry-set-parent-search-enabled-deferred", deferred, StringComparison.Ordinal);
    }

    [Fact]
    public void NativeSourcesCollectBoundAddressesAndContainCppAndSehGuards()
    {
        string execution = ReadSource("native", "src", "tensorrt", "common", "execution_context_synchronous_inference.inc");
        string trt8Registry = ReadSource("native", "src", "tensorrt", "v8", "modules", "plugin", "trt8_global_plugin_registry_inventory.inc");
        string globalRegistry = ReadSource("native", "src", "tensorrt", "common", "global_runtime_plugin_probe.inc");
        string trt8Api = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11Api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");

        Assert.Contains("context->getTensorAddress(tensor_name)", execution, StringComparison.Ordinal);
        Assert.Contains("std::vector<void*> bindings", execution, StringComparison.Ordinal);
        Assert.Contains("engine.getNbBindings()", execution, StringComparison.Ordinal);
        Assert.Contains("engine.getNbIOTensors()", execution, StringComparison.Ordinal);
        Assert.Contains("capture_vendor_seh_exception_code", execution + trt8Registry + globalRegistry, StringComparison.Ordinal);
        Assert.Contains("report_vendor_exception", execution + trt8Registry + globalRegistry, StringComparison.Ordinal);
        Assert.Contains("copy_string_to_buffer", trt8Registry, StringComparison.Ordinal);
        Assert.Contains("validate_index", trt8Registry, StringComparison.Ordinal);
        Assert.Contains("setParentSearchEnabled", trt8Registry + globalRegistry, StringComparison.Ordinal);
        Assert.Contains("get_trt8_global_creator_field_count_body", trt8Registry, StringComparison.Ordinal);
        Assert.Contains("kMaximumTrt8GlobalPluginFieldCount", trt8Registry, StringComparison.Ordinal);
        Assert.Contains("#define JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR 8", trt8Api, StringComparison.Ordinal);
        Assert.Contains("#define JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR 10", trt10Api, StringComparison.Ordinal);
        Assert.Contains("#define JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR 11", trt11Api, StringComparison.Ordinal);
        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR", execution, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfaceIsPointerFreeAndEnqueueV2SynchronizesBeforeReturning()
    {
        string nativeBridge = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Inference", "NativeBridgeApi.SynchronousInference.cs");
        string executionContext = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.SynchronousInference.cs");
        string bindings = ReadSource("src", "JYPPX.TensorRtSharp", "Inference", "TensorRtInferenceBindings.cs");
        string environment = ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.cs");

        Assert.Contains("TensorRtApiLine.TensorRt8", nativeBridge, StringComparison.Ordinal);
        Assert.Contains("TensorRtApiLine.TensorRt10", nativeBridge, StringComparison.Ordinal);
        Assert.Contains("TensorRtApiLine.TensorRt11", nativeBridge, StringComparison.Ordinal);
        Assert.Contains("public void ExecuteV2()", executionContext, StringComparison.Ordinal);
        Assert.Contains("public void ExecuteLegacy(int batchSize)", executionContext, StringComparison.Ordinal);
        Assert.Contains("internal void EnqueueV2(CudaStream stream)", executionContext, StringComparison.Ordinal);
        Assert.Contains("public TensorRtInferenceExecutionSummary ExecuteV2", bindings, StringComparison.Ordinal);
        Assert.Contains("public TensorRtInferenceExecutionSummary ExecuteLegacy", bindings, StringComparison.Ordinal);
        Assert.Contains("public TensorRtInferenceExecutionSummary EnqueueV2AndSynchronize", bindings, StringComparison.Ordinal);
        Assert.Contains("_context.EnqueueV2(stream);\n        stream.Synchronize();", bindings.Replace("\r\n", "\n"), StringComparison.Ordinal);
        Assert.Contains("public static bool IsGlobalPluginRegistryParentSearchEnabled", environment, StringComparison.Ordinal);
        Assert.Contains("public static void SetGlobalPluginRegistryParentSearchEnabled", environment, StringComparison.Ordinal);
        Assert.Contains("public static bool TrySetGlobalPluginRegistryParentSearchEnabled", environment, StringComparison.Ordinal);
        Assert.Contains("parent-search readback mismatch", environment, StringComparison.Ordinal);
        string globalRegistryInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Plugins", "NativeBridgeApi.GlobalPluginRegistry.cs");
        Assert.Contains("IsOptionalTrt8GlobalCreatorFieldFailure", globalRegistryInterop, StringComparison.Ordinal);
        Assert.Contains("int? recursiveCreatorCount = line == TensorRtApiLine.TensorRt8", globalRegistryInterop, StringComparison.Ordinal);
        Assert.Contains("? null\n            : GetGlobalPluginRegistryRecursiveCreatorCount(line);", globalRegistryInterop.Replace("\r\n", "\n"), StringComparison.Ordinal);
        Assert.DoesNotContain("if (line == TensorRtApiLine.TensorRt8)\n        {\n            return 0;", globalRegistryInterop.Replace("\r\n", "\n"), StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", executionContext + bindings + environment, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", executionContext + bindings + environment, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", executionContext + bindings + environment, StringComparison.Ordinal);
    }

    [Fact]
    public void SmokeRunnersExerciseExecutionAndRestoredRegistryControl()
    {
        string inferenceSmoke = ReadSource("smoke", "InferenceBindingsSmokeRunner", "Program.cs");
        string registrySmoke = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");

        Assert.Contains("bindings.ExecuteV2", inferenceSmoke, StringComparison.Ordinal);
        Assert.Contains("bindings.EnqueueV2AndSynchronize", inferenceSmoke, StringComparison.Ordinal);
        Assert.Contains("LegacyExecute Skipped=True Reason=ExplicitBatchIdentityEngine", inferenceSmoke, StringComparison.Ordinal);
        Assert.Contains("EnsureOutputMatches(\"executeV2\"", inferenceSmoke, StringComparison.Ordinal);
        Assert.Contains("EnsureOutputMatches(\"enqueueV2\"", inferenceSmoke, StringComparison.Ordinal);
        Assert.Contains("ValidateGlobalParentSearchRoundTrip", registrySmoke, StringComparison.Ordinal);
        Assert.Contains("finally", registrySmoke, StringComparison.Ordinal);
        Assert.Contains("SetGlobalPluginRegistryParentSearchEnabled(line, original)", registrySmoke, StringComparison.Ordinal);
        Assert.DoesNotContain("TensorRt8GlobalPluginRegistrySkipped=True", registrySmoke, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageAliasesPreferImplementedEntriesAndRetainDeferredHistory()
    {
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("\"Global::getPluginRegistry\" = @(\"id:*global-plugin-registry-exists\")", script, StringComparison.Ordinal);
        Assert.Contains("\"IPluginRegistry::setParentSearchEnabled\" = @(\"id:*global-plugin-registry-set-parent-search-enabled\")", script, StringComparison.Ordinal);
        Assert.Contains("\"IExecutionContext::execute\" = @(\"id:*execution-context-execute-legacy-safe\")", script, StringComparison.Ordinal);
        Assert.Contains("\"IExecutionContext::executeV2\" = @(\"id:*execution-context-execute-v2-safe\")", script, StringComparison.Ordinal);
        Assert.Contains("\"IExecutionContext::enqueueV2\" = @(\"id:*execution-context-enqueue-v2-safe\")", script, StringComparison.Ordinal);
        Assert.Contains("\"Global::getPluginRegistry\" = @(\"id:*global-get-plugin-registry-deferred\")", script, StringComparison.Ordinal);
        Assert.Contains("\"IExecutionContext::executeV2\" = @(\"id:*execution-context-execute-v2-deferred\")", script, StringComparison.Ordinal);
        Assert.Contains("\"IPluginRegistry::setParentSearchEnabled\" = @(\"id:*plugin-registry-set-parent-search-enabled-deferred\")", script, StringComparison.Ordinal);

        int matcherStart = script.IndexOf("function Find-MatchedManifestApis", StringComparison.Ordinal);
        int explicitPriorityStart = script.IndexOf("if ($interfaceKey -in @(", matcherStart, StringComparison.Ordinal);
        int heuristicStart = script.IndexOf("foreach ($candidate in $methodCandidates)", explicitPriorityStart, StringComparison.Ordinal);
        Assert.True(matcherStart >= 0 && explicitPriorityStart > matcherStart && heuristicStart > explicitPriorityStart);

        string explicitPriorityBlock = script.Substring(explicitPriorityStart, heuristicStart - explicitPriorityStart);
        Assert.Contains("\"Global::getPluginRegistry\"", explicitPriorityBlock, StringComparison.Ordinal);
        Assert.Contains("\"IPluginRegistry::setParentSearchEnabled\"", explicitPriorityBlock, StringComparison.Ordinal);
        Assert.Contains("\"IExecutionContext::execute\"", explicitPriorityBlock, StringComparison.Ordinal);
        Assert.Contains("\"IExecutionContext::executeV2\"", explicitPriorityBlock, StringComparison.Ordinal);
        Assert.Contains("\"IExecutionContext::enqueueV2\"", explicitPriorityBlock, StringComparison.Ordinal);
        Assert.Contains("Find-ExplicitTensorRtInterfaceAliasApis", explicitPriorityBlock, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageRowsAreImplementedWithDeferredHistory()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("\"Global\",\"getPluginRegistry\",\"Global::getPluginRegistry\",\"global\",\"implemented-with-deferred-history\"", comparison, StringComparison.Ordinal);
        Assert.Contains("\"IPluginRegistry\",\"setParentSearchEnabled\",\"IPluginRegistry::setParentSearchEnabled\",\"plugin\",\"implemented-with-deferred-history\"", comparison, StringComparison.Ordinal);
        Assert.Contains("\"IExecutionContext\",\"execute\",\"IExecutionContext::execute\",\"engine-context\",\"implemented-with-deferred-history\"", comparison, StringComparison.Ordinal);
        Assert.Contains("\"IExecutionContext\",\"executeV2\",\"IExecutionContext::executeV2\",\"engine-context\",\"implemented-with-deferred-history\"", comparison, StringComparison.Ordinal);
        Assert.Contains("\"IExecutionContext\",\"enqueueV2\",\"IExecutionContext::enqueueV2\",\"engine-context\",\"implemented-with-deferred-history\"", comparison, StringComparison.Ordinal);
    }

}
