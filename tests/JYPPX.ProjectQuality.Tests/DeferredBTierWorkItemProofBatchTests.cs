using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredBTierWorkItemProofBatchTests
{
    [Fact]
    public void AllCurrentBTierWorkItemsHaveSafeAlternativeProofWithoutDeletingDeferredHistory()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredReadOnlyApiCandidatePlan.ps1"), "-IncludeMediumRisk", "-MaxItems", "60");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierProofClosureDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierAliasProofClosureRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierImplementationWorkPackage.ps1"));

        string packageJson = ReadSource("artifacts", "interface-coverage", "deferred-btier-implementation-work-package.json");
        using JsonDocument package = JsonDocument.Parse(packageJson);

        JsonElement root = package.RootElement;
        Assert.Equal("deferred-btier-implementation-work-package", root.GetProperty("recordKind").GetString());
        Assert.Equal("source-quality-proof-closed", root.GetProperty("workPackageState").GetString());
        Assert.Equal(51, root.GetProperty("closedWorkItemCount").GetInt32());
        Assert.Equal(0, root.GetProperty("remainingWorkItemCount").GetInt32());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("canDeleteDeferredRecords").GetBoolean());

        Dictionary<string, JsonElement> workItems = root.GetProperty("workItems")
            .EnumerateArray()
            .ToDictionary(static item => item.GetProperty("workItemId").GetString()!, static item => item);

        Assert.Equal(ExpectedWorkItemIds, workItems.Keys);
        Assert.Equal("IBuilderConfig::getAvgTimingIterations", workItems["btier-018"].GetProperty("interface").GetString());
        Assert.Equal("IParser::getError", workItems["btier-019"].GetProperty("interface").GetString());
        Assert.Equal("IParserRefitter::getError", workItems["btier-021"].GetProperty("interface").GetString());
        Assert.Equal("IBuilder::getMaxBatchSize", workItems["btier-022"].GetProperty("interface").GetString());
        Assert.Equal("IParser::getError", workItems["btier-040"].GetProperty("interface").GetString());
        Assert.Equal("IBinaryProtoBlob::getData", workItems["btier-041"].GetProperty("interface").GetString());
        Assert.Equal("IBinaryProtoBlob::getDataType", workItems["btier-042"].GetProperty("interface").GetString());
        Assert.Equal("IBinaryProtoBlob::getDimensions", workItems["btier-043"].GetProperty("interface").GetString());
        Assert.Equal("IUffParser::getUffRequiredVersionMajor", workItems["btier-044"].GetProperty("interface").GetString());
        Assert.Equal("IUffParser::getUffRequiredVersionMinor", workItems["btier-045"].GetProperty("interface").GetString());
        Assert.Equal("IUffParser::getUffRequiredVersionPatch", workItems["btier-046"].GetProperty("interface").GetString());
        Assert.Equal("IBuilderConfig::getTilingOptimizationLevel", workItems["btier-047"].GetProperty("interface").GetString());
        Assert.Equal("ICudaEngine::hasImplicitBatchDimension", workItems["btier-048"].GetProperty("interface").GetString());
        Assert.Equal("IExecutionContext::getNvtxVerbosity", workItems["btier-049"].GetProperty("interface").GetString());
        Assert.Equal("IParser::getError", workItems["btier-050"].GetProperty("interface").GetString());
        Assert.Equal("IParserRefitter::getError", workItems["btier-051"].GetProperty("interface").GetString());

        string manifests = string.Join(
            Environment.NewLine,
            ReadSources("native", "manifests", "tensorrt", "v8", "*.json"),
            ReadSources("native", "manifests", "tensorrt", "v10", "*.json"),
            ReadSources("native", "manifests", "tensorrt", "v11", "*.json"));
        string coverage = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-coverage.csv");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        string nativeHeaders = ReadSources("native", "include", "jyppx", "tensorrt", "*.h");
        string nativeSources = ReadSources("native", "src", "tensorrt", "*.cpp", "*.inc");
        string managedApi = ReadSources("src", "JYPPX.TensorRtSharp", "*.cs");
        string focusedManagedApi = string.Join(
            Environment.NewLine,
            ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.Trt11BoundaryControls.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11RuntimeControls.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Layers", "TensorRtLayer.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtProfiler.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserRefitter.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.CallbackInterfaceInfo.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.DeploymentMetadata.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11BoundaryControls.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11RuntimeControls.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.OnnxParserSupport.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.ParserRefitterDiagnostics.cs"));
        string projectQualityTests = ReadSources("tests", "JYPPX.ProjectQuality.Tests", "*.cs");
        string manualDesignGroups = ReadSource("docs", "articles", "zh-cn", "deferred-manual-design-groups.md");
        string latestClosureArticle = ReadSource("docs", "articles", "zh-cn", "deferred-btier-46-50-proof-closure.md");

        foreach ((string id, JsonElement item) in workItems)
        {
            Assert.Equal("B - safe-alternative-or-alias", item.GetProperty("safetyTier").GetString());
            Assert.Equal("source-quality-proof-closed", item.GetProperty("workItemState").GetString());
            Assert.Equal("artifacts/interface-coverage/deferred-btier-work-item-proof-closure-ledger.json", item.GetProperty("closureProofRecord").GetString());
            Assert.False(item.GetProperty("canDeleteDeferredRecord").GetBoolean());
            Assert.False(item.GetProperty("canPromoteReleaseProof").GetBoolean());
            Assert.False(item.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(item.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());

            string[] safeAlternativeIds = Strings(item, "safeAlternativeManifestIds");
            string[] deferredHistoryIds = Strings(item, "deferredHistoryManifestIds");
            Assert.NotEmpty(safeAlternativeIds);
            Assert.NotEmpty(deferredHistoryIds);

            foreach (string manifestId in safeAlternativeIds)
            {
                Assert.Contains(manifestId, manifests, StringComparison.Ordinal);
                Assert.Contains(manifestId, coverage + comparison, StringComparison.Ordinal);
            }

            foreach (string manifestId in deferredHistoryIds)
            {
                Assert.Contains(manifestId, manifests, StringComparison.Ordinal);
                Assert.Contains(manifestId, coverage + comparison, StringComparison.Ordinal);
                Assert.Contains("deferred", manifestId, StringComparison.Ordinal);
            }

            if (int.Parse(id["btier-".Length..]) <= 45)
            {
                Assert.Contains(id, manualDesignGroups, StringComparison.Ordinal);
            }
            else
            {
                Assert.Contains(id, latestClosureArticle, StringComparison.Ordinal);
            }
        }

        Assert.Contains("\"IExecutionContext\",\"getName\",\"IExecutionContext::getName\",\"engine-context\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IProfiler\",\"getInterfaceInfo\",\"IProfiler::getInterfaceInfo\",\"diagnostics\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"ICudaEngine\",\"getProfileShape\",\"ICudaEngine::getProfileShape\",\"engine-context\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"ILayer\",\"getInput\",\"ILayer::getInput\",\"network-layer\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilder\",\"getMaxDLABatchSize\",\"IBuilder::getMaxDLABatchSize\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilder\",\"getMaxThreads\",\"IBuilder::getMaxThreads\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilder\",\"isNetworkSupported\",\"IBuilder::isNetworkSupported\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"canRunOnDLA\",\"IBuilderConfig::canRunOnDLA\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"getAvgTimingIterations\",\"IBuilderConfig::getAvgTimingIterations\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"getDefaultDeviceType\",\"IBuilderConfig::getDefaultDeviceType\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"getDeviceType\",\"IBuilderConfig::getDeviceType\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"getDLACore\",\"IBuilderConfig::getDLACore\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"getL2LimitForTiling\",\"IBuilderConfig::getL2LimitForTiling\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"getMaxNbTactics\",\"IBuilderConfig::getMaxNbTactics\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"getQuantizationFlag\",\"IBuilderConfig::getQuantizationFlag\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"getQuantizationFlags\",\"IBuilderConfig::getQuantizationFlags\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"getFlag\",\"IBuilderConfig::getFlag\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"getFlags\",\"IBuilderConfig::getFlags\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"getMaxWorkspaceSize\",\"IBuilderConfig::getMaxWorkspaceSize\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"getMinTimingIterations\",\"IBuilderConfig::getMinTimingIterations\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"ICudaEngine\",\"getHardwareCompatibilityLevel\",\"ICudaEngine::getHardwareCompatibilityLevel\",\"engine-context\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"ICudaEngine\",\"getProfileDimensions\",\"ICudaEngine::getProfileDimensions\",\"engine-context\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IExecutionContext\",\"getNvtxVerbosity\",\"IExecutionContext::getNvtxVerbosity\",\"engine-context\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IParser\",\"getError\",\"IParser::getError\",\"onnx-parser\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IParser\",\"isSubgraphSupported\",\"IParser::isSubgraphSupported\",\"onnx-parser\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IParserRefitter\",\"getError\",\"IParserRefitter::getError\",\"onnx-parser\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);
        Assert.Contains("\"IBuilder\",\"getMaxBatchSize\",\"IBuilder::getMaxBatchSize\",\"builder\",\"implemented-with-deferred-history\"", coverage + comparison, StringComparison.Ordinal);

        foreach (string nativeEntryPoint in NativeEntryPoints)
        {
            Assert.Contains(nativeEntryPoint, nativeHeaders, StringComparison.Ordinal);
            Assert.Contains(nativeEntryPoint, managedApi, StringComparison.Ordinal);
        }

        foreach (string nativeSourceToken in NativeSourceTokens)
        {
            Assert.Contains(nativeSourceToken, nativeSources, StringComparison.Ordinal);
        }

        foreach (string wrapperToken in WrapperTokens)
        {
            Assert.Contains(wrapperToken, managedApi, StringComparison.Ordinal);
        }

        foreach (string testToken in ExistingProofTokens)
        {
            Assert.Contains(testToken, projectQualityTests, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("public IntPtr", focusedManagedApi, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", focusedManagedApi, StringComparison.Ordinal);
        Assert.Contains("B-tier 前 12 项 proof 批量收口", manualDesignGroups, StringComparison.Ordinal);
        Assert.Contains("B-tier 后 12 项 proof 批量收口", manualDesignGroups, StringComparison.Ordinal);
        Assert.Contains("B-tier 第三批 proof 批量收口", manualDesignGroups, StringComparison.Ordinal);
        Assert.Contains("B-tier 第四批 proof 批量收口", manualDesignGroups, StringComparison.Ordinal);
        Assert.Contains("不能删除 deferred history", manualDesignGroups, StringComparison.Ordinal);
        Assert.Contains("不是 runtime proof", manualDesignGroups, StringComparison.Ordinal);
    }

    private static readonly string[] ExpectedWorkItemIds =
    {
        "btier-001",
        "btier-002",
        "btier-003",
        "btier-004",
        "btier-005",
        "btier-006",
        "btier-007",
        "btier-008",
        "btier-009",
        "btier-010",
        "btier-011",
        "btier-012",
        "btier-013",
        "btier-014",
        "btier-015",
        "btier-016",
        "btier-017",
        "btier-018",
        "btier-019",
        "btier-020",
        "btier-021",
        "btier-022",
        "btier-023",
        "btier-024",
        "btier-025",
        "btier-026",
        "btier-027",
        "btier-028",
        "btier-029",
        "btier-030",
        "btier-031",
        "btier-032",
        "btier-033",
        "btier-034",
        "btier-035",
        "btier-036",
        "btier-037",
        "btier-038",
        "btier-039",
        "btier-040",
        "btier-041",
        "btier-042",
        "btier-043",
        "btier-044",
        "btier-045",
        "btier-046",
        "btier-047",
        "btier-048",
        "btier-049",
        "btier-050",
        "btier-051",
    };

    private static readonly string[] NativeEntryPoints =
    {
        "jyppx_trt10_execution_context_get_name",
        "jyppx_trt8_execution_context_get_name",
        "jyppx_trt11_profiler_get_interface_info",
        "jyppx_trt8_cuda_engine_get_profile_shape_values",
        "jyppx_trt8_engine_get_profile_shape",
        "jyppx_trt8_layer_get_input",
        "jyppx_trt8_layer_get_input_count",
        "jyppx_trt8_rnn_v2_layer_get_input_mode",
        "jyppx_trt10_builder_get_max_dla_batch_size",
        "jyppx_trt10_builder_get_max_threads",
        "jyppx_trt10_builder_is_network_supported",
        "jyppx_trt10_builder_config_can_run_on_dla",
        "jyppx_trt10_builder_config_get_average_timing_iterations",
        "jyppx_trt10_builder_config_get_default_device_type",
        "jyppx_trt10_builder_config_get_layer_device_type",
        "jyppx_trt10_builder_config_is_layer_device_type_set",
        "jyppx_trt10_builder_config_reset_layer_device_type",
        "jyppx_trt10_builder_config_set_default_device_type",
        "jyppx_trt10_builder_config_set_layer_device_type",
        "jyppx_trt10_builder_config_get_dla_core",
        "jyppx_trt10_builder_config_get_l2_limit_for_tiling",
        "jyppx_trt10_builder_config_get_max_nb_tactics",
        "jyppx_trt10_builder_config_get_quantization_flag",
        "jyppx_trt10_builder_config_get_quantization_flags",
        "jyppx_trt10_builder_config_get_tiling_optimization_level",
        "jyppx_trt10_cuda_engine_has_implicit_batch_dimension",
        "jyppx_trt10_execution_context_get_nvtx_verbosity",
        "jyppx_trt10_onnx_parser_get_error",
        "jyppx_trt10_onnx_parser_get_error_count",
        "jyppx_trt10_parser_refitter_get_error",
        "jyppx_trt10_parser_refitter_get_error_count",
        "jyppx_trt11_builder_config_get_average_timing_iterations",
        "jyppx_trt11_onnx_parser_get_error",
        "jyppx_trt11_onnx_parser_is_subgraph_supported",
        "jyppx_trt11_parser_refitter_get_error",
        "jyppx_trt8_builder_get_max_batch_size",
        "jyppx_trt8_builder_get_max_dla_batch_size",
        "jyppx_trt8_builder_get_max_threads",
        "jyppx_trt8_builder_is_network_supported",
        "jyppx_trt8_builder_config_can_run_on_dla",
        "jyppx_trt8_builder_config_get_average_timing_iterations",
        "jyppx_trt8_builder_config_get_default_device_type",
        "jyppx_trt8_builder_config_get_layer_device_type",
        "jyppx_trt8_builder_config_is_layer_device_type_set",
        "jyppx_trt8_builder_config_get_dla_core",
        "jyppx_trt8_builder_config_get_flag",
        "jyppx_trt8_builder_config_get_flags",
        "jyppx_trt8_builder_config_get_max_workspace_size",
        "jyppx_trt8_builder_config_get_min_timing_iterations",
        "jyppx_trt8_builder_config_get_quantization_flag",
        "jyppx_trt8_builder_config_get_quantization_flags",
        "jyppx_trt8_engine_get_hardware_compatibility_level",
        "jyppx_trt8_execution_context_get_nvtx_verbosity",
        "jyppx_trt8_onnx_parser_get_error",
        "jyppx_trt8_onnx_parser_get_error_count",
    };

    private static readonly string[] NativeSourceTokens =
    {
        "jyppx_trt10_execution_context_get_name",
        "jyppx_trt8_execution_context_get_name",
        "jyppx_trt11_profiler_get_interface_info",
        "jyppx_trt8_cuda_engine_get_profile_shape_values",
        "jyppx_trt8_engine_get_profile_shape",
        "jyppx_trt8_layer_get_input",
        "jyppx_trt8_layer_get_input_count",
        "jyppx_trt8_rnn_v2_layer_get_input_mode",
        "jyppx_trt10_builder_get_max_dla_batch_size",
        "jyppx_trt10_builder_get_max_threads",
        "jyppx_trt10_builder_is_network_supported",
        "jyppx_trt10_builder_config_can_run_on_dla",
        "jyppx_trt10_builder_config_get_average_timing_iterations",
        "jyppx_trt10_builder_config_get_default_device_type",
        "jyppx_trt10_builder_config_get_layer_device_type",
        "jyppx_trt10_builder_config_is_layer_device_type_set",
        "jyppx_trt10_builder_config_reset_layer_device_type",
        "jyppx_trt10_builder_config_set_default_device_type",
        "jyppx_trt10_builder_config_set_layer_device_type",
        "jyppx_trt10_builder_config_get_dla_core",
        "jyppx_trt10_builder_config_get_l2_limit_for_tiling",
        "jyppx_trt10_builder_config_get_max_nb_tactics",
        "jyppx_trt10_builder_config_get_quantization_flag",
        "jyppx_trt10_builder_config_get_quantization_flags",
        "jyppx_trt10_builder_config_get_tiling_optimization_level",
        "jyppx_trt10_cuda_engine_has_implicit_batch_dimension",
        "jyppx_trt10_execution_context_get_nvtx_verbosity",
        "jyppx_trt10_onnx_parser_get_error",
        "jyppx_trt10_onnx_parser_get_error_count",
        "JYPPX_TRT_PARSER_REFITTER_API(get_error)",
        "JYPPX_TRT_PARSER_REFITTER_API(get_error_count)",
        "jyppx_trt11_builder_config_get_average_timing_iterations",
        "jyppx_trt11_onnx_parser_get_error",
        "JYPPX_TRT_ONNX_PARSER_API(is_subgraph_supported)",
        "jyppx_trt11_parser_refitter_get_error",
        "jyppx_trt8_builder_get_max_batch_size",
        "jyppx_trt8_builder_get_max_dla_batch_size",
        "jyppx_trt8_builder_get_max_threads",
        "jyppx_trt8_builder_is_network_supported",
        "jyppx_trt8_builder_config_can_run_on_dla",
        "jyppx_trt8_builder_config_get_average_timing_iterations",
        "jyppx_trt8_builder_config_get_default_device_type",
        "jyppx_trt8_builder_config_get_layer_device_type",
        "jyppx_trt8_builder_config_is_layer_device_type_set",
        "jyppx_trt8_builder_config_get_dla_core",
        "jyppx_trt8_builder_config_get_flag",
        "jyppx_trt8_builder_config_get_flags",
        "jyppx_trt8_builder_config_get_max_workspace_size",
        "jyppx_trt8_builder_config_get_min_timing_iterations",
        "jyppx_trt8_builder_config_get_quantization_flag",
        "jyppx_trt8_builder_config_get_quantization_flags",
        "jyppx_trt8_engine_get_hardware_compatibility_level",
        "jyppx_trt8_execution_context_get_nvtx_verbosity",
        "jyppx_trt8_onnx_parser_get_error",
        "jyppx_trt8_onnx_parser_get_error_count",
    };

    private static readonly string[] WrapperTokens =
    {
        "public string Name",
        "NativeBridgeApi.GetExecutionContextName",
        "GetProfilerInterfaceInfo",
        "public TensorRtDims GetProfileShape",
        "public int[] GetProfileShapeValues",
        "public int InputCount => NativeBridgeApi.GetLayerInputCount",
        "public TensorRtTensor GetInput",
        "public int MaxThreads",
        "public bool IsNetworkSupported",
        "public TensorRtDeviceType GetLayerDeviceType",
        "public bool IsLayerDeviceTypeSet",
        "public int GetAverageTimingIterations",
        "public TensorRtDeviceType GetDefaultDeviceType",
        "public int GetDlaCore",
        "public long GetL2LimitForTiling",
        "GetTilingOptimizationLevel",
        "public int GetMaxTactics",
        "public bool GetQuantizationFlag",
        "public TensorRtQuantizationFlags GetQuantizationFlags",
        "public int MaxBatchSizeCompatibility",
        "public int MaxDlaBatchSize",
        "public TensorRtBuilderFlags GetFlags",
        "public ulong MaxWorkspaceSizeCompatibilityInBytes",
        "public int MinTimingIterationsCompatibility",
        "public TensorRtHardwareCompatibilityLevel EngineHardwareCompatibilityLevel",
        "public bool HasImplicitBatchDimensionCompatibility",
        "public TensorRtProfilingVerbosity GetNvtxVerbosity",
        "public int ErrorCount",
        "GetError",
        "IsSubgraphSupported",
        "TensorRtOnnxParserRefitter",
    };

    private static readonly string[] ExistingProofTokens =
    {
        "builder_payload->getMaxDLABatchSize()",
        "builder_payload->getMaxThreads()",
        "builder_payload->isNetworkSupported(*network_payload, *config_payload)",
        "config_payload->canRunOnDLA(layer_payload)",
        "config_payload->getDefaultDeviceType()",
        "config_payload->getL2LimitForTiling()",
        "config_payload->getTilingOptimizationLevel()",
        "config_payload->setMaxNbTactics(max_tactics);",
        "config_payload->getFlag(static_cast<nvinfer1::BuilderFlag>(flag))",
        "config_payload->getFlags()",
        "config_payload->getMaxWorkspaceSize()",
        "config_payload->getMinTimingIterations()",
        "engine_payload->getHardwareCompatibilityLevel()",
        "engine_payload->hasImplicitBatchDimension()",
        "context_payload->getNvtxVerbosity()",
        "config_payload->getQuantizationFlag(static_cast<nvinfer1::QuantizationFlag>(flag))",
        "builder.MaxDlaBatchSize",
        "builder.MaxBatchSizeCompatibility",
        "config.GetDlaCore()",
        "config.GetQuantizationFlags()",
        "Parser/ParserRefitter diagnostic snapshots are copied managed surface proof only",
        "TensorRtBuilderConfigAliasesCoverOfficialScalarGetterNames",
        "InterfaceCoverageMatrixSeparatesPromotionsFromDeferredHistory",
        "TensorRtPluginAliasesCoverCreatorMetadataInventoryFalseNegativeRows",
    };

    private static string[] Strings(JsonElement item, string propertyName)
    {
        return item.GetProperty(propertyName)
            .EnumerateArray()
            .Select(static value => value.GetString()!)
            .ToArray();
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);

        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)!;
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"PowerShell command failed with exit code {process.ExitCode}:{Environment.NewLine}{output}{error}");
        return output + error;
    }

    private static string ReadSources(params string[] pathPartsAndPatterns)
    {
        List<string> parts = pathPartsAndPatterns.ToList();
        List<string> patterns = new();
        while (parts.Count > 0 && parts[^1].Contains('*', StringComparison.Ordinal))
        {
            patterns.Insert(0, parts[^1]);
            parts.RemoveAt(parts.Count - 1);
        }

        string root = Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray());
        SearchOption option = patterns.Any(static pattern => pattern is "*.cpp" or "*.inc" or "*.cs")
            ? SearchOption.AllDirectories
            : SearchOption.TopDirectoryOnly;

        return string.Join(
            Environment.NewLine,
            patterns.SelectMany(pattern => Directory.EnumerateFiles(root, pattern, option))
                .Order(StringComparer.Ordinal)
                .Select(File.ReadAllText));
    }
}
