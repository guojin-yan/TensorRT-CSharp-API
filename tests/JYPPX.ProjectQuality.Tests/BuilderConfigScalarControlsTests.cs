using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class BuilderConfigScalarControlsTests
{
    [Fact]
    public void TensorRt8And10ScalarControlManifestsPromoteRealAbiWithoutDeletingDeferredRecords()
    {
        string manifest8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-builder-config-scalar-controls.manifest.json");
        string manifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-builder-config-scalar-controls.manifest.json");
        string minimalManifest8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-minimal.manifest.json");
        string minimalManifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-minimal.manifest.json");
        string deferred8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-twenty-third-batch-deferred-coverage.manifest.json");
        string deferred10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-third-batch-deferred-coverage.manifest.json");

        foreach (string manifest in new[] { manifest8, manifest10 })
        {
            Assert.Contains("builder_get_max_dla_batch_size", manifest);
            Assert.Contains("builder_set_max_threads", manifest);
            Assert.Contains("builder_get_max_threads", manifest);
            Assert.Contains("builder_is_network_supported", manifest);
            Assert.Contains("builder_config_set_default_device_type", manifest);
            Assert.Contains("builder_config_get_default_device_type", manifest);
            Assert.Contains("builder_config_set_dla_core", manifest);
            Assert.Contains("builder_config_get_dla_core", manifest);
            Assert.Contains("builder_config_can_run_on_dla", manifest);
            Assert.Contains("builder_config_get_flags", manifest);
            Assert.Contains("builder_config_set_quantization_flags", manifest);
            Assert.Contains("builder_config_get_quantization_flags", manifest);
            Assert.Contains("builder_config_clear_quantization_flag", manifest);
            Assert.Contains("builder_config_set_quantization_flag", manifest);
            Assert.Contains("builder_config_get_quantization_flag", manifest);
            Assert.DoesNotContain("_deferred", manifest);
        }

        Assert.Contains("builder_get_max_batch_size", manifest8);
        Assert.Contains("builder_config_get_max_workspace_size", manifest8);
        Assert.Contains("builder_config_get_min_timing_iterations", manifest8);
        Assert.Contains("builder_config_set_tiling_optimization_level", manifest10);
        Assert.Contains("builder_config_get_tiling_optimization_level", manifest10);
        Assert.Contains("builder_config_set_l2_limit_for_tiling", manifest10);
        Assert.Contains("builder_config_get_l2_limit_for_tiling", manifest10);
        Assert.Contains("builder_config_set_max_nb_tactics", manifest10);
        Assert.Contains("builder_config_get_max_nb_tactics", manifest10);
        Assert.Contains("trt8-builder-config-has-algorithm-selector", minimalManifest8);
        Assert.Contains("trt8-builder-config-has-int8-calibrator", minimalManifest8);
        Assert.Contains("trt10-builder-config-has-algorithm-selector", minimalManifest10);
        Assert.Contains("trt10-builder-config-has-int8-calibrator", minimalManifest10);

        Assert.Contains("trt8-builder-get-max-batch-size-deferred", deferred8);
        Assert.Contains("trt8-builder-is-network-supported-deferred", deferred8);
        Assert.Contains("trt8-builder-config-get-flags-deferred", deferred8);
        Assert.Contains("trt8-builder-config-get-max-workspace-size-deferred", deferred8);
        Assert.Contains("trt8-builder-config-get-min-timing-iterations-deferred", deferred8);
        Assert.Contains("trt8-builder-get-max-threads-deferred", deferred8);
        Assert.Contains("trt8-builder-config-get-default-device-type-deferred", deferred8);
        Assert.Contains("trt8-builder-config-set-quantization-flags-deferred", deferred8);
        Assert.Contains("trt8-builder-config-get-quantization-flag-deferred", deferred8);
        Assert.Contains("trt8-builder-config-get-algorithm-selector-deferred", deferred8);
        Assert.Contains("trt8-builder-config-get-int8-calibrator-deferred", deferred8);
        Assert.Contains("trt10-builder-is-network-supported-deferred", deferred10);
        Assert.Contains("trt10-builder-get-max-threads-deferred", deferred10);
        Assert.Contains("trt10-builder-config-get-default-device-type-deferred", deferred10);
        Assert.Contains("trt10-builder-config-set-quantization-flags-deferred", deferred10);
        Assert.Contains("trt10-builder-config-get-quantization-flag-deferred", deferred10);
        Assert.Contains("trt10-builder-config-set-tiling-optimization-level-deferred", deferred10);
        Assert.Contains("trt10-builder-config-get-max-nb-tactics-deferred", deferred10);
        Assert.Contains("trt10-builder-config-get-algorithm-selector-deferred", deferred10);
        Assert.Contains("trt10-builder-config-get-int8-calibrator-deferred", deferred10);
    }

    [Fact]
    public void NativeHeadersAndSourcesExposeSafeScalarParametersForTensorRt8And10()
    {
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string api8 = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string api10 = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string builderConfig8 = ReadSource("native", "src", "tensorrt", "v8", "modules", "builder", "builder_config.inc");
        string builderConfig10 = ReadSource("native", "src", "tensorrt", "v10", "modules", "builder", "builder_config.inc");

        foreach (string header in new[] { header8, header10 })
        {
            Assert.Contains("builder_get_max_dla_batch_size(JYPPX_TensorRtBuilder* builder, int32_t* out_size)", header);
            Assert.Contains("builder_set_max_threads(JYPPX_TensorRtBuilder* builder, int32_t max_threads, JYPPX_Boolean* out_set)", header);
            Assert.Contains("builder_get_max_threads(JYPPX_TensorRtBuilder* builder, int32_t* out_max_threads)", header);
            Assert.Contains("builder_is_network_supported(JYPPX_TensorRtBuilder* builder, JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_supported)", header);
            Assert.Contains("builder_config_set_default_device_type(JYPPX_TensorRtBuilderConfig* config, int32_t device_type)", header);
            Assert.Contains("builder_config_get_default_device_type(JYPPX_TensorRtBuilderConfig* config, int32_t* out_device_type)", header);
            Assert.Contains("builder_config_can_run_on_dla(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_can_run)", header);
            Assert.Contains("builder_config_get_flags(JYPPX_TensorRtBuilderConfig* config, uint32_t* out_flags)", header);
            Assert.Contains("builder_config_set_quantization_flags(JYPPX_TensorRtBuilderConfig* config, uint32_t flags)", header);
            Assert.Contains("builder_config_get_quantization_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag, JYPPX_Boolean* out_enabled)", header);
            Assert.Contains("builder_config_has_algorithm_selector(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_has_selector)", header);
            Assert.Contains("builder_config_has_int8_calibrator(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_has_calibrator)", header);
        }

        Assert.Contains("builder_get_max_batch_size(JYPPX_TensorRtBuilder* builder, int32_t* out_max_batch_size)", header8);
        Assert.Contains("builder_config_get_max_workspace_size(JYPPX_TensorRtBuilderConfig* config, size_t* out_workspace_size)", header8);
        Assert.Contains("builder_config_get_min_timing_iterations(JYPPX_TensorRtBuilderConfig* config, int32_t* out_iterations)", header8);
        Assert.Contains("builder_config_set_tiling_optimization_level(JYPPX_TensorRtBuilderConfig* config, int32_t level, JYPPX_Boolean* out_set)", header10);
        Assert.Contains("builder_config_get_l2_limit_for_tiling(JYPPX_TensorRtBuilderConfig* config, int64_t* out_bytes)", header10);
        Assert.Contains("builder_config_set_max_nb_tactics(JYPPX_TensorRtBuilderConfig* config, int32_t max_tactics)", header10);

        foreach (string api in new[] { api8, api10 })
        {
            Assert.Contains("builder_payload->getMaxDLABatchSize()", api);
            Assert.Contains("builder_payload->setMaxThreads(max_threads)", api);
            Assert.Contains("builder_payload->getMaxThreads()", api);
            Assert.Contains("Builder max_threads must be greater than or equal to 1.", api);
            Assert.Contains("builder_payload->isNetworkSupported(*network_payload, *config_payload)", api);
        }

        Assert.Contains("builder_payload->getMaxBatchSize()", api8);
        Assert.Contains("config_payload->getMaxWorkspaceSize()", builderConfig8);
        Assert.Contains("config_payload->getMinTimingIterations()", builderConfig8);

        foreach (string builderConfig in new[] { builderConfig8, builderConfig10 })
        {
            Assert.Contains("config_payload->setDefaultDeviceType(static_cast<nvinfer1::DeviceType>(device_type));", builderConfig);
            Assert.Contains("config_payload->getDefaultDeviceType()", builderConfig);
            Assert.Contains("config_payload->setDLACore(dla_core);", builderConfig);
            Assert.Contains("config_payload->getDLACore()", builderConfig);
            Assert.Contains("config_payload->canRunOnDLA(layer_payload)", builderConfig);
            Assert.Contains("config_payload->getFlags()", builderConfig);
            Assert.Contains("config_payload->setQuantizationFlags(static_cast<nvinfer1::QuantizationFlags>(flags));", builderConfig);
            Assert.Contains("config_payload->getQuantizationFlag(static_cast<nvinfer1::QuantizationFlag>(flag))", builderConfig);
            Assert.Contains("config_payload->getAlgorithmSelector() != nullptr", builderConfig);
            Assert.Contains("config_payload->getInt8Calibrator() != nullptr", builderConfig);
            Assert.Contains("TensorRT device type must be 0 (GPU) or 1 (DLA).", builderConfig);
        }

        Assert.Contains("config_payload->setTilingOptimizationLevel(static_cast<nvinfer1::TilingOptimizationLevel>(level))", builderConfig10);
        Assert.Contains("config_payload->getL2LimitForTiling()", builderConfig10);
        Assert.Contains("config_payload->setMaxNbTactics(max_tactics);", builderConfig10);
        Assert.Contains("Builder config max tactics must be greater than or equal to zero.", builderConfig10);
    }

    [Fact]
    public void ManagedInteropAndPublicApiRouteScalarControlsAcrossTensorRt8_10_11()
    {
        string boundaryInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11BoundaryControls.cs");
        string runtimeInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Builder", "NativeBridgeApi.BuilderConfigRuntimeControls.cs");
        string coreInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.cs");
        string diagnosticsInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Builder", "NativeBridgeApi.BuilderConfigDiagnostics.cs");
        string builderApi = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.Trt11BoundaryControls.cs");
        string builderConfigApi = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11RuntimeControls.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11Diagnostics.cs");

        Assert.Contains("jyppx_trt8_builder_get_max_threads", boundaryInterop);
        Assert.Contains("jyppx_trt10_builder_get_max_threads", boundaryInterop);
        Assert.Contains("jyppx_trt11_builder_get_max_threads", boundaryInterop);
        Assert.Contains("jyppx_trt8_builder_get_max_batch_size", boundaryInterop);
        Assert.Contains("jyppx_trt8_builder_is_network_supported", boundaryInterop);
        Assert.Contains("jyppx_trt10_builder_is_network_supported", boundaryInterop);
        Assert.Contains("jyppx_trt8_builder_config_set_default_device_type", runtimeInterop);
        Assert.Contains("jyppx_trt10_builder_config_get_default_device_type", runtimeInterop);
        Assert.Contains("jyppx_trt8_builder_config_get_dla_core", runtimeInterop);
        Assert.Contains("jyppx_trt10_builder_config_set_dla_core", runtimeInterop);
        Assert.Contains("jyppx_trt8_builder_config_get_flags", runtimeInterop);
        Assert.Contains("jyppx_trt10_builder_config_get_flags", runtimeInterop);
        Assert.Contains("jyppx_trt8_builder_config_set_quantization_flags", runtimeInterop);
        Assert.Contains("jyppx_trt10_builder_config_get_quantization_flag", runtimeInterop);
        Assert.Contains("TensorRT 11 removed this deprecated API", runtimeInterop);
        Assert.Contains("jyppx_trt10_builder_config_set_tiling_optimization_level", runtimeInterop);
        Assert.Contains("jyppx_trt11_builder_config_get_l2_limit_for_tiling", runtimeInterop);
        Assert.Contains("jyppx_trt10_builder_config_set_max_nb_tactics", runtimeInterop);
        Assert.Contains("GetMaxWorkspaceSizeCompatibility", coreInterop);
        Assert.Contains("GetMinTimingIterationsCompatibility", coreInterop);
        Assert.Contains("HasBuilderConfigAlgorithmSelectorCompatibility", coreInterop);
        Assert.Contains("HasBuilderConfigInt8CalibratorCompatibility", coreInterop);
        Assert.Contains("jyppx_trt8_builder_config_has_algorithm_selector", coreInterop);
        Assert.Contains("jyppx_trt10_builder_config_has_algorithm_selector", coreInterop);
        Assert.Contains("jyppx_trt8_builder_config_has_int8_calibrator", coreInterop);
        Assert.Contains("jyppx_trt10_builder_config_has_int8_calibrator", coreInterop);
        Assert.Contains("TensorRT 11 callback ownership remains deferred", coreInterop);
        Assert.Contains("jyppx_trt8_builder_config_can_run_on_dla", diagnosticsInterop);
        Assert.Contains("jyppx_trt10_builder_config_can_run_on_dla", diagnosticsInterop);
        Assert.Contains("jyppx_trt11_builder_config_can_run_on_dla", diagnosticsInterop);

        Assert.Contains("Gets TensorRT's current builder worker-thread limit.", builderApi);
        Assert.Contains("Sets TensorRT's builder worker-thread limit.", builderApi);
        Assert.Contains("public int MaxThreads", builderApi);
        Assert.Contains("public int MaxBatchSizeCompatibility", builderApi);
        Assert.Contains("public bool SetMaxThreads(int maxThreads)", builderApi);
        Assert.Contains("public void SetDefaultDeviceType", builderConfigApi);
        Assert.Contains("public TensorRtDeviceType GetDefaultDeviceType", builderConfigApi);
        Assert.Contains("public TensorRtBuilderFlags GetFlags", builderConfigApi);
        Assert.Contains("public bool CanRunOnDla", builderConfigApi);
        Assert.Contains("public void SetQuantizationFlags", builderConfigApi);
        Assert.Contains("public bool GetQuantizationFlag", builderConfigApi);
        Assert.Contains("public ulong MaxWorkspaceSizeCompatibilityInBytes", builderConfigApi);
        Assert.Contains("public int MinTimingIterationsCompatibility", builderConfigApi);
        Assert.Contains("public bool HasAlgorithmSelectorCompatibility", builderConfigApi);
        Assert.Contains("public bool HasInt8CalibratorCompatibility", builderConfigApi);
        Assert.Contains("does not transfer ownership", builderConfigApi);
        Assert.Contains("不暴露 borrowed", builderConfigApi);
        Assert.Contains("public bool SetTilingOptimizationLevel", builderConfigApi);
        Assert.Contains("TensorRT 10/11 tiling optimization", builderConfigApi);
        Assert.DoesNotContain("public IntPtr", builderApi + builderConfigApi);
        Assert.DoesNotContain("public nint", builderApi + builderConfigApi);

        string enums = ReadSource("src", "JYPPX.TensorRtSharp", "Core", "TensorRtEnums.cs");
        Assert.Contains("public enum TensorRtQuantizationFlag", enums);
        Assert.Contains("public enum TensorRtQuantizationFlags : uint", enums);
    }

    [Fact]
    public void NetworkBuilderSmokeCoversScalarControlProbe()
    {
        string program = ReadSource("smoke", "NetworkBuilderSmokeRunner", "Program.cs");

        Assert.Contains("ProbeBuilderScalarControls(builder, config)", program);
        Assert.Contains("builder.MaxThreads", program);
        Assert.Contains("builder.SetMaxThreads(targetMaxThreads)", program);
        Assert.Contains("builder.MaxDlaBatchSize", program);
        Assert.Contains("builder.MaxBatchSizeCompatibility", program);
        Assert.Contains("config.SetDefaultDeviceType(TensorRtDeviceType.Gpu);", program);
        Assert.Contains("config.GetDefaultDeviceType()", program);
        Assert.Contains("config.GetFlags()", program);
        Assert.Contains("config.GetDlaCore()", program);
        Assert.Contains("ProbeBuilderConfigLegacyCompatibility(config)", program);
        Assert.Contains("const ulong workspaceMemoryPoolLimit = 64UL * 1024UL * 1024UL", program);
        Assert.Contains("config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, workspaceMemoryPoolLimit)", program);
        Assert.Contains("config.GetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace)", program);
        Assert.Contains("WorkspaceMemoryPoolLimit=", program);
        Assert.Contains("config.MaxWorkspaceSizeCompatibilityInBytes", program);
        Assert.Contains("config.MinTimingIterationsCompatibility", program);
        Assert.Contains("ProbeQuantizationFlags(config)", program);
        Assert.Contains("config.GetQuantizationFlags()", program);
        Assert.Contains("config.SetQuantizationFlag(TensorRtQuantizationFlag.CalibrateBeforeFusion)", program);
        Assert.Contains("ProbeTilingControls(config)", program);
        Assert.Contains("ProbeBuilderConfigCallbackPresence(config)", program);
        Assert.Contains("config.HasAlgorithmSelectorCompatibility", program);
        Assert.Contains("config.HasInt8CalibratorCompatibility", program);
        Assert.Contains("config.SetTilingOptimizationLevel(TensorRtTilingOptimizationLevel.None)", program);
        Assert.Contains("config.GetMaxTactics()", program);
        Assert.Contains("config.CanRunOnDla(layer)", program);
        Assert.Contains("config.GetDeploymentSnapshot()", program);
        Assert.Contains("DeploymentSnapshot=", program);
        Assert.Contains("deploymentSnapshot.SerializedPluginSnapshot.Count", program);
        Assert.Contains("ScalarControls=", program);
        Assert.Contains("BuilderFlags=", program);
        Assert.Contains("MaxBatchCompatibility=", program);
        Assert.Contains("QuantizationFlags=", program);
        Assert.Contains("TilingControls=", program);
        Assert.Contains("CallbackPresence=AlgorithmSelector:", program);
        Assert.Contains("Int8Calibrator:", program);
        Assert.Contains("LayerDla=", program);

        string snapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfigDeploymentSnapshot.cs");
        Assert.Contains("public TensorRtBuilderConfigSerializedPluginSnapshot SerializedPluginSnapshot", snapshot);
        Assert.Contains("plugins={SerializedPluginSnapshot.Count}/{SerializedPluginSnapshot.PluginLibraryPaths.Count}", snapshot);
    }

    [Fact]
    public void InterfaceCoverageMatrixSeparatesRealImplementationFromDeferredHistory()
    {
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("function Test-DeferredManifestId", script);
        Assert.Contains("function Get-ResolvedImplementationStatus", script);
        Assert.Contains("implemented-with-deferred-history", script);
        Assert.Contains("deferred-only", script);
        Assert.Contains("\"IBuilderConfig::getAvgTimingIterations\" = @(\"id:*builder-config-get-average-timing-iterations\")", script);
        Assert.Contains("\"IBuilderConfig::setAvgTimingIterations\" = @(\"id:*builder-config-set-average-timing-iterations\")", script);
        Assert.Contains("\"IBuilderConfig::getBuilderOptimizationLevel\" = @(\"id:*builder-config-get-optimization-level\")", script);
        Assert.Contains("\"IBuilderConfig::setBuilderOptimizationLevel\" = @(\"id:*builder-config-set-optimization-level\")", script);
        Assert.Contains("\"IBuilderConfig::getAvgTimingIterations\" = @(\"id:*builder-config-get-avg-timing-iterations-deferred\")", script);
        Assert.Contains("\"IBuilderConfig::setAvgTimingIterations\" = @(\"id:*builder-config-set-avg-timing-iterations-deferred\")", script);
        Assert.Contains("\"IBuilderConfig::getBuilderOptimizationLevel\" = @(\"id:*builder-config-get-builder-optimization-level-deferred\")", script);
        Assert.Contains("\"IBuilderConfig::setBuilderOptimizationLevel\" = @(\"id:*builder-config-set-builder-optimization-level-deferred\")", script);
        Assert.Contains("tensorrt-interface-comparison.csv", script);
        Assert.Contains("cuda-runtime-interface-comparison.csv", script);

        string audit = ReadSource("artifacts", "interface-coverage", "trt-builder-config-scalar-candidate-audit.md");
        string auditJson = ReadSource("artifacts", "interface-coverage", "trt-builder-config-scalar-candidate-audit.json");
        Assert.Contains("promote-by-explicit-alias-history", audit);
        Assert.Contains("TRT8", audit);
        Assert.Contains("TRT10", audit);
        Assert.Contains("TRT11", audit);
        Assert.Contains("deferred records remain", audit);
        Assert.Contains("\"decision\": \"promote-by-explicit-alias-history\"", auditJson);
        Assert.Contains("IBuilderConfig::getAvgTimingIterations", auditJson);
        Assert.Contains("IBuilderConfig::setBuilderOptimizationLevel", auditJson);

        string coverage = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-coverage.csv");
        Assert.Contains("\"ImplementationStatus\"", coverage);
        Assert.Contains("\"IBuilder\",\"getMaxThreads\",\"IBuilder::getMaxThreads\",\"builder\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("\"IBuilder\",\"setMaxThreads\",\"IBuilder::setMaxThreads\",\"builder\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("\"IBuilder\",\"isNetworkSupported\",\"IBuilder::isNetworkSupported\",\"builder\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("\"IBuilderConfig\",\"getFlags\",\"IBuilderConfig::getFlags\",\"builder\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("\"IBuilderConfig\",\"setAvgTimingIterations\",\"IBuilderConfig::setAvgTimingIterations\",\"builder\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("\"IBuilderConfig\",\"setBuilderOptimizationLevel\",\"IBuilderConfig::setBuilderOptimizationLevel\",\"builder\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("\"IBuilderConfig\",\"getMaxWorkspaceSize\",\"IBuilderConfig::getMaxWorkspaceSize\",\"builder\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("\"IBuilderConfig\",\"getMinTimingIterations\",\"IBuilderConfig::getMinTimingIterations\",\"builder\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("\"IBuilderConfig\",\"getAlgorithmSelector\",\"IBuilderConfig::getAlgorithmSelector\",\"builder\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("\"IBuilderConfig\",\"getInt8Calibrator\",\"IBuilderConfig::getInt8Calibrator\",\"builder\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("jyppx-trt10-builder-get-max-threads;trt10-builder-get-max-threads-deferred", coverage);
        Assert.Contains("jyppx-trt8-builder-set-max-threads;trt8-builder-set-max-threads-deferred", coverage);
        Assert.Contains("jyppx-trt10-builder-is-network-supported;trt10-builder-is-network-supported-deferred", coverage);
        Assert.Contains("trt10-builder-config-set-average-timing-iterations;trt10-builder-config-set-avg-timing-iterations-deferred", coverage);
        Assert.Contains("trt10-builder-config-set-builder-optimization-level-deferred;trt10-builder-config-set-optimization-level", coverage);
        Assert.Contains("jyppx-trt8-builder-config-get-flags;trt8-builder-config-get-flags-deferred", coverage);
        Assert.Contains("trt8-builder-config-get-algorithm-selector-deferred;trt8-builder-config-has-algorithm-selector", coverage);
        Assert.Contains("trt10-builder-config-get-int8-calibrator-deferred;trt10-builder-config-has-int8-calibrator", coverage);
        Assert.Contains("jyppx-trt8-builder-config-get-max-workspace-size;trt8-builder-config-get-max-workspace-size-deferred", coverage);
        Assert.Contains("jyppx-trt8-builder-config-get-min-timing-iterations;trt8-builder-config-get-min-timing-iterations-deferred", coverage);

        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        Assert.Contains("\"ImplementationStatus\"", comparison);
        Assert.Contains("\"IBuilder\",\"getMaxThreads\",\"IBuilder::getMaxThreads\",\"builder\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IBuilder\",\"setMaxThreads\",\"IBuilder::setMaxThreads\",\"builder\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IBuilder\",\"isNetworkSupported\",\"IBuilder::isNetworkSupported\",\"builder\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IBuilderConfig\",\"getFlags\",\"IBuilderConfig::getFlags\",\"builder\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IBuilderConfig\",\"setAvgTimingIterations\",\"IBuilderConfig::setAvgTimingIterations\",\"builder\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IBuilderConfig\",\"setBuilderOptimizationLevel\",\"IBuilderConfig::setBuilderOptimizationLevel\",\"builder\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IBuilderConfig\",\"getMaxWorkspaceSize\",\"IBuilderConfig::getMaxWorkspaceSize\",\"builder\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IBuilderConfig\",\"getMinTimingIterations\",\"IBuilderConfig::getMinTimingIterations\",\"builder\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IBuilderConfig\",\"getAlgorithmSelector\",\"IBuilderConfig::getAlgorithmSelector\",\"builder\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("\"IBuilderConfig\",\"getInt8Calibrator\",\"IBuilderConfig::getInt8Calibrator\",\"builder\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("jyppx_trt10_builder_get_max_threads;jyppx_trt10_builder_get_max_threads_deferred", comparison);
        Assert.Contains("jyppx_trt8_builder_set_max_threads;jyppx_trt8_builder_set_max_threads_deferred", comparison);
        Assert.Contains("jyppx_trt10_builder_is_network_supported;jyppx_trt10_builder_is_network_supported_deferred", comparison);
        Assert.Contains("jyppx_trt10_builder_config_set_average_timing_iterations;jyppx_trt10_builder_config_set_avg_timing_iterations_deferred", comparison);
        Assert.Contains("jyppx_trt10_builder_config_set_builder_optimization_level_deferred;jyppx_trt10_builder_config_set_optimization_level", comparison);
        Assert.Contains("jyppx_trt8_builder_config_get_flags;jyppx_trt8_builder_config_get_flags_deferred", comparison);
        Assert.Contains("jyppx_trt8_builder_config_get_algorithm_selector_deferred;jyppx_trt8_builder_config_has_algorithm_selector", comparison);
        Assert.Contains("jyppx_trt10_builder_config_get_int8_calibrator_deferred;jyppx_trt10_builder_config_has_int8_calibrator", comparison);
        Assert.Contains("jyppx_trt8_builder_config_get_max_workspace_size;jyppx_trt8_builder_config_get_max_workspace_size_deferred", comparison);
        Assert.Contains("jyppx_trt8_builder_config_get_min_timing_iterations;jyppx_trt8_builder_config_get_min_timing_iterations_deferred", comparison);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
