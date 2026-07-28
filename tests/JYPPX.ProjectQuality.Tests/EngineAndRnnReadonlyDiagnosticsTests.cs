using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class EngineAndRnnReadonlyDiagnosticsTests
{
    [Fact]
    public void TensorRt8And10ReadonlyManifestsPromoteSafeApisWithoutDeletingDeferredRecords()
    {
        string trt10Manifest = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-execution-context-readonly-controls.manifest.json");
        string trt10ProfileTensorValuesManifest = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-engine-profile-tensor-values-readonly.manifest.json");
        string trt8PluginManifest = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-builder-config-plugin-serialization-readonly.manifest.json");
        string trt8RnnManifest = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-rnnv2-layer-readonly-diagnostics.manifest.json");
        string deferred10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-third-batch-deferred-coverage.manifest.json");
        string deferred8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-twenty-third-batch-deferred-coverage.manifest.json");
        string rnnDeferred8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-ninth-batch-network-layer-deferred.manifest.json");

        Assert.Contains("jyppx-trt10-cuda-engine-has-implicit-batch-dimension", trt10Manifest);
        Assert.Contains("jyppx_trt10_cuda_engine_has_implicit_batch_dimension", trt10Manifest);
        Assert.Contains("\"type\": \"JYPPX_Boolean*\", \"direction\": \"out\"", trt10Manifest);
        Assert.Contains("jyppx-trt10-engine-get-profile-tensor-values", trt10ProfileTensorValuesManifest);
        Assert.Contains("jyppx-trt10-engine-get-profile-tensor-values-v2", trt10ProfileTensorValuesManifest);
        Assert.Contains("\"type\": \"int32_t*\", \"direction\": \"out\", \"managedType\": \"IntPtr\"", trt10ProfileTensorValuesManifest);
        Assert.Contains("\"type\": \"int64_t*\", \"direction\": \"out\", \"managedType\": \"IntPtr\"", trt10ProfileTensorValuesManifest);
        Assert.Contains("trt8-cuda-engine-get-profile-shape-values", deferred8);
        Assert.Contains("jyppx_trt8_cuda_engine_get_profile_shape_values", deferred8);
        Assert.Contains("\"type\": \"int32_t*\"", deferred8);
        Assert.Contains("trt8-execution-context-get-shape-binding", deferred8);
        Assert.Contains("jyppx_trt8_execution_context_get_shape_binding", deferred8);
        Assert.Contains("jyppx-trt8-builder-config-get-nb-plugins-to-serialize", trt8PluginManifest);
        Assert.Contains("jyppx_trt8_builder_config_get_nb_plugins_to_serialize", trt8PluginManifest);

        foreach (string entryPoint in new[]
        {
            "jyppx_trt8_rnn_v2_layer_get_layer_count",
            "jyppx_trt8_rnn_v2_layer_get_hidden_size",
            "jyppx_trt8_rnn_v2_layer_get_data_length",
            "jyppx_trt8_rnn_v2_layer_get_max_seq_length",
            "jyppx_trt8_rnn_v2_layer_get_operation",
            "jyppx_trt8_rnn_v2_layer_get_direction",
            "jyppx_trt8_rnn_v2_layer_get_input_mode",
            "jyppx_trt8_rnn_v2_layer_get_cell_state",
            "jyppx_trt8_rnn_v2_layer_get_hidden_state",
            "jyppx_trt8_rnn_v2_layer_get_sequence_lengths",
            "jyppx_trt8_rnn_v2_layer_get_weights_for_gate_copy",
            "jyppx_trt8_rnn_v2_layer_get_bias_for_gate_copy"
        })
        {
            Assert.Contains(entryPoint, trt8RnnManifest);
        }

        Assert.Contains("trt10-cuda-engine-has-implicit-batch-dimension-deferred", deferred10);
        Assert.Contains("trt10-cuda-engine-get-profile-tensor-values-deferred", deferred10);
        Assert.Contains("trt10-cuda-engine-get-profile-tensor-values-v2-deferred", deferred10);
        Assert.Contains("trt8-cuda-engine-get-profile-shape-values-deferred", deferred8);
        Assert.Contains("trt8-execution-context-get-shape-binding-deferred", deferred8);
        Assert.Contains("trt8-builder-config-get-nb-plugins-to-serialize-deferred", deferred8);
        Assert.Contains("trt8-rnnv2-layer-get-layer-count-deferred", rnnDeferred8);
        Assert.Contains("trt8-rnnv2-layer-get-operation-deferred", rnnDeferred8);
        Assert.Contains("trt8-rnnv2-layer-get-data-length-deferred", rnnDeferred8);
        Assert.DoesNotContain("_deferred", trt10Manifest + trt10ProfileTensorValuesManifest + trt8PluginManifest + trt8RnnManifest);
    }

    [Fact]
    public void NativeHeadersAndSourcesExposeOnlySafeScalarAndBooleanResults()
    {
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string api8 = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string builderConfig8 = ReadSource("native", "src", "tensorrt", "v8", "modules", "builder", "builder_config.inc");
        string rnn8 = ReadSource("native", "src", "tensorrt", "v8", "modules", "layers", "rnn_v2_readonly.inc");
        string api10 = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt10ProfileTensorValues = ReadSource("native", "src", "tensorrt", "v10", "modules", "deployment", "engine_profile_tensor_values.inc");

        Assert.Contains("jyppx_trt10_cuda_engine_has_implicit_batch_dimension(JYPPX_TensorRtCudaEngine* engine, JYPPX_Boolean* out_has_implicit_batch)", header10);
        Assert.Contains("jyppx_trt10_engine_get_profile_tensor_values(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t selector, int32_t* output_values, int32_t output_count, int32_t* out_count)", header10);
        Assert.Contains("jyppx_trt10_engine_get_profile_tensor_values_v2(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t selector, int64_t* output_values, int32_t output_count, int32_t* out_count)", header10);
        Assert.Contains("jyppx_trt8_cuda_engine_get_profile_shape_values(JYPPX_TensorRtCudaEngine* engine, int32_t binding_index, int32_t profile_index, int32_t selector, int32_t* output_values, int32_t output_count, int32_t* out_count)", header8);
        Assert.Contains("jyppx_trt8_execution_context_get_shape_binding(JYPPX_TensorRtExecutionContext* context, int32_t binding_index, int32_t* output_values, int32_t output_count, int32_t* out_count)", header8);
        Assert.Contains("jyppx_trt8_builder_config_get_nb_plugins_to_serialize(JYPPX_TensorRtBuilderConfig* config, int32_t* out_count)", header8);
        Assert.Contains("jyppx_trt8_rnn_v2_layer_get_layer_count(JYPPX_TensorRtLayer* layer, int32_t* out_layer_count)", header8);
        Assert.Contains("jyppx_trt8_rnn_v2_layer_get_data_length(JYPPX_TensorRtLayer* layer, int32_t* out_data_length)", header8);
        Assert.Contains("jyppx_trt8_rnn_v2_layer_get_input_mode(JYPPX_TensorRtLayer* layer, int32_t* out_input_mode)", header8);
        Assert.Contains("jyppx_trt8_rnn_v2_layer_get_cell_state(", header8);
        Assert.Contains("jyppx_trt8_rnn_v2_layer_get_hidden_state(", header8);
        Assert.Contains("jyppx_trt8_rnn_v2_layer_get_sequence_lengths(", header8);
        Assert.Contains("jyppx_trt8_rnn_v2_layer_get_weights_for_gate_copy(", header8);
        Assert.Contains("jyppx_trt8_rnn_v2_layer_get_bias_for_gate_copy(", header8);

        Assert.Contains("engine_payload->hasImplicitBatchDimension()", api10);
        Assert.Contains("modules/deployment/engine_profile_tensor_values.inc", api10);
        Assert.Contains("engine_payload->getProfileTensorValues(", trt10ProfileTensorValues);
        Assert.Contains("engine_payload->getProfileTensorValuesV2(", trt10ProfileTensorValues);
        Assert.Contains("engine_payload->getProfileShapeValues(profile_index, binding_index", api8);
        Assert.Contains("context_payload->getShapeBinding(binding_index, scratch)", api8);
        Assert.Contains("output_values[i] = values[i];", trt10ProfileTensorValues);
        Assert.Contains("output_values[i] = values[i];", api8);
        Assert.Contains("output_values[i] = scratch[i];", api8);
        Assert.Contains("JYPPX_STATUS_BUFFER_TOO_SMALL", trt10ProfileTensorValues);
        Assert.Contains("JYPPX_STATUS_BUFFER_TOO_SMALL", api8);
        Assert.Contains("config_payload->getNbPluginsToSerialize()", builderConfig8);
        Assert.Contains("layer->getType() != nvinfer1::LayerType::kRNN_V2", api8);
        Assert.Contains("rnn_layer->getLayerCount()", rnn8);
        Assert.Contains("rnn_layer->getHiddenSize()", rnn8);
        Assert.Contains("rnn_layer->getDataLength()", rnn8);
        Assert.Contains("rnn_layer->getMaxSeqLength()", rnn8);
        Assert.Contains("rnn_layer->getOperation()", rnn8);
        Assert.Contains("rnn_layer->getDirection()", rnn8);
        Assert.Contains("rnn_layer->getInputMode()", rnn8);
        Assert.Contains("rnn_layer->getCellState()", rnn8);
        Assert.Contains("rnn_layer->getHiddenState()", rnn8);
        Assert.Contains("rnn_layer->getSequenceLengths()", rnn8);
        Assert.Contains("rnn_layer->getWeightsForGate(", rnn8);
        Assert.Contains("rnn_layer->getBiasForGate(", rnn8);
        Assert.Contains("JYPPX_STATUS_BUFFER_TOO_SMALL", rnn8);
        Assert.DoesNotContain("JYPPX_TensorRtPluginCreator**", header8 + header10);
    }

    [Fact]
    public void ManagedInteropAndPublicApiExposeCompatibilityQueriesWithoutRawPointers()
    {
        string runtimeInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11RuntimeControls.cs");
        string diagnosticsInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11Diagnostics.cs");
        string rnnInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt8RnnV2Diagnostics.cs");
        string engineApi = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.cs");
        string contextApi = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.cs");
        string engineProfileValuesApi = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.Trt11FifteenthBatch.cs");
        string engineProfileValuesSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngineProfileTensorValuesSnapshot.cs");
        string engineDeploymentSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngineDeploymentSnapshot.cs");
        string engineProfileValuesInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11FifteenthBatch.cs");
        string deploymentInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.DeploymentMetadata.cs");
        string builderConfigApi = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11Diagnostics.cs");
        string layerApi = ReadSource("src", "JYPPX.TensorRtSharp", "Layers", "TensorRtLayer.Trt8RnnV2Diagnostics.cs");
        string tensorApi = ReadSource("src", "JYPPX.TensorRtSharp", "Network", "TensorRtTensor.cs");
        string snapshotApi = ReadSource("src", "JYPPX.TensorRtSharp", "Layers", "TensorRtRnnV2GateWeightsSnapshot.cs");
        string ownerLease = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Handles", "SafeTensorRtObjectHandleLease.cs");
        string enums = ReadSource("src", "JYPPX.TensorRtSharp", "Core", "TensorRtEnums.cs");
        string smoke = ReadSource("smoke", "NetworkBuilderSmokeRunner", "Program.cs");
        string tensorRtSmoke = ReadSource("smoke", "TensorRtSmokeRunner", "Program.cs");

        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_cuda_engine_has_implicit_batch_dimension", runtimeInterop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_cuda_engine_has_implicit_batch_dimension", runtimeInterop);
        Assert.Contains("HasEngineImplicitBatchDimensionCompatibility", runtimeInterop);
        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_nb_plugins_to_serialize", diagnosticsInterop);
        Assert.Contains("SerializedPluginPathCountCompatibility", builderConfigApi);
        Assert.Contains("TensorRT 8/10/11 support caller-buffer path copying", builderConfigApi);

        Assert.Contains("GetRnnV2LayerCount", rnnInterop);
        Assert.Contains("jyppx_trt8_rnn_v2_layer_get_layer_count", rnnInterop);
        Assert.Contains("GetRnnV2DataLength", rnnInterop);
        Assert.Contains("jyppx_trt8_rnn_v2_layer_get_data_length", rnnInterop);
        Assert.Contains("available only for TensorRT 8 RNNv2 layers", rnnInterop);
        Assert.Contains("public int GetRnnV2LayerCount()", layerApi);
        Assert.Contains("public int GetRnnV2DataLength()", layerApi);
        Assert.Contains("does not expose TensorRT-owned tensor or weight pointers", layerApi);
        Assert.Contains("public TensorRtRnnOperation GetRnnV2Operation()", layerApi);
        Assert.Contains("public TensorRtRnnDirection GetRnnV2Direction()", layerApi);
        Assert.Contains("public TensorRtRnnInputMode GetRnnV2InputMode()", layerApi);
        Assert.Contains("public TensorRtTensor? GetRnnV2CellState()", layerApi);
        Assert.Contains("public TensorRtTensor? GetRnnV2HiddenState()", layerApi);
        Assert.Contains("public TensorRtTensor? GetRnnV2SequenceLengths()", layerApi);
        Assert.Contains("public TensorRtRnnV2GateWeightsSnapshot GetRnnV2WeightsForGate(", layerApi);
        Assert.Contains("public TensorRtRnnV2GateWeightsSnapshot GetRnnV2BiasForGate(", layerApi);
        Assert.Contains("public bool IsOwnerLifetimeBound", tensorApi);
        Assert.Contains("public sealed class TensorRtRnnV2GateWeightsSnapshot", snapshotApi);
        Assert.Contains("return (byte[])_values.Clone()", snapshotApi);
        Assert.Contains("owner.DangerousAddRef(ref addedRef)", ownerLease);
        Assert.Contains("_owner.DangerousRelease()", ownerLease);
        Assert.Contains("public bool HasImplicitBatchDimensionCompatibility", engineApi);
        Assert.Contains("public int[] GetProfileShapeValues(int bindingIndex, int profileIndex, TensorRtOptimizationProfileSelector selector)", engineApi);
        Assert.Contains("public int[] GetShapeBinding(int bindingIndex)", contextApi);
        Assert.Contains("public bool SetInputShapeBinding(int bindingIndex, IReadOnlyList<int> values)", contextApi);
        Assert.Contains("GetEngineProfileShapeValues", deploymentInterop);
        Assert.Contains("GetExecutionContextShapeBinding", deploymentInterop);
        Assert.Contains("jyppx_trt8_cuda_engine_get_profile_shape_values", deploymentInterop);
        Assert.Contains("jyppx_trt8_execution_context_get_shape_binding", deploymentInterop);
        Assert.Contains("GCHandle.Alloc(values, GCHandleType.Pinned)", deploymentInterop);
        Assert.Contains("public int[] GetProfileTensorValues(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector)", engineProfileValuesApi);
        Assert.Contains("public int[] GetProfileTensorValues(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector, int valueCount)", engineProfileValuesApi);
        Assert.Contains("public long[] GetProfileTensorValuesV2(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector)", engineProfileValuesApi);
        Assert.Contains("public long[] GetProfileTensorValuesV2(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector, int valueCount)", engineProfileValuesApi);
        Assert.Contains("public TensorRtEngineProfileTensorValuesSnapshot GetProfileTensorValuesSnapshot(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector)", engineProfileValuesApi);
        Assert.Contains("public TensorRtEngineProfileTensorValuesSnapshot GetProfileTensorValuesSnapshot(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector, int valueCount)", engineProfileValuesApi);
        Assert.Contains("TryGetProfileTensorValuesSnapshot", engineProfileValuesApi);
        Assert.Contains("InferProfileTensorValueCount(tensorName)", engineProfileValuesApi);
        Assert.Contains("TensorRtDims shape = GetTensorShape(tensorName);", engineProfileValuesApi);
        Assert.Contains("call the overload that accepts valueCount", engineProfileValuesApi);
        Assert.Contains("caller-owned arrays and never exposes TensorRT-owned pointers", engineProfileValuesApi);
        Assert.Contains("List<TensorRtEngineProfileTensorValuesSnapshot> profileTensorValues", engineProfileValuesApi);
        Assert.Contains("profileTensorValues.Add(GetProfileTensorValuesSnapshot", engineProfileValuesApi);
        Assert.Contains("IReadOnlyList<TensorRtEngineProfileTensorValuesSnapshot> ProfileTensorValues", engineDeploymentSnapshot);
        Assert.Contains("profileTensorValues={ProfileTensorValues.Count}", engineDeploymentSnapshot);
        Assert.Contains("public sealed class TensorRtEngineProfileTensorValuesSnapshot", engineProfileValuesSnapshot);
        Assert.Contains("public IReadOnlyList<int> LegacyInt32Values", engineProfileValuesSnapshot);
        Assert.Contains("public IReadOnlyList<long> ValuesV2", engineProfileValuesSnapshot);
        Assert.Contains("public bool HasAnyValues", engineProfileValuesSnapshot);
        Assert.Contains("public IReadOnlyList<string> Diagnostics", engineProfileValuesSnapshot);
        Assert.Contains("EnsureTensorRt10(line, nameof(GetEngineProfileTensorValues))", engineProfileValuesInterop);
        Assert.Contains("EnsureTensorRt10Or11(line, nameof(GetEngineProfileTensorValuesV2))", engineProfileValuesInterop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt10_engine_get_profile_tensor_values(", engineProfileValuesInterop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt10_engine_get_profile_tensor_values_v2(", engineProfileValuesInterop);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_engine_get_profile_tensor_values_v2(", engineProfileValuesInterop);
        Assert.Contains("GCHandle.Alloc(values, GCHandleType.Pinned)", engineProfileValuesInterop);
        Assert.Contains("public enum TensorRtRnnOperation", enums);
        Assert.Contains("public enum TensorRtRnnDirection", enums);
        Assert.Contains("public enum TensorRtRnnInputMode", enums);
        Assert.Contains("public enum TensorRtRnnGateType", enums);

        Assert.Contains("ProbeSerializedPluginPaths(config)", smoke);
        Assert.Contains("config.GetSerializedPluginSnapshot()", smoke);
        Assert.Contains("engine.HasImplicitBatchDimensionCompatibility", smoke);
        Assert.Contains("engine.GetProfileTensorValuesSnapshot", tensorRtSmoke);
        Assert.Contains("ProbeTrt8LegacyShapeBindingBoundary", tensorRtSmoke);
        Assert.Contains("ProbeTrt8LegacyContextShapeBindingBoundary", tensorRtSmoke);
        Assert.Contains("Trt8LegacyShapeBinding=", tensorRtSmoke);
        Assert.Contains("Trt8LegacyContextShapeBinding=", tensorRtSmoke);
        Assert.Contains("TryGetProfileTensorValuesSnapshot", tensorRtSmoke);
        Assert.Contains("EngineProfileTensorSnapshot=", tensorRtSmoke);
        Assert.Contains("DeploymentSnapshotEvidence=", tensorRtSmoke);
        Assert.Contains("engineSnapshot.ProfileTensorValues.Count", tensorRtSmoke);
        Assert.DoesNotContain("public IntPtr", engineApi + contextApi + engineProfileValuesApi + engineProfileValuesSnapshot + engineDeploymentSnapshot + builderConfigApi + layerApi + rnnInterop + deploymentInterop);
        Assert.DoesNotContain("public nint", engineApi + contextApi + engineProfileValuesApi + engineProfileValuesSnapshot + engineDeploymentSnapshot + builderConfigApi + layerApi + rnnInterop + deploymentInterop);
    }

    [Fact]
    public void InterfaceCoverageMatrixSeparatesPromotionsFromDeferredHistory()
    {
        string coverage = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-coverage.csv");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        foreach (string matrix in new[] { coverage, comparison })
        {
            Assert.Contains("\"ICudaEngine\",\"hasImplicitBatchDimension\",\"ICudaEngine::hasImplicitBatchDimension\",\"engine-context\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"ICudaEngine\",\"getProfileTensorValues\",\"ICudaEngine::getProfileTensorValues\",\"engine-context\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"ICudaEngine\",\"getProfileTensorValuesV2\",\"ICudaEngine::getProfileTensorValuesV2\",\"engine-context\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"ICudaEngine\",\"getProfileShapeValues\",\"ICudaEngine::getProfileShapeValues\",\"engine-context\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"IExecutionContext\",\"getShapeBinding\",\"IExecutionContext::getShapeBinding\",\"engine-context\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"IBuilderConfig\",\"getNbPluginsToSerialize\",\"IBuilderConfig::getNbPluginsToSerialize\",\"builder\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"IRNNv2Layer\",\"getLayerCount\",\"IRNNv2Layer::getLayerCount\",\"network-layer\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"IRNNv2Layer\",\"getDataLength\",\"IRNNv2Layer::getDataLength\",\"network-layer\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"IRNNv2Layer\",\"getOperation\",\"IRNNv2Layer::getOperation\",\"network-layer\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"IRNNv2Layer\",\"getBiasForGate\",\"IRNNv2Layer::getBiasForGate\",\"network-layer\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"IRNNv2Layer\",\"getCellState\",\"IRNNv2Layer::getCellState\",\"network-layer\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"IRNNv2Layer\",\"getHiddenState\",\"IRNNv2Layer::getHiddenState\",\"network-layer\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"IRNNv2Layer\",\"getSequenceLengths\",\"IRNNv2Layer::getSequenceLengths\",\"network-layer\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"IRNNv2Layer\",\"getWeightsForGate\",\"IRNNv2Layer::getWeightsForGate\",\"network-layer\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("jyppx_trt10_cuda_engine_has_implicit_batch_dimension;jyppx_trt10_cuda_engine_has_implicit_batch_dimension_deferred", matrix);
            Assert.Contains("jyppx_trt10_engine_get_profile_tensor_values", matrix);
            Assert.Contains("jyppx_trt10_cuda_engine_get_profile_tensor_values_deferred", matrix);
            Assert.Contains("jyppx_trt10_engine_get_profile_tensor_values_v2", matrix);
            Assert.Contains("jyppx_trt10_cuda_engine_get_profile_tensor_values_v2_deferred", matrix);
            Assert.Contains("jyppx_trt8_cuda_engine_get_profile_shape_values", matrix);
            Assert.Contains("jyppx_trt8_cuda_engine_get_profile_shape_values_deferred", matrix);
            Assert.Contains("jyppx_trt8_execution_context_get_shape_binding", matrix);
            Assert.Contains("jyppx_trt8_execution_context_get_shape_binding_deferred", matrix);
            Assert.Contains("jyppx_trt8_builder_config_get_nb_plugins_to_serialize;jyppx_trt8_builder_config_get_nb_plugins_to_serialize_deferred", matrix);
            Assert.Contains("jyppx_trt8_rnn_v2_layer_get_layer_count;jyppx_trt8_rnn_v2_layer_get_layer_count_deferred", matrix);
            Assert.Contains("jyppx_trt8_rnn_v2_layer_get_data_length;jyppx_trt8_rnn_v2_layer_get_data_length_deferred", matrix);
            Assert.Contains("jyppx_trt8_rnn_v2_layer_get_bias_for_gate_copy;jyppx_trt8_rnn_v2_layer_get_bias_for_gate_deferred", matrix);
            Assert.Contains("jyppx_trt8_rnn_v2_layer_get_cell_state;jyppx_trt8_rnn_v2_layer_get_cell_state_deferred", matrix);
            Assert.Contains("jyppx_trt8_rnn_v2_layer_get_hidden_state;jyppx_trt8_rnn_v2_layer_get_hidden_state_deferred", matrix);
            Assert.Contains("jyppx_trt8_rnn_v2_layer_get_sequence_lengths;jyppx_trt8_rnn_v2_layer_get_sequence_lengths_deferred", matrix);
            Assert.Contains("jyppx_trt8_rnn_v2_layer_get_weights_for_gate_copy;jyppx_trt8_rnn_v2_layer_get_weights_for_gate_deferred", matrix);
        }
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
