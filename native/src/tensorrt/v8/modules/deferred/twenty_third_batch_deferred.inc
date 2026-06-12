JYPPX_StatusCode jyppx_trt8_cuda_engine_destroy_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "ICudaEngine::destroy deferred: lifetime is governed by bridge SafeHandle/common destroy semantics rather than per-interface vendor destroy calls");
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_get_error_recorder_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "ICudaEngine::getErrorRecorder deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_get_hardware_compatibility_level_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "ICudaEngine::getHardwareCompatibilityLevel deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_get_profile_shape_values_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "ICudaEngine::getProfileShapeValues deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_set_error_recorder_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "ICudaEngine::setErrorRecorder deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_set_gpu_allocator_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "ICudaEngine::setGpuAllocator deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_engine_inspector_get_error_recorder_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IEngineInspector::getErrorRecorder deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_engine_inspector_set_error_recorder_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IEngineInspector::setErrorRecorder deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_execution_context_destroy_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::destroy deferred: lifetime is governed by bridge SafeHandle/common destroy semantics rather than per-interface vendor destroy calls");
}

JYPPX_StatusCode jyppx_trt8_execution_context_enqueue_v2_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::enqueueV2 deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_execution_context_execute_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::execute deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_execution_context_execute_v2_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::executeV2 deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_execution_context_get_error_buffer_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::getErrorBuffer deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_execution_context_get_error_recorder_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::getErrorRecorder deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_execution_context_get_nvtx_verbosity_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::getNvtxVerbosity deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_execution_context_get_output_allocator_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::getOutputAllocator deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_execution_context_get_shape_binding_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::getShapeBinding deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_execution_context_get_temporary_storage_allocator_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::getTemporaryStorageAllocator deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_execution_context_no_copy_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::INoCopy deferred: lifetime is governed by bridge SafeHandle/common destroy semantics rather than per-interface vendor destroy calls");
}

JYPPX_StatusCode jyppx_trt8_execution_context_set_aux_streams_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::setAuxStreams deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_execution_context_set_error_buffer_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::setErrorBuffer deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_execution_context_set_error_recorder_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::setErrorRecorder deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_execution_context_set_input_shape_binding_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::setInputShapeBinding deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_execution_context_set_nvtx_verbosity_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::setNvtxVerbosity deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_execution_context_set_output_allocator_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::setOutputAllocator deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_execution_context_set_temporary_storage_allocator_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::setTemporaryStorageAllocator deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_builder_build_engine_with_config_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::buildEngineWithConfig deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_can_run_on_dla_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::canRunOnDLA deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_clear_quantization_flag_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::clearQuantizationFlag deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_destroy_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::destroy deferred: lifetime is governed by bridge SafeHandle/common destroy semantics rather than per-interface vendor destroy calls");
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_algorithm_selector_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getAlgorithmSelector deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_avg_timing_iterations_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getAvgTimingIterations deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_builder_optimization_level_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getBuilderOptimizationLevel deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_default_device_type_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getDefaultDeviceType deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_dla_core_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getDLACore deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_flags_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getFlags deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_int8_calibrator_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getInt8Calibrator deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_max_workspace_size_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getMaxWorkspaceSize deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_min_timing_iterations_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getMinTimingIterations deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_nb_plugins_to_serialize_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getNbPluginsToSerialize deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_plugin_to_serialize_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getPluginToSerialize deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_quantization_flag_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getQuantizationFlag deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_quantization_flags_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getQuantizationFlags deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_algorithm_selector_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setAlgorithmSelector deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_avg_timing_iterations_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setAvgTimingIterations deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_builder_optimization_level_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setBuilderOptimizationLevel deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_default_device_type_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setDefaultDeviceType deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_dla_core_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setDLACore deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_flags_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setFlags deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_int8_calibrator_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setInt8Calibrator deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_max_workspace_size_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setMaxWorkspaceSize deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_min_timing_iterations_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setMinTimingIterations deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_plugins_to_serialize_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setPluginsToSerialize deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_quantization_flag_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setQuantizationFlag deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_quantization_flags_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setQuantizationFlags deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_destroy_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::destroy deferred: lifetime is governed by bridge SafeHandle/common destroy semantics rather than per-interface vendor destroy calls");
}

JYPPX_StatusCode jyppx_trt8_builder_get_error_recorder_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::getErrorRecorder deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_builder_get_logger_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::getLogger deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_builder_get_max_batch_size_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::getMaxBatchSize deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_get_max_dla_batch_size_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::getMaxDLABatchSize deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_get_max_threads_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::getMaxThreads deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_get_plugin_registry_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::getPluginRegistry deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_is_network_supported_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::isNetworkSupported deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_set_error_recorder_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::setErrorRecorder deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_builder_set_gpu_allocator_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::setGpuAllocator deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt8_builder_set_max_batch_size_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::setMaxBatchSize deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_builder_set_max_threads_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::setMaxThreads deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_expr_builder_constant_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExprBuilder::constant deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_expr_builder_operation_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExprBuilder::operation deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt8_onnx_config_add_verbosity_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::addVerbosity deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

