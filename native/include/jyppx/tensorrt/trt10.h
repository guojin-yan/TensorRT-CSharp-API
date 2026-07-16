#pragma once

#include <stddef.h>
#include <stdint.h>

#include "jyppx/cuda/types.h"
#include "jyppx/tensorrt/types.h"

/* BEGIN TRT10 SAFE LIFECYCLE AND ERROR METADATA DECLARATIONS */
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_build_engine_with_config(JYPPX_TensorRtBuilder* builder, JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtCudaEngine** out_engine);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_error_code_get_exclusive_upper_bound(int32_t* out_exclusive_upper_bound);
/* END TRT10 SAFE LIFECYCLE AND ERROR METADATA DECLARATIONS */

/* BEGIN TRT10 PLUGIN V2 LAYER METADATA SNAPSHOT DECLARATIONS */
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_get_plugin_type(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_get_plugin_version(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_get_plugin_namespace(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_get_serialization_size(JYPPX_TensorRtLayer* layer, size_t* out_serialization_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_get_tensor_rt_version(JYPPX_TensorRtLayer* layer, int32_t* out_tensor_rt_version);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_get_plugin_output_count(JYPPX_TensorRtLayer* layer, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_get_capability_presence(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_has_ext, JYPPX_Boolean* out_has_io_ext, JYPPX_Boolean* out_has_dynamic_ext);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_get_legacy_output_dimensions(JYPPX_TensorRtLayer* layer, int32_t output_index, JYPPX_TensorRtDims* out_dimensions);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_get_legacy_workspace_size(JYPPX_TensorRtLayer* layer, int32_t max_batch_size, size_t* out_workspace_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_supports_legacy_format(JYPPX_TensorRtLayer* layer, int32_t data_type, int32_t tensor_format, JYPPX_Boolean* out_supported);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_get_output_data_type(JYPPX_TensorRtLayer* layer, int32_t output_index, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_can_broadcast_input_across_batch(JYPPX_TensorRtLayer* layer, int32_t input_index, JYPPX_Boolean* out_can_broadcast);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_is_output_broadcast_across_batch(JYPPX_TensorRtLayer* layer, int32_t output_index, const uint8_t* input_is_broadcasted, size_t input_count, JYPPX_Boolean* out_is_broadcast);
/* END TRT10 PLUGIN V2 LAYER METADATA SNAPSHOT DECLARATIONS */

/* BEGIN TRT10 PLUGIN V3 LAYER METADATA SNAPSHOT DECLARATIONS */
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_plugin_interface_info(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_capability_presence(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_has_core, JYPPX_Boolean* out_has_build, JYPPX_Boolean* out_has_runtime);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_core_plugin_name(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_core_plugin_version(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_core_plugin_namespace(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_core_interface_info(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_build_interface_info(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_build_nb_outputs(JYPPX_TensorRtLayer* layer, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_build_nb_tactics(JYPPX_TensorRtLayer* layer, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_build_format_combination_limit(JYPPX_TensorRtLayer* layer, int32_t* out_limit);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_build_timing_cache_id(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_build_metadata_string(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_runtime_interface_info(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_copy_dynamic_format_support(JYPPX_TensorRtLayer* layer, uint8_t* output_support, size_t output_capacity, size_t* out_required_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_copy_io_ext_format_support(JYPPX_TensorRtLayer* layer, uint8_t* output_support, size_t output_capacity, size_t* out_required_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_build_io_counts(JYPPX_TensorRtLayer* layer, int32_t* out_input_count, int32_t* out_output_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_copy_build_output_data_types(JYPPX_TensorRtLayer* layer, int32_t* output_types, size_t output_capacity, size_t* out_required_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_copy_build_aliased_inputs(JYPPX_TensorRtLayer* layer, int32_t* output_aliases, size_t output_capacity, size_t* out_required_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_copy_build_current_format_support(JYPPX_TensorRtLayer* layer, uint8_t* output_support, size_t output_capacity, size_t* out_required_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_runtime_serialization_field_count(JYPPX_TensorRtLayer* layer, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_runtime_serialization_field_name(JYPPX_TensorRtLayer* layer, int32_t field_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_runtime_serialization_field_metadata(JYPPX_TensorRtLayer* layer, int32_t field_index, int32_t* out_field_type, int32_t* out_length, JYPPX_Boolean* out_has_data);
/* END TRT10 PLUGIN V3 LAYER METADATA SNAPSHOT DECLARATIONS */

JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_query_adapter_info(JYPPX_TensorRtAdapterInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_logger_create(JYPPX_TensorRtLogger** out_logger);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_logger_create_with_callback(JYPPX_TensorRtLoggerCallback callback, void* user_state, int32_t minimum_severity, JYPPX_TensorRtLogger** out_logger);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_logger_emit_diagnostic(JYPPX_TensorRtLogger* logger, int32_t severity, const char* message, JYPPX_Boolean* out_callback_failed);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_progress_monitor_create_with_callback(JYPPX_TensorRtProgressMonitorCallback callback, void* user_state, JYPPX_TensorRtProgressMonitor** out_monitor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_progress_monitor_emit_diagnostic(JYPPX_TensorRtProgressMonitor* monitor, int32_t event_kind, const char* phase_name, const char* parent_phase, int32_t step, int32_t nb_steps, JYPPX_Boolean* out_should_continue, JYPPX_Boolean* out_callback_failed);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_progress_monitor_get_interface_info(JYPPX_TensorRtProgressMonitor* monitor, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_progress_monitor_get_api_language(JYPPX_TensorRtProgressMonitor* monitor, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_get_error_recorder_versioned_metadata(JYPPX_TensorRtRuntime* runtime, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_error_recorder_versioned_metadata(JYPPX_TensorRtRefitter* refitter, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_error_recorder_versioned_metadata(JYPPX_TensorRtCudaEngine* engine, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_error_recorder_versioned_metadata(JYPPX_TensorRtExecutionContext* context, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_get_error_recorder_versioned_metadata(JYPPX_TensorRtBuilder* builder, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_get_error_recorder_versioned_metadata(JYPPX_TensorRtNetworkDefinition* network, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_inspector_get_error_recorder_versioned_metadata(JYPPX_TensorRtEngineInspector* inspector, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_output_allocator_api_language(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_temporary_storage_allocator_api_language(JYPPX_TensorRtExecutionContext* context, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_debug_listener_api_language(JYPPX_TensorRtExecutionContext* context, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_progress_monitor_versioned_metadata(JYPPX_TensorRtBuilderConfig* config, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_profiler_create_with_callback(JYPPX_TensorRtProfilerCallback callback, void* user_state, JYPPX_TensorRtProfiler** out_profiler);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_profiler_emit_diagnostic(JYPPX_TensorRtProfiler* profiler, const char* layer_name, float milliseconds, JYPPX_Boolean* out_callback_failed);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_allocator_owner_dry_run_create(JYPPX_TensorRtAllocatorOwner** out_owner);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_allocator_owner_dry_run_emit_diagnostic(JYPPX_TensorRtAllocatorOwner* owner, uint64_t size, uint64_t alignment, const char* reason, JYPPX_TensorRtAllocatorOwnerDiagnosticInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_allocator_owner_dry_run_get_info(JYPPX_TensorRtAllocatorOwner* owner, JYPPX_TensorRtAllocatorOwnerDiagnosticInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_allocator_owner_dry_run_get_state(JYPPX_TensorRtAllocatorOwner* owner, JYPPX_TensorRtAllocatorOwnerStateInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_allocator_owner_dry_run_attach_intent(JYPPX_TensorRtAllocatorOwner* owner, const char* target_kind, JYPPX_TensorRtAllocatorOwnerStateInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_allocator_owner_dry_run_detach_intent(JYPPX_TensorRtAllocatorOwner* owner, const char* target_kind, JYPPX_TensorRtAllocatorOwnerStateInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_allocator_owner_dry_run_ledger_record_allocation_intent(JYPPX_TensorRtAllocatorOwner* owner, uint64_t size, uint64_t alignment, uint64_t stream_value, JYPPX_TensorRtAllocatorOwnerStateInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_allocator_owner_dry_run_ledger_record_release_intent(JYPPX_TensorRtAllocatorOwner* owner, uint64_t allocation_id, uint64_t stream_value, JYPPX_TensorRtAllocatorOwnerStateInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_create(JYPPX_TensorRtLogger* logger, JYPPX_TensorRtRuntime** out_runtime);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_set_dla_core(JYPPX_TensorRtRuntime* runtime, int32_t dla_core);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_get_dla_core(JYPPX_TensorRtRuntime* runtime, int32_t* out_dla_core);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_get_dla_core_count(JYPPX_TensorRtRuntime* runtime, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_set_max_threads(JYPPX_TensorRtRuntime* runtime, int32_t max_threads, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_get_max_threads(JYPPX_TensorRtRuntime* runtime, int32_t* out_max_threads);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_set_temporary_directory(JYPPX_TensorRtRuntime* runtime, const char* directory_path);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_get_temporary_directory(JYPPX_TensorRtRuntime* runtime, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_clear_temporary_directory(JYPPX_TensorRtRuntime* runtime);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_set_tempfile_control_flags(JYPPX_TensorRtRuntime* runtime, uint32_t flags);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_get_tempfile_control_flags(JYPPX_TensorRtRuntime* runtime, uint32_t* out_flags);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_set_engine_host_code_allowed(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean allowed);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_get_engine_host_code_allowed(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_allowed);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_has_error_recorder(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_has_recorder);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_has_logger(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_has_logger);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_get_error_recorder_snapshot_info(JYPPX_TensorRtRuntime* runtime, JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_get_error_recorder_error(JYPPX_TensorRtRuntime* runtime, int32_t index, JYPPX_TensorRtErrorRecordInfo* out_error);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_clear_error_recorder(JYPPX_TensorRtRuntime* runtime);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_clear_gpu_allocator(JYPPX_TensorRtRuntime* runtime);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_registry_exists(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_exists);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_registry_get_creator_count(JYPPX_TensorRtRuntime* runtime, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_registry_has_error_recorder(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_has_recorder);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_registry_is_parent_search_enabled(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_creator_get_name(JYPPX_TensorRtRuntime* runtime, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_creator_get_version(JYPPX_TensorRtRuntime* runtime, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_creator_get_namespace(JYPPX_TensorRtRuntime* runtime, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_creator_get_interface_info(JYPPX_TensorRtRuntime* runtime, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_creator_get_api_language(JYPPX_TensorRtRuntime* runtime, int32_t creator_index, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_creator_get_field_count(JYPPX_TensorRtRuntime* runtime, int32_t creator_index, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_creator_get_field_name(JYPPX_TensorRtRuntime* runtime, int32_t creator_index, int32_t field_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_creator_get_field_metadata(JYPPX_TensorRtRuntime* runtime, int32_t creator_index, int32_t field_index, int32_t* out_field_type, int32_t* out_length, JYPPX_Boolean* out_has_data);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_creator_lookup(JYPPX_TensorRtRuntime* runtime, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, JYPPX_Boolean* out_found);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_plugin_creator_lookup_get_api_language(JYPPX_TensorRtRuntime* runtime, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_create(JYPPX_TensorRtLogger* logger, JYPPX_TensorRtBuilder** out_builder);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_progress_monitor(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtProgressMonitor* monitor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_has_progress_monitor(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_has_monitor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_clear_progress_monitor(JYPPX_TensorRtBuilderConfig* config);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_clear_plugins_to_serialize(JYPPX_TensorRtBuilderConfig* config);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_plugins_to_serialize(JYPPX_TensorRtBuilderConfig* config, const char** paths, int32_t path_count, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_plugin_to_serialize_count(JYPPX_TensorRtBuilderConfig* config, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_nb_plugins_to_serialize(JYPPX_TensorRtBuilderConfig* config, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_plugin_to_serialize(JYPPX_TensorRtBuilderConfig* config, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_platform_has_fast_fp16(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_supported);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_platform_has_fast_int8(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_supported);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_platform_has_tf32(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_supported);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_get_dla_core_count(JYPPX_TensorRtBuilder* builder, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_get_max_dla_batch_size(JYPPX_TensorRtBuilder* builder, int32_t* out_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_set_max_threads(JYPPX_TensorRtBuilder* builder, int32_t max_threads, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_get_max_threads(JYPPX_TensorRtBuilder* builder, int32_t* out_max_threads);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_is_network_supported(JYPPX_TensorRtBuilder* builder, JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_supported);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_has_logger(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_has_logger);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_plugin_registry_get_creator_count(JYPPX_TensorRtBuilder* builder, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_plugin_registry_has_error_recorder(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_has_recorder);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_plugin_registry_is_parent_search_enabled(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_plugin_creator_get_name(JYPPX_TensorRtBuilder* builder, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_plugin_creator_get_version(JYPPX_TensorRtBuilder* builder, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_plugin_creator_get_namespace(JYPPX_TensorRtBuilder* builder, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_plugin_creator_get_interface_info(JYPPX_TensorRtBuilder* builder, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_plugin_creator_get_api_language(JYPPX_TensorRtBuilder* builder, int32_t creator_index, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_plugin_creator_get_field_count(JYPPX_TensorRtBuilder* builder, int32_t creator_index, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_plugin_creator_get_field_name(JYPPX_TensorRtBuilder* builder, int32_t creator_index, int32_t field_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_plugin_creator_get_field_metadata(JYPPX_TensorRtBuilder* builder, int32_t creator_index, int32_t field_index, int32_t* out_field_type, int32_t* out_length, JYPPX_Boolean* out_has_data);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_plugin_creator_lookup_get_api_language(JYPPX_TensorRtBuilder* builder, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_get_infer_lib_version(int32_t* out_version);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_get_infer_lib_major_version(int32_t* out_version);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_get_infer_lib_minor_version(int32_t* out_version);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_get_infer_lib_patch_version(int32_t* out_version);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_get_infer_lib_build_version(int32_t* out_version);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_get_onnx_parser_version(int32_t* out_version);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_has_logger(JYPPX_Boolean* out_has_logger);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_registry_exists(JYPPX_Boolean* out_exists);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_registry_get_creator_count(int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_registry_get_recursive_creator_count(int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_registry_has_error_recorder(JYPPX_Boolean* out_has_recorder);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_registry_is_parent_search_enabled(JYPPX_Boolean* out_enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_get_name(int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_get_version(int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_get_namespace(int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_get_interface_info(int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_get_api_language(int32_t creator_index, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_get_field_count(int32_t creator_index, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_get_field_name(int32_t creator_index, int32_t field_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_get_field_metadata(int32_t creator_index, int32_t field_index, int32_t* out_field_type, int32_t* out_length, JYPPX_Boolean* out_has_data);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_lookup(const char* plugin_name, const char* plugin_version, const char* plugin_namespace, JYPPX_Boolean* out_found);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_lookup_get_interface_info(const char* plugin_name, const char* plugin_version, const char* plugin_namespace, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_lookup_get_api_language(const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_lookup_get_field_count(const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_lookup_get_field_name(const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t field_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_global_plugin_creator_lookup_get_field_metadata(const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t field_index, int32_t* out_field_type, int32_t* out_length, JYPPX_Boolean* out_has_data);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_registry_exists(int32_t capability, JYPPX_Boolean* out_exists);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_registry_get_creator_count(int32_t capability, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_registry_get_recursive_creator_count(int32_t capability, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_registry_has_error_recorder(int32_t capability, JYPPX_Boolean* out_has_recorder);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_registry_is_parent_search_enabled(int32_t capability, JYPPX_Boolean* out_enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_get_name(int32_t capability, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_get_version(int32_t capability, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_get_namespace(int32_t capability, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_get_interface_info(int32_t capability, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_get_api_language(int32_t capability, int32_t creator_index, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_get_field_count(int32_t capability, int32_t creator_index, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_get_field_name(int32_t capability, int32_t creator_index, int32_t field_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_get_field_metadata(int32_t capability, int32_t creator_index, int32_t field_index, int32_t* out_field_type, int32_t* out_length, JYPPX_Boolean* out_has_data);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_lookup(int32_t capability, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, JYPPX_Boolean* out_found);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_lookup_get_interface_info(int32_t capability, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_lookup_get_api_language(int32_t capability, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t* out_api_language);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_lookup_get_field_count(int32_t capability, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_lookup_get_field_name(int32_t capability, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t field_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_capability_plugin_creator_lookup_get_field_metadata(int32_t capability, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t field_index, int32_t* out_field_type, int32_t* out_length, JYPPX_Boolean* out_has_data);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_create_config(JYPPX_TensorRtBuilder* builder, JYPPX_TensorRtBuilderConfig** out_config);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_create_network(JYPPX_TensorRtBuilder* builder, uint32_t creation_flags, JYPPX_TensorRtNetworkDefinition** out_network);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_create_optimization_profile(JYPPX_TensorRtBuilder* builder, JYPPX_TensorRtOptimizationProfile** out_profile);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_optimization_profile_set_shape(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, const JYPPX_TensorRtDims* dims);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_optimization_profile_get_shape(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, JYPPX_TensorRtDims* out_dims);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_optimization_profile_set_shape_values(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, const int32_t* values, int32_t value_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_optimization_profile_get_shape_value_count(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_optimization_profile_get_shape_values(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, int32_t* output_values, int32_t output_count, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_optimization_profile_set_shape_values_v2(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, const int64_t* values, int32_t value_count, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_optimization_profile_get_shape_value_count_v2(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_optimization_profile_get_shape_values_v2(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, int64_t* output_values, int32_t output_count, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_optimization_profile_set_extra_memory_target(JYPPX_TensorRtOptimizationProfile* profile, float target);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_optimization_profile_get_extra_memory_target(JYPPX_TensorRtOptimizationProfile* profile, float* out_target);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_optimization_profile_is_valid(JYPPX_TensorRtOptimizationProfile* profile, JYPPX_Boolean* out_is_valid);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_add_optimization_profile(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtOptimizationProfile* profile, int32_t* out_profile_index);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_profile_stream(JYPPX_TensorRtBuilderConfig* config, JYPPX_CudaStream* stream);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_is_profile_stream_set(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_is_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_optimization_profile_count(JYPPX_TensorRtBuilderConfig* config, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_calibration_profile(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtOptimizationProfile* profile);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_has_calibration_profile(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_has_profile);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_has_algorithm_selector(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_has_selector);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_has_int8_calibrator(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_has_calibrator);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag, JYPPX_Boolean enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag, JYPPX_Boolean* out_enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_engine_capability(JYPPX_TensorRtBuilderConfig* config, int32_t capability);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_engine_capability(JYPPX_TensorRtBuilderConfig* config, int32_t* out_capability);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_preview_feature(JYPPX_TensorRtBuilderConfig* config, int32_t feature, JYPPX_Boolean enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_preview_feature(JYPPX_TensorRtBuilderConfig* config, int32_t feature, JYPPX_Boolean* out_enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_hardware_compatibility_level(JYPPX_TensorRtBuilderConfig* config, int32_t level);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_hardware_compatibility_level(JYPPX_TensorRtBuilderConfig* config, int32_t* out_level);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_runtime_platform(JYPPX_TensorRtBuilderConfig* config, int32_t platform);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_runtime_platform(JYPPX_TensorRtBuilderConfig* config, int32_t* out_platform);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_memory_pool_limit(JYPPX_TensorRtBuilderConfig* config, int32_t pool, size_t pool_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_memory_pool_limit(JYPPX_TensorRtBuilderConfig* config, int32_t pool, size_t* out_pool_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_optimization_level(JYPPX_TensorRtBuilderConfig* config, int32_t level);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_optimization_level(JYPPX_TensorRtBuilderConfig* config, int32_t* out_level);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_profiling_verbosity(JYPPX_TensorRtBuilderConfig* config, int32_t verbosity);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_profiling_verbosity(JYPPX_TensorRtBuilderConfig* config, int32_t* out_verbosity);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_max_aux_streams(JYPPX_TensorRtBuilderConfig* config, int32_t stream_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_max_aux_streams(JYPPX_TensorRtBuilderConfig* config, int32_t* out_stream_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_average_timing_iterations(JYPPX_TensorRtBuilderConfig* config, int32_t iterations);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_average_timing_iterations(JYPPX_TensorRtBuilderConfig* config, int32_t* out_iterations);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_tactic_sources(JYPPX_TensorRtBuilderConfig* config, uint32_t tactic_sources);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_tactic_sources(JYPPX_TensorRtBuilderConfig* config, uint32_t* out_tactic_sources);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_default_device_type(JYPPX_TensorRtBuilderConfig* config, int32_t device_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_default_device_type(JYPPX_TensorRtBuilderConfig* config, int32_t* out_device_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_dla_core(JYPPX_TensorRtBuilderConfig* config, int32_t dla_core);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_dla_core(JYPPX_TensorRtBuilderConfig* config, int32_t* out_dla_core);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_can_run_on_dla(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_can_run);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_flags(JYPPX_TensorRtBuilderConfig* config, uint32_t* out_flags);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_quantization_flags(JYPPX_TensorRtBuilderConfig* config, uint32_t flags);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_quantization_flags(JYPPX_TensorRtBuilderConfig* config, uint32_t* out_flags);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_clear_quantization_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_quantization_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_quantization_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag, JYPPX_Boolean* out_enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_tiling_optimization_level(JYPPX_TensorRtBuilderConfig* config, int32_t level, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_tiling_optimization_level(JYPPX_TensorRtBuilderConfig* config, int32_t* out_level);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_l2_limit_for_tiling(JYPPX_TensorRtBuilderConfig* config, int64_t bytes, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_l2_limit_for_tiling(JYPPX_TensorRtBuilderConfig* config, int64_t* out_bytes);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_max_nb_tactics(JYPPX_TensorRtBuilderConfig* config, int32_t max_tactics);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_max_nb_tactics(JYPPX_TensorRtBuilderConfig* config, int32_t* out_max_tactics);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_layer_device_type(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, int32_t device_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_layer_device_type(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, int32_t* out_device_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_is_layer_device_type_set(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_is_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_reset_layer_device_type(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_create_timing_cache(JYPPX_TensorRtBuilderConfig* config, const void* blob, size_t blob_size, JYPPX_TensorRtTimingCache** out_cache);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_timing_cache(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtTimingCache* cache, JYPPX_Boolean ignore_mismatch);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_timing_cache_serialize(JYPPX_TensorRtTimingCache* cache, JYPPX_TensorRtHostMemory** out_host_memory);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_timing_cache_combine(JYPPX_TensorRtTimingCache* cache, JYPPX_TensorRtTimingCache* input_cache, JYPPX_Boolean ignore_mismatch, JYPPX_Boolean* out_combined);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_timing_cache_reset(JYPPX_TensorRtTimingCache* cache, JYPPX_Boolean* out_reset);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_timing_cache_query_key_count(JYPPX_TensorRtTimingCache* cache, int64_t* out_key_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_timing_cache_copy_keys(JYPPX_TensorRtTimingCache* cache, uint8_t* output_keys, int64_t key_capacity, int64_t* out_key_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_timing_cache_query(JYPPX_TensorRtTimingCache* cache, const uint8_t* key_data, int32_t key_size, uint64_t* out_tactic_hash, float* out_timing_msec, JYPPX_Boolean* out_found);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_timing_cache_update(JYPPX_TensorRtTimingCache* cache, const uint8_t* key_data, int32_t key_size, uint64_t tactic_hash, float timing_msec, JYPPX_Boolean* out_updated);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_host_memory_get_type(JYPPX_TensorRtHostMemory* host_memory, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_input(JYPPX_TensorRtNetworkDefinition* network, const char* name, int32_t data_type, const JYPPX_TensorRtDims* dims, JYPPX_TensorRtTensor** out_tensor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_mark_output(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_unmark_output(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_get_input_count(JYPPX_TensorRtNetworkDefinition* network, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_get_output_count(JYPPX_TensorRtNetworkDefinition* network, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_get_layer_count(JYPPX_TensorRtNetworkDefinition* network, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_get_layer(JYPPX_TensorRtNetworkDefinition* network, int32_t index, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_get_name(JYPPX_TensorRtNetworkDefinition* network, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_set_name(JYPPX_TensorRtNetworkDefinition* network, const char* name);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_get_flags(JYPPX_TensorRtNetworkDefinition* network, uint32_t* out_flags);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_has_implicit_batch_dimension(JYPPX_TensorRtNetworkDefinition* network, JYPPX_Boolean* out_has_implicit_batch_dimension);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_get_flag(JYPPX_TensorRtNetworkDefinition* network, int32_t flag, JYPPX_Boolean* out_enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_get_input(JYPPX_TensorRtNetworkDefinition* network, int32_t index, JYPPX_TensorRtTensor** out_tensor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_get_output(JYPPX_TensorRtNetworkDefinition* network, int32_t index, JYPPX_TensorRtTensor** out_tensor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_identity(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_constant(JYPPX_TensorRtNetworkDefinition* network, const JYPPX_TensorRtDims* dims, int32_t data_type, const void* values, size_t value_count, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_convolution_nd(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t output_maps, const JYPPX_TensorRtDims* kernel_size, int32_t data_type, const void* kernel_values, size_t kernel_value_count, const void* bias_values, size_t bias_value_count, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_deconvolution_nd(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t output_maps, const JYPPX_TensorRtDims* kernel_size, int32_t data_type, const void* kernel_values, size_t kernel_value_count, const void* bias_values, size_t bias_value_count, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_scale_nd(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t mode, int32_t data_type, const void* shift_values, size_t shift_value_count, const void* scale_values, size_t scale_value_count, const void* power_values, size_t power_value_count, int32_t channel_axis, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_padding_nd(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, const JYPPX_TensorRtDims* pre_padding, const JYPPX_TensorRtDims* post_padding, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_elementwise(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* left_tensor, JYPPX_TensorRtTensor* right_tensor, int32_t operation, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_matrix_multiply(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* left_tensor, int32_t left_operation, JYPPX_TensorRtTensor* right_tensor, int32_t right_operation, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_matrix_multiply_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t input_index, int32_t operation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_matrix_multiply_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t input_index, int32_t* out_operation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_shuffle(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_shuffle_layer_set_reshape_dimensions(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* dims);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_shuffle_layer_get_reshape_dimensions(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_dims);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_shuffle_layer_set_first_transpose(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* permutation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_shuffle_layer_get_first_transpose(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_permutation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_shuffle_layer_set_second_transpose(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* permutation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_shuffle_layer_get_second_transpose(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_permutation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_shuffle_layer_set_zero_is_placeholder(JYPPX_TensorRtLayer* layer, JYPPX_Boolean zero_is_placeholder);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_shuffle_layer_get_zero_is_placeholder(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_zero_is_placeholder);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_reduce(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t operation, uint32_t axes, int32_t keep_dimensions, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_reduce_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t* out_operation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_reduce_layer_get_axes(JYPPX_TensorRtLayer* layer, uint32_t* out_axes);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_reduce_layer_get_keep_dimensions(JYPPX_TensorRtLayer* layer, int32_t* out_keep_dimensions);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_concatenation(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor** input_tensors, int32_t input_count, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_concatenation_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_concatenation_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_slice(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, const JYPPX_TensorRtDims* start, const JYPPX_TensorRtDims* size, const JYPPX_TensorRtDims* stride, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_slice_layer_set_start(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* start);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_slice_layer_get_start(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_start);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_slice_layer_set_size(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_slice_layer_get_size(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_slice_layer_set_stride(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* stride);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_slice_layer_get_stride(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_stride);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_slice_layer_set_mode(JYPPX_TensorRtLayer* layer, int32_t mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_slice_layer_get_mode(JYPPX_TensorRtLayer* layer, int32_t* out_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_slice_layer_set_axes(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* axes);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_slice_layer_get_axes(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_axes);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_softmax(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_softmax_layer_set_axes(JYPPX_TensorRtLayer* layer, uint32_t axes);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_softmax_layer_get_axes(JYPPX_TensorRtLayer* layer, uint32_t* out_axes);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_unary(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t operation, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_unary_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t* out_operation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_topk(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t operation, int32_t k, uint32_t axes, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_topk_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t* out_operation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_topk_layer_get_k(JYPPX_TensorRtLayer* layer, int32_t* out_k);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_topk_layer_get_axes(JYPPX_TensorRtLayer* layer, uint32_t* out_axes);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_gather(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* data_tensor, JYPPX_TensorRtTensor* indices_tensor, int32_t axis, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_gather_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_activation(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t activation_type, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_activation_layer_get_type(JYPPX_TensorRtLayer* layer, int32_t* out_activation_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_pooling_nd(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t pooling_type, const JYPPX_TensorRtDims* window_size, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_get_type(JYPPX_TensorRtLayer* layer, int32_t* out_pooling_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_set_window_size_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* window_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_get_window_size_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_window_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_set_stride_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* stride);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_get_stride_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_stride);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_set_padding_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_get_padding_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_lrn(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t window_size, float alpha, float beta, float k, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_lrn_layer_get_window_size(JYPPX_TensorRtLayer* layer, int32_t* out_window_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_lrn_layer_set_window_size(JYPPX_TensorRtLayer* layer, int32_t window_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_lrn_layer_get_alpha(JYPPX_TensorRtLayer* layer, float* out_alpha);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_lrn_layer_set_alpha(JYPPX_TensorRtLayer* layer, float alpha);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_lrn_layer_get_beta(JYPPX_TensorRtLayer* layer, float* out_beta);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_lrn_layer_set_beta(JYPPX_TensorRtLayer* layer, float beta);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_lrn_layer_get_k(JYPPX_TensorRtLayer* layer, float* out_k);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_lrn_layer_set_k(JYPPX_TensorRtLayer* layer, float k);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_quantize(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* scale_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_dequantize(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* scale_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_scatter(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* data_tensor, JYPPX_TensorRtTensor* indices_tensor, JYPPX_TensorRtTensor* updates_tensor, int32_t mode, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_gather_v2(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* data_tensor, JYPPX_TensorRtTensor* indices_tensor, int32_t mode, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_ragged_softmax(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* bounds_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_fill_v2(JYPPX_TensorRtNetworkDefinition* network, const JYPPX_TensorRtDims* dimensions, int32_t operation, int32_t output_type, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_quantize_v2(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* scale_tensor, int32_t output_type, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_dequantize_v2(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* scale_tensor, int32_t output_type, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_parametric_relu(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* slope_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_non_zero(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_squeeze(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* axes_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_unsqueeze(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* axes_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_mark_debug(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_marked);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_unmark_debug(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_unmarked);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_is_debug_tensor(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_debug);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_mark_output_for_shapes(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_marked);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_unmark_output_for_shapes(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_unmarked);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_mark_weights_refittable(JYPPX_TensorRtNetworkDefinition* network, const char* weights_name, JYPPX_Boolean* out_marked);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_unmark_weights_refittable(JYPPX_TensorRtNetworkDefinition* network, const char* weights_name, JYPPX_Boolean* out_unmarked);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_are_weights_marked_refittable(JYPPX_TensorRtNetworkDefinition* network, const char* weights_name, JYPPX_Boolean* out_marked);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_set_weights_name(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtLayer* constant_layer, const char* weights_name, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_plugin_v2_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_plugin_v3_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_get_error_recorder_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_set_error_recorder_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v2_layer_get_plugin_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_plugin_v3_layer_get_plugin_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_dynamic_quantize(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t axis, int32_t block_size, int32_t output_type, int32_t scale_type, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_cumulative(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* axis_tensor, int32_t operation, JYPPX_Boolean exclusive, JYPPX_Boolean reverse, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_reverse_sequence(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* sequence_lengths_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_quantize_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_quantize_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_quantize_layer_set_to_type(JYPPX_TensorRtLayer* layer, int32_t data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_quantize_layer_get_to_type(JYPPX_TensorRtLayer* layer, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_dequantize_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_dequantize_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_dequantize_layer_set_to_type(JYPPX_TensorRtLayer* layer, int32_t data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_dequantize_layer_get_to_type(JYPPX_TensorRtLayer* layer, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_resize(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_set_output_dimensions(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* dimensions);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_get_output_dimensions(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_dimensions);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_set_mode(JYPPX_TensorRtLayer* layer, int32_t resize_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_get_mode(JYPPX_TensorRtLayer* layer, int32_t* out_resize_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_set_scales(JYPPX_TensorRtLayer* layer, const float* scales, int32_t scale_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_get_scales(JYPPX_TensorRtLayer* layer, float* scales, int32_t scale_capacity, int32_t* out_scale_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_shape(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_select(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* condition_tensor, JYPPX_TensorRtTensor* then_tensor, JYPPX_TensorRtTensor* else_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_fill(JYPPX_TensorRtNetworkDefinition* network, const JYPPX_TensorRtDims* dimensions, int32_t operation, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_set_dimensions(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* dimensions);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_get_dimensions(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_dimensions);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t operation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t* out_operation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_set_alpha(JYPPX_TensorRtLayer* layer, double alpha);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_get_alpha(JYPPX_TensorRtLayer* layer, double* out_alpha);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_set_beta(JYPPX_TensorRtLayer* layer, double beta);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_get_beta(JYPPX_TensorRtLayer* layer, double* out_beta);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_set_to_type(JYPPX_TensorRtLayer* layer, int32_t data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_get_to_type(JYPPX_TensorRtLayer* layer, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_set_alpha_int64(JYPPX_TensorRtLayer* layer, int64_t alpha);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_get_alpha_int64(JYPPX_TensorRtLayer* layer, int64_t* out_alpha);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_set_beta_int64(JYPPX_TensorRtLayer* layer, int64_t beta);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_get_beta_int64(JYPPX_TensorRtLayer* layer, int64_t* out_beta);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_fill_layer_is_alpha_beta_int64(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_is_int64);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_reverse_sequence_layer_set_batch_axis(JYPPX_TensorRtLayer* layer, int32_t axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_reverse_sequence_layer_get_batch_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_reverse_sequence_layer_set_sequence_axis(JYPPX_TensorRtLayer* layer, int32_t axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_reverse_sequence_layer_get_sequence_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_cast(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t to_type, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_nms(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* boxes_tensor, JYPPX_TensorRtTensor* scores_tensor, JYPPX_TensorRtTensor* max_output_boxes_per_class_tensor, int32_t indices_type, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_einsum(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor** input_tensors, int32_t input_count, const char* equation, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_one_hot(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* indices_tensor, JYPPX_TensorRtTensor* values_tensor, JYPPX_TensorRtTensor* depth_tensor, int32_t axis, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_assertion(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* condition_tensor, const char* message, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_grid_sample(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* grid_tensor, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_normalization_v2(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* scale_tensor, JYPPX_TensorRtTensor* bias_tensor, uint32_t axes, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_gather_layer_set_mode(JYPPX_TensorRtLayer* layer, int32_t mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_gather_layer_get_mode(JYPPX_TensorRtLayer* layer, int32_t* out_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_gather_layer_set_nb_elementwise_dims(JYPPX_TensorRtLayer* layer, int32_t elementwise_dims);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_gather_layer_get_nb_elementwise_dims(JYPPX_TensorRtLayer* layer, int32_t* out_elementwise_dims);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_scatter_layer_set_mode(JYPPX_TensorRtLayer* layer, int32_t mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_scatter_layer_get_mode(JYPPX_TensorRtLayer* layer, int32_t* out_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_scatter_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_scatter_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_one_hot_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_one_hot_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_assertion_layer_set_message(JYPPX_TensorRtLayer* layer, const char* message);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_assertion_layer_get_message(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_grid_sample_layer_set_interpolation_mode(JYPPX_TensorRtLayer* layer, int32_t interpolation_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_grid_sample_layer_get_interpolation_mode(JYPPX_TensorRtLayer* layer, int32_t* out_interpolation_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_grid_sample_layer_set_align_corners(JYPPX_TensorRtLayer* layer, JYPPX_Boolean align_corners);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_grid_sample_layer_get_align_corners(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_align_corners);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_grid_sample_layer_set_sample_mode(JYPPX_TensorRtLayer* layer, int32_t sample_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_grid_sample_layer_get_sample_mode(JYPPX_TensorRtLayer* layer, int32_t* out_sample_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_dynamic_quantize_layer_set_to_type(JYPPX_TensorRtLayer* layer, int32_t data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_dynamic_quantize_layer_get_to_type(JYPPX_TensorRtLayer* layer, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_dynamic_quantize_layer_set_scale_type(JYPPX_TensorRtLayer* layer, int32_t data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_dynamic_quantize_layer_get_scale_type(JYPPX_TensorRtLayer* layer, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_dynamic_quantize_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_dynamic_quantize_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_dynamic_quantize_layer_set_block_size(JYPPX_TensorRtLayer* layer, int32_t block_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_dynamic_quantize_layer_get_block_size(JYPPX_TensorRtLayer* layer, int32_t* out_block_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cumulative_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t operation, JYPPX_Boolean* out_success);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cumulative_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t* out_operation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cumulative_layer_set_exclusive(JYPPX_TensorRtLayer* layer, JYPPX_Boolean exclusive);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cumulative_layer_get_exclusive(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_exclusive);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cumulative_layer_set_reverse(JYPPX_TensorRtLayer* layer, JYPPX_Boolean reverse);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cumulative_layer_get_reverse(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_reverse);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_normalization_layer_set_epsilon(JYPPX_TensorRtLayer* layer, double epsilon);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_normalization_layer_get_epsilon(JYPPX_TensorRtLayer* layer, double* out_epsilon);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_normalization_layer_set_axes(JYPPX_TensorRtLayer* layer, uint32_t axes);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_normalization_layer_get_axes(JYPPX_TensorRtLayer* layer, uint32_t* out_axes);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_normalization_layer_set_nb_groups(JYPPX_TensorRtLayer* layer, int64_t group_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_normalization_layer_get_nb_groups(JYPPX_TensorRtLayer* layer, int64_t* out_group_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_normalization_layer_set_compute_precision(JYPPX_TensorRtLayer* layer, int32_t data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_normalization_layer_get_compute_precision(JYPPX_TensorRtLayer* layer, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cast_layer_set_to_type(JYPPX_TensorRtLayer* layer, int32_t data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cast_layer_get_to_type(JYPPX_TensorRtLayer* layer, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_nms_layer_set_bounding_box_format(JYPPX_TensorRtLayer* layer, int32_t format);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_nms_layer_get_bounding_box_format(JYPPX_TensorRtLayer* layer, int32_t* out_format);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_nms_layer_set_topk_box_limit(JYPPX_TensorRtLayer* layer, int32_t limit);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_nms_layer_get_topk_box_limit(JYPPX_TensorRtLayer* layer, int32_t* out_limit);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_constant_layer_set_dimensions(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* dimensions);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_constant_layer_get_dimensions(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_dimensions);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_einsum_layer_set_equation(JYPPX_TensorRtLayer* layer, const char* equation, JYPPX_Boolean* out_success);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_einsum_layer_get_equation(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_get_output(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_TensorRtTensor** out_tensor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_get_input(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_TensorRtTensor** out_tensor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_get_input_count(JYPPX_TensorRtLayer* layer, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_get_output_count(JYPPX_TensorRtLayer* layer, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_get_type(JYPPX_TensorRtLayer* layer, int32_t* out_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_get_name(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_set_name(JYPPX_TensorRtLayer* layer, const char* name);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_get_metadata(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_set_metadata(JYPPX_TensorRtLayer* layer, const char* metadata);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_set_precision(JYPPX_TensorRtLayer* layer, int32_t data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_get_precision(JYPPX_TensorRtLayer* layer, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_precision_is_set(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_is_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_reset_precision(JYPPX_TensorRtLayer* layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_set_output_type(JYPPX_TensorRtLayer* layer, int32_t index, int32_t data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_get_output_type(JYPPX_TensorRtLayer* layer, int32_t index, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_output_type_is_set(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_Boolean* out_is_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_layer_reset_output_type(JYPPX_TensorRtLayer* layer, int32_t index);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_constant_layer_get_weights_info(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtWeightsInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_get_kernel_weights_info(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtWeightsInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_get_bias_weights_info(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtWeightsInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_get_kernel_weights_info(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtWeightsInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_get_bias_weights_info(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtWeightsInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_scale_layer_get_shift_weights_info(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtWeightsInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_scale_layer_get_scale_weights_info(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtWeightsInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_scale_layer_get_power_weights_info(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtWeightsInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_get_name(JYPPX_TensorRtTensor* tensor, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_set_name(JYPPX_TensorRtTensor* tensor, const char* name);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_get_data_type(JYPPX_TensorRtTensor* tensor, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_set_data_type(JYPPX_TensorRtTensor* tensor, int32_t data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_get_shape(JYPPX_TensorRtTensor* tensor, JYPPX_TensorRtDims* out_dims);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_set_shape(JYPPX_TensorRtTensor* tensor, const JYPPX_TensorRtDims* dims);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_get_dimensions(JYPPX_TensorRtTensor* tensor, JYPPX_TensorRtDims* out_dims);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_set_dimensions(JYPPX_TensorRtTensor* tensor, const JYPPX_TensorRtDims* dims);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_get_location(JYPPX_TensorRtTensor* tensor, int32_t* out_location);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_set_location(JYPPX_TensorRtTensor* tensor, int32_t location);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_get_allowed_formats(JYPPX_TensorRtTensor* tensor, uint32_t* out_formats);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_set_allowed_formats(JYPPX_TensorRtTensor* tensor, uint32_t formats);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_set_dynamic_range(JYPPX_TensorRtTensor* tensor, float min, float max);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_dynamic_range_is_set(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_get_dynamic_range_min(JYPPX_TensorRtTensor* tensor, float* out_min);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_get_dynamic_range_max(JYPPX_TensorRtTensor* tensor, float* out_max);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_reset_dynamic_range(JYPPX_TensorRtTensor* tensor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_get_broadcast_across_batch(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_broadcast_across_batch);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_set_broadcast_across_batch(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean broadcast_across_batch);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_is_network_input(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_network_input);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_is_network_output(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_network_output);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_is_shape_tensor(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_shape_tensor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_is_execution_tensor(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_execution_tensor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_get_dimension_name(JYPPX_TensorRtTensor* tensor, int32_t dimension_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_set_dimension_name(JYPPX_TensorRtTensor* tensor, int32_t dimension_index, const char* name);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_tensor_clear_dimension_name(JYPPX_TensorRtTensor* tensor, int32_t dimension_index);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_loop(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtLoop** out_loop);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_network_add_if_conditional(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtIfConditional** out_conditional);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_loop_set_name(JYPPX_TensorRtLoop* loop, const char* name);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_loop_get_name(JYPPX_TensorRtLoop* loop, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_loop_add_recurrence(JYPPX_TensorRtLoop* loop, JYPPX_TensorRtTensor* initial_value, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_loop_add_trip_limit(JYPPX_TensorRtLoop* loop, JYPPX_TensorRtTensor* tensor, int32_t limit, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_loop_add_iterator(JYPPX_TensorRtLoop* loop, JYPPX_TensorRtTensor* tensor, int32_t axis, JYPPX_Boolean reverse, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_loop_add_output(JYPPX_TensorRtLoop* loop, JYPPX_TensorRtTensor* tensor, int32_t output_kind, int32_t axis, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_if_conditional_set_name(JYPPX_TensorRtIfConditional* conditional, const char* name);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_if_conditional_get_name(JYPPX_TensorRtIfConditional* conditional, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_if_conditional_set_condition(JYPPX_TensorRtIfConditional* conditional, JYPPX_TensorRtTensor* condition, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_if_conditional_add_input(JYPPX_TensorRtIfConditional* conditional, JYPPX_TensorRtTensor* input, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_if_conditional_add_output(JYPPX_TensorRtIfConditional* conditional, JYPPX_TensorRtTensor* true_output, JYPPX_TensorRtTensor* false_output, JYPPX_TensorRtLayer** out_layer);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_loop_boundary_layer_get_loop_name(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_if_conditional_boundary_layer_get_conditional_name(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_loop_output_layer_get_kind(JYPPX_TensorRtLayer* layer, int32_t* out_kind);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_loop_output_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_loop_output_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_trip_limit_layer_get_kind(JYPPX_TensorRtLayer* layer, int32_t* out_kind);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_iterator_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_iterator_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_iterator_layer_set_reverse(JYPPX_TensorRtLayer* layer, JYPPX_Boolean reverse);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_iterator_layer_get_reverse(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_reverse);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_create(JYPPX_TensorRtLogger* logger, JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtOnnxParser** out_parser);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_parse_from_file(JYPPX_TensorRtOnnxParser* parser, const char* file_path, int32_t verbosity, JYPPX_Boolean* out_parsed);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_parse_from_memory(JYPPX_TensorRtOnnxParser* parser, const void* model_data, size_t model_size, const char* model_path, JYPPX_Boolean* out_parsed);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_get_error_count(JYPPX_TensorRtOnnxParser* parser, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_get_error(JYPPX_TensorRtOnnxParser* parser, int32_t index, JYPPX_TensorRtParserErrorInfo* out_error);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_error_get_code(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_code);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_error_get_line(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_line);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_error_get_node(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_node);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_error_get_description(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_error_get_file(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_error_get_function(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_error_get_node_name(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_error_get_node_operator(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_error_get_local_function_stack_size(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_error_get_local_function_stack_entry(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t stack_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_clear_errors(JYPPX_TensorRtOnnxParser* parser);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_get_flags(JYPPX_TensorRtOnnxParser* parser, uint32_t* out_flags);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_set_flags(JYPPX_TensorRtOnnxParser* parser, uint32_t flags);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_get_flag(JYPPX_TensorRtOnnxParser* parser, int32_t flag, JYPPX_Boolean* out_enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_set_flag(JYPPX_TensorRtOnnxParser* parser, int32_t flag);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_clear_flag(JYPPX_TensorRtOnnxParser* parser, int32_t flag);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_supports_operator(JYPPX_TensorRtOnnxParser* parser, const char* operator_name, JYPPX_Boolean* out_supported);
/* BEGIN TRT10 TWENTY-NINTH BATCH ONNX PARSER SUPPORT DECLARATIONS */
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_get_used_vc_plugin_library_count(JYPPX_TensorRtOnnxParser* parser, int64_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_get_used_vc_plugin_library(JYPPX_TensorRtOnnxParser* parser, int64_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_supports_model_v2(JYPPX_TensorRtOnnxParser* parser, const void* model_data, size_t model_size, const char* model_path, JYPPX_Boolean* out_supported);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_get_subgraph_count(JYPPX_TensorRtOnnxParser* parser, int64_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_is_subgraph_supported(JYPPX_TensorRtOnnxParser* parser, int64_t index, JYPPX_Boolean* out_supported);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_get_subgraph_node_count(JYPPX_TensorRtOnnxParser* parser, int64_t index, int64_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_get_subgraph_node(JYPPX_TensorRtOnnxParser* parser, int64_t subgraph_index, int64_t node_index, int64_t* out_node);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_layer_output_tensor_exists(JYPPX_TensorRtOnnxParser* parser, const char* layer_name, int64_t output_index, JYPPX_Boolean* out_exists);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_get_supported_subgraph_count(JYPPX_TensorRtOnnxParser* parser, int64_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_parser_get_unsupported_subgraph_count(JYPPX_TensorRtOnnxParser* parser, int64_t* out_count);
/* END TRT10 TWENTY-NINTH BATCH ONNX PARSER SUPPORT DECLARATIONS */
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_create(JYPPX_TensorRtOnnxConfig** out_config);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_get_model_dtype(JYPPX_TensorRtOnnxConfig* config, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_set_model_dtype(JYPPX_TensorRtOnnxConfig* config, int32_t data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_get_verbosity_level(JYPPX_TensorRtOnnxConfig* config, int32_t* out_verbosity);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_set_verbosity_level(JYPPX_TensorRtOnnxConfig* config, int32_t verbosity);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_add_verbosity(JYPPX_TensorRtOnnxConfig* config);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_reduce_verbosity(JYPPX_TensorRtOnnxConfig* config);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_get_model_file_name(JYPPX_TensorRtOnnxConfig* config, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_set_model_file_name(JYPPX_TensorRtOnnxConfig* config, const char* value);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_get_text_file_name(JYPPX_TensorRtOnnxConfig* config, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_set_text_file_name(JYPPX_TensorRtOnnxConfig* config, const char* value);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_get_full_text_file_name(JYPPX_TensorRtOnnxConfig* config, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_set_full_text_file_name(JYPPX_TensorRtOnnxConfig* config, const char* value);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_get_print_layer_info(JYPPX_TensorRtOnnxConfig* config, JYPPX_Boolean* out_enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_set_print_layer_info(JYPPX_TensorRtOnnxConfig* config, JYPPX_Boolean enabled);
/* BEGIN TRT10 PARSER REFITTER DIAGNOSTICS DECLARATIONS */
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_parser_refitter_create(JYPPX_TensorRtRefitter* refitter, JYPPX_TensorRtLogger* logger, JYPPX_TensorRtOnnxParserRefitter** out_parser_refitter);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_parser_refitter_get_error_count(JYPPX_TensorRtOnnxParserRefitter* parser_refitter, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_parser_refitter_get_error(JYPPX_TensorRtOnnxParserRefitter* parser_refitter, int32_t index, JYPPX_TensorRtParserErrorInfo* out_error);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_parser_refitter_error_get_description(JYPPX_TensorRtOnnxParserRefitter* parser_refitter, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_parser_refitter_error_get_file(JYPPX_TensorRtOnnxParserRefitter* parser_refitter, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_parser_refitter_error_get_function(JYPPX_TensorRtOnnxParserRefitter* parser_refitter, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_parser_refitter_error_get_node_name(JYPPX_TensorRtOnnxParserRefitter* parser_refitter, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_parser_refitter_error_get_node_operator(JYPPX_TensorRtOnnxParserRefitter* parser_refitter, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_parser_refitter_error_get_local_function_stack_size(JYPPX_TensorRtOnnxParserRefitter* parser_refitter, int32_t index, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_parser_refitter_error_get_local_function_stack_entry(JYPPX_TensorRtOnnxParserRefitter* parser_refitter, int32_t index, int32_t stack_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_parser_refitter_clear_errors(JYPPX_TensorRtOnnxParserRefitter* parser_refitter);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_parser_refitter_refit_from_bytes(JYPPX_TensorRtOnnxParserRefitter* parser_refitter, const void* model_data, size_t model_size, const char* model_path, JYPPX_Boolean* out_refitted);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_parser_refitter_refit_from_file(JYPPX_TensorRtOnnxParserRefitter* parser_refitter, const char* file_path, JYPPX_Boolean* out_refitted);
/* END TRT10 PARSER REFITTER DIAGNOSTICS DECLARATIONS */
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_build_serialized_network(JYPPX_TensorRtBuilder* builder, JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtHostMemory** out_host_memory);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_host_memory_get_size(JYPPX_TensorRtHostMemory* host_memory, size_t* out_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_host_memory_copy_to_buffer(JYPPX_TensorRtHostMemory* host_memory, void* destination, size_t destination_size, size_t* out_bytes_written);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_deserialize_engine(JYPPX_TensorRtRuntime* runtime, const void* engine_data, size_t engine_size, JYPPX_TensorRtCudaEngine** out_engine);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_deserialize_host_memory(JYPPX_TensorRtRuntime* runtime, JYPPX_TensorRtHostMemory* host_memory, JYPPX_TensorRtCudaEngine** out_engine);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_serialize(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtHostMemory** out_host_memory);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_create_serialization_config(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtSerializationConfig** out_config);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_serialize_with_config(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtSerializationConfig* config, JYPPX_TensorRtHostMemory** out_host_memory);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_create_runtime_config(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtRuntimeConfig** out_config);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_create_execution_context_with_runtime_config(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtRuntimeConfig* runtime_config, JYPPX_TensorRtExecutionContext** out_context);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_serialization_config_set_flags(JYPPX_TensorRtSerializationConfig* config, uint32_t flags, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_serialization_config_get_flags(JYPPX_TensorRtSerializationConfig* config, uint32_t* out_flags);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_serialization_config_set_flag(JYPPX_TensorRtSerializationConfig* config, int32_t flag, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_serialization_config_clear_flag(JYPPX_TensorRtSerializationConfig* config, int32_t flag, JYPPX_Boolean* out_cleared);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_serialization_config_get_flag(JYPPX_TensorRtSerializationConfig* config, int32_t flag, JYPPX_Boolean* out_enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_config_set_execution_context_allocation_strategy(JYPPX_TensorRtRuntimeConfig* config, int32_t strategy);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_runtime_config_get_execution_context_allocation_strategy(JYPPX_TensorRtRuntimeConfig* config, int32_t* out_strategy);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_streamable_weights_size(JYPPX_TensorRtCudaEngine* engine, int64_t* out_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_minimum_weight_streaming_budget(JYPPX_TensorRtCudaEngine* engine, int64_t* out_budget);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_set_weight_streaming_budget_v2(JYPPX_TensorRtCudaEngine* engine, int64_t budget, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_weight_streaming_budget_v2(JYPPX_TensorRtCudaEngine* engine, int64_t* out_budget);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_weight_streaming_automatic_budget(JYPPX_TensorRtCudaEngine* engine, int64_t* out_budget);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_weight_streaming_scratch_memory_size(JYPPX_TensorRtCudaEngine* engine, int64_t* out_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_hardware_compatibility_level(JYPPX_TensorRtCudaEngine* engine, int32_t* out_level);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_io_tensor_count(JYPPX_TensorRtCudaEngine* engine, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_io_tensor_info(JYPPX_TensorRtCudaEngine* engine, int32_t index, JYPPX_TensorRtTensorInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_device_memory_size(JYPPX_TensorRtCudaEngine* engine, size_t* out_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_device_memory_size_for_profile(JYPPX_TensorRtCudaEngine* engine, int32_t profile_index, size_t* out_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_device_memory_size_v2(JYPPX_TensorRtCudaEngine* engine, size_t* out_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_device_memory_size_for_profile_v2(JYPPX_TensorRtCudaEngine* engine, int32_t profile_index, size_t* out_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_nb_aux_streams(JYPPX_TensorRtCudaEngine* engine, int32_t* out_stream_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_is_debug_tensor(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, JYPPX_Boolean* out_is_debug_tensor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_optimization_profile_count(JYPPX_TensorRtCudaEngine* engine, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_io_tensor_name(JYPPX_TensorRtCudaEngine* engine, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_index(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_index);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_data_type(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_data_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_shape(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, JYPPX_TensorRtDims* out_shape);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_io_mode(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_io_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_create_inspector(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtEngineInspector** out_inspector);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_create_execution_context(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtExecutionContext** out_context);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_create_execution_context_without_device_memory(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtExecutionContext** out_context);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_inspector_set_execution_context(JYPPX_TensorRtEngineInspector* inspector, JYPPX_TensorRtExecutionContext* context);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_inspector_get_engine_information(JYPPX_TensorRtEngineInspector* inspector, int32_t format, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_inspector_get_layer_information(JYPPX_TensorRtEngineInspector* inspector, int32_t layer_index, int32_t format, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_input_shape(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, const JYPPX_TensorRtDims* dims);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_tensor_address(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_CudaMemory* memory);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_input_tensor_address(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_CudaMemory* memory);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_output_tensor_address(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_CudaMemory* memory);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_enqueue_async(JYPPX_TensorRtExecutionContext* context, JYPPX_CudaStream* stream);

JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_reduce_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t operation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_reduce_layer_set_axes(JYPPX_TensorRtLayer* layer, uint32_t axes);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_reduce_layer_set_keep_dimensions(JYPPX_TensorRtLayer* layer, JYPPX_Boolean keep_dimensions);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_unary_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t operation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_topk_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t operation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_topk_layer_set_k(JYPPX_TensorRtLayer* layer, int32_t k);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_topk_layer_set_axes(JYPPX_TensorRtLayer* layer, uint32_t axes);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_gather_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_elementwise_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t* out_operation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_elementwise_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t operation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_activation_layer_set_type(JYPPX_TensorRtLayer* layer, int32_t activation_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_activation_layer_get_alpha(JYPPX_TensorRtLayer* layer, double* out_alpha);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_activation_layer_set_alpha(JYPPX_TensorRtLayer* layer, double alpha);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_activation_layer_get_beta(JYPPX_TensorRtLayer* layer, double* out_beta);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_activation_layer_set_beta(JYPPX_TensorRtLayer* layer, double beta);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_set_type(JYPPX_TensorRtLayer* layer, int32_t pooling_type);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_get_blend_factor(JYPPX_TensorRtLayer* layer, double* out_blend_factor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_set_blend_factor(JYPPX_TensorRtLayer* layer, double blend_factor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_get_average_count_excludes_padding(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_excludes_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_set_average_count_excludes_padding(JYPPX_TensorRtLayer* layer, JYPPX_Boolean excludes_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_get_pre_padding(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_pre_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_set_pre_padding(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* pre_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_get_post_padding(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_post_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_set_post_padding(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* post_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_get_padding_mode(JYPPX_TensorRtLayer* layer, int32_t* out_padding_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_pooling_layer_set_padding_mode(JYPPX_TensorRtLayer* layer, int32_t padding_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_get_nb_output_maps(JYPPX_TensorRtLayer* layer, int32_t* out_output_maps);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_set_nb_output_maps(JYPPX_TensorRtLayer* layer, int32_t output_maps);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_get_nb_groups(JYPPX_TensorRtLayer* layer, int32_t* out_groups);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_set_nb_groups(JYPPX_TensorRtLayer* layer, int32_t groups);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_get_stride_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_stride);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_set_stride_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* stride);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_get_padding_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_set_padding_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_get_pre_padding(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_pre_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_set_pre_padding(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* pre_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_get_post_padding(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_post_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_set_post_padding(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* post_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_get_dilation_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_dilation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_set_dilation_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* dilation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_get_padding_mode(JYPPX_TensorRtLayer* layer, int32_t* out_padding_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_convolution_layer_set_padding_mode(JYPPX_TensorRtLayer* layer, int32_t padding_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_get_nb_output_maps(JYPPX_TensorRtLayer* layer, int32_t* out_output_maps);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_set_nb_output_maps(JYPPX_TensorRtLayer* layer, int32_t output_maps);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_get_nb_groups(JYPPX_TensorRtLayer* layer, int32_t* out_groups);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_set_nb_groups(JYPPX_TensorRtLayer* layer, int32_t groups);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_get_kernel_size_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_kernel_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_set_kernel_size_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* kernel_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_get_stride_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_stride);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_set_stride_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* stride);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_get_padding_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_set_padding_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_get_pre_padding(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_pre_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_set_pre_padding(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* pre_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_get_post_padding(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_post_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_set_post_padding(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* post_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_get_dilation_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_dilation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_set_dilation_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* dilation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_get_padding_mode(JYPPX_TensorRtLayer* layer, int32_t* out_padding_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_deconvolution_layer_set_padding_mode(JYPPX_TensorRtLayer* layer, int32_t padding_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_scale_layer_get_mode(JYPPX_TensorRtLayer* layer, int32_t* out_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_scale_layer_set_mode(JYPPX_TensorRtLayer* layer, int32_t mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_scale_layer_get_channel_axis(JYPPX_TensorRtLayer* layer, int32_t* out_channel_axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_scale_layer_set_channel_axis(JYPPX_TensorRtLayer* layer, int32_t channel_axis);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_padding_layer_get_pre_padding_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_pre_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_padding_layer_set_pre_padding_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* pre_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_padding_layer_get_post_padding_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_post_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_padding_layer_set_post_padding_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* post_padding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_get_coordinate_transformation(JYPPX_TensorRtLayer* layer, int32_t* out_coordinate_transformation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_set_coordinate_transformation(JYPPX_TensorRtLayer* layer, int32_t coordinate_transformation);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_get_selector_for_single_pixel(JYPPX_TensorRtLayer* layer, int32_t* out_selector);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_set_selector_for_single_pixel(JYPPX_TensorRtLayer* layer, int32_t selector);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_get_nearest_rounding(JYPPX_TensorRtLayer* layer, int32_t* out_rounding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_set_nearest_rounding(JYPPX_TensorRtLayer* layer, int32_t rounding);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_get_cubic_coeff(JYPPX_TensorRtLayer* layer, double* out_cubic_coeff);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_set_cubic_coeff(JYPPX_TensorRtLayer* layer, double cubic_coeff);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_get_exclude_outside(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_exclude_outside);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_set_exclude_outside(JYPPX_TensorRtLayer* layer, JYPPX_Boolean exclude_outside);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_get_resize_mode(JYPPX_TensorRtLayer* layer, int32_t* out_resize_mode);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_resize_layer_set_resize_mode(JYPPX_TensorRtLayer* layer, int32_t resize_mode);

JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_layer_count(JYPPX_TensorRtCudaEngine* engine, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_is_refittable(JYPPX_TensorRtCudaEngine* engine, JYPPX_Boolean* out_refittable);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_create_refitter(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtLogger* logger, JYPPX_TensorRtRefitter** out_refitter);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_missing_count(JYPPX_TensorRtRefitter* refitter, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_all_count(JYPPX_TensorRtRefitter* refitter, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_missing_entries(JYPPX_TensorRtRefitter* refitter, JYPPX_TensorRtRefitEntryInfo* output_entries, int32_t output_count, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_all_entries(JYPPX_TensorRtRefitter* refitter, JYPPX_TensorRtRefitEntryInfo* output_entries, int32_t output_count, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_set_weights(JYPPX_TensorRtRefitter* refitter, const char* layer_name, int32_t role, int32_t data_type, const void* values, int64_t value_count, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_refit_cuda_engine(JYPPX_TensorRtRefitter* refitter, JYPPX_Boolean* out_refitted);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_set_max_threads(JYPPX_TensorRtRefitter* refitter, int32_t max_threads, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_max_threads(JYPPX_TensorRtRefitter* refitter, int32_t* out_max_threads);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_has_logger(JYPPX_TensorRtRefitter* refitter, JYPPX_Boolean* out_has_logger);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_has_error_recorder(JYPPX_TensorRtRefitter* refitter, JYPPX_Boolean* out_has_recorder);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_clear_error_recorder(JYPPX_TensorRtRefitter* refitter);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_set_dynamic_range(JYPPX_TensorRtRefitter* refitter, const char* tensor_name, float min, float max, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_dynamic_range_min(JYPPX_TensorRtRefitter* refitter, const char* tensor_name, float* out_min);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_dynamic_range_max(JYPPX_TensorRtRefitter* refitter, const char* tensor_name, float* out_max);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_dynamic_range_tensor_count(JYPPX_TensorRtRefitter* refitter, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_dynamic_range_tensor_entries(JYPPX_TensorRtRefitter* refitter, JYPPX_TensorRtRefitEntryInfo* output_entries, int32_t output_count, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_set_named_weights(JYPPX_TensorRtRefitter* refitter, const char* weights_name, int32_t data_type, const void* values, int64_t value_count, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_missing_weights_count(JYPPX_TensorRtRefitter* refitter, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_all_weights_count(JYPPX_TensorRtRefitter* refitter, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_missing_weights_entries(JYPPX_TensorRtRefitter* refitter, JYPPX_TensorRtRefitEntryInfo* output_entries, int32_t output_count, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_all_weights_entries(JYPPX_TensorRtRefitter* refitter, JYPPX_TensorRtRefitEntryInfo* output_entries, int32_t output_count, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_refit_cuda_engine_async(JYPPX_TensorRtRefitter* refitter, JYPPX_CudaStream* stream, JYPPX_Boolean* out_refitted);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_set_weights_validation(JYPPX_TensorRtRefitter* refitter, JYPPX_Boolean enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_weights_validation(JYPPX_TensorRtRefitter* refitter, JYPPX_Boolean* out_enabled);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_unset_named_weights(JYPPX_TensorRtRefitter* refitter, const char* weights_name, JYPPX_Boolean* out_unset);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_weights_location(JYPPX_TensorRtRefitter* refitter, const char* weights_name, int32_t* out_location);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_named_weights_info(JYPPX_TensorRtRefitter* refitter, const char* weights_name, JYPPX_TensorRtWeightsInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_refitter_get_weights_prototype_info(JYPPX_TensorRtRefitter* refitter, const char* weights_name, JYPPX_TensorRtWeightsInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_name(JYPPX_TensorRtCudaEngine* engine, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_location(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_location);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_is_shape_inference_io(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, JYPPX_Boolean* out_is_shape_inference_io);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_bytes_per_component(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_bytes);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_bytes_per_component_for_profile(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t* out_bytes);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_components_per_element(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_components);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_components_per_element_for_profile(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t* out_components);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_format(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_format);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_format_for_profile(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t* out_format);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_format_desc(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_format_desc_for_profile(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_vectorized_dim(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_dim);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tensor_vectorized_dim_for_profile(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t* out_dim);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_profile_shape(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t selector, JYPPX_TensorRtDims* out_shape);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_engine_capability(JYPPX_TensorRtCudaEngine* engine, int32_t* out_capability);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_tactic_sources(JYPPX_TensorRtCudaEngine* engine, uint32_t* out_tactic_sources);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_profiling_verbosity(JYPPX_TensorRtCudaEngine* engine, int32_t* out_verbosity);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_max_batch_size(JYPPX_TensorRtCudaEngine* engine, int32_t* out_max_batch_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cuda_engine_has_implicit_batch_dimension(JYPPX_TensorRtCudaEngine* engine, JYPPX_Boolean* out_has_implicit_batch);

JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_tensor_shape(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_TensorRtDims* out_shape);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_tensor_strides(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_TensorRtDims* out_strides);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_max_output_size(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, int64_t* out_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_is_tensor_address_bound(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean* out_bound);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_output_allocator_interface_info(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_temporary_storage_allocator_interface_info(JYPPX_TensorRtExecutionContext* context, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_debug_listener_interface_info(JYPPX_TensorRtExecutionContext* context, char* output_buffer, size_t output_buffer_size, size_t* out_required_size, int32_t* out_major, int32_t* out_minor);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_callback_state_snapshot(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_TensorRtExecutionContextCallbackStateInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_clear_callback_state(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_TensorRtExecutionContextCallbackStateInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_tensor_debug_state(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean debug_state);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_tensor_debug_state(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean* out_debug_state);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_all_tensors_debug_state(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean debug_state);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_debug_sync(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean debug_sync);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_debug_sync(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_debug_sync);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_nvtx_verbosity(JYPPX_TensorRtExecutionContext* context, int32_t verbosity, JYPPX_Boolean* out_set);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_nvtx_verbosity(JYPPX_TensorRtExecutionContext* context, int32_t* out_verbosity);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_name(JYPPX_TensorRtExecutionContext* context, const char* name);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_name(JYPPX_TensorRtExecutionContext* context, char* output_buffer, size_t output_buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_optimization_profile(JYPPX_TensorRtExecutionContext* context, int32_t* out_profile_index);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_optimization_profile_async(JYPPX_TensorRtExecutionContext* context, int32_t profile_index, JYPPX_CudaStream* stream);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_all_input_dimensions_specified(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_specified);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_all_input_shapes_specified(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_specified);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_infer_shapes(JYPPX_TensorRtExecutionContext* context, int32_t* out_missing_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_enqueue_emits_profile(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean enqueue_emits_profile);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_enqueue_emits_profile(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_enqueue_emits_profile);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_report_to_profiler(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_reported);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_profiler(JYPPX_TensorRtExecutionContext* context, JYPPX_TensorRtProfiler* profiler);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_clear_profiler(JYPPX_TensorRtExecutionContext* context);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_has_profiler(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_has_profiler);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_device_memory(JYPPX_TensorRtExecutionContext* context, JYPPX_CudaMemory* memory);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_device_memory_size(JYPPX_TensorRtExecutionContext* context, size_t* out_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_update_device_memory_size_for_shapes(JYPPX_TensorRtExecutionContext* context, size_t* out_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_persistent_cache_limit(JYPPX_TensorRtExecutionContext* context, size_t cache_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_persistent_cache_limit(JYPPX_TensorRtExecutionContext* context, size_t* out_cache_size);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_input_consumed_event(JYPPX_TensorRtExecutionContext* context, JYPPX_CudaEvent* event);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_profile_tensor_values(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t selector, int32_t* output_values, int32_t output_count, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_profile_tensor_values_v2(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t selector, int64_t* output_values, int32_t output_count, int32_t* out_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_error_recorder_snapshot_info(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_get_error_recorder_error(JYPPX_TensorRtCudaEngine* engine, int32_t index, JYPPX_TensorRtErrorRecordInfo* out_error);

/* BEGIN TRT10 TWENTY-THIRD BATCH GENERATED DECLARATIONS */
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cuda_engine_get_error_recorder_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cuda_engine_get_minimum_weight_streaming_budget_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cuda_engine_get_profile_tensor_values_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cuda_engine_get_profile_tensor_values_v2_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cuda_engine_has_implicit_batch_dimension_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_cuda_engine_set_error_recorder_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_inspector_get_error_recorder_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_engine_inspector_set_error_recorder_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_execute_v2_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_debug_listener_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_error_recorder_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_nvtx_verbosity_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_output_allocator_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_error_recorder_snapshot_info(JYPPX_TensorRtExecutionContext* context, JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_error_recorder_error(JYPPX_TensorRtExecutionContext* context, int32_t index, JYPPX_TensorRtErrorRecordInfo* out_error);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_get_temporary_storage_allocator_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_no_copy_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_aux_streams(JYPPX_TensorRtExecutionContext* context, JYPPX_CudaStream** streams, int32_t stream_count);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_clear_aux_streams(JYPPX_TensorRtExecutionContext* context);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_aux_streams_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_debug_listener_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_device_memory_v2_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_error_recorder_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_nvtx_verbosity_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_output_allocator_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_execution_context_set_temporary_storage_allocator_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_build_engine_with_config_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_can_run_on_dla_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_clear_quantization_flag_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_algorithm_selector_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_avg_timing_iterations_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_builder_optimization_level_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_default_device_type_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_dla_core_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_int8_calibrator_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_l2_limit_for_tiling_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_max_nb_tactics_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_nb_plugins_to_serialize_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_plugin_to_serialize_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_progress_monitor_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_quantization_flag_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_quantization_flags_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_get_tiling_optimization_level_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_algorithm_selector_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_avg_timing_iterations_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_builder_optimization_level_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_default_device_type_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_dla_core_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_int8_calibrator_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_l2_limit_for_tiling_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_max_nb_tactics_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_plugins_to_serialize_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_progress_monitor_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_quantization_flag_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_quantization_flags_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_config_set_tiling_optimization_level_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_get_error_recorder_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_get_logger_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_get_max_dla_batch_size_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_get_max_threads_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_get_plugin_registry_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_is_network_supported_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_set_error_recorder_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_set_gpu_allocator_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_set_max_threads_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_expr_builder_constant_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_expr_builder_declare_size_tensor_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_expr_builder_operation_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_add_verbosity_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_get_full_text_file_name_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_get_model_dtype_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_get_model_file_name_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_get_print_layer_info_deferred(void);
JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_onnx_config_get_text_file_name_deferred(void);
/* END TRT10 TWENTY-THIRD BATCH GENERATED DECLARATIONS */

/* BEGIN TRT10 CROSS-VERSION SECOND BATCH PLUGIN DEFERRED DECLARATIONS */
#define JYPPX_TRT10_PLUGIN_DEFERRED_DECL(function_name) JYPPX_C_API(JYPPX_StatusCode) function_name(void);
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_creator_set_plugin_namespace_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_creator_v3_one_create_plugin_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_registry_acquire_plugin_resource_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_registry_deregister_creator_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_registry_deregister_library_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_registry_get_all_creators_recursive_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_registry_get_plugin_creator_list_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_registry_load_library_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_registry_register_creator_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_registry_release_plugin_resource_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_resource_clone_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_resource_get_interface_info_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_resource_release_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_resource_context_get_error_recorder_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_resource_context_get_gpu_allocator_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_clone_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_configure_with_format_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_destroy_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_enqueue_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_get_nb_outputs_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_get_output_dimensions_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_get_plugin_namespace_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_get_plugin_type_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_get_plugin_version_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_get_serialization_size_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_get_tensor_rt_version_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_get_workspace_size_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_initialize_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_serialize_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_set_plugin_namespace_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_supports_format_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_terminate_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_dynamic_ext_can_broadcast_input_across_batch_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_dynamic_ext_clone_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_dynamic_ext_configure_plugin_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_dynamic_ext_enqueue_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_dynamic_ext_get_output_dimensions_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_dynamic_ext_get_workspace_size_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_dynamic_ext_is_output_broadcast_across_batch_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_dynamic_ext_return_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_dynamic_ext_supports_format_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_dynamic_ext_supports_format_combination_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_ext_can_broadcast_input_across_batch_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_ext_clone_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_ext_configure_plugin_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_ext_get_output_data_type_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_ext_get_tensor_rt_version_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_ext_is_output_broadcast_across_batch_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_io_ext_configure_plugin_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_io_ext_get_tensor_rt_version_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_io_ext_supports_format_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v2_io_ext_supports_format_combination_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_clone_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_get_capability_interface_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_get_interface_info_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_build_configure_plugin_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_build_get_aliased_input_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_build_get_format_combination_limit_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_build_get_interface_info_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_build_get_metadata_string_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_build_get_nb_outputs_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_build_get_nb_tactics_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_build_get_output_data_types_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_build_get_output_shapes_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_build_get_timing_cache_id_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_build_get_valid_tactics_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_build_get_workspace_size_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_build_supports_format_combination_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_core_get_interface_info_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_core_get_plugin_name_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_core_get_plugin_namespace_deferred)
JYPPX_TRT10_PLUGIN_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_core_get_plugin_version_deferred)
#undef JYPPX_TRT10_PLUGIN_DEFERRED_DECL
/* END TRT10 CROSS-VERSION SECOND BATCH PLUGIN DEFERRED DECLARATIONS */

/* BEGIN TRT10 CROSS-VERSION THIRD BATCH OTHER DEFERRED DECLARATIONS */
#define JYPPX_TRT10_OTHER_DEFERRED_DECL(function_name) JYPPX_C_API(JYPPX_StatusCode) function_name(void);
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_get_algorithm_io_info_by_index_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_get_algorithm_variant_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_get_timing_m_sec_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_get_workspace_size_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_context_get_dimensions_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_context_get_name_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_context_get_nb_inputs_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_context_get_nb_outputs_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_io_info_get_components_per_element_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_io_info_get_data_type_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_io_info_get_strides_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_io_info_get_vectorized_dim_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_selector_get_interface_info_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_selector_report_algorithms_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_selector_select_algorithms_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_variant_get_implementation_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_algorithm_variant_get_tactic_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_debug_listener_get_interface_info_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_debug_listener_process_debug_tensor_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_dimension_expr_get_constant_value_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_dimension_expr_is_constant_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_dimension_expr_is_size_tensor_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_gpu_allocator_allocate_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_gpu_allocator_deallocate_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_gpu_allocator_get_interface_info_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_gpu_allocator_reallocate_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_gpu_async_allocator_allocate_async_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_gpu_async_allocator_deallocate_async_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_gpu_async_allocator_get_interface_info_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_calibrator_get_algorithm_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_calibrator_get_batch_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_calibrator_get_batch_size_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_calibrator_read_calibration_cache_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_calibrator_write_calibration_cache_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_entropy_calibrator_get_algorithm_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_entropy_calibrator_get_interface_info_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_entropy_calibrator2_get_algorithm_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_entropy_calibrator2_get_interface_info_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_legacy_calibrator_get_algorithm_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_legacy_calibrator_get_interface_info_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_legacy_calibrator_get_quantile_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_legacy_calibrator_get_regression_cutoff_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_legacy_calibrator_read_histogram_cache_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_legacy_calibrator_write_histogram_cache_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_min_max_calibrator_get_algorithm_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_int8_min_max_calibrator_get_interface_info_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_output_allocator_get_interface_info_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_output_allocator_notify_shape_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_output_allocator_reallocate_output_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_progress_monitor_get_interface_info_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_progress_monitor_phase_finish_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_progress_monitor_phase_start_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_progress_monitor_step_complete_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_stream_reader_get_interface_info_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_stream_reader_read_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_stream_reader_v2_get_interface_info_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_stream_reader_v2_read_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_stream_reader_v2_seek_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_versioned_interface_get_api_language_deferred)
JYPPX_TRT10_OTHER_DEFERRED_DECL(jyppx_trt10_versioned_interface_get_interface_info_deferred)
#undef JYPPX_TRT10_OTHER_DEFERRED_DECL
/* END TRT10 CROSS-VERSION THIRD BATCH OTHER DEFERRED DECLARATIONS */

/* BEGIN TRT10 CROSS-VERSION FOURTH BATCH ONNX PARSER/GLOBAL DEFERRED DECLARATIONS */
#define JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL(function_name) JYPPX_C_API(JYPPX_StatusCode) function_name(void);
JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL(jyppx_trt10_onnx_config_create_deferred)
JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL(jyppx_trt10_onnx_config_get_verbosity_level_deferred)
JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL(jyppx_trt10_onnx_config_reduce_verbosity_deferred)
JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL(jyppx_trt10_onnx_config_set_verbosity_level_deferred)
JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL(jyppx_trt10_parser_parse_with_weight_descriptors_deferred)
JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL(jyppx_trt10_parser_refitter_create_deferred)
JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL(jyppx_trt10_parser_refitter_clear_errors_deferred)
JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL(jyppx_trt10_parser_refitter_get_error_deferred)
JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL(jyppx_trt10_parser_refitter_get_nb_errors_deferred)
JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL(jyppx_trt10_parser_refitter_refit_from_bytes_deferred)
JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL(jyppx_trt10_parser_refitter_refit_from_file_deferred)
JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL(jyppx_trt10_global_init_lib_nvinfer_plugins_deferred)
#undef JYPPX_TRT10_ONNX_GLOBAL_DEFERRED_DECL
/* END TRT10 CROSS-VERSION FOURTH BATCH ONNX PARSER/GLOBAL DEFERRED DECLARATIONS */

/* BEGIN TRT10 CROSS-VERSION FIFTH BATCH RUNTIME SERIALIZATION DEFERRED DECLARATIONS */
#define JYPPX_TRT10_RUNTIME_SERIALIZATION_DEFERRED_DECL(function_name) JYPPX_C_API(JYPPX_StatusCode) function_name(void);
JYPPX_TRT10_RUNTIME_SERIALIZATION_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_runtime_attach_to_context_deferred)
JYPPX_TRT10_RUNTIME_SERIALIZATION_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_runtime_enqueue_deferred)
JYPPX_TRT10_RUNTIME_SERIALIZATION_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_runtime_get_fields_to_serialize_deferred)
JYPPX_TRT10_RUNTIME_SERIALIZATION_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_runtime_get_interface_info_deferred)
JYPPX_TRT10_RUNTIME_SERIALIZATION_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_runtime_on_shape_change_deferred)
JYPPX_TRT10_RUNTIME_SERIALIZATION_DEFERRED_DECL(jyppx_trt10_plugin_v3_one_runtime_set_tactic_deferred)
JYPPX_TRT10_RUNTIME_SERIALIZATION_DEFERRED_DECL(jyppx_trt10_runtime_deserialize_cuda_engine_v2_deferred)
JYPPX_TRT10_RUNTIME_SERIALIZATION_DEFERRED_DECL(jyppx_trt10_runtime_get_logger_deferred)
JYPPX_TRT10_RUNTIME_SERIALIZATION_DEFERRED_DECL(jyppx_trt10_runtime_get_plugin_registry_deferred)
JYPPX_TRT10_RUNTIME_SERIALIZATION_DEFERRED_DECL(jyppx_trt10_runtime_load_runtime_deferred)
#undef JYPPX_TRT10_RUNTIME_SERIALIZATION_DEFERRED_DECL
/* END TRT10 CROSS-VERSION FIFTH BATCH RUNTIME SERIALIZATION DEFERRED DECLARATIONS */

/* BEGIN TRT10 CROSS-VERSION SIXTH BATCH DIAGNOSTICS/REFITTER DEFERRED DECLARATIONS */
#define JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_DECL(function_name) JYPPX_C_API(JYPPX_StatusCode) function_name(void);
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_DECL(jyppx_trt10_error_recorder_dec_ref_count_deferred)
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_DECL(jyppx_trt10_error_recorder_enum_max_deferred)
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_DECL(jyppx_trt10_error_recorder_get_error_code_deferred)
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_DECL(jyppx_trt10_error_recorder_get_error_desc_deferred)
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_DECL(jyppx_trt10_error_recorder_get_interface_info_deferred)
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_DECL(jyppx_trt10_error_recorder_get_nb_errors_deferred)
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_DECL(jyppx_trt10_error_recorder_has_overflowed_deferred)
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_DECL(jyppx_trt10_error_recorder_inc_ref_count_deferred)
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_DECL(jyppx_trt10_logger_finder_find_logger_deferred)
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_DECL(jyppx_trt10_profiler_report_layer_time_deferred)
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_DECL(jyppx_trt10_refitter_set_named_weights_with_location_deferred)
#undef JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_DECL
/* END TRT10 CROSS-VERSION SIXTH BATCH DIAGNOSTICS/REFITTER DEFERRED DECLARATIONS */
