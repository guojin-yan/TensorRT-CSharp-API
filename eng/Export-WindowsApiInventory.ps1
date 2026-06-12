[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot "artifacts\api-inventory"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$manifestRoot = Join-Path $RepositoryRoot "native\manifests"
$manifestFiles = Get-ChildItem -Path $manifestRoot -Recurse -Filter "*.manifest.json" | Sort-Object FullName

$apis = @()
foreach ($file in $manifestFiles) {
  $document = Get-Content -LiteralPath $file.FullName -Raw -Encoding utf8 | ConvertFrom-Json
  foreach ($api in $document.apis) {
    $apis += [pscustomobject]@{
      id = [string]$api.id
      module = [string]$document.module
      versionLine = [string]$document.versionLine
      entryPoint = [string]$api.entryPoint
      ownership = [string]$api.ownership
      manualOverride = [bool]$api.manualOverride
      versionGuard = [string]$api.versionGuard
      wrapperKind = [string]$api.wrapperKind
      bindingRole = [string]$api.bindingRole
      manifest = $file.FullName.Substring($RepositoryRoot.Length + 1).Replace("\", "/")
      parameterCount = @($api.parameters).Count
    }
  }
}

$expectedCuda = @(
  "jyppx_cuda_query_runtime_info",
  "jyppx_cuda_get_device_count",
  "jyppx_cuda_get_device_info",
  "jyppx_cuda_get_current_device",
  "jyppx_cuda_set_device",
  "jyppx_cuda_get_memory_info",
  "jyppx_cuda_stream_create",
  "jyppx_cuda_stream_query",
  "jyppx_cuda_stream_wait_event",
  "jyppx_cuda_stream_synchronize",
  "jyppx_cuda_stream_destroy",
  "jyppx_cuda_event_create",
  "jyppx_cuda_event_record",
  "jyppx_cuda_event_synchronize",
  "jyppx_cuda_event_elapsed_time",
  "jyppx_cuda_event_destroy",
  "jyppx_cuda_memory_alloc",
  "jyppx_cuda_memory_get_size",
  "jyppx_cuda_memory_get_device_pointer",
  "jyppx_cuda_memory_copy_from_host",
  "jyppx_cuda_memory_copy_to_host",
  "jyppx_cuda_memory_copy_from_host_async",
  "jyppx_cuda_memory_copy_to_host_async",
  "jyppx_cuda_memory_copy_device_to_device",
  "jyppx_cuda_memory_copy_device_to_device_async",
  "jyppx_cuda_memory_memset",
  "jyppx_cuda_memory_memset_async",
  "jyppx_cuda_memory_free",
  "jyppx_cuda_pinned_memory_alloc",
  "jyppx_cuda_pinned_memory_alloc_with_flags",
  "jyppx_cuda_pinned_memory_register",
  "jyppx_cuda_pinned_memory_get_size",
  "jyppx_cuda_pinned_memory_get_host_pointer",
  "jyppx_cuda_pinned_memory_get_mapped_device_pointer",
  "jyppx_cuda_pinned_memory_get_flags",
  "jyppx_cuda_pinned_memory_free"
)

$expectedTensorRt = @(
  "jyppx_trt_object_destroy",
  "jyppx_trt10_query_adapter_info",
  "jyppx_trt10_logger_create",
  "jyppx_trt10_runtime_create",
  "jyppx_trt10_builder_create",
  "jyppx_trt10_builder_create_config",
  "jyppx_trt10_builder_create_network",
  "jyppx_trt10_builder_build_serialized_network",
  "jyppx_trt10_host_memory_get_size",
  "jyppx_trt10_runtime_deserialize_engine",
  "jyppx_trt10_runtime_deserialize_host_memory",
  "jyppx_trt10_engine_get_io_tensor_count",
  "jyppx_trt10_engine_get_io_tensor_info",
  "jyppx_trt10_engine_get_device_memory_size",
  "jyppx_trt10_engine_get_optimization_profile_count",
  "jyppx_trt10_engine_get_io_tensor_name",
  "jyppx_trt10_engine_get_tensor_index",
  "jyppx_trt10_engine_get_tensor_data_type",
  "jyppx_trt10_engine_get_tensor_shape",
  "jyppx_trt10_engine_get_tensor_io_mode",
  "jyppx_trt10_engine_create_execution_context",
  "jyppx_trt10_builder_create_optimization_profile",
  "jyppx_trt10_optimization_profile_set_shape",
  "jyppx_trt10_builder_config_add_optimization_profile",
  "jyppx_trt10_builder_config_set_flag",
  "jyppx_trt10_builder_config_get_flag",
  "jyppx_trt10_builder_config_set_memory_pool_limit",
  "jyppx_trt10_builder_config_get_memory_pool_limit",
  "jyppx_trt10_builder_config_create_timing_cache",
  "jyppx_trt10_builder_config_set_timing_cache",
  "jyppx_trt10_timing_cache_serialize",
  "jyppx_trt10_builder_config_set_optimization_level",
  "jyppx_trt10_builder_config_get_optimization_level",
  "jyppx_trt10_builder_config_set_profiling_verbosity",
  "jyppx_trt10_builder_config_get_profiling_verbosity",
  "jyppx_trt10_builder_config_set_max_aux_streams",
  "jyppx_trt10_builder_config_get_max_aux_streams",
  "jyppx_trt10_builder_config_set_average_timing_iterations",
  "jyppx_trt10_builder_config_get_average_timing_iterations",
  "jyppx_trt10_builder_config_set_tactic_sources",
  "jyppx_trt10_builder_config_get_tactic_sources",
  "jyppx_trt10_network_add_input",
  "jyppx_trt10_network_mark_output",
  "jyppx_trt10_network_get_input_count",
  "jyppx_trt10_network_get_output_count",
  "jyppx_trt10_network_get_input",
  "jyppx_trt10_network_get_output",
  "jyppx_trt10_network_add_identity",
  "jyppx_trt10_network_add_constant",
  "jyppx_trt10_network_add_elementwise",
  "jyppx_trt10_network_add_matrix_multiply",
  "jyppx_trt10_matrix_multiply_layer_set_operation",
  "jyppx_trt10_matrix_multiply_layer_get_operation",
  "jyppx_trt10_network_add_shuffle",
  "jyppx_trt10_shuffle_layer_set_reshape_dimensions",
  "jyppx_trt10_shuffle_layer_get_reshape_dimensions",
  "jyppx_trt10_shuffle_layer_set_zero_is_placeholder",
  "jyppx_trt10_shuffle_layer_get_zero_is_placeholder",
  "jyppx_trt10_network_add_reduce",
  "jyppx_trt10_reduce_layer_get_operation",
  "jyppx_trt10_reduce_layer_get_axes",
  "jyppx_trt10_reduce_layer_get_keep_dimensions",
  "jyppx_trt10_network_add_concatenation",
  "jyppx_trt10_concatenation_layer_set_axis",
  "jyppx_trt10_concatenation_layer_get_axis",
  "jyppx_trt10_network_add_slice",
  "jyppx_trt10_slice_layer_set_start",
  "jyppx_trt10_slice_layer_get_start",
  "jyppx_trt10_slice_layer_set_size",
  "jyppx_trt10_slice_layer_get_size",
  "jyppx_trt10_slice_layer_set_stride",
  "jyppx_trt10_slice_layer_get_stride",
  "jyppx_trt10_slice_layer_set_mode",
  "jyppx_trt10_slice_layer_get_mode",
  "jyppx_trt10_network_add_softmax",
  "jyppx_trt10_softmax_layer_set_axes",
  "jyppx_trt10_softmax_layer_get_axes",
  "jyppx_trt10_network_add_unary",
  "jyppx_trt10_unary_layer_get_operation",
  "jyppx_trt10_network_add_topk",
  "jyppx_trt10_topk_layer_get_operation",
  "jyppx_trt10_topk_layer_get_k",
  "jyppx_trt10_topk_layer_get_axes",
  "jyppx_trt10_network_add_gather",
  "jyppx_trt10_gather_layer_get_axis",
  "jyppx_trt10_network_add_activation",
  "jyppx_trt10_activation_layer_get_type",
  "jyppx_trt10_network_add_pooling_nd",
  "jyppx_trt10_pooling_layer_get_type",
  "jyppx_trt10_pooling_layer_set_window_size_nd",
  "jyppx_trt10_pooling_layer_get_window_size_nd",
  "jyppx_trt10_pooling_layer_set_stride_nd",
  "jyppx_trt10_pooling_layer_get_stride_nd",
  "jyppx_trt10_pooling_layer_set_padding_nd",
  "jyppx_trt10_pooling_layer_get_padding_nd",
  "jyppx_trt10_network_add_resize",
  "jyppx_trt10_resize_layer_set_output_dimensions",
  "jyppx_trt10_resize_layer_get_output_dimensions",
  "jyppx_trt10_resize_layer_set_mode",
  "jyppx_trt10_resize_layer_get_mode",
  "jyppx_trt10_resize_layer_set_scales",
  "jyppx_trt10_resize_layer_get_scales",
  "jyppx_trt10_network_add_shape",
  "jyppx_trt10_network_add_select",
  "jyppx_trt10_network_add_fill",
  "jyppx_trt10_fill_layer_set_dimensions",
  "jyppx_trt10_fill_layer_get_dimensions",
  "jyppx_trt10_fill_layer_set_operation",
  "jyppx_trt10_fill_layer_get_operation",
  "jyppx_trt10_fill_layer_set_alpha",
  "jyppx_trt10_fill_layer_get_alpha",
  "jyppx_trt10_fill_layer_set_beta",
  "jyppx_trt10_fill_layer_get_beta",
  "jyppx_trt10_network_add_quantize",
  "jyppx_trt10_network_add_dequantize",
  "jyppx_trt10_quantize_layer_get_axis",
  "jyppx_trt10_quantize_layer_set_axis",
  "jyppx_trt10_dequantize_layer_get_axis",
  "jyppx_trt10_dequantize_layer_set_axis",
  "jyppx_trt10_layer_get_input",
  "jyppx_trt10_layer_get_input_count",
  "jyppx_trt10_layer_get_output_count",
  "jyppx_trt10_layer_get_type",
  "jyppx_trt10_layer_get_name",
  "jyppx_trt10_layer_set_name",
  "jyppx_trt10_layer_get_output",
  "jyppx_trt10_tensor_get_name",
  "jyppx_trt10_tensor_set_name",
  "jyppx_trt10_tensor_get_data_type",
  "jyppx_trt10_tensor_get_shape",
  "jyppx_trt10_tensor_set_shape",
  "jyppx_trt10_onnx_parser_create",
  "jyppx_trt10_onnx_parser_parse_from_file",
  "jyppx_trt10_onnx_parser_parse_from_memory",
  "jyppx_trt10_onnx_parser_get_error_count",
  "jyppx_trt10_onnx_parser_get_error",
  "jyppx_trt10_engine_create_inspector",
  "jyppx_trt10_engine_inspector_get_engine_information",
  "jyppx_trt10_execution_context_set_input_shape",
  "jyppx_trt10_execution_context_set_tensor_address",
  "jyppx_trt10_execution_context_enqueue_async",
  "jyppx_trt8_query_adapter_info",
  "jyppx_trt8_logger_create",
  "jyppx_trt8_runtime_create",
  "jyppx_trt8_builder_create",
  "jyppx_trt8_builder_create_config",
  "jyppx_trt8_builder_create_network",
  "jyppx_trt8_builder_build_serialized_network",
  "jyppx_trt8_host_memory_get_size",
  "jyppx_trt8_runtime_deserialize_engine",
  "jyppx_trt8_runtime_deserialize_host_memory",
  "jyppx_trt8_engine_get_io_tensor_count",
  "jyppx_trt8_engine_get_io_tensor_info",
  "jyppx_trt8_engine_get_device_memory_size",
  "jyppx_trt8_engine_get_optimization_profile_count",
  "jyppx_trt8_engine_get_io_tensor_name",
  "jyppx_trt8_engine_get_tensor_index",
  "jyppx_trt8_engine_get_tensor_data_type",
  "jyppx_trt8_engine_get_tensor_shape",
  "jyppx_trt8_engine_get_tensor_io_mode",
  "jyppx_trt8_engine_create_execution_context",
  "jyppx_trt8_builder_create_optimization_profile",
  "jyppx_trt8_optimization_profile_set_shape",
  "jyppx_trt8_builder_config_add_optimization_profile",
  "jyppx_trt8_builder_config_set_flag",
  "jyppx_trt8_builder_config_get_flag",
  "jyppx_trt8_builder_config_set_memory_pool_limit",
  "jyppx_trt8_builder_config_get_memory_pool_limit",
  "jyppx_trt8_builder_config_create_timing_cache",
  "jyppx_trt8_builder_config_set_timing_cache",
  "jyppx_trt8_timing_cache_serialize",
  "jyppx_trt8_builder_config_set_optimization_level",
  "jyppx_trt8_builder_config_get_optimization_level",
  "jyppx_trt8_builder_config_set_profiling_verbosity",
  "jyppx_trt8_builder_config_get_profiling_verbosity",
  "jyppx_trt8_builder_config_set_max_aux_streams",
  "jyppx_trt8_builder_config_get_max_aux_streams",
  "jyppx_trt8_builder_config_set_average_timing_iterations",
  "jyppx_trt8_builder_config_get_average_timing_iterations",
  "jyppx_trt8_builder_config_set_tactic_sources",
  "jyppx_trt8_builder_config_get_tactic_sources",
  "jyppx_trt8_network_add_input",
  "jyppx_trt8_network_mark_output",
  "jyppx_trt8_network_get_input_count",
  "jyppx_trt8_network_get_output_count",
  "jyppx_trt8_network_get_input",
  "jyppx_trt8_network_get_output",
  "jyppx_trt8_network_add_identity",
  "jyppx_trt8_network_add_constant",
  "jyppx_trt8_network_add_elementwise",
  "jyppx_trt8_network_add_matrix_multiply",
  "jyppx_trt8_matrix_multiply_layer_set_operation",
  "jyppx_trt8_matrix_multiply_layer_get_operation",
  "jyppx_trt8_network_add_shuffle",
  "jyppx_trt8_shuffle_layer_set_reshape_dimensions",
  "jyppx_trt8_shuffle_layer_get_reshape_dimensions",
  "jyppx_trt8_shuffle_layer_set_zero_is_placeholder",
  "jyppx_trt8_shuffle_layer_get_zero_is_placeholder",
  "jyppx_trt8_network_add_reduce",
  "jyppx_trt8_reduce_layer_get_operation",
  "jyppx_trt8_reduce_layer_get_axes",
  "jyppx_trt8_reduce_layer_get_keep_dimensions",
  "jyppx_trt8_network_add_concatenation",
  "jyppx_trt8_concatenation_layer_set_axis",
  "jyppx_trt8_concatenation_layer_get_axis",
  "jyppx_trt8_network_add_slice",
  "jyppx_trt8_slice_layer_set_start",
  "jyppx_trt8_slice_layer_get_start",
  "jyppx_trt8_slice_layer_set_size",
  "jyppx_trt8_slice_layer_get_size",
  "jyppx_trt8_slice_layer_set_stride",
  "jyppx_trt8_slice_layer_get_stride",
  "jyppx_trt8_slice_layer_set_mode",
  "jyppx_trt8_slice_layer_get_mode",
  "jyppx_trt8_network_add_softmax",
  "jyppx_trt8_softmax_layer_set_axes",
  "jyppx_trt8_softmax_layer_get_axes",
  "jyppx_trt8_network_add_unary",
  "jyppx_trt8_unary_layer_get_operation",
  "jyppx_trt8_network_add_topk",
  "jyppx_trt8_topk_layer_get_operation",
  "jyppx_trt8_topk_layer_get_k",
  "jyppx_trt8_topk_layer_get_axes",
  "jyppx_trt8_network_add_gather",
  "jyppx_trt8_gather_layer_get_axis",
  "jyppx_trt8_network_add_activation",
  "jyppx_trt8_activation_layer_get_type",
  "jyppx_trt8_network_add_pooling_nd",
  "jyppx_trt8_pooling_layer_get_type",
  "jyppx_trt8_pooling_layer_set_window_size_nd",
  "jyppx_trt8_pooling_layer_get_window_size_nd",
  "jyppx_trt8_pooling_layer_set_stride_nd",
  "jyppx_trt8_pooling_layer_get_stride_nd",
  "jyppx_trt8_pooling_layer_set_padding_nd",
  "jyppx_trt8_pooling_layer_get_padding_nd",
  "jyppx_trt8_network_add_resize",
  "jyppx_trt8_resize_layer_set_output_dimensions",
  "jyppx_trt8_resize_layer_get_output_dimensions",
  "jyppx_trt8_resize_layer_set_mode",
  "jyppx_trt8_resize_layer_get_mode",
  "jyppx_trt8_resize_layer_set_scales",
  "jyppx_trt8_resize_layer_get_scales",
  "jyppx_trt8_network_add_shape",
  "jyppx_trt8_network_add_select",
  "jyppx_trt8_network_add_fill",
  "jyppx_trt8_fill_layer_set_dimensions",
  "jyppx_trt8_fill_layer_get_dimensions",
  "jyppx_trt8_fill_layer_set_operation",
  "jyppx_trt8_fill_layer_get_operation",
  "jyppx_trt8_fill_layer_set_alpha",
  "jyppx_trt8_fill_layer_get_alpha",
  "jyppx_trt8_fill_layer_set_beta",
  "jyppx_trt8_fill_layer_get_beta",
  "jyppx_trt8_network_add_quantize",
  "jyppx_trt8_network_add_dequantize",
  "jyppx_trt8_quantize_layer_get_axis",
  "jyppx_trt8_quantize_layer_set_axis",
  "jyppx_trt8_dequantize_layer_get_axis",
  "jyppx_trt8_dequantize_layer_set_axis",
  "jyppx_trt8_layer_get_input",
  "jyppx_trt8_layer_get_input_count",
  "jyppx_trt8_layer_get_output_count",
  "jyppx_trt8_layer_get_type",
  "jyppx_trt8_layer_get_name",
  "jyppx_trt8_layer_set_name",
  "jyppx_trt8_layer_get_output",
  "jyppx_trt8_tensor_get_name",
  "jyppx_trt8_tensor_set_name",
  "jyppx_trt8_tensor_get_data_type",
  "jyppx_trt8_tensor_get_shape",
  "jyppx_trt8_tensor_set_shape",
  "jyppx_trt8_onnx_parser_create",
  "jyppx_trt8_onnx_parser_parse_from_file",
  "jyppx_trt8_onnx_parser_parse_from_memory",
  "jyppx_trt8_onnx_parser_get_error_count",
  "jyppx_trt8_onnx_parser_get_error",
  "jyppx_trt8_engine_create_inspector",
  "jyppx_trt8_engine_inspector_get_engine_information",
  "jyppx_trt8_execution_context_set_input_shape",
  "jyppx_trt8_execution_context_set_binding_dimensions",
  "jyppx_trt8_execution_context_set_tensor_address",
  "jyppx_trt8_execution_context_enqueue_async"
)

$deferredTensorRt = @(
  "Profiler / ErrorRecorder",
  "Calibrator and quantization helpers",
  "Plugin registry and custom plugin lifecycle",
  "Advanced normalization APIs beyond LRN",
  "Grouped deconvolution output validation beyond current metadata smoke",
  "Advanced execution context profiling and debug tensor APIs"
)

function Get-CoverageRows {
  param(
    [string[]]$Expected,
    [object[]]$ActualApis
  )

  foreach ($entryPoint in $Expected) {
    $match = $ActualApis | Where-Object { $_.entryPoint -eq $entryPoint } | Select-Object -First 1
    [pscustomobject]@{
      entryPoint = $entryPoint
      status = if ($match) { "covered" } else { "missing" }
      module = if ($match) { $match.module } else { "" }
      versionLine = if ($match) { $match.versionLine } else { "" }
      manifest = if ($match) { $match.manifest } else { "" }
    }
  }
}

$cudaCoverage = @(Get-CoverageRows -Expected $expectedCuda -ActualApis $apis)
$tensorRtCoverage = @(Get-CoverageRows -Expected $expectedTensorRt -ActualApis $apis)

$report = [pscustomobject]@{
  generatedAt = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  totalApiCount = $apis.Count
  moduleCounts = $apis | Group-Object module | ForEach-Object { [pscustomobject]@{ module = $_.Name; count = $_.Count } }
  versionLineCounts = $apis | Group-Object module, versionLine | ForEach-Object {
    [pscustomobject]@{
      group = $_.Name
      count = $_.Count
    }
  }
  cudaDeploymentCoverage = $cudaCoverage
  tensorRtDeploymentCoverage = $tensorRtCoverage
  deferredTensorRtAreas = $deferredTensorRt
  highLevelManagedObjects = [pscustomobject]@{
    cuda = @("CudaDevice", "CudaDeviceScope", "CudaDeviceInfo", "CudaMemoryInfo", "CudaStream", "CudaEvent", "CudaGraph", "CudaGraphExec", "CudaMemory", "CudaManagedMemory", "CudaPinnedMemory", "CudaRegisteredHostMemory", "CudaPitchedMemory", "CudaArray", "CudaMipmappedArray", "CudaMemoryPool")
    tensorRt = @("TensorRtLogger", "TensorRtRuntime", "TensorRtBuilder", "TensorRtBuilderConfig", "TensorRtNetworkDefinition", "TensorRtTensor", "TensorRtLayer", "TensorRtWeights", "TensorRtOptimizationProfile", "TensorRtOnnxParser", "TensorRtHostMemory", "TensorRtTimingCache", "TensorRtEngine", "TensorRtEngineInspector", "TensorRtExecutionContext")
  }
  windowsSamples = @("CudaSmokeRunner", "CudaGraphSmokeRunner", "TensorRtSmokeRunner", "RefitWeightsSmokeRunner", "OnnxToEngineSmokeRunner", "NetworkBuilderSmokeRunner", "NetworkLayersSmokeRunner", "NetworkShapeOpsSmokeRunner", "NetworkConcatSliceSmokeRunner", "NetworkSoftmaxTopKSmokeRunner", "NetworkActivationPoolingResizeSmokeRunner", "NetworkMatrixFillSelectSmokeRunner", "NetworkConvolutionScaleSmokeRunner", "NetworkDeconvolutionSmokeRunner", "NetworkLrnSmokeRunner", "NetworkQuantizeDequantizeSmokeRunner", "LifecycleSmokeRunner")
}

$jsonPath = Join-Path $OutputDirectory "windows-api-inventory.json"
$markdownPath = Join-Path $OutputDirectory "windows-api-inventory.md"

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Windows API Inventory")
$lines.Add("")
$lines.Add("Generated: $($report.generatedAt)")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- Total manifest APIs: $($report.totalApiCount)")
foreach ($group in $report.moduleCounts) {
  $lines.Add("- $($group.module): $($group.count)")
}
$lines.Add("")
$lines.Add("## CUDA Deployment Coverage")
$lines.Add("")
$lines.Add("| Entrypoint | Status | Manifest |")
$lines.Add("| --- | --- | --- |")
foreach ($row in $cudaCoverage) {
  $lines.Add("| ``" + $row.entryPoint + "`` | " + $row.status + " | " + $row.manifest + " |")
}
$lines.Add("")
$lines.Add("## TensorRT Deployment Coverage")
$lines.Add("")
$lines.Add("| Entrypoint | Status | Manifest |")
$lines.Add("| --- | --- | --- |")
foreach ($row in $tensorRtCoverage) {
  $lines.Add("| ``" + $row.entryPoint + "`` | " + $row.status + " | " + $row.manifest + " |")
}
$lines.Add("")
$lines.Add("## Current Managed Object Model")
$lines.Add("")
$lines.Add("- CUDA: " + ($report.highLevelManagedObjects.cuda -join ", "))
$lines.Add("- TensorRT: " + ($report.highLevelManagedObjects.tensorRt -join ", "))
$lines.Add("")
$lines.Add("## Windows Smoke Runners")
$lines.Add("")
foreach ($sample in $report.windowsSamples) {
  $lines.Add("- $sample")
}
$lines.Add("")
$lines.Add("## Deferred TensorRT Areas")
$lines.Add("")
foreach ($area in $deferredTensorRt) {
  $lines.Add("- $area")
}
$lines.Add("")
$lines.Add("## Decision")
$lines.Add("")
$lines.Add("Windows API expansion should continue through manifest-driven native ABI, generated NativeMethods, high-level managed wrappers, and smoke validation. Linux packaging remains out of scope for the current API-completion phase.")

[System.IO.File]::WriteAllLines($markdownPath, $lines, $utf8)

Write-Host "Windows API inventory report written to $jsonPath"
Write-Host "Windows API inventory report written to $markdownPath"
