#pragma once

#include <stddef.h>
#include <stdint.h>

#include "jyppx/common/bridge_exports.h"
#include "jyppx/common/status.h"

typedef enum JYPPX_TensorRtLine
{
    JYPPX_TENSORRT_LINE_UNKNOWN = 0,
    JYPPX_TENSORRT_LINE_8 = 8,
    JYPPX_TENSORRT_LINE_10 = 10,
    JYPPX_TENSORRT_LINE_11 = 11
} JYPPX_TensorRtLine;

typedef enum JYPPX_TensorRtObjectKind
{
    JYPPX_TENSORRT_OBJECT_KIND_UNKNOWN = 0,
    JYPPX_TENSORRT_OBJECT_KIND_LOGGER = 1,
    JYPPX_TENSORRT_OBJECT_KIND_RUNTIME = 2,
    JYPPX_TENSORRT_OBJECT_KIND_BUILDER = 3,
    JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG = 4,
    JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION = 5,
    JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY = 6,
    JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE = 7,
    JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT = 8,
    JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER = 9,
    JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE = 10,
    JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR = 11,
    JYPPX_TENSORRT_OBJECT_KIND_TIMING_CACHE = 12,
    JYPPX_TENSORRT_OBJECT_KIND_TENSOR = 13,
    JYPPX_TENSORRT_OBJECT_KIND_LAYER = 14,
    JYPPX_TENSORRT_OBJECT_KIND_REFITTER = 15,
    JYPPX_TENSORRT_OBJECT_KIND_LOOP = 16,
    JYPPX_TENSORRT_OBJECT_KIND_IF_CONDITIONAL = 17,
    JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG = 18,
    JYPPX_TENSORRT_OBJECT_KIND_RUNTIME_CONFIG = 19,
    JYPPX_TENSORRT_OBJECT_KIND_ATTENTION = 20,
    JYPPX_TENSORRT_OBJECT_KIND_PROGRESS_MONITOR = 21,
    JYPPX_TENSORRT_OBJECT_KIND_PROFILER = 22,
    JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER_REFITTER = 23,
    JYPPX_TENSORRT_OBJECT_KIND_ALLOCATOR_CALLBACK_OWNER = 24,
    JYPPX_TENSORRT_OBJECT_KIND_ONNX_CONFIG = 25,
    JYPPX_TENSORRT_OBJECT_KIND_DEBUG_LISTENER_CALLBACK_OWNER = 26,
    JYPPX_TENSORRT_OBJECT_KIND_OUTPUT_ALLOCATOR_CALLBACK_OWNER = 27,
    JYPPX_TENSORRT_OBJECT_KIND_GPU_ALLOCATOR_CALLBACK_OWNER = 28
} JYPPX_TensorRtObjectKind;

typedef enum JYPPX_TensorRtProgressMonitorEventKind
{
    JYPPX_TENSORRT_PROGRESS_MONITOR_EVENT_UNKNOWN = 0,
    JYPPX_TENSORRT_PROGRESS_MONITOR_EVENT_PHASE_START = 1,
    JYPPX_TENSORRT_PROGRESS_MONITOR_EVENT_STEP_COMPLETE = 2,
    JYPPX_TENSORRT_PROGRESS_MONITOR_EVENT_PHASE_FINISH = 3
} JYPPX_TensorRtProgressMonitorEventKind;

typedef enum JYPPX_TensorRtIOMode
{
    JYPPX_TENSORRT_IO_MODE_UNKNOWN = 0,
    JYPPX_TENSORRT_IO_MODE_INPUT = 1,
    JYPPX_TENSORRT_IO_MODE_OUTPUT = 2
} JYPPX_TensorRtIOMode;

typedef struct JYPPX_TensorRtObjectBase JYPPX_TensorRtObjectBase;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtLogger;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtRuntime;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtBuilder;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtBuilderConfig;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtNetworkDefinition;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtHostMemory;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtCudaEngine;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtExecutionContext;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtOnnxParser;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtOptimizationProfile;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtEngineInspector;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtTimingCache;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtTensor;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtLayer;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtRefitter;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtLoop;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtIfConditional;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtSerializationConfig;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtRuntimeConfig;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtAttention;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtProgressMonitor;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtProfiler;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtOnnxParserRefitter;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtAllocatorOwner;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtOnnxConfig;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtDebugListenerOwner;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtOutputAllocatorOwner;
typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtGpuAllocatorOwner;

typedef JYPPX_StatusCode (*JYPPX_TensorRtLoggerCallback)(
    int32_t severity,
    const char* message,
    size_t message_length,
    void* user_state);

typedef JYPPX_StatusCode (*JYPPX_TensorRtProgressMonitorCallback)(
    int32_t event_kind,
    const char* phase_name,
    size_t phase_name_length,
    const char* parent_phase,
    size_t parent_phase_length,
    int32_t step,
    int32_t nb_steps,
    JYPPX_Boolean* out_should_continue,
    void* user_state);

typedef JYPPX_StatusCode (*JYPPX_TensorRtProfilerCallback)(
    const char* layer_name,
    size_t layer_name_length,
    float milliseconds,
    void* user_state);

typedef JYPPX_StatusCode (*JYPPX_TensorRtDebugListenerCallback)(
    uint32_t line,
    const char* tensor_name,
    size_t tensor_name_length,
    int32_t data_type,
    int32_t location,
    int32_t shape_rank,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3,
    int64_t dim4,
    int64_t dim5,
    int64_t dim6,
    int64_t dim7,
    void* user_state);

typedef enum JYPPX_TensorRtOutputAllocatorCallbackKind
{
    JYPPX_TENSORRT_OUTPUT_ALLOCATOR_CALLBACK_UNKNOWN = 0,
    JYPPX_TENSORRT_OUTPUT_ALLOCATOR_CALLBACK_NOTIFY_SHAPE = 1,
    JYPPX_TENSORRT_OUTPUT_ALLOCATOR_CALLBACK_REALLOCATE_OUTPUT = 2
} JYPPX_TensorRtOutputAllocatorCallbackKind;

typedef JYPPX_StatusCode (*JYPPX_TensorRtOutputAllocatorCallback)(
    uint32_t line,
    int32_t callback_kind,
    const char* tensor_name,
    size_t tensor_name_length,
    uint64_t requested_size,
    uint64_t alignment,
    JYPPX_Boolean has_current_memory,
    JYPPX_Boolean has_stream,
    int32_t shape_rank,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3,
    int64_t dim4,
    int64_t dim5,
    int64_t dim6,
    int64_t dim7,
    JYPPX_Boolean* out_should_allocate,
    void* user_state);

typedef enum JYPPX_TensorRtGpuAllocatorCallbackKind
{
    JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_UNKNOWN = 0,
    JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_ALLOCATE = 1,
    JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_REALLOCATE = 2,
    JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_DEALLOCATE = 3,
    JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_ALLOCATE_ASYNC = 4,
    JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_DEALLOCATE_ASYNC = 5
} JYPPX_TensorRtGpuAllocatorCallbackKind;

typedef enum JYPPX_TensorRtGpuAllocatorAttachmentTarget
{
    JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_NONE = 0,
    JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_RUNTIME = 1,
    JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_BUILDER = 2
} JYPPX_TensorRtGpuAllocatorAttachmentTarget;

typedef JYPPX_StatusCode (*JYPPX_TensorRtGpuAllocatorCallback)(
    uint32_t line,
    int32_t callback_kind,
    uint64_t requested_size,
    uint64_t alignment,
    uint32_t allocator_flags,
    JYPPX_Boolean has_current_memory,
    JYPPX_Boolean has_stream,
    JYPPX_Boolean* out_should_proceed,
    void* user_state);

typedef struct JYPPX_TensorRtAdapterInfo
{
    uint32_t line;
    JYPPX_Boolean vendor_dependency_available;
    JYPPX_Boolean runtime_creation_supported;
    JYPPX_Boolean builder_creation_supported;
    JYPPX_Boolean network_creation_supported;
    JYPPX_Boolean engine_deserialization_supported;
    const char* detected_version;
    const char* status_message;
} JYPPX_TensorRtAdapterInfo;

typedef struct JYPPX_TensorRtDims
{
    int32_t nb_dims;
    int32_t d[8];
} JYPPX_TensorRtDims;

typedef struct JYPPX_TensorRtDims64
{
    int32_t nb_dims;
    int64_t d[8];
} JYPPX_TensorRtDims64;

typedef struct JYPPX_TensorRtTensorInfo
{
    int32_t index;
    char name[256];
    int32_t data_type;
    int32_t io_mode;
    JYPPX_TensorRtDims shape;
} JYPPX_TensorRtTensorInfo;

typedef struct JYPPX_TensorRtParserErrorInfo
{
    int32_t index;
    int32_t code;
    int32_t line;
    int32_t node;
    char description[1024];
    char file[256];
    char function_name[256];
    char node_name[256];
    char node_operator[128];
} JYPPX_TensorRtParserErrorInfo;

typedef struct JYPPX_TensorRtRefitEntryInfo
{
    char layer_name[256];
    int32_t role;
} JYPPX_TensorRtRefitEntryInfo;

typedef struct JYPPX_TensorRtWeightsInfo
{
    int32_t data_type;
    int64_t count;
    JYPPX_Boolean has_values;
} JYPPX_TensorRtWeightsInfo;

typedef struct JYPPX_TensorRtErrorRecorderSnapshotInfo
{
    JYPPX_Boolean has_recorder;
    int32_t error_count;
    JYPPX_Boolean has_overflowed;
    JYPPX_Boolean interface_info_available;
    int32_t interface_info_major;
    int32_t interface_info_minor;
    char interface_info_kind[128];
} JYPPX_TensorRtErrorRecorderSnapshotInfo;

typedef struct JYPPX_TensorRtErrorRecordInfo
{
    int32_t index;
    int32_t code;
    char description[1024];
} JYPPX_TensorRtErrorRecordInfo;

typedef struct JYPPX_TensorRtAllocatorOwnerDiagnosticInfo
{
    uint32_t line;
    uint64_t invocation_count;
    uint64_t failure_count;
    int32_t last_status;
    JYPPX_Boolean is_attached;
    uint64_t last_size;
    uint64_t last_alignment;
    char last_diagnostic[1024];
} JYPPX_TensorRtAllocatorOwnerDiagnosticInfo;

typedef struct JYPPX_TensorRtAllocatorOwnerStateInfo
{
    uint32_t line;
    uint64_t owner_id;
    uint64_t state_transition_count;
    uint64_t ledger_allocation_count;
    uint64_t ledger_release_count;
    uint64_t ledger_failure_count;
    uint64_t last_allocation_id;
    uint64_t last_release_allocation_id;
    uint64_t last_size;
    uint64_t last_alignment;
    uint64_t last_stream_value;
    int32_t attach_state;
    int32_t last_status;
    JYPPX_Boolean is_attached;
    JYPPX_Boolean has_live_allocation;
    char last_operation[64];
    char last_diagnostic[1024];
} JYPPX_TensorRtAllocatorOwnerStateInfo;

typedef struct JYPPX_TensorRtDebugListenerOwnerInfo
{
    uint32_t line;
    uint64_t owner_id;
    uint64_t invocation_count;
    uint64_t failure_count;
    uint64_t in_flight_callback_count;
    uint64_t max_in_flight_callback_count;
    uint64_t attach_count;
    uint64_t detach_count;
    int32_t last_status;
    JYPPX_Boolean is_attached;
    JYPPX_Boolean last_callback_succeeded;
    int32_t last_data_type;
    int32_t last_location;
    int32_t last_shape_rank;
    int64_t last_shape[8];
    char last_tensor_name[256];
    char last_diagnostic[1024];
} JYPPX_TensorRtDebugListenerOwnerInfo;

typedef struct JYPPX_TensorRtOutputAllocatorOwnerInfo
{
    uint32_t line;
    uint64_t owner_id;
    uint64_t invocation_count;
    uint64_t notify_shape_count;
    uint64_t reallocate_output_count;
    uint64_t failure_count;
    uint64_t in_flight_callback_count;
    uint64_t max_in_flight_callback_count;
    uint64_t attach_count;
    uint64_t detach_count;
    uint64_t allocation_count;
    uint64_t reuse_count;
    uint64_t release_count;
    uint64_t live_allocation_count;
    uint64_t live_allocation_bytes;
    uint64_t peak_live_allocation_bytes;
    uint64_t last_requested_size;
    uint64_t last_alignment;
    int32_t last_status;
    JYPPX_Boolean is_attached;
    JYPPX_Boolean last_callback_succeeded;
    JYPPX_Boolean last_allocation_succeeded;
    JYPPX_Boolean last_had_current_memory;
    JYPPX_Boolean last_had_stream;
    int32_t last_callback_kind;
    int32_t last_shape_rank;
    int64_t last_shape[8];
    char last_tensor_name[256];
    char last_diagnostic[1024];
} JYPPX_TensorRtOutputAllocatorOwnerInfo;

typedef struct JYPPX_TensorRtGpuAllocatorOwnerInfo
{
    uint32_t line;
    uint64_t owner_id;
    uint64_t invocation_count;
    uint64_t allocate_count;
    uint64_t reallocate_count;
    uint64_t deallocate_count;
    uint64_t allocate_async_count;
    uint64_t deallocate_async_count;
    uint64_t rejected_count;
    uint64_t callback_failure_count;
    uint64_t cuda_failure_count;
    uint64_t in_flight_callback_count;
    uint64_t max_in_flight_callback_count;
    uint64_t attach_count;
    uint64_t detach_count;
    uint64_t live_allocation_count;
    uint64_t live_allocation_bytes;
    uint64_t peak_live_allocation_bytes;
    uint64_t last_requested_size;
    uint64_t last_alignment;
    uint32_t last_allocator_flags;
    int32_t last_status;
    int32_t attachment_target;
    int32_t last_callback_kind;
    JYPPX_Boolean is_attached;
    JYPPX_Boolean last_callback_succeeded;
    JYPPX_Boolean last_operation_succeeded;
    JYPPX_Boolean last_had_current_memory;
    JYPPX_Boolean last_had_stream;
    char last_diagnostic[1024];
} JYPPX_TensorRtGpuAllocatorOwnerInfo;

typedef struct JYPPX_TensorRtRuntimeCreateDiagnosticInfo
{
    uint32_t line;
    JYPPX_Boolean attempted;
    JYPPX_Boolean logger_handle_present;
    JYPPX_Boolean logger_payload_present;
    JYPPX_Boolean create_infer_runtime_returned_non_null;
    JYPPX_Boolean create_infer_runtime_returned_null;
    int32_t last_status;
    int32_t tensor_rt_available;
    int32_t expected_major;
    int32_t bridge_built_major;
    char detected_version[64];
    JYPPX_Boolean logger_callback_available;
    uint32_t logger_message_count;
    int32_t last_logger_severity;
    char last_logger_message[1024];
    char create_runtime_phase[128];
    char native_detail[1024];
    char diagnostic[1024];
} JYPPX_TensorRtRuntimeCreateDiagnosticInfo;

typedef struct JYPPX_TensorRtExecutionContextCallbackStateInfo
{
    uint32_t line;
    JYPPX_Boolean has_output_allocator;
    JYPPX_Boolean has_temporary_storage_allocator;
    JYPPX_Boolean has_debug_listener;
    JYPPX_Boolean output_allocator_interface_info_available;
    JYPPX_Boolean temporary_storage_allocator_interface_info_available;
    JYPPX_Boolean debug_listener_interface_info_available;
    JYPPX_Boolean output_allocator_clear_supported;
    JYPPX_Boolean temporary_storage_allocator_clear_supported;
    JYPPX_Boolean debug_listener_clear_supported;
    JYPPX_Boolean output_allocator_cleared;
    JYPPX_Boolean temporary_storage_allocator_cleared;
    JYPPX_Boolean debug_listener_cleared;
    int32_t output_allocator_interface_major;
    int32_t output_allocator_interface_minor;
    int32_t temporary_storage_allocator_interface_major;
    int32_t temporary_storage_allocator_interface_minor;
    int32_t debug_listener_interface_major;
    int32_t debug_listener_interface_minor;
    int32_t last_status;
    char output_allocator_interface_kind[128];
    char temporary_storage_allocator_interface_kind[128];
    char debug_listener_interface_kind[128];
    char last_operation[64];
    char last_diagnostic[1024];
} JYPPX_TensorRtExecutionContextCallbackStateInfo;

JYPPX_C_API(JYPPX_StatusCode) jyppx_trt_object_destroy(JYPPX_TensorRtObjectBase* object);
