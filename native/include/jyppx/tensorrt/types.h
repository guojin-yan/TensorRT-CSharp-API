#pragma once

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
    JYPPX_TENSORRT_OBJECT_KIND_ATTENTION = 20
} JYPPX_TensorRtObjectKind;

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

JYPPX_C_API(JYPPX_StatusCode) jyppx_trt_object_destroy(JYPPX_TensorRtObjectBase* object);
