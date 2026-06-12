#pragma once

#include <cstdint>

#include "jyppx/common/status.h"
#include "jyppx/tensorrt/types.h"

struct JYPPX_TensorRtObjectBase
{
    uint32_t magic;
    uint32_t line;
    uint32_t kind;
    void* payload;
    void (*destroy_payload)(void*);
};

namespace jyppx::tensorrt
{
constexpr uint32_t kObjectMagic = 0x4A595452U;

JYPPX_StatusCode validate_output_pointer(void* pointer, const char* name);
JYPPX_StatusCode validate_handle(const JYPPX_TensorRtObjectBase* object, JYPPX_TensorRtLine expected_line, JYPPX_TensorRtObjectKind expected_kind, const char* parameter_name);
JYPPX_TensorRtObjectBase* create_object(JYPPX_TensorRtLine line, JYPPX_TensorRtObjectKind kind);
void attach_payload(JYPPX_TensorRtObjectBase* object, void* payload, void (*destroy_payload)(void*));
void* get_payload(const JYPPX_TensorRtObjectBase* object);
const char* line_to_version_text(JYPPX_TensorRtLine line);
JYPPX_StatusCode report_vendor_missing(JYPPX_TensorRtLine line, const char* feature_name);
JYPPX_StatusCode report_vendor_mismatch(JYPPX_TensorRtLine requested_line, int32_t detected_major, const char* feature_name);
JYPPX_StatusCode report_not_implemented(JYPPX_TensorRtLine line, const char* feature_name);
void fill_adapter_info(JYPPX_TensorRtAdapterInfo* out_info, JYPPX_TensorRtLine line);
}
