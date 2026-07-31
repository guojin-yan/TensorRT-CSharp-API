#include "jyppx/tensorrt/trt11.h"

#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../common/object.hpp"
#include "../../common/error_state.hpp"
#include "../../cuda/object.hpp"

#if JYPPX_HAS_TENSORRT
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#endif

#if JYPPX_HAS_TENSORRT_ONNX_CONFIG
#include <NvOnnxConfig.h>
#endif

#if JYPPX_HAS_TENSORRT_ONNXPARSER
#include <NvOnnxParser.h>
#endif

#ifndef JYPPX_TENSORRT_VERSION_MAJOR_NUM
#define JYPPX_TENSORRT_VERSION_MAJOR_NUM 0
#endif

#ifndef JYPPX_TENSORRT_VERSION_TEXT
#define JYPPX_TENSORRT_VERSION_TEXT "not-detected"
#endif

#ifndef JYPPX_HAS_TENSORRT_ONNXPARSER
#define JYPPX_HAS_TENSORRT_ONNXPARSER 0
#endif

#ifndef JYPPX_HAS_TENSORRT_ONNX_CONFIG
#define JYPPX_HAS_TENSORRT_ONNX_CONFIG 0
#endif

namespace
{
constexpr JYPPX_TensorRtLine kLine = JYPPX_TENSORRT_LINE_11;

#if JYPPX_HAS_TENSORRT
class ManagedLogger final : public nvinfer1::ILogger
{
public:
    ManagedLogger() = default;

    ManagedLogger(JYPPX_TensorRtLoggerCallback callback, void* user_state, const int32_t minimum_severity) noexcept
        : callback_(callback),
          user_state_(user_state),
          minimum_severity_(minimum_severity)
    {
    }

    void log(Severity severity, char const* msg) noexcept override
    {
        if (msg == nullptr)
        {
            return;
        }

        last_callback_failed_ = false;
        const auto severity_value = static_cast<int32_t>(severity);
        ++message_count_;
        last_severity_ = severity_value;
        copy_c_string_noexcept(msg, last_message_, sizeof(last_message_));
        if (severity_value > minimum_severity_)
        {
            return;
        }

        try
        {
            if (callback_ != nullptr)
            {
                const auto callback_status = callback_(severity_value, msg, std::strlen(msg), user_state_);
                if (callback_status != JYPPX_STATUS_OK)
                {
                    last_callback_failed_ = true;
                    std::ostringstream builder;
                    builder << "Managed TensorRT logger callback returned status " << static_cast<int32_t>(callback_status) << ".";
                    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
                }

                return;
            }

            if (severity <= Severity::kWARNING)
            {
                std::ostringstream builder;
                builder << "TensorRT logger: " << msg;
                jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
            }
        }
        catch (const std::exception& exception)
        {
            last_callback_failed_ = true;
            std::ostringstream builder;
            builder << "TensorRT logger callback boundary caught a native exception: " << exception.what();
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        }
        catch (...)
        {
            last_callback_failed_ = true;
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT logger callback boundary caught an unknown native exception.");
        }
    }

    bool last_callback_failed() const noexcept
    {
        return last_callback_failed_;
    }

    bool callback_available() const noexcept
    {
        return callback_ != nullptr;
    }

    uint32_t message_count() const noexcept
    {
        return message_count_;
    }

    int32_t last_severity() const noexcept
    {
        return last_severity_;
    }

    const char* last_message() const noexcept
    {
        return last_message_;
    }

private:
    static void copy_c_string_noexcept(const char* source, char* destination, const size_t capacity) noexcept
    {
        if (destination == nullptr || capacity == 0)
        {
            return;
        }

        std::memset(destination, 0, capacity);
        if (source == nullptr)
        {
            return;
        }

        const size_t length = std::strlen(source);
        const size_t copy_length = length < capacity - 1 ? length : capacity - 1;
        std::memcpy(destination, source, copy_length);
        destination[copy_length] = '\0';
    }

    JYPPX_TensorRtLoggerCallback callback_{nullptr};
    void* user_state_{nullptr};
    int32_t minimum_severity_{static_cast<int32_t>(Severity::kWARNING)};
    bool last_callback_failed_{false};
    uint32_t message_count_{0};
    int32_t last_severity_{0};
    char last_message_[1024]{};
};

class ManagedProgressMonitor final : public nvinfer1::IProgressMonitor
{
public:
    ManagedProgressMonitor(JYPPX_TensorRtProgressMonitorCallback callback, void* user_state) noexcept
        : callback_(callback),
          user_state_(user_state)
    {
    }

    void phaseStart(char const* phase_name, char const* parent_phase, int32_t nb_steps) noexcept override
    {
        JYPPX_Boolean should_continue = JYPPX_TRUE;
        notify(JYPPX_TENSORRT_PROGRESS_MONITOR_EVENT_PHASE_START, phase_name, parent_phase, -1, nb_steps, &should_continue);
    }

    bool stepComplete(char const* phase_name, int32_t step) noexcept override
    {
        JYPPX_Boolean should_continue = JYPPX_TRUE;
        notify(JYPPX_TENSORRT_PROGRESS_MONITOR_EVENT_STEP_COMPLETE, phase_name, nullptr, step, 0, &should_continue);
        return should_continue != JYPPX_FALSE;
    }

    void phaseFinish(char const* phase_name) noexcept override
    {
        JYPPX_Boolean should_continue = JYPPX_TRUE;
        notify(JYPPX_TENSORRT_PROGRESS_MONITOR_EVENT_PHASE_FINISH, phase_name, nullptr, -1, 0, &should_continue);
    }

    bool emit_diagnostic(
        const int32_t event_kind,
        const char* phase_name,
        const char* parent_phase,
        const int32_t step,
        const int32_t nb_steps,
        JYPPX_Boolean* out_should_continue) noexcept
    {
        if (out_should_continue == nullptr)
        {
            return false;
        }

        *out_should_continue = JYPPX_TRUE;
        switch (event_kind)
        {
        case JYPPX_TENSORRT_PROGRESS_MONITOR_EVENT_PHASE_START:
            phaseStart(phase_name, parent_phase, nb_steps);
            return true;
        case JYPPX_TENSORRT_PROGRESS_MONITOR_EVENT_STEP_COMPLETE:
            *out_should_continue = stepComplete(phase_name, step) ? JYPPX_TRUE : JYPPX_FALSE;
            return true;
        case JYPPX_TENSORRT_PROGRESS_MONITOR_EVENT_PHASE_FINISH:
            phaseFinish(phase_name);
            return true;
        default:
            return false;
        }
    }

    bool last_callback_failed() const noexcept
    {
        return last_callback_failed_;
    }

private:
    static size_t safe_length(const char* value) noexcept
    {
        return value != nullptr ? std::strlen(value) : 0U;
    }

    void notify(
        const int32_t event_kind,
        const char* phase_name,
        const char* parent_phase,
        const int32_t step,
        const int32_t nb_steps,
        JYPPX_Boolean* out_should_continue) noexcept
    {
        last_callback_failed_ = false;
        if (out_should_continue != nullptr)
        {
            *out_should_continue = JYPPX_TRUE;
        }

        try
        {
            if (callback_ == nullptr)
            {
                return;
            }

            JYPPX_Boolean local_continue = JYPPX_TRUE;
            JYPPX_Boolean* continue_output = out_should_continue != nullptr ? out_should_continue : &local_continue;
            const auto callback_status = callback_(
                event_kind,
                phase_name,
                safe_length(phase_name),
                parent_phase,
                safe_length(parent_phase),
                step,
                nb_steps,
                continue_output,
                user_state_);
            if (callback_status != JYPPX_STATUS_OK)
            {
                last_callback_failed_ = true;
                if (continue_output != nullptr)
                {
                    *continue_output = JYPPX_TRUE;
                }

                std::ostringstream builder;
                builder << "Managed TensorRT progress monitor callback returned status " << static_cast<int32_t>(callback_status) << ".";
                jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
            }
        }
        catch (const std::exception& exception)
        {
            last_callback_failed_ = true;
            if (out_should_continue != nullptr)
            {
                *out_should_continue = JYPPX_TRUE;
            }

            std::ostringstream builder;
            builder << "TensorRT progress monitor callback boundary caught a native exception: " << exception.what();
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        }
        catch (...)
        {
            last_callback_failed_ = true;
            if (out_should_continue != nullptr)
            {
                *out_should_continue = JYPPX_TRUE;
            }

            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT progress monitor callback boundary caught an unknown native exception.");
        }
    }

    JYPPX_TensorRtProgressMonitorCallback callback_{nullptr};
    void* user_state_{nullptr};
    bool last_callback_failed_{false};
};

class ManagedProfiler final : public nvinfer1::IProfiler
{
public:
    ManagedProfiler(JYPPX_TensorRtProfilerCallback callback, void* user_state) noexcept
        : callback_(callback),
          user_state_(user_state)
    {
    }

    void reportLayerTime(char const* layer_name, float milliseconds) noexcept override
    {
        last_callback_failed_ = false;

        try
        {
            if (callback_ == nullptr)
            {
                return;
            }

            const char* safe_layer_name = layer_name != nullptr ? layer_name : "";
            const auto callback_status = callback_(safe_layer_name, std::strlen(safe_layer_name), milliseconds, user_state_);
            if (callback_status != JYPPX_STATUS_OK)
            {
                last_callback_failed_ = true;
                std::ostringstream builder;
                builder << "Managed TensorRT profiler callback returned status " << static_cast<int32_t>(callback_status) << ".";
                jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
            }
        }
        catch (const std::exception& exception)
        {
            last_callback_failed_ = true;
            std::ostringstream builder;
            builder << "TensorRT profiler callback boundary caught a native exception: " << exception.what();
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        }
        catch (...)
        {
            last_callback_failed_ = true;
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT profiler callback boundary caught an unknown native exception.");
        }
    }

    void emit_diagnostic(const char* layer_name, const float milliseconds) noexcept
    {
        reportLayerTime(layer_name, milliseconds);
    }

    bool last_callback_failed() const noexcept
    {
        return last_callback_failed_;
    }

private:
    JYPPX_TensorRtProfilerCallback callback_{nullptr};
    void* user_state_{nullptr};
    bool last_callback_failed_{false};
};

struct LayerReferencePayload
{
    nvinfer1::ILayer* layer{};
    std::vector<uint8_t> owned_data;
    std::vector<std::vector<uint8_t>> owned_weight_blobs;
    int32_t reduce_operation{-1};
    uint32_t reduce_axes{};
    int32_t reduce_keep_dimensions{};
    uint32_t softmax_axes{};
    int32_t unary_operation{-1};
    int32_t topk_operation{-1};
    int32_t topk_k{};
    uint32_t topk_axes{};
    int32_t gather_axis{};
    std::vector<int64_t> owned_dist_collective_groups;
};

struct LoopReferencePayload
{
    nvinfer1::ILoop* loop{};
};

struct IfConditionalReferencePayload
{
    nvinfer1::IIfConditional* conditional{};
};

struct AttentionReferencePayload
{
    nvinfer1::IAttention* attention{};
};

template <typename TObject>
void destroy_payload(void* payload)
{
    delete static_cast<TObject*>(payload);
}

void destroy_layer_reference_payload(void* payload)
{
    delete static_cast<LayerReferencePayload*>(payload);
}

void destroy_loop_reference_payload(void* payload)
{
    delete static_cast<LoopReferencePayload*>(payload);
}

void destroy_if_conditional_reference_payload(void* payload)
{
    delete static_cast<IfConditionalReferencePayload*>(payload);
}

void destroy_attention_reference_payload(void* payload)
{
    delete static_cast<AttentionReferencePayload*>(payload);
}

#if JYPPX_HAS_TENSORRT_ONNX_CONFIG
void destroy_onnx_config_payload(void* payload)
{
    delete static_cast<nvonnxparser::IOnnxConfig*>(payload);
}
#endif

#if JYPPX_HAS_TENSORRT_ONNXPARSER
std::mutex g_onnx_parser_support_mutex;
std::unordered_set<const nvonnxparser::IParser*> g_onnx_parser_support_ready;

void mark_onnx_parser_support_ready(const nvonnxparser::IParser* parser, const bool ready)
{
    std::lock_guard<std::mutex> lock(g_onnx_parser_support_mutex);
    if (ready)
    {
        g_onnx_parser_support_ready.insert(parser);
    }
    else
    {
        g_onnx_parser_support_ready.erase(parser);
    }
}

bool is_onnx_parser_support_ready(const nvonnxparser::IParser* parser)
{
    std::lock_guard<std::mutex> lock(g_onnx_parser_support_mutex);
    return g_onnx_parser_support_ready.find(parser) != g_onnx_parser_support_ready.end();
}

void destroy_parser_payload(void* payload)
{
    auto* parser = static_cast<nvonnxparser::IParser*>(payload);
    mark_onnx_parser_support_ready(parser, false);
    delete parser;
}
#endif

template <typename TObject>
TObject* get_payload(const JYPPX_TensorRtObjectBase* object)
{
    return static_cast<TObject*>(jyppx::tensorrt::get_payload(object));
}

JYPPX_StatusCode report_null_vendor_object(const char* feature_name)
{
    std::ostringstream builder;
    builder << feature_name << " returned a null TensorRT object.";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
    return JYPPX_STATUS_RUNTIME_ERROR;
}

#if JYPPX_HAS_TENSORRT
JYPPX_StatusCode create_infer_runtime_with_seh_guard(ManagedLogger& logger, nvinfer1::IRuntime** out_runtime)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_runtime = nvinfer1::createInferRuntime(logger);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_runtime = nullptr;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, "runtime creation", seh_exception_code);
    }
#else
    *out_runtime = nvinfer1::createInferRuntime(logger);
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode create_infer_builder_with_seh_guard(ManagedLogger& logger, nvinfer1::IBuilder** out_builder)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_builder = nvinfer1::createInferBuilder(logger);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_builder = nullptr;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, "builder creation", seh_exception_code);
    }
#else
    *out_builder = nvinfer1::createInferBuilder(logger);
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode create_infer_runtime_with_guard(ManagedLogger& logger, nvinfer1::IRuntime** out_runtime)
{
    try
    {
        return create_infer_runtime_with_seh_guard(logger, out_runtime);
    }
    catch (const std::exception& exception)
    {
        *out_runtime = nullptr;
        return jyppx::tensorrt::report_vendor_exception(kLine, "runtime creation", exception.what());
    }
    catch (...)
    {
        *out_runtime = nullptr;
        return jyppx::tensorrt::report_vendor_exception(kLine, "runtime creation", "unknown native exception");
    }
}

JYPPX_StatusCode create_infer_builder_with_guard(ManagedLogger& logger, nvinfer1::IBuilder** out_builder)
{
    try
    {
        return create_infer_builder_with_seh_guard(logger, out_builder);
    }
    catch (const std::exception& exception)
    {
        *out_builder = nullptr;
        return jyppx::tensorrt::report_vendor_exception(kLine, "builder creation", exception.what());
    }
    catch (...)
    {
        *out_builder = nullptr;
        return jyppx::tensorrt::report_vendor_exception(kLine, "builder creation", "unknown native exception");
    }
}
#endif

JYPPX_StatusCode create_handle_with_payload(
    JYPPX_TensorRtObjectBase** out_handle,
    const JYPPX_TensorRtObjectKind kind,
    void* payload,
    void (*destroyer)(void*))
{
    auto* handle = jyppx::tensorrt::create_object(kLine, kind);
    if (handle == nullptr)
    {
        if (destroyer != nullptr && payload != nullptr)
        {
            destroyer(payload);
        }

        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    jyppx::tensorrt::attach_payload(handle, payload, destroyer);
    *out_handle = handle;
    return JYPPX_STATUS_OK;
}

void copy_dims(const nvinfer1::Dims& source, JYPPX_TensorRtDims* destination)
{
    destination->nb_dims = source.nbDims;
    for (int32_t i = 0; i < 8; ++i)
    {
        destination->d[i] = i < source.nbDims ? static_cast<int32_t>(source.d[i]) : 0;
    }
}

void copy_dims64(const nvinfer1::Dims& source, JYPPX_TensorRtDims64* destination)
{
    destination->nb_dims = source.nbDims;
    for (int32_t i = 0; i < 8; ++i)
    {
        destination->d[i] = i < source.nbDims ? static_cast<int64_t>(source.d[i]) : 0;
    }
}

void copy_tensor_name(const char* source, char (&destination)[256])
{
    std::memset(destination, 0, 256);
    if (source == nullptr)
    {
        return;
    }

    const size_t length = std::strlen(source);
    const size_t copy_length = length < 255 ? length : 255;
    std::memcpy(destination, source, copy_length);
    destination[copy_length] = '\0';
}

void copy_c_string(const char* source, char* destination, const size_t capacity)
{
    if (destination == nullptr || capacity == 0)
    {
        return;
    }

    std::memset(destination, 0, capacity);
    if (source == nullptr)
    {
        return;
    }

    const size_t length = std::strlen(source);
    const size_t copy_length = length < capacity - 1 ? length : capacity - 1;
    std::memcpy(destination, source, copy_length);
    destination[copy_length] = '\0';
}

void reset_runtime_create_diagnostic_info(JYPPX_TensorRtRuntimeCreateDiagnosticInfo* out_info) noexcept
{
    if (out_info == nullptr)
    {
        return;
    }

    std::memset(out_info, 0, sizeof(*out_info));
    out_info->line = static_cast<uint32_t>(kLine);
    out_info->last_status = static_cast<int32_t>(JYPPX_STATUS_NOT_READY);
    out_info->tensor_rt_available = JYPPX_HAS_TENSORRT ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->expected_major = 11;
    out_info->bridge_built_major = JYPPX_TENSORRT_VERSION_MAJOR_NUM;
    out_info->last_logger_severity = 0;
    copy_c_string(JYPPX_TENSORRT_VERSION_TEXT, out_info->detected_version, sizeof(out_info->detected_version));
    copy_c_string("not-attempted", out_info->create_runtime_phase, sizeof(out_info->create_runtime_phase));
    copy_c_string("Runtime create diagnostic has not reached createInferRuntime.", out_info->native_detail, sizeof(out_info->native_detail));
    copy_c_string("Runtime create diagnostic was not attempted.", out_info->diagnostic, sizeof(out_info->diagnostic));
}

void set_runtime_create_diagnostic_message(
    JYPPX_TensorRtRuntimeCreateDiagnosticInfo* out_info,
    const char* message) noexcept
{
    if (out_info == nullptr)
    {
        return;
    }

    copy_c_string(message, out_info->diagnostic, sizeof(out_info->diagnostic));
}

#if JYPPX_HAS_TENSORRT
void copy_logger_runtime_create_diagnostic(
    const ManagedLogger& logger,
    JYPPX_TensorRtRuntimeCreateDiagnosticInfo* out_info) noexcept
{
    if (out_info == nullptr)
    {
        return;
    }

    out_info->logger_callback_available = logger.callback_available() ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->logger_message_count = logger.message_count();
    out_info->last_logger_severity = logger.last_severity();
    copy_c_string(logger.last_message(), out_info->last_logger_message, sizeof(out_info->last_logger_message));
}
#endif

JYPPX_StatusCode copy_string_to_buffer(const char* value, char* output_buffer, const size_t output_buffer_size, size_t* out_required_size)
{
    const char* safe_value = value != nullptr ? value : "";
    const size_t required_size = std::strlen(safe_value) + 1;
    *out_required_size = required_size;

    if (output_buffer == nullptr || output_buffer_size == 0)
    {
        return JYPPX_STATUS_OK;
    }

    if (output_buffer_size < required_size)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Output string buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    std::memcpy(output_buffer, safe_value, required_size);
    return JYPPX_STATUS_OK;
}

#if JYPPX_HAS_TENSORRT
JYPPX_StatusCode copy_interface_info_to_buffer(
    const nvinfer1::InterfaceInfo& info,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor)
{
    *out_major = static_cast<int32_t>(info.major);
    *out_minor = static_cast<int32_t>(info.minor);
    return copy_string_to_buffer(info.kind, output_buffer, output_buffer_size, out_required_size);
}

template <typename TObject>
JYPPX_StatusCode get_callback_interface_info_with_seh_guard(
    const TObject* object,
    nvinfer1::InterfaceInfo* out_info,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_info = object->getInterfaceInfo();
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_info = nvinfer1::InterfaceInfo{"", 0, 0};
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_info = object->getInterfaceInfo();
    return JYPPX_STATUS_OK;
#endif
}

template <typename TObject>
JYPPX_StatusCode get_callback_interface_info(
    const TObject* object,
    nvinfer1::InterfaceInfo* out_info,
    const char* feature_name)
{
    try
    {
        return get_callback_interface_info_with_seh_guard(object, out_info, feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_info = nvinfer1::InterfaceInfo{"", 0, 0};
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_info = nvinfer1::InterfaceInfo{"", 0, 0};
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}

template <typename TObject>
JYPPX_StatusCode get_callback_api_language_with_seh_guard(
    const TObject* object,
    int32_t* out_api_language,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_api_language = static_cast<int32_t>(object->getAPILanguage());
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_api_language = 0;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    (void)feature_name;
    *out_api_language = static_cast<int32_t>(object->getAPILanguage());
    return JYPPX_STATUS_OK;
#endif
}

template <typename TObject>
JYPPX_StatusCode get_callback_api_language(
    const TObject* object,
    int32_t* out_api_language,
    const char* feature_name)
{
    try
    {
        return get_callback_api_language_with_seh_guard(object, out_api_language, feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_api_language = 0;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_api_language = 0;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}
#endif

JYPPX_StatusCode validate_c_string(const char* value, const char* parameter_name)
{
    if (value != nullptr && value[0] != '\0')
    {
        return JYPPX_STATUS_OK;
    }

    std::ostringstream builder;
    builder << parameter_name << " must not be null or empty.";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
    return JYPPX_STATUS_INVALID_ARGUMENT;
}

JYPPX_StatusCode validate_named_tensor(const char* tensor_name)
{
    return validate_c_string(tensor_name, "tensor_name");
}

JYPPX_StatusCode validate_index(const int32_t index, const int32_t count, const char* parameter_name)
{
    if (index >= 0 && index < count)
    {
        return JYPPX_STATUS_OK;
    }

    std::ostringstream builder;
    builder << parameter_name << " index " << index << " is outside the valid range [0, " << count << ").";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
    return JYPPX_STATUS_INVALID_ARGUMENT;
}

JYPPX_StatusCode make_dims(const JYPPX_TensorRtDims* source, nvinfer1::Dims* destination, const char* parameter_name)
{
    if (source == nullptr)
    {
        std::ostringstream builder;
        builder << parameter_name << " must not be null.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (source->nb_dims < 0 || source->nb_dims > 8)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT dimensions must contain between 0 and 8 dimensions.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    destination->nbDims = source->nb_dims;
    for (int32_t i = 0; i < 8; ++i)
    {
        destination->d[i] = source->d[i];
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode make_permutation(const JYPPX_TensorRtDims* source, nvinfer1::Permutation* destination, const char* parameter_name)
{
    if (source == nullptr)
    {
        std::ostringstream builder;
        builder << parameter_name << " must not be null.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (source->nb_dims < 0 || source->nb_dims > 8)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT permutation must contain between 0 and 8 entries.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    for (int32_t i = 0; i < 8; ++i)
    {
        destination->order[i] = i;
    }

    for (int32_t i = 0; i < source->nb_dims; ++i)
    {
        if (source->d[i] < 0 || source->d[i] >= source->nb_dims)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT permutation entries must be in the range [0, rank).");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        destination->order[i] = source->d[i];
    }

    return JYPPX_STATUS_OK;
}

int32_t get_layer_input_rank(nvinfer1::ILayer* layer)
{
    if (layer == nullptr || layer->getNbInputs() <= 0 || layer->getInput(0) == nullptr)
    {
        return 8;
    }

    const int32_t rank = layer->getInput(0)->getDimensions().nbDims;
    return rank >= 0 && rank <= 8 ? rank : 8;
}

int32_t get_shuffle_second_transpose_rank(nvinfer1::IShuffleLayer* layer)
{
    if (layer == nullptr)
    {
        return 8;
    }

    const int32_t reshape_rank = layer->getReshapeDimensions().nbDims;
    if (reshape_rank > 0 && reshape_rank <= 8)
    {
        return reshape_rank;
    }

    auto* output = layer->getOutput(0);
    if (output != nullptr)
    {
        const int32_t output_rank = output->getDimensions().nbDims;
        if (output_rank >= 0 && output_rank <= 8)
        {
            return output_rank;
        }
    }

    return get_layer_input_rank(layer);
}

void copy_permutation(const nvinfer1::Permutation& source, const int32_t rank, JYPPX_TensorRtDims* destination)
{
    const int32_t safe_rank = rank >= 0 && rank <= 8 ? rank : 8;
    destination->nb_dims = safe_rank;
    for (int32_t i = 0; i < 8; ++i)
    {
        destination->d[i] = i < safe_rank ? source.order[i] : 0;
    }
}

size_t get_data_type_size(const int32_t data_type)
{
    switch (static_cast<nvinfer1::DataType>(data_type))
    {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
        return 1;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT64:
        return 8;
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    case nvinfer1::DataType::kUINT8:
        return 1;
#endif
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 11
    case nvinfer1::DataType::kFP8:
    case nvinfer1::DataType::kINT4:
        return 1;
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kFP4:
        return 2;
#endif
    default:
        return 0;
    }
}

JYPPX_StatusCode copy_weights(const int32_t data_type, const void* values, const size_t value_count, std::vector<uint8_t>* out_owned_data)
{
    if (out_owned_data == nullptr)
    {
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const size_t element_size = get_data_type_size(data_type);
    if (element_size == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT weights data type is not supported by the active TensorRT line.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (value_count == 0)
    {
        out_owned_data->clear();
        return JYPPX_STATUS_OK;
    }

    if (values == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT weights values must not be null when value_count is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    out_owned_data->resize(value_count * element_size);
    std::memcpy(out_owned_data->data(), values, out_owned_data->size());
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode copy_optional_weights(const int32_t data_type, const void* values, const size_t value_count, const char* name, const bool required, std::vector<uint8_t>* out_owned_data)
{
    if (out_owned_data == nullptr)
    {
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    out_owned_data->clear();
    if (values == nullptr || value_count == 0)
    {
        if (required)
        {
            std::ostringstream builder;
            builder << name << " weights must not be null or empty.";
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        return JYPPX_STATUS_OK;
    }

    return copy_weights(data_type, values, value_count, out_owned_data);
}

JYPPX_StatusCode create_tensor_reference_handle(nvinfer1::ITensor* tensor, JYPPX_TensorRtTensor** out_tensor)
{
    if (tensor == nullptr)
    {
        return report_null_vendor_object("TensorRT tensor reference creation");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    const JYPPX_StatusCode status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, tensor, nullptr);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_tensor = reinterpret_cast<JYPPX_TensorRtTensor*>(handle);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode create_layer_reference_handle(
    nvinfer1::ILayer* layer,
    JYPPX_TensorRtLayer** out_layer,
    std::vector<uint8_t> owned_data = {},
    const int32_t reduce_operation = -1,
    const uint32_t reduce_axes = 0,
    const int32_t reduce_keep_dimensions = 0,
    const uint32_t softmax_axes = 0,
    const int32_t unary_operation = -1,
    const int32_t topk_operation = -1,
    const int32_t topk_k = 0,
    const uint32_t topk_axes = 0,
    const int32_t gather_axis = 0,
    std::vector<std::vector<uint8_t>> owned_weight_blobs = {})
{
    if (layer == nullptr)
    {
        return report_null_vendor_object("TensorRT layer reference creation");
    }

    auto* payload = new (std::nothrow) LayerReferencePayload();
    if (payload == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    payload->layer = layer;
    payload->owned_data = std::move(owned_data);
    payload->reduce_operation = reduce_operation;
    payload->reduce_axes = reduce_axes;
    payload->reduce_keep_dimensions = reduce_keep_dimensions;
    payload->softmax_axes = softmax_axes;
    payload->unary_operation = unary_operation;
    payload->topk_operation = topk_operation;
    payload->topk_k = topk_k;
    payload->topk_axes = topk_axes;
    payload->gather_axis = gather_axis;
    payload->owned_weight_blobs = std::move(owned_weight_blobs);

    JYPPX_TensorRtObjectBase* handle = nullptr;
    const JYPPX_StatusCode status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_LAYER, payload, &destroy_layer_reference_payload);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_layer = reinterpret_cast<JYPPX_TensorRtLayer*>(handle);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode create_loop_reference_handle(nvinfer1::ILoop* loop, JYPPX_TensorRtLoop** out_loop)
{
    if (loop == nullptr)
    {
        return report_null_vendor_object("TensorRT loop reference creation");
    }

    auto* payload = new (std::nothrow) LoopReferencePayload();
    if (payload == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    payload->loop = loop;
    JYPPX_TensorRtObjectBase* handle = nullptr;
    const JYPPX_StatusCode status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_LOOP, payload, &destroy_loop_reference_payload);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_loop = reinterpret_cast<JYPPX_TensorRtLoop*>(handle);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode create_if_conditional_reference_handle(nvinfer1::IIfConditional* conditional, JYPPX_TensorRtIfConditional** out_conditional)
{
    if (conditional == nullptr)
    {
        return report_null_vendor_object("TensorRT if-conditional reference creation");
    }

    auto* payload = new (std::nothrow) IfConditionalReferencePayload();
    if (payload == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    payload->conditional = conditional;
    JYPPX_TensorRtObjectBase* handle = nullptr;
    const JYPPX_StatusCode status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_IF_CONDITIONAL, payload, &destroy_if_conditional_reference_payload);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_conditional = reinterpret_cast<JYPPX_TensorRtIfConditional*>(handle);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode create_attention_reference_handle(nvinfer1::IAttention* attention, JYPPX_TensorRtAttention** out_attention)
{
    if (attention == nullptr)
    {
        return report_null_vendor_object("TensorRT attention reference creation");
    }

    auto* payload = new (std::nothrow) AttentionReferencePayload();
    if (payload == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    payload->attention = attention;
    JYPPX_TensorRtObjectBase* handle = nullptr;
    const JYPPX_StatusCode status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_ATTENTION, payload, &destroy_attention_reference_payload);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_attention = reinterpret_cast<JYPPX_TensorRtAttention*>(handle);
    return JYPPX_STATUS_OK;
}

nvinfer1::IAttention* get_attention_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* payload = get_payload<AttentionReferencePayload>(object);
    return payload != nullptr ? payload->attention : nullptr;
}

nvinfer1::ILayer* get_layer_payload(JYPPX_TensorRtLayer* layer)
{
    auto* payload = get_payload<LayerReferencePayload>(layer);
    return payload != nullptr ? payload->layer : nullptr;
}

LayerReferencePayload* get_layer_reference_payload(const JYPPX_TensorRtObjectBase* object)
{
    return get_payload<LayerReferencePayload>(object);
}

nvinfer1::ILayer* get_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* payload = get_payload<LayerReferencePayload>(object);
    return payload != nullptr ? payload->layer : nullptr;
}

nvinfer1::IConvolutionLayer* get_convolution_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kCONVOLUTION)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IConvolutionLayer*>(layer);
}

nvinfer1::IDeconvolutionLayer* get_deconvolution_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kDECONVOLUTION)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IDeconvolutionLayer*>(layer);
}

nvinfer1::IScaleLayer* get_scale_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kSCALE)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IScaleLayer*>(layer);
}

nvinfer1::IPaddingLayer* get_padding_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kPADDING)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IPaddingLayer*>(layer);
}

nvinfer1::ILRNLayer* get_lrn_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kLRN)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::ILRNLayer*>(layer);
}

nvinfer1::IQuantizeLayer* get_quantize_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kQUANTIZE)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IQuantizeLayer*>(layer);
}

nvinfer1::IDequantizeLayer* get_dequantize_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kDEQUANTIZE)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IDequantizeLayer*>(layer);
}

JYPPX_StatusCode get_engine_payload_ext(JYPPX_TensorRtCudaEngine* engine, nvinfer1::ICudaEngine** out_payload, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        std::ostringstream builder;
        builder << "Engine handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_payload = engine_payload;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode get_context_payload_ext(JYPPX_TensorRtExecutionContext* context, nvinfer1::IExecutionContext** out_payload, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    if (context_payload == nullptr)
    {
        std::ostringstream builder;
        builder << "Execution context handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_payload = context_payload;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode ensure_minimal_identity_network(nvinfer1::INetworkDefinition& network)
{
    if (network.getNbOutputs() > 0)
    {
        return JYPPX_STATUS_OK;
    }

    nvinfer1::ITensor* input_tensor = nullptr;
    if (network.getNbInputs() > 0)
    {
        input_tensor = network.getInput(0);
    }
    else
    {
        nvinfer1::Dims dims{};
        dims.nbDims = 4;
        dims.d[0] = 1;
        dims.d[1] = 1;
        dims.d[2] = 1;
        dims.d[3] = 1;
        input_tensor = network.addInput("input_0", nvinfer1::DataType::kFLOAT, dims);
    }

    if (input_tensor == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Failed to create or retrieve the smoke-test input tensor.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    nvinfer1::IIdentityLayer* identity = network.addIdentity(*input_tensor);
    if (identity == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Failed to add an identity layer to the smoke-test network.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    nvinfer1::ITensor* output_tensor = identity->getOutput(0);
    if (output_tensor == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Failed to retrieve the smoke-test network output tensor.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    network.markOutput(*output_tensor);
    return JYPPX_STATUS_OK;
}
#endif
}

JYPPX_StatusCode jyppx_trt11_query_adapter_info(JYPPX_TensorRtAdapterInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    jyppx::tensorrt::fill_adapter_info(out_info, kLine);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_logger_create(JYPPX_TensorRtLogger** out_logger)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_logger, "out_logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_logger = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* logger = new (std::nothrow) ManagedLogger();
    if (logger == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, logger, &destroy_payload<ManagedLogger>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_logger = reinterpret_cast<JYPPX_TensorRtLogger*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "logger creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_logger_create_with_callback(
    JYPPX_TensorRtLoggerCallback callback,
    void* user_state,
    int32_t minimum_severity,
    JYPPX_TensorRtLogger** out_logger)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_logger, "out_logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_logger = nullptr;
    if (callback == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Managed logger callback must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto* logger = new (std::nothrow) ManagedLogger(callback, user_state, minimum_severity);
    if (logger == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, logger, &destroy_payload<ManagedLogger>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_logger = reinterpret_cast<JYPPX_TensorRtLogger*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed logger callback creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_logger_emit_diagnostic(
    JYPPX_TensorRtLogger* logger,
    int32_t severity,
    const char* message,
    JYPPX_Boolean* out_callback_failed)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_callback_failed, "out_callback_failed");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_callback_failed = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(logger, kLine, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, "logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(message, "message");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* logger_payload = get_payload<ManagedLogger>(logger);
    if (logger_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Logger handle does not carry a TensorRT logger payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    logger_payload->log(static_cast<nvinfer1::ILogger::Severity>(severity), message);
    *out_callback_failed = logger_payload->last_callback_failed() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed logger diagnostic emission");
#endif
}

JYPPX_StatusCode jyppx_trt11_logger_get_interface_info(
    JYPPX_TensorRtLogger* logger,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_major, "out_major");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_minor, "out_minor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;
    *out_major = 0;
    *out_minor = 0;
    status = jyppx::tensorrt::validate_handle(logger, kLine, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, "logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11
    auto* logger_payload = get_payload<ManagedLogger>(logger);
    if (logger_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Logger handle does not carry a TensorRT logger payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::InterfaceInfo info{"", 0, 0};
    status = get_callback_interface_info(logger_payload, &info, "managed logger interface info query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_interface_info_to_buffer(info, output_buffer, output_buffer_size, out_required_size, out_major, out_minor);
#elif JYPPX_HAS_TENSORRT
    (void)logger;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed logger interface info query");
#else
    (void)logger;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed logger interface info query");
#endif
}

JYPPX_StatusCode jyppx_trt11_logger_get_api_language(
    JYPPX_TensorRtLogger* logger,
    int32_t* out_api_language)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_api_language, "out_api_language");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_api_language = 0;
    status = jyppx::tensorrt::validate_handle(logger, kLine, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, "logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11
    auto* logger_payload = get_payload<ManagedLogger>(logger);
    if (logger_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Logger handle does not carry a TensorRT logger payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return get_callback_api_language(logger_payload, out_api_language, "managed logger API language query");
#elif JYPPX_HAS_TENSORRT
    (void)logger;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed logger API language query");
#else
    (void)logger;
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed logger API language query");
#endif
}

JYPPX_StatusCode jyppx_trt11_progress_monitor_create_with_callback(
    JYPPX_TensorRtProgressMonitorCallback callback,
    void* user_state,
    JYPPX_TensorRtProgressMonitor** out_monitor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_monitor, "out_monitor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_monitor = nullptr;
    if (callback == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Managed progress monitor callback must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 11)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed progress monitor callback creation");
    }

    auto* monitor = new (std::nothrow) ManagedProgressMonitor(callback, user_state);
    if (monitor == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_PROGRESS_MONITOR, monitor, &destroy_payload<ManagedProgressMonitor>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_monitor = reinterpret_cast<JYPPX_TensorRtProgressMonitor*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed progress monitor callback creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_progress_monitor_emit_diagnostic(
    JYPPX_TensorRtProgressMonitor* monitor,
    int32_t event_kind,
    const char* phase_name,
    const char* parent_phase,
    int32_t step,
    int32_t nb_steps,
    JYPPX_Boolean* out_should_continue,
    JYPPX_Boolean* out_callback_failed)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_should_continue, "out_should_continue");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_callback_failed, "out_callback_failed");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_should_continue = JYPPX_TRUE;
    *out_callback_failed = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(monitor, kLine, JYPPX_TENSORRT_OBJECT_KIND_PROGRESS_MONITOR, "monitor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(phase_name, "phase_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 11)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed progress monitor diagnostic emission");
    }

    auto* monitor_payload = get_payload<ManagedProgressMonitor>(monitor);
    if (monitor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Progress monitor handle does not carry a TensorRT progress monitor payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (!monitor_payload->emit_diagnostic(event_kind, phase_name, parent_phase, step, nb_steps, out_should_continue))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unsupported TensorRT progress monitor diagnostic event kind.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_callback_failed = monitor_payload->last_callback_failed() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)event_kind;
    (void)parent_phase;
    (void)step;
    (void)nb_steps;
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed progress monitor diagnostic emission");
#endif
}

JYPPX_StatusCode jyppx_trt11_progress_monitor_get_interface_info(
    JYPPX_TensorRtProgressMonitor* monitor,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_major, "out_major");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_minor, "out_minor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;
    *out_major = 0;
    *out_minor = 0;
    status = jyppx::tensorrt::validate_handle(monitor, kLine, JYPPX_TENSORRT_OBJECT_KIND_PROGRESS_MONITOR, "monitor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11
    auto* monitor_payload = get_payload<ManagedProgressMonitor>(monitor);
    if (monitor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Progress monitor handle does not carry a TensorRT progress monitor payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::InterfaceInfo info{"", 0, 0};
    status = get_callback_interface_info(monitor_payload, &info, "managed progress monitor interface info query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_interface_info_to_buffer(info, output_buffer, output_buffer_size, out_required_size, out_major, out_minor);
#elif JYPPX_HAS_TENSORRT
    (void)monitor;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed progress monitor interface info query");
#else
    (void)monitor;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed progress monitor interface info query");
#endif
}

JYPPX_StatusCode jyppx_trt11_progress_monitor_get_api_language(
    JYPPX_TensorRtProgressMonitor* monitor,
    int32_t* out_api_language)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_api_language, "out_api_language");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_api_language = 0;
    status = jyppx::tensorrt::validate_handle(monitor, kLine, JYPPX_TENSORRT_OBJECT_KIND_PROGRESS_MONITOR, "monitor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11
    auto* monitor_payload = get_payload<ManagedProgressMonitor>(monitor);
    if (monitor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Progress monitor handle does not carry a TensorRT progress monitor payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return get_callback_api_language(monitor_payload, out_api_language, "managed progress monitor API language query");
#elif JYPPX_HAS_TENSORRT
    (void)monitor;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed progress monitor API language query");
#else
    (void)monitor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed progress monitor API language query");
#endif
}

JYPPX_StatusCode jyppx_trt11_profiler_create_with_callback(
    JYPPX_TensorRtProfilerCallback callback,
    void* user_state,
    JYPPX_TensorRtProfiler** out_profiler)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_profiler, "out_profiler");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_profiler = nullptr;
    if (callback == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Managed profiler callback must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 11)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed profiler callback creation");
    }

    auto* profiler = new (std::nothrow) ManagedProfiler(callback, user_state);
    if (profiler == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_PROFILER, profiler, &destroy_payload<ManagedProfiler>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_profiler = reinterpret_cast<JYPPX_TensorRtProfiler*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed profiler callback creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_profiler_emit_diagnostic(
    JYPPX_TensorRtProfiler* profiler,
    const char* layer_name,
    float milliseconds,
    JYPPX_Boolean* out_callback_failed)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_callback_failed, "out_callback_failed");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_callback_failed = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(profiler, kLine, JYPPX_TENSORRT_OBJECT_KIND_PROFILER, "profiler");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(layer_name, "layer_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 11)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed profiler diagnostic emission");
    }

    auto* profiler_payload = get_payload<ManagedProfiler>(profiler);
    if (profiler_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profiler handle does not carry a TensorRT profiler payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    profiler_payload->emit_diagnostic(layer_name, milliseconds);
    *out_callback_failed = profiler_payload->last_callback_failed() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)milliseconds;
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed profiler diagnostic emission");
#endif
}

JYPPX_StatusCode jyppx_trt11_profiler_get_interface_info(
    JYPPX_TensorRtProfiler* profiler,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_major, "out_major");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_minor, "out_minor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;
    *out_major = 0;
    *out_minor = 0;
    status = jyppx::tensorrt::validate_handle(profiler, kLine, JYPPX_TENSORRT_OBJECT_KIND_PROFILER, "profiler");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11
    auto* profiler_payload = get_payload<ManagedProfiler>(profiler);
    if (profiler_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profiler handle does not carry a TensorRT profiler payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::InterfaceInfo info{"", 0, 0};
    status = get_callback_interface_info(profiler_payload, &info, "managed profiler interface info query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_interface_info_to_buffer(info, output_buffer, output_buffer_size, out_required_size, out_major, out_minor);
#elif JYPPX_HAS_TENSORRT
    (void)profiler;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed profiler interface info query");
#else
    (void)profiler;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed profiler interface info query");
#endif
}

JYPPX_StatusCode jyppx_trt11_profiler_get_api_language(
    JYPPX_TensorRtProfiler* profiler,
    int32_t* out_api_language)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_api_language, "out_api_language");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_api_language = 0;
    status = jyppx::tensorrt::validate_handle(profiler, kLine, JYPPX_TENSORRT_OBJECT_KIND_PROFILER, "profiler");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11
    auto* profiler_payload = get_payload<ManagedProfiler>(profiler);
    if (profiler_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profiler handle does not carry a TensorRT profiler payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return get_callback_api_language(profiler_payload, out_api_language, "managed profiler API language query");
#elif JYPPX_HAS_TENSORRT
    (void)profiler;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed profiler API language query");
#else
    (void)profiler;
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed profiler API language query");
#endif
}

JYPPX_StatusCode jyppx_trt11_runtime_create(JYPPX_TensorRtLogger* logger, JYPPX_TensorRtRuntime** out_runtime)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_runtime, "out_runtime");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(logger, kLine, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, "logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_runtime = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* logger_payload = get_payload<ManagedLogger>(logger);
    if (logger_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Logger handle does not carry a TensorRT logger payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IRuntime* runtime = nullptr;
    status = create_infer_runtime_with_guard(*logger_payload, &runtime);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (runtime == nullptr)
    {
        return report_null_vendor_object("createInferRuntime");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, runtime, &destroy_payload<nvinfer1::IRuntime>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_runtime = reinterpret_cast<JYPPX_TensorRtRuntime*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_runtime_create_diagnostic(
    JYPPX_TensorRtLogger* logger,
    JYPPX_TensorRtRuntimeCreateDiagnosticInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    reset_runtime_create_diagnostic_info(out_info);
    out_info->logger_handle_present = logger != nullptr ? JYPPX_TRUE : JYPPX_FALSE;

    status = jyppx::tensorrt::validate_handle(logger, kLine, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, "logger");
    if (status != JYPPX_STATUS_OK)
    {
        out_info->last_status = static_cast<int32_t>(status);
        set_runtime_create_diagnostic_message(out_info, "Logger handle validation failed before createInferRuntime.");
        return JYPPX_STATUS_OK;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11
    try
    {
        auto* logger_payload = get_payload<ManagedLogger>(logger);
        out_info->logger_payload_present = logger_payload != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        if (logger_payload == nullptr)
        {
            out_info->last_status = static_cast<int32_t>(JYPPX_STATUS_INVALID_STATE);
            copy_c_string("logger-payload-missing", out_info->create_runtime_phase, sizeof(out_info->create_runtime_phase));
            copy_c_string("Logger handle validation passed, but the handle did not carry a ManagedLogger payload.", out_info->native_detail, sizeof(out_info->native_detail));
            set_runtime_create_diagnostic_message(out_info, "Logger handle does not carry a TensorRT logger payload.");
            return JYPPX_STATUS_OK;
        }

        nvinfer1::IRuntime* runtime = nullptr;
        out_info->attempted = JYPPX_TRUE;
        copy_logger_runtime_create_diagnostic(*logger_payload, out_info);
        copy_c_string("before-createInferRuntime", out_info->create_runtime_phase, sizeof(out_info->create_runtime_phase));
        copy_c_string("About to call nvinfer1::createInferRuntime through the guarded TRT11 bridge path.", out_info->native_detail, sizeof(out_info->native_detail));
        status = create_infer_runtime_with_guard(*logger_payload, &runtime);
        copy_logger_runtime_create_diagnostic(*logger_payload, out_info);
        out_info->last_status = static_cast<int32_t>(status);
        out_info->create_infer_runtime_returned_non_null = runtime != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->create_infer_runtime_returned_null = runtime == nullptr ? JYPPX_TRUE : JYPPX_FALSE;

        if (runtime != nullptr)
        {
            delete runtime;
            copy_c_string("after-createInferRuntime-non-null-destroyed", out_info->create_runtime_phase, sizeof(out_info->create_runtime_phase));
            copy_c_string("createInferRuntime returned a non-null runtime; the diagnostic destroyed it immediately without exposing the native pointer.", out_info->native_detail, sizeof(out_info->native_detail));
            set_runtime_create_diagnostic_message(out_info, "createInferRuntime returned a non-null runtime; diagnostic destroyed the temporary runtime immediately.");
            return JYPPX_STATUS_OK;
        }

        if (status == JYPPX_STATUS_OK)
        {
            copy_c_string("after-createInferRuntime-null-guard-ok", out_info->create_runtime_phase, sizeof(out_info->create_runtime_phase));
            copy_c_string("The guarded native call completed with OK status, but TensorRT returned a null IRuntime pointer. Inspect logger message fields and vendor initialization state.", out_info->native_detail, sizeof(out_info->native_detail));
            set_runtime_create_diagnostic_message(out_info, "createInferRuntime returned null while the native guard reported OK.");
        }
        else
        {
            copy_c_string("after-createInferRuntime-status-failed", out_info->create_runtime_phase, sizeof(out_info->create_runtime_phase));
            copy_c_string("The guarded native createInferRuntime call returned a failing bridge status before a runtime was created.", out_info->native_detail, sizeof(out_info->native_detail));
            set_runtime_create_diagnostic_message(out_info, "createInferRuntime failed before returning a runtime; see last_status and bridge last-error state.");
        }

        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        out_info->last_status = static_cast<int32_t>(JYPPX_STATUS_RUNTIME_ERROR);
        out_info->create_infer_runtime_returned_null = JYPPX_TRUE;
        copy_c_string("native-exception", out_info->create_runtime_phase, sizeof(out_info->create_runtime_phase));
        copy_c_string("A native C++ exception was caught inside the diagnostic wrapper.", out_info->native_detail, sizeof(out_info->native_detail));
        set_runtime_create_diagnostic_message(out_info, exception.what());
        return JYPPX_STATUS_OK;
    }
    catch (...)
    {
        out_info->last_status = static_cast<int32_t>(JYPPX_STATUS_RUNTIME_ERROR);
        out_info->create_infer_runtime_returned_null = JYPPX_TRUE;
        copy_c_string("native-unknown-exception", out_info->create_runtime_phase, sizeof(out_info->create_runtime_phase));
        copy_c_string("An unknown native exception was caught inside the diagnostic wrapper.", out_info->native_detail, sizeof(out_info->native_detail));
        set_runtime_create_diagnostic_message(out_info, "Unknown native exception while diagnosing createInferRuntime.");
        return JYPPX_STATUS_OK;
    }
#elif JYPPX_HAS_TENSORRT
    out_info->last_status = static_cast<int32_t>(JYPPX_STATUS_NOT_SUPPORTED);
    copy_c_string("version-guard-not-supported", out_info->create_runtime_phase, sizeof(out_info->create_runtime_phase));
    copy_c_string("The bridge was built with TensorRT support, but not against TensorRT major version 11.", out_info->native_detail, sizeof(out_info->native_detail));
    set_runtime_create_diagnostic_message(out_info, "TRT11 runtime create diagnostic requires a bridge built against TensorRT 11.");
    return JYPPX_STATUS_OK;
#else
    out_info->last_status = static_cast<int32_t>(JYPPX_STATUS_DEPENDENCY_MISSING);
    copy_c_string("tensorrt-dependency-missing", out_info->create_runtime_phase, sizeof(out_info->create_runtime_phase));
    copy_c_string("The native bridge was built without TensorRT support.", out_info->native_detail, sizeof(out_info->native_detail));
    set_runtime_create_diagnostic_message(out_info, "TensorRT dependency is not available in this native bridge.");
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_create(JYPPX_TensorRtLogger* logger, JYPPX_TensorRtBuilder** out_builder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_builder, "out_builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(logger, kLine, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, "logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_builder = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* logger_payload = get_payload<ManagedLogger>(logger);
    if (logger_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Logger handle does not carry a TensorRT logger payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IBuilder* builder = nullptr;
    status = create_infer_builder_with_guard(*logger_payload, &builder);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (builder == nullptr)
    {
        return report_null_vendor_object("createInferBuilder");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, builder, &destroy_payload<nvinfer1::IBuilder>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_builder = reinterpret_cast<JYPPX_TensorRtBuilder*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_create_config(JYPPX_TensorRtBuilder* builder, JYPPX_TensorRtBuilderConfig** out_config)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_config, "out_config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_config = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IBuilderConfig* config = builder_payload->createBuilderConfig();
    if (config == nullptr)
    {
        return report_null_vendor_object("createBuilderConfig");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, config, &destroy_payload<nvinfer1::IBuilderConfig>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_config = reinterpret_cast<JYPPX_TensorRtBuilderConfig*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_create_network(JYPPX_TensorRtBuilder* builder, uint32_t creation_flags, JYPPX_TensorRtNetworkDefinition** out_network)
{
    (void)creation_flags;
    auto status = jyppx::tensorrt::validate_output_pointer(out_network, "out_network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_network = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::INetworkDefinition* network = builder_payload->createNetworkV2(0U);
    if (network == nullptr)
    {
        return report_null_vendor_object("createNetworkV2");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, network, &destroy_payload<nvinfer1::INetworkDefinition>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_network = reinterpret_cast<JYPPX_TensorRtNetworkDefinition*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network definition creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_build_serialized_network(JYPPX_TensorRtBuilder* builder, JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtHostMemory** out_host_memory)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_host_memory, "out_host_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_host_memory = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (builder_payload == nullptr || network_payload == nullptr || config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "One or more TensorRT handles do not carry the expected vendor payloads.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    status = ensure_minimal_identity_network(*network_payload);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::IHostMemory* host_memory = builder_payload->buildSerializedNetwork(*network_payload, *config_payload);
    if (host_memory == nullptr)
    {
        return report_null_vendor_object("buildSerializedNetwork");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY, host_memory, &destroy_payload<nvinfer1::IHostMemory>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_host_memory = reinterpret_cast<JYPPX_TensorRtHostMemory*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "serialized network build");
#endif
}

JYPPX_StatusCode jyppx_trt11_runtime_deserialize_engine(JYPPX_TensorRtRuntime* runtime, const void* engine_data, size_t engine_size, JYPPX_TensorRtCudaEngine** out_engine)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_engine, "out_engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (engine_data == nullptr || engine_size == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Serialized engine data must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_engine = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime handle does not carry a TensorRT runtime payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::ICudaEngine* engine = runtime_payload->deserializeCudaEngine(engine_data, engine_size);
    if (engine == nullptr)
    {
        return report_null_vendor_object("deserializeCudaEngine");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, engine, &destroy_payload<nvinfer1::ICudaEngine>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_engine = reinterpret_cast<JYPPX_TensorRtCudaEngine*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine deserialization");
#endif
}

JYPPX_StatusCode jyppx_trt11_runtime_deserialize_host_memory(JYPPX_TensorRtRuntime* runtime, JYPPX_TensorRtHostMemory* host_memory, JYPPX_TensorRtCudaEngine** out_engine)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_engine, "out_engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(host_memory, kLine, JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY, "host_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* host_memory_payload = get_payload<nvinfer1::IHostMemory>(host_memory);
    if (host_memory_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Host memory handle does not carry a TensorRT host-memory payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return jyppx_trt11_runtime_deserialize_engine(runtime, host_memory_payload->data(), host_memory_payload->size(), out_engine);
#else
    *out_engine = nullptr;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine deserialization from host memory");
#endif
}

JYPPX_StatusCode jyppx_trt11_host_memory_get_size(JYPPX_TensorRtHostMemory* host_memory, size_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(host_memory, kLine, JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY, "host_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* host_memory_payload = get_payload<nvinfer1::IHostMemory>(host_memory);
    if (host_memory_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Host memory handle does not carry a TensorRT host-memory payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_size = host_memory_payload->size();
    return JYPPX_STATUS_OK;
#else
    *out_size = 0;
    return jyppx::tensorrt::report_vendor_missing(kLine, "host memory size query");
#endif
}

JYPPX_StatusCode jyppx_trt11_host_memory_copy_to_buffer(JYPPX_TensorRtHostMemory* host_memory, void* destination, size_t destination_size, size_t* out_bytes_written)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_bytes_written, "out_bytes_written");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_bytes_written = 0;

    status = jyppx::tensorrt::validate_handle(host_memory, kLine, JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY, "host_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (destination == nullptr && destination_size > 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Destination buffer must not be null when destination size is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto* host_memory_payload = get_payload<nvinfer1::IHostMemory>(host_memory);
    if (host_memory_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Host memory handle does not carry a TensorRT host-memory payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const size_t source_size = host_memory_payload->size();
    *out_bytes_written = source_size <= destination_size ? source_size : destination_size;
    if (*out_bytes_written > 0)
    {
        std::memcpy(destination, host_memory_payload->data(), *out_bytes_written);
    }

    return source_size <= destination_size ? JYPPX_STATUS_OK : JYPPX_STATUS_BUFFER_TOO_SMALL;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "host memory copy");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_io_tensor_count(JYPPX_TensorRtCudaEngine* engine, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine handle does not carry a TensorRT engine payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = engine_payload->getNbIOTensors();
    return JYPPX_STATUS_OK;
#else
    *out_count = 0;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine I/O tensor count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_io_tensor_info(JYPPX_TensorRtCudaEngine* engine, int32_t index, JYPPX_TensorRtTensorInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine handle does not carry a TensorRT engine payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int32_t count = engine_payload->getNbIOTensors();
    if (index >= count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const char* name = engine_payload->getIOTensorName(index);
    if (name == nullptr)
    {
        return report_null_vendor_object("getIOTensorName");
    }

    std::memset(out_info, 0, sizeof(*out_info));
    out_info->index = index;
    copy_tensor_name(name, out_info->name);
    out_info->data_type = static_cast<int32_t>(engine_payload->getTensorDataType(name));
    const nvinfer1::TensorIOMode mode = engine_payload->getTensorIOMode(name);
    out_info->io_mode = mode == nvinfer1::TensorIOMode::kINPUT
        ? JYPPX_TENSORRT_IO_MODE_INPUT
        : (mode == nvinfer1::TensorIOMode::kOUTPUT ? JYPPX_TENSORRT_IO_MODE_OUTPUT : JYPPX_TENSORRT_IO_MODE_UNKNOWN);
    copy_dims(engine_payload->getTensorShape(name), &out_info->shape);
    return JYPPX_STATUS_OK;
#else
    std::memset(out_info, 0, sizeof(*out_info));
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine I/O tensor info query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_create_execution_context(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtExecutionContext** out_context)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_context, "out_context");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_context = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine handle does not carry a TensorRT engine payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IExecutionContext* context = engine_payload->createExecutionContext();
    if (context == nullptr)
    {
        return report_null_vendor_object("createExecutionContext");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, context, &destroy_payload<nvinfer1::IExecutionContext>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_context = reinterpret_cast<JYPPX_TensorRtExecutionContext*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_create_optimization_profile(JYPPX_TensorRtBuilder* builder, JYPPX_TensorRtOptimizationProfile** out_profile)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_profile, "out_profile");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_profile = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    auto* profile = builder_payload->createOptimizationProfile();
    if (profile == nullptr) { return report_null_vendor_object("createOptimizationProfile"); }
    JYPPX_TensorRtObjectBase* handle = nullptr;
    // TensorRT 11 documents optimization profiles as builder-owned objects.
    // The bridge handle is only a reference wrapper and must not delete the vendor payload.
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, profile, nullptr);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_profile = reinterpret_cast<JYPPX_TensorRtOptimizationProfile*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_optimization_profile_set_shape(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, const JYPPX_TensorRtDims* dims)
{
    auto status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (selector < 0 || selector > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile selector must be 0 (min), 1 (opt), or 2 (max).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_dims{};
    status = make_dims(dims, &native_dims, "dims");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    if (!profile_payload->setDimensions(input_name, static_cast<nvinfer1::OptProfileSelector>(selector), native_dims))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IOptimizationProfile::setDimensions returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape set");
#endif
}

JYPPX_StatusCode jyppx_trt11_optimization_profile_get_shape(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, JYPPX_TensorRtDims* out_dims)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dims, "out_dims");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (selector < 0 || selector > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile selector must be 0 (min), 1 (opt), or 2 (max).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    copy_dims(profile_payload->getDimensions(input_name, static_cast<nvinfer1::OptProfileSelector>(selector)), out_dims);
    return JYPPX_STATUS_OK;
#else
    std::memset(out_dims, 0, sizeof(*out_dims));
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape query");
#endif
}

JYPPX_StatusCode jyppx_trt11_optimization_profile_set_shape_values(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, const int32_t* values, int32_t value_count)
{
    auto status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (selector < 0 || selector > 2 || values == nullptr || value_count <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Shape value selector must be [0,2] and values must be non-empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    std::vector<int64_t> converted(static_cast<size_t>(value_count));
    for (int32_t i = 0; i < value_count; ++i) { converted[static_cast<size_t>(i)] = values[i]; }
    if (!profile_payload->setShapeValuesV2(input_name, static_cast<nvinfer1::OptProfileSelector>(selector), converted.data(), value_count))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IOptimizationProfile::setShapeValuesV2 returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape values set");
#endif
}

JYPPX_StatusCode jyppx_trt11_optimization_profile_get_shape_value_count(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    *out_count = profile_payload->getNbShapeValues(input_name);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape value count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_optimization_profile_get_shape_values(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, int32_t* output_values, int32_t output_count, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (selector < 0 || selector > 2 || output_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Shape value selector must be [0,2] and output count must be non-negative.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    const int32_t count = profile_payload->getNbShapeValues(input_name);
    *out_count = count;
    if (count <= 0) { return JYPPX_STATUS_OK; }
    if (output_values == nullptr || output_count < count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Shape values output buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }
    const int64_t* values = profile_payload->getShapeValuesV2(input_name, static_cast<nvinfer1::OptProfileSelector>(selector));
    if (values == nullptr) { return JYPPX_STATUS_OK; }
    for (int32_t i = 0; i < count; ++i) { output_values[i] = static_cast<int32_t>(values[i]); }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape values query");
#endif
}

JYPPX_StatusCode jyppx_trt11_optimization_profile_set_extra_memory_target(JYPPX_TensorRtOptimizationProfile* profile, float target)
{
    auto status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    if (!profile_payload->setExtraMemoryTarget(target))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IOptimizationProfile::setExtraMemoryTarget returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile extra memory target set");
#endif
}

JYPPX_StatusCode jyppx_trt11_optimization_profile_get_extra_memory_target(JYPPX_TensorRtOptimizationProfile* profile, float* out_target)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_target, "out_target");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_target = 0;
#if JYPPX_HAS_TENSORRT
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_target = profile_payload->getExtraMemoryTarget();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile extra memory target query");
#endif
}

JYPPX_StatusCode jyppx_trt11_optimization_profile_is_valid(JYPPX_TensorRtOptimizationProfile* profile, JYPPX_Boolean* out_is_valid)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_valid, "out_is_valid");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_valid = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_is_valid = profile_payload->isValid() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile validation query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_add_optimization_profile(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtOptimizationProfile* profile, int32_t* out_profile_index)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_profile_index, "out_profile_index");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_profile_index = -1;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (config_payload == nullptr || profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config or optimization profile handle does not carry the expected payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    const int32_t index = config_payload->addOptimizationProfile(profile_payload);
    if (index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IBuilderConfig::addOptimizationProfile returned a negative index.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    *out_profile_index = index;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config add optimization profile");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_profile_stream(JYPPX_TensorRtBuilderConfig* config, JYPPX_CudaStream* stream)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* stream_payload = reinterpret_cast<jyppx::cuda::StreamObject*>(stream);
    if (config_payload == nullptr || stream_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config or CUDA stream handle does not carry the expected payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    config_payload->setProfileStream(stream_payload->handle);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT builder config profile stream set");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config profile stream set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_is_profile_stream_set(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_is_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_set, "out_is_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_set = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_is_set = config_payload->getProfileStream() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config profile stream query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_optimization_profile_count(JYPPX_TensorRtBuilderConfig* config, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_count = config_payload->getNbOptimizationProfiles();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config optimization profile count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag, JYPPX_Boolean enabled)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (flag < 0) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder flag must be non-negative."); return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    const auto native_flag = static_cast<nvinfer1::BuilderFlag>(flag);
    if (enabled == JYPPX_FALSE) { config_payload->clearFlag(native_flag); } else { config_payload->setFlag(native_flag); }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config flag set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (flag < 0) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder flag must be non-negative."); return JYPPX_STATUS_INVALID_ARGUMENT; }
    *out_enabled = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_enabled = config_payload->getFlag(static_cast<nvinfer1::BuilderFlag>(flag)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config flag query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_engine_capability(JYPPX_TensorRtBuilderConfig* config, int32_t capability)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (capability < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setEngineCapability(static_cast<nvinfer1::EngineCapability>(capability));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config engine capability set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_engine_capability(JYPPX_TensorRtBuilderConfig* config, int32_t* out_capability)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_capability, "out_capability");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_capability = 0;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_capability = static_cast<int32_t>(config_payload->getEngineCapability());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config engine capability query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_preview_feature(JYPPX_TensorRtBuilderConfig* config, int32_t feature, JYPPX_Boolean enabled)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (feature < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setPreviewFeature(static_cast<nvinfer1::PreviewFeature>(feature), enabled != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config preview feature set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_preview_feature(JYPPX_TensorRtBuilderConfig* config, int32_t feature, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (feature < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    *out_enabled = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_enabled = config_payload->getPreviewFeature(static_cast<nvinfer1::PreviewFeature>(feature)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config preview feature query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_hardware_compatibility_level(JYPPX_TensorRtBuilderConfig* config, int32_t level)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (level < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setHardwareCompatibilityLevel(static_cast<nvinfer1::HardwareCompatibilityLevel>(level));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config hardware compatibility level set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_hardware_compatibility_level(JYPPX_TensorRtBuilderConfig* config, int32_t* out_level)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_level, "out_level");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_level = 0;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_level = static_cast<int32_t>(config_payload->getHardwareCompatibilityLevel());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config hardware compatibility level query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_runtime_platform(JYPPX_TensorRtBuilderConfig* config, int32_t platform)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (platform < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setRuntimePlatform(static_cast<nvinfer1::RuntimePlatform>(platform));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config runtime platform set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_runtime_platform(JYPPX_TensorRtBuilderConfig* config, int32_t* out_platform)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_platform, "out_platform");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_platform = 0;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_platform = static_cast<int32_t>(config_payload->getRuntimePlatform());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config runtime platform query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_memory_pool_limit(JYPPX_TensorRtBuilderConfig* config, int32_t pool, size_t pool_size)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (pool < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setMemoryPoolLimit(static_cast<nvinfer1::MemoryPoolType>(pool), pool_size);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config memory pool limit set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_memory_pool_limit(JYPPX_TensorRtBuilderConfig* config, int32_t pool, size_t* out_pool_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_pool_size, "out_pool_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (pool < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    *out_pool_size = 0;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_pool_size = config_payload->getMemoryPoolLimit(static_cast<nvinfer1::MemoryPoolType>(pool));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config memory pool limit query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_optimization_level(JYPPX_TensorRtBuilderConfig* config, int32_t level)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setBuilderOptimizationLevel(level);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config optimization level set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_optimization_level(JYPPX_TensorRtBuilderConfig* config, int32_t* out_level)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_level, "out_level");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_level = 0;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_level = config_payload->getBuilderOptimizationLevel();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config optimization level query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_profiling_verbosity(JYPPX_TensorRtBuilderConfig* config, int32_t verbosity)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (verbosity < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setProfilingVerbosity(static_cast<nvinfer1::ProfilingVerbosity>(verbosity));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config profiling verbosity set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_profiling_verbosity(JYPPX_TensorRtBuilderConfig* config, int32_t* out_verbosity)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_verbosity, "out_verbosity");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_verbosity = 0;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_verbosity = static_cast<int32_t>(config_payload->getProfilingVerbosity());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config profiling verbosity query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_max_aux_streams(JYPPX_TensorRtBuilderConfig* config, int32_t stream_count)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (stream_count < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    if (!config_payload->setMaxAuxStreams(stream_count))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IBuilderConfig::setMaxAuxStreams returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config max aux streams set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_max_aux_streams(JYPPX_TensorRtBuilderConfig* config, int32_t* out_stream_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_stream_count, "out_stream_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_stream_count = 0;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_stream_count = config_payload->getMaxAuxStreams();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config max aux streams query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_average_timing_iterations(JYPPX_TensorRtBuilderConfig* config, int32_t iterations)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (iterations <= 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setAvgTimingIterations(iterations);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config average timing iterations set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_average_timing_iterations(JYPPX_TensorRtBuilderConfig* config, int32_t* out_iterations)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_iterations, "out_iterations");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_iterations = 0;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_iterations = config_payload->getAvgTimingIterations();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config average timing iterations query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_tactic_sources(JYPPX_TensorRtBuilderConfig* config, uint32_t tactic_sources)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    if (!config_payload->setTacticSources(static_cast<nvinfer1::TacticSources>(tactic_sources)))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IBuilderConfig::setTacticSources returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config tactic sources set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_tactic_sources(JYPPX_TensorRtBuilderConfig* config, uint32_t* out_tactic_sources)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tactic_sources, "out_tactic_sources");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tactic_sources = 0;
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_tactic_sources = static_cast<uint32_t>(config_payload->getTacticSources());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config tactic sources query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_create_timing_cache(JYPPX_TensorRtBuilderConfig* config, const void* blob, size_t blob_size, JYPPX_TensorRtTimingCache** out_cache)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cache, "out_cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cache = nullptr;
    if (blob == nullptr && blob_size != 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    auto* cache = config_payload->createTimingCache(blob, blob_size);
    if (cache == nullptr) { return report_null_vendor_object("createTimingCache"); }
    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_TIMING_CACHE, cache, &destroy_payload<nvinfer1::ITimingCache>);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cache = reinterpret_cast<JYPPX_TensorRtTimingCache*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config timing cache creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_timing_cache(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtTimingCache* cache, JYPPX_Boolean ignore_mismatch)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(cache, kLine, JYPPX_TENSORRT_OBJECT_KIND_TIMING_CACHE, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* cache_payload = get_payload<nvinfer1::ITimingCache>(cache);
    if (config_payload == nullptr || cache_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    if (!config_payload->setTimingCache(*cache_payload, ignore_mismatch != JYPPX_FALSE))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IBuilderConfig::setTimingCache returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config timing cache set");
#endif
}

static JYPPX_StatusCode get_builder_config_and_layer_payload(
    JYPPX_TensorRtBuilderConfig* config,
    JYPPX_TensorRtLayer* layer,
    nvinfer1::IBuilderConfig** out_config,
    nvinfer1::ILayer** out_layer,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    *out_config = get_payload<nvinfer1::IBuilderConfig>(config);
    *out_layer = get_layer_payload(layer);
    if (*out_config == nullptr || *out_layer == nullptr)
    {
        std::ostringstream builder;
        builder << "Builder config or layer handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }
    return JYPPX_STATUS_OK;
#else
    (void)feature_name;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config layer payload query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_layer_device_type(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, int32_t device_type)
{
    if (device_type < 0 || device_type > 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT device type must be 0 (GPU) or 1 (DLA).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilderConfig* config_payload = nullptr;
    nvinfer1::ILayer* layer_payload = nullptr;
    auto status = get_builder_config_and_layer_payload(config, layer, &config_payload, &layer_payload, "layer device type set");
    if (status != JYPPX_STATUS_OK) { return status; }
    config_payload->setDeviceType(layer_payload, static_cast<nvinfer1::DeviceType>(device_type));
    return JYPPX_STATUS_OK;
#else
    (void)config;
    (void)layer;
    (void)device_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer device type set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_layer_device_type(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, int32_t* out_device_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_device_type, "out_device_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_device_type = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilderConfig* config_payload = nullptr;
    nvinfer1::ILayer* layer_payload = nullptr;
    status = get_builder_config_and_layer_payload(config, layer, &config_payload, &layer_payload, "layer device type query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_device_type = static_cast<int32_t>(config_payload->getDeviceType(layer_payload));
    return JYPPX_STATUS_OK;
#else
    (void)config;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer device type query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_is_layer_device_type_set(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_is_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_set, "out_is_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_set = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilderConfig* config_payload = nullptr;
    nvinfer1::ILayer* layer_payload = nullptr;
    status = get_builder_config_and_layer_payload(config, layer, &config_payload, &layer_payload, "layer device type set-state query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_set = config_payload->isDeviceTypeSet(layer_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)config;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer device type set-state query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_reset_layer_device_type(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilderConfig* config_payload = nullptr;
    nvinfer1::ILayer* layer_payload = nullptr;
    auto status = get_builder_config_and_layer_payload(config, layer, &config_payload, &layer_payload, "layer device type reset");
    if (status != JYPPX_STATUS_OK) { return status; }
    config_payload->resetDeviceType(layer_payload);
    return JYPPX_STATUS_OK;
#else
    (void)config;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer device type reset");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_calibration_profile(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtOptimizationProfile* profile)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK) { return status; }
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 11 removed explicit calibrator profile APIs; use explicit quantization flows instead.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt11_builder_config_has_calibration_profile(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_has_profile)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_profile, "out_has_profile");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_profile = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 11 removed explicit calibrator profile APIs; no calibration profile is tracked by this bridge.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

#if JYPPX_HAS_TENSORRT
constexpr int32_t kTimingCacheKeySize = 16;

JYPPX_StatusCode get_timing_cache_payload(
    JYPPX_TensorRtTimingCache* cache,
    nvinfer1::ITimingCache** out_cache,
    const char* parameter_name)
{
    auto status = jyppx::tensorrt::validate_handle(cache, kLine, JYPPX_TENSORRT_OBJECT_KIND_TIMING_CACHE, parameter_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cache = get_payload<nvinfer1::ITimingCache>(cache);
    if (*out_cache == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode make_timing_cache_key(const uint8_t* key_data, int32_t key_size, nvinfer1::TimingCacheKey* out_key)
{
    if (key_data == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT timing cache key data must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (key_size != kTimingCacheKeySize)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT timing cache keys must be exactly 16 bytes.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    std::memcpy(out_key->data, key_data, kTimingCacheKeySize);
    return JYPPX_STATUS_OK;
}
#endif

JYPPX_StatusCode jyppx_trt11_timing_cache_serialize(JYPPX_TensorRtTimingCache* cache, JYPPX_TensorRtHostMemory** out_host_memory)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_host_memory, "out_host_memory");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(cache, kLine, JYPPX_TENSORRT_OBJECT_KIND_TIMING_CACHE, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_host_memory = nullptr;
#if JYPPX_HAS_TENSORRT
    auto* cache_payload = get_payload<nvinfer1::ITimingCache>(cache);
    if (cache_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    auto* serialized = cache_payload->serialize();
    if (serialized == nullptr) { return report_null_vendor_object("ITimingCache::serialize"); }
    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY, serialized, &destroy_payload<nvinfer1::IHostMemory>);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_host_memory = reinterpret_cast<JYPPX_TensorRtHostMemory*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache serialization");
#endif
}

JYPPX_StatusCode jyppx_trt11_timing_cache_combine(JYPPX_TensorRtTimingCache* cache, JYPPX_TensorRtTimingCache* input_cache, JYPPX_Boolean ignore_mismatch, JYPPX_Boolean* out_combined)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_combined, "out_combined");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_combined = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITimingCache* input_cache_payload = nullptr;
    status = get_timing_cache_payload(input_cache, &input_cache_payload, "input_cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_combined = cache_payload->combine(*input_cache_payload, ignore_mismatch != JYPPX_FALSE) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)cache;
    (void)input_cache;
    (void)ignore_mismatch;
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache combine");
#endif
}

JYPPX_StatusCode jyppx_trt11_timing_cache_reset(JYPPX_TensorRtTimingCache* cache, JYPPX_Boolean* out_reset)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_reset, "out_reset");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_reset = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_reset = cache_payload->reset() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)cache;
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache reset");
#endif
}

JYPPX_StatusCode jyppx_trt11_timing_cache_query_key_count(JYPPX_TensorRtTimingCache* cache, int64_t* out_key_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_key_count, "out_key_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_key_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int64_t count = cache_payload->queryKeys(nullptr, 0);
    if (count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ITimingCache::queryKeys returned an error.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    *out_key_count = count;
    return JYPPX_STATUS_OK;
#else
    (void)cache;
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache key count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_timing_cache_copy_keys(JYPPX_TensorRtTimingCache* cache, uint8_t* output_keys, int64_t key_capacity, int64_t* out_key_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_key_count, "out_key_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_key_count = 0;
    if (key_capacity < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT timing cache key capacity must not be negative.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (key_capacity > 0 && output_keys == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT timing cache output key buffer must not be null when capacity is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::vector<nvinfer1::TimingCacheKey> keys(static_cast<size_t>(key_capacity));
    nvinfer1::TimingCacheKey* key_buffer = keys.empty() ? nullptr : keys.data();
    const int64_t count = cache_payload->queryKeys(key_buffer, key_capacity);
    if (count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ITimingCache::queryKeys returned an error.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    const int64_t copy_count = count < key_capacity ? count : key_capacity;
    if (copy_count > 0)
    {
        std::memcpy(output_keys, keys.data(), static_cast<size_t>(copy_count) * kTimingCacheKeySize);
    }

    *out_key_count = count;
    return JYPPX_STATUS_OK;
#else
    (void)cache;
    (void)output_keys;
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache key copy");
#endif
}

JYPPX_StatusCode jyppx_trt11_timing_cache_query(JYPPX_TensorRtTimingCache* cache, const uint8_t* key_data, int32_t key_size, uint64_t* out_tactic_hash, float* out_timing_msec, JYPPX_Boolean* out_found)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tactic_hash, "out_tactic_hash");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_timing_msec, "out_timing_msec");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_found, "out_found");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tactic_hash = 0;
    *out_timing_msec = 0.0F;
    *out_found = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::TimingCacheKey key{};
    status = make_timing_cache_key(key_data, key_size, &key);
    if (status != JYPPX_STATUS_OK) { return status; }
    const nvinfer1::TimingCacheValue value = cache_payload->query(key);
    *out_tactic_hash = value.tacticHash;
    *out_timing_msec = value.timingMSec;
    *out_found = value.tacticHash == nvinfer1::TimingCacheValue::kINVALID_TACTIC_HASH ? JYPPX_FALSE : JYPPX_TRUE;
    return JYPPX_STATUS_OK;
#else
    (void)cache;
    (void)key_data;
    (void)key_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache value query");
#endif
}

JYPPX_StatusCode jyppx_trt11_timing_cache_update(JYPPX_TensorRtTimingCache* cache, const uint8_t* key_data, int32_t key_size, uint64_t tactic_hash, float timing_msec, JYPPX_Boolean* out_updated)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_updated, "out_updated");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_updated = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::TimingCacheKey key{};
    status = make_timing_cache_key(key_data, key_size, &key);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::TimingCacheValue value{};
    value.tacticHash = tactic_hash;
    value.timingMSec = timing_msec;
    *out_updated = cache_payload->update(key, value) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)cache;
    (void)key_data;
    (void)key_size;
    (void)tactic_hash;
    (void)timing_msec;
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache value update");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_input(JYPPX_TensorRtNetworkDefinition* network, const char* name, int32_t data_type, const JYPPX_TensorRtDims* dims, JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (data_type < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    *out_tensor = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_dims{};
    status = make_dims(dims, &native_dims, "dims");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    return create_tensor_reference_handle(network_payload->addInput(name, static_cast<nvinfer1::DataType>(data_type), native_dims), out_tensor);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add input");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_mark_output(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor)
{
    auto status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (network_payload == nullptr || tensor_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    network_payload->markOutput(*tensor_payload);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network mark output");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_unmark_output(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor)
{
    auto status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (network_payload == nullptr || tensor_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    network_payload->unmarkOutput(*tensor_payload);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network unmark output");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_get_input_count(JYPPX_TensorRtNetworkDefinition* network, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_count = network_payload->getNbInputs();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network input count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_get_output_count(JYPPX_TensorRtNetworkDefinition* network, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_count = network_payload->getNbOutputs();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network output count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_get_layer_count(JYPPX_TensorRtNetworkDefinition* network, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_count = network_payload->getNbLayers();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network layer count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_get_layer(JYPPX_TensorRtNetworkDefinition* network, int32_t index, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    status = validate_index(index, network_payload->getNbLayers(), "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->getLayer(index), out_layer);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network layer query");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_get_name(JYPPX_TensorRtNetworkDefinition* network, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    return copy_string_to_buffer(network_payload->getName(), output_buffer, output_buffer_size, out_required_size);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network name query");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_set_name(JYPPX_TensorRtNetworkDefinition* network, const char* name)
{
    auto status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    network_payload->setName(name);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network name set");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_get_flags(JYPPX_TensorRtNetworkDefinition* network, uint32_t* out_flags)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_flags = 0;
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_flags = network_payload->getFlags();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network flags query");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_has_implicit_batch_dimension(JYPPX_TensorRtNetworkDefinition* network, JYPPX_Boolean* out_has_implicit_batch_dimension)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_implicit_batch_dimension, "out_has_implicit_batch_dimension");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_implicit_batch_dimension = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const bool has_implicit_batch_dimension = network_payload->hasImplicitBatchDimension();
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    *out_has_implicit_batch_dimension = has_implicit_batch_dimension ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network implicit batch dimension query");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_get_flag(JYPPX_TensorRtNetworkDefinition* network, int32_t flag, JYPPX_Boolean* out_enabled)
{
    if (flag < 0 || flag > 31) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_enabled = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_enabled = network_payload->getFlag(static_cast<nvinfer1::NetworkDefinitionCreationFlag>(flag)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network flag query");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_get_input(JYPPX_TensorRtNetworkDefinition* network, int32_t index, JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    status = validate_index(index, network_payload->getNbInputs(), "input");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_tensor_reference_handle(network_payload->getInput(index), out_tensor);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network input query");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_get_output(JYPPX_TensorRtNetworkDefinition* network, int32_t index, JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    status = validate_index(index, network_payload->getNbOutputs(), "output");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_tensor_reference_handle(network_payload->getOutput(index), out_tensor);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network output query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_optimization_profile_count(JYPPX_TensorRtCudaEngine* engine, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine optimization profile count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = engine_payload->getNbOptimizationProfiles();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine optimization profile count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_tensor_data_type(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine tensor data type query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = static_cast<int32_t>(engine_payload->getTensorDataType(tensor_name));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tensor data type query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_tensor_shape(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, JYPPX_TensorRtDims* out_shape)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_shape, "out_shape");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine tensor shape query");
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_dims(engine_payload->getTensorShape(tensor_name), out_shape);
    return JYPPX_STATUS_OK;
#else
    std::memset(out_shape, 0, sizeof(*out_shape));
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tensor shape query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_tensor_io_mode(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_io_mode)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_io_mode, "out_io_mode");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_io_mode = JYPPX_TENSORRT_IO_MODE_UNKNOWN;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine tensor I/O mode query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const nvinfer1::TensorIOMode mode = engine_payload->getTensorIOMode(tensor_name);
    *out_io_mode = mode == nvinfer1::TensorIOMode::kINPUT ? JYPPX_TENSORRT_IO_MODE_INPUT : (mode == nvinfer1::TensorIOMode::kOUTPUT ? JYPPX_TENSORRT_IO_MODE_OUTPUT : JYPPX_TENSORRT_IO_MODE_UNKNOWN);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tensor I/O mode query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_tensor_location(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_location)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_location, "out_location");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_location = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine tensor location query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_location = static_cast<int32_t>(engine_payload->getTensorLocation(tensor_name));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tensor location query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_tensor_format(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_format)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_format, "out_format");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_format = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine tensor format query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_format = static_cast<int32_t>(engine_payload->getTensorFormat(tensor_name));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tensor format query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_tensor_format_description(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine tensor format description query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(engine_payload->getTensorFormatDesc(tensor_name), output_buffer, output_buffer_size, out_required_size);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tensor format description query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_profile_shape(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t selector, JYPPX_TensorRtDims* out_shape)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_shape, "out_shape");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (profile_index < 0 || selector < 0 || selector > 2) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine profile shape query");
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_dims(engine_payload->getProfileShape(tensor_name, profile_index, static_cast<nvinfer1::OptProfileSelector>(selector)), out_shape);
    return JYPPX_STATUS_OK;
#else
    std::memset(out_shape, 0, sizeof(*out_shape));
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine profile shape query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_create_inspector(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtEngineInspector** out_inspector)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_inspector, "out_inspector");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_inspector = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine inspector creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* inspector = engine_payload->createEngineInspector();
    if (inspector == nullptr) { return report_null_vendor_object("createEngineInspector"); }
    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR, inspector, &destroy_payload<nvinfer1::IEngineInspector>);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_inspector = reinterpret_cast<JYPPX_TensorRtEngineInspector*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_inspector_set_execution_context(JYPPX_TensorRtEngineInspector* inspector, JYPPX_TensorRtExecutionContext* context)
{
    auto status = jyppx::tensorrt::validate_handle(inspector, kLine, JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR, "inspector");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* inspector_payload = get_payload<nvinfer1::IEngineInspector>(inspector);
    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    if (inspector_payload == nullptr || context_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    inspector_payload->setExecutionContext(context_payload);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector execution context set");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_inspector_get_engine_information(JYPPX_TensorRtEngineInspector* inspector, int32_t format, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(inspector, kLine, JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR, "inspector");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (format < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT
    auto* inspector_payload = get_payload<nvinfer1::IEngineInspector>(inspector);
    if (inspector_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    return copy_string_to_buffer(inspector_payload->getEngineInformation(static_cast<nvinfer1::LayerInformationFormat>(format)), output_buffer, output_buffer_size, out_required_size);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector information query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_input_shape(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, const JYPPX_TensorRtDims* dims)
{
    auto status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_dims{};
    status = make_dims(dims, &native_dims, "dims");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context input shape set");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (!context_payload->setInputShape(tensor_name, native_dims))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setInputShape returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context input shape set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_tensor_shape(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_TensorRtDims* out_shape)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_shape, "out_shape");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor shape query");
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_dims(context_payload->getTensorShape(tensor_name), out_shape);
    return JYPPX_STATUS_OK;
#else
    std::memset(out_shape, 0, sizeof(*out_shape));
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context tensor shape query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_tensor_strides(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_TensorRtDims* out_strides)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_strides, "out_strides");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor strides query");
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_dims(context_payload->getTensorStrides(tensor_name), out_strides);
    return JYPPX_STATUS_OK;
#else
    std::memset(out_strides, 0, sizeof(*out_strides));
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context tensor strides query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_all_input_dimensions_specified(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_specified)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_specified, "out_specified");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_specified = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context input dimensions specified query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_specified = context_payload->allInputDimensionsSpecified() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context input dimensions specified query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_all_input_shapes_specified(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_specified)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_specified, "out_specified");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_specified = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 11 removed allInputShapesSpecified; use inferShapes and named tensor shape APIs.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_tensor_address(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_CudaMemory* memory)
{
    auto status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor address set");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* memory_payload = reinterpret_cast<jyppx::cuda::MemoryObject*>(memory);
    if (memory_payload == nullptr || memory_payload->pointer == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    if (!context_payload->setTensorAddress(tensor_name, memory_payload->pointer))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setTensorAddress returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context tensor address set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_input_tensor_address(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_CudaMemory* memory)
{
    auto status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context input tensor address set");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* memory_payload = reinterpret_cast<jyppx::cuda::MemoryObject*>(memory);
    if (memory_payload == nullptr || memory_payload->pointer == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    if (!context_payload->setInputTensorAddress(tensor_name, memory_payload->pointer))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setInputTensorAddress returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context input tensor address set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_output_tensor_address(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_CudaMemory* memory)
{
    auto status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context output tensor address set");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* memory_payload = reinterpret_cast<jyppx::cuda::MemoryObject*>(memory);
    if (memory_payload == nullptr || memory_payload->pointer == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    if (!context_payload->setOutputTensorAddress(tensor_name, memory_payload->pointer))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setOutputTensorAddress returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context output tensor address set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_is_tensor_address_set(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean* out_bound)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_bound, "out_bound");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_bound = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor address query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_bound = context_payload->getTensorAddress(tensor_name) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context tensor address query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_is_tensor_address_bound(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean* out_bound)
{
    return jyppx_trt11_execution_context_is_tensor_address_set(context, tensor_name, out_bound);
}

JYPPX_StatusCode jyppx_trt11_execution_context_infer_shapes(JYPPX_TensorRtExecutionContext* context, int32_t* out_missing_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_missing_count, "out_missing_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_missing_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context shape inference");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int32_t missing = context_payload->inferShapes(0, nullptr);
    if (missing < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::inferShapes failed.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    *out_missing_count = missing;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context shape inference");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_optimization_profile(JYPPX_TensorRtExecutionContext* context, int32_t* out_profile_index)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_profile_index, "out_profile_index");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_profile_index = -1;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context optimization profile query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_profile_index = context_payload->getOptimizationProfile();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context optimization profile query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_optimization_profile_async(JYPPX_TensorRtExecutionContext* context, int32_t profile_index, JYPPX_CudaStream* stream)
{
    if (profile_index < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context async profile selection");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* stream_payload = reinterpret_cast<jyppx::cuda::StreamObject*>(stream);
    if (stream_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    if (!context_payload->setOptimizationProfileAsync(profile_index, stream_payload->handle))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setOptimizationProfileAsync returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT execution context async profile selection");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context async profile selection");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_max_output_size(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, int64_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = -1;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context maximum output size query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = context_payload->getMaxOutputSize(tensor_name);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context maximum output size query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_device_memory(JYPPX_TensorRtExecutionContext* context, JYPPX_CudaMemory* memory)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    auto* memory_payload = reinterpret_cast<jyppx::cuda::MemoryObject*>(memory);
    if (context_payload == nullptr || memory_payload == nullptr || memory_payload->pointer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context or CUDA memory handle does not carry the expected payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    context_payload->setDeviceMemoryV2(memory_payload->pointer, static_cast<int64_t>(memory_payload->size));
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT execution context device memory set");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context device memory set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_device_memory_size(JYPPX_TensorRtExecutionContext* context, size_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = 0;
    status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(
        JYPPX_ERROR_CATEGORY_TENSORRT,
        "TensorRT 11 IExecutionContext does not expose a context-scoped device memory size query. "
        "Use engine device memory size or updateDeviceMemorySizeForShapes after shapes are specified.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context device memory size query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_update_device_memory_size_for_shapes(JYPPX_TensorRtExecutionContext* context, size_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context shape device memory size update");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = context_payload->updateDeviceMemorySizeForShapes();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context shape device memory size update");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_persistent_cache_limit(JYPPX_TensorRtExecutionContext* context, size_t cache_size)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context persistent cache limit set");
    if (status != JYPPX_STATUS_OK) { return status; }
    context_payload->setPersistentCacheLimit(cache_size);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context persistent cache limit set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_persistent_cache_limit(JYPPX_TensorRtExecutionContext* context, size_t* out_cache_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cache_size, "out_cache_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cache_size = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context persistent cache limit query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cache_size = context_payload->getPersistentCacheLimit();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context persistent cache limit query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_input_consumed_event(JYPPX_TensorRtExecutionContext* context, JYPPX_CudaEvent* event)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_event(event, "event");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    auto* event_payload = reinterpret_cast<jyppx::cuda::EventObject*>(event);
    if (context_payload == nullptr || event_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context or CUDA event handle does not carry the expected payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (!context_payload->setInputConsumedEvent(event_payload->handle))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setInputConsumedEvent returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT execution context input consumed event set");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context input consumed event set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_enqueue_async(JYPPX_TensorRtExecutionContext* context, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context enqueue");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* stream_payload = reinterpret_cast<jyppx::cuda::StreamObject*>(stream);
    if (stream_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    if (!context_payload->enqueueV3(stream_payload->handle))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::enqueueV3 returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT execution context enqueue");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context enqueue");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_debug_sync(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean debug_sync)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context debug sync set");
    if (status != JYPPX_STATUS_OK) { return status; }
    context_payload->setDebugSync(debug_sync != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#else
    (void)context;
    (void)debug_sync;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context debug sync set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_debug_sync(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_debug_sync)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_debug_sync, "out_debug_sync");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_debug_sync = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context debug sync query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_debug_sync = context_payload->getDebugSync() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context debug sync query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_name(JYPPX_TensorRtExecutionContext* context, const char* name)
{
    auto status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context name set");
    if (status != JYPPX_STATUS_OK) { return status; }
    context_payload->setName(name);
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context name set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_name(JYPPX_TensorRtExecutionContext* context, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context name query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(context_payload->getName(), output_buffer, output_buffer_size, out_required_size);
#else
    (void)context;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context name query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_enqueue_emits_profile(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_enqueue_emits_profile)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enqueue_emits_profile, "out_enqueue_emits_profile");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_enqueue_emits_profile = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context enqueue emits profile query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_enqueue_emits_profile = context_payload->getEnqueueEmitsProfile() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context enqueue emits profile query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_enqueue_emits_profile(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean enqueue_emits_profile)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context enqueue emits profile set");
    if (status != JYPPX_STATUS_OK) { return status; }
    context_payload->setEnqueueEmitsProfile(enqueue_emits_profile != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#else
    (void)context;
    (void)enqueue_emits_profile;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context enqueue emits profile set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_profiler(JYPPX_TensorRtExecutionContext* context, JYPPX_TensorRtProfiler* profiler)
{
#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 11)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context profiler attach");
    }

    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context profiler attach");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(profiler, kLine, JYPPX_TENSORRT_OBJECT_KIND_PROFILER, "profiler");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* profiler_payload = get_payload<ManagedProfiler>(profiler);
    if (profiler_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profiler handle does not carry a TensorRT profiler payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        context_payload->setProfiler(profiler_payload);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "execution context profiler attach", exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "execution context profiler attach", "unknown native exception");
    }
#else
    (void)context;
    (void)profiler;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context profiler attach");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_report_to_profiler(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_reported)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_reported, "out_reported");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_reported = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context profiler report");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_reported = context_payload->reportToProfiler() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context profiler report");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_tensor_debug_state(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean debug_state)
{
    auto status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor debug state set");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (!context_payload->setTensorDebugState(tensor_name, debug_state != JYPPX_FALSE))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setTensorDebugState returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    (void)context;
    (void)debug_state;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context tensor debug state set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_tensor_debug_state(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean* out_debug_state)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_debug_state, "out_debug_state");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_debug_state = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor debug state query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_debug_state = context_payload->getDebugState(tensor_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context tensor debug state query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_all_tensors_debug_state(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean debug_state)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context all tensors debug state set");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (!context_payload->setAllTensorsDebugState(debug_state != JYPPX_FALSE))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setAllTensorsDebugState returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    (void)context;
    (void)debug_state;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context all tensors debug state set");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_create(JYPPX_TensorRtLogger* logger, JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtOnnxParser** out_parser)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_parser, "out_parser");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(logger, kLine, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, "logger");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_parser = nullptr;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* logger_payload = get_payload<ManagedLogger>(logger);
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (logger_payload == nullptr || network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Logger or network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    auto* parser = nvonnxparser::createParser(*network_payload, *logger_payload);
    if (parser == nullptr) { return report_null_vendor_object("nvonnxparser::createParser"); }
    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, parser, &destroy_parser_payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_parser = reinterpret_cast<JYPPX_TensorRtOnnxParser*>(handle);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_parse_from_file(JYPPX_TensorRtOnnxParser* parser, const char* file_path, int32_t verbosity, JYPPX_Boolean* out_parsed)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_parsed, "out_parsed");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_c_string(file_path, "file_path");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_parsed = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    mark_onnx_parser_support_ready(parser_payload, false);
    *out_parsed = parser_payload->parseFromFile(file_path, verbosity) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser parse from file");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_parse_from_memory(JYPPX_TensorRtOnnxParser* parser, const void* model_data, size_t model_size, const char* model_path, JYPPX_Boolean* out_parsed)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_parsed, "out_parsed");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (model_data == nullptr || model_size == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX model data must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    *out_parsed = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    mark_onnx_parser_support_ready(parser_payload, false);
    *out_parsed = parser_payload->parse(model_data, model_size, model_path) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser parse from memory");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_load_model_proto(JYPPX_TensorRtOnnxParser* parser, const void* model_data, size_t model_size, const char* model_path, JYPPX_Boolean* out_loaded)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_loaded, "out_loaded");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (model_data == nullptr || model_size == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX model proto data must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    *out_loaded = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        mark_onnx_parser_support_ready(parser_payload, false);
        *out_loaded = parser_payload->loadModelProto(model_data, model_size, model_path) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        std::ostringstream builder;
        builder << "ONNX parser loadModelProto caught a native exception: " << exception.what();
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    catch (...)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser loadModelProto caught an unknown native exception.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
#elif JYPPX_HAS_TENSORRT
    (void)model_path;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)model_path;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser model proto load");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_load_initializer(JYPPX_TensorRtOnnxParser* parser, const char* name, const void* data, size_t data_size, JYPPX_Boolean* out_loaded)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_loaded, "out_loaded");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (data == nullptr || data_size == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX initializer data must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    *out_loaded = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        mark_onnx_parser_support_ready(parser_payload, false);
        *out_loaded = parser_payload->loadInitializer(name, data, data_size) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        std::ostringstream builder;
        builder << "ONNX parser loadInitializer caught a native exception: " << exception.what();
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    catch (...)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser loadInitializer caught an unknown native exception.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser initializer load");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_parse_model_proto(JYPPX_TensorRtOnnxParser* parser, JYPPX_Boolean* out_parsed)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_parsed, "out_parsed");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_parsed = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        mark_onnx_parser_support_ready(parser_payload, false);
        *out_parsed = parser_payload->parseModelProto() ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        std::ostringstream builder;
        builder << "ONNX parser parseModelProto caught a native exception: " << exception.what();
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    catch (...)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser parseModelProto caught an unknown native exception.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser model proto parse");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_get_error_count(JYPPX_TensorRtOnnxParser* parser, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_count = parser_payload->getNbErrors();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser error count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_clear_errors(JYPPX_TensorRtOnnxParser* parser)
{
    auto status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    parser_payload->clearErrors();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser error clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_get_flags(JYPPX_TensorRtOnnxParser* parser, uint32_t* out_flags)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_flags = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_flags = static_cast<uint32_t>(parser_payload->getFlags());
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser flags query");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_set_flags(JYPPX_TensorRtOnnxParser* parser, uint32_t flags)
{
    auto status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    parser_payload->setFlags(static_cast<nvonnxparser::OnnxParserFlags>(flags));
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser flags set");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_supports_operator(JYPPX_TensorRtOnnxParser* parser, const char* operator_name, JYPPX_Boolean* out_supported)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_supported, "out_supported");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_c_string(operator_name, "operator_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_supported = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_supported = parser_payload->supportsOperator(operator_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser operator support query");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_get_flag(JYPPX_TensorRtOnnxParser* parser, int32_t flag, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    *out_enabled = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_enabled = parser_payload->getFlag(static_cast<nvonnxparser::OnnxParserFlag>(flag)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser flag query");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_set_flag(JYPPX_TensorRtOnnxParser* parser, int32_t flag)
{
    auto status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    parser_payload->setFlag(static_cast<nvonnxparser::OnnxParserFlag>(flag));
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser flag set");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_clear_flag(JYPPX_TensorRtOnnxParser* parser, int32_t flag)
{
    auto status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    parser_payload->clearFlag(static_cast<nvonnxparser::OnnxParserFlag>(flag));
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser flag clear");
#endif
}

#define JYPPX_TRT_EXPECTED_MAJOR 11
#define JYPPX_TRT_ONNX_PARSER_API(name) jyppx_trt11_onnx_parser_##name
#include "../common/onnx_parser_support.inc"

#define JYPPX_TRT_EXPECTED_MAJOR 11
#define JYPPX_TRT_PARSER_REFITTER_API(name) jyppx_trt11_parser_refitter_##name
#include "../common/parser_refitter_diagnostics.inc"

JYPPX_StatusCode jyppx_trt11_builder_get_dla_core_count(JYPPX_TensorRtBuilder* builder, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry the expected TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = builder_payload->getNbDLACores();
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder DLA core count query");
#endif
}

static JYPPX_StatusCode trt11_report_removed_builder_capability(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_supported, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_supported, "out_supported");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_supported = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::ostringstream message;
    message << "TensorRT 11 removed " << feature_name << "; query concrete build flags and device capabilities instead.";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, message.str().c_str());
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt11_builder_platform_has_fast_fp16(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_supported)
{
    return trt11_report_removed_builder_capability(builder, out_supported, "IBuilder::platformHasFastFp16");
}

JYPPX_StatusCode jyppx_trt11_builder_platform_has_fast_int8(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_supported)
{
    return trt11_report_removed_builder_capability(builder, out_supported, "IBuilder::platformHasFastInt8");
}

JYPPX_StatusCode jyppx_trt11_builder_platform_has_tf32(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_supported)
{
    return trt11_report_removed_builder_capability(builder, out_supported, "IBuilder::platformHasTf32");
}

JYPPX_StatusCode jyppx_trt11_engine_create_execution_context_without_device_memory(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtExecutionContext** out_context)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_context, "out_context");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_context = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine execution context creation without device memory");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* context = engine_payload->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
    if (context == nullptr) { return report_null_vendor_object("createExecutionContext(kUSER_MANAGED)"); }
    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, context, &destroy_payload<nvinfer1::IExecutionContext>);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_context = reinterpret_cast<JYPPX_TensorRtExecutionContext*>(handle);
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine execution context creation without device memory");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_device_memory_size_v2(JYPPX_TensorRtCudaEngine* engine, size_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine device memory size v2 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int64_t size = engine_payload->getDeviceMemorySizeV2();
    *out_size = size > 0 ? static_cast<size_t>(size) : 0;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine device memory size v2 query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_device_memory_size(JYPPX_TensorRtCudaEngine* engine, size_t* out_size)
{
    return jyppx_trt11_engine_get_device_memory_size_v2(engine, out_size);
}

JYPPX_StatusCode jyppx_trt11_engine_get_device_memory_size_for_profile_v2(JYPPX_TensorRtCudaEngine* engine, int32_t profile_index, size_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (profile_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_size = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine profile device memory size v2 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int64_t size = engine_payload->getDeviceMemorySizeForProfileV2(profile_index);
    *out_size = size > 0 ? static_cast<size_t>(size) : 0;
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine profile device memory size v2 query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_device_memory_size_for_profile(JYPPX_TensorRtCudaEngine* engine, int32_t profile_index, size_t* out_size)
{
    return jyppx_trt11_engine_get_device_memory_size_for_profile_v2(engine, profile_index, out_size);
}

JYPPX_StatusCode jyppx_trt11_engine_get_layer_count(JYPPX_TensorRtCudaEngine* engine, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine layer count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = engine_payload->getNbLayers();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine layer count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_is_refittable(JYPPX_TensorRtCudaEngine* engine, JYPPX_Boolean* out_refittable)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_refittable, "out_refittable");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_refittable = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine refittable query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_refittable = engine_payload->isRefittable() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine refittable query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_name(JYPPX_TensorRtCudaEngine* engine, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine name query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(engine_payload->getName(), output_buffer, output_buffer_size, out_required_size);
#else
    (void)engine;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine name query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_io_tensor_name(
    JYPPX_TensorRtCudaEngine* engine,
    int32_t index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine I/O tensor name query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int32_t count = engine_payload->getNbIOTensors();
    if (index >= count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return copy_string_to_buffer(engine_payload->getIOTensorName(index), output_buffer, output_buffer_size, out_required_size);
#else
    (void)engine;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine I/O tensor name query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_tensor_index(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_index)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_index, "out_index");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_index = -1;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine tensor index query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int32_t count = engine_payload->getNbIOTensors();
    for (int32_t index = 0; index < count; ++index)
    {
        const char* const name = engine_payload->getIOTensorName(index);
        if (name != nullptr && std::strcmp(name, tensor_name) == 0)
        {
            *out_index = index;
            return JYPPX_STATUS_OK;
        }
    }

    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor name was not found in the engine I/O tensor list.");
    return JYPPX_STATUS_INVALID_ARGUMENT;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tensor index query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_max_batch_size(JYPPX_TensorRtCudaEngine* engine, int32_t* out_max_batch_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_max_batch_size, "out_max_batch_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_max_batch_size = 0;
    status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK) { return status; }
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 11 removed ICudaEngine::getMaxBatchSize; explicit-batch engines should use named I/O tensor shapes.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt11_engine_is_debug_tensor(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, JYPPX_Boolean* out_is_debug_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_debug_tensor, "out_is_debug_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_debug_tensor = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine debug tensor query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_debug_tensor = engine_payload->isDebugTensor(tensor_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine debug tensor query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_engine_capability(JYPPX_TensorRtCudaEngine* engine, int32_t* out_capability)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_capability, "out_capability");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_capability = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine capability query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_capability = static_cast<int32_t>(engine_payload->getEngineCapability());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine capability query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_tactic_sources(JYPPX_TensorRtCudaEngine* engine, uint32_t* out_tactic_sources)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tactic_sources, "out_tactic_sources");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tactic_sources = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine tactic sources query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tactic_sources = static_cast<uint32_t>(engine_payload->getTacticSources());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tactic sources query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_profiling_verbosity(JYPPX_TensorRtCudaEngine* engine, int32_t* out_verbosity)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_verbosity, "out_verbosity");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_verbosity = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine profiling verbosity query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_verbosity = static_cast<int32_t>(engine_payload->getProfilingVerbosity());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine profiling verbosity query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_nb_aux_streams(JYPPX_TensorRtCudaEngine* engine, int32_t* out_stream_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_stream_count, "out_stream_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_stream_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine auxiliary stream count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_stream_count = engine_payload->getNbAuxStreams();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine auxiliary stream count query");
#endif
}

#define JYPPX_TRT11_ENGINE_TENSOR_INT_QUERY(FUNCTION_NAME, OUTPUT_NAME, FEATURE_NAME, EXPR) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        status = validate_named_tensor(tensor_name); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUTPUT_NAME = 0; \
        nvinfer1::ICudaEngine* engine_payload = nullptr; \
        status = get_engine_payload_ext(engine, &engine_payload, FEATURE_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUTPUT_NAME = static_cast<int32_t>(EXPR); \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_TRT11_ENGINE_TENSOR_INT_QUERY(jyppx_trt11_engine_get_tensor_bytes_per_component, out_bytes, "engine tensor bytes-per-component query", engine_payload->getTensorBytesPerComponent(tensor_name))
JYPPX_TRT11_ENGINE_TENSOR_INT_QUERY(jyppx_trt11_engine_get_tensor_components_per_element, out_components, "engine tensor components-per-element query", engine_payload->getTensorComponentsPerElement(tensor_name))
JYPPX_TRT11_ENGINE_TENSOR_INT_QUERY(jyppx_trt11_engine_get_tensor_vectorized_dim, out_dim, "engine tensor vectorized dimension query", engine_payload->getTensorVectorizedDim(tensor_name))
JYPPX_TRT11_ENGINE_TENSOR_INT_QUERY(jyppx_trt11_engine_is_shape_inference_io, out_is_shape_inference_io, "engine shape-inference I/O query", engine_payload->isShapeInferenceIO(tensor_name) ? 1 : 0)
#endif
#undef JYPPX_TRT11_ENGINE_TENSOR_INT_QUERY

JYPPX_StatusCode jyppx_trt11_engine_get_tensor_format_desc(
    JYPPX_TensorRtCudaEngine* engine,
    const char* tensor_name,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    return jyppx_trt11_engine_get_tensor_format_description(engine, tensor_name, output_buffer, output_buffer_size, out_required_size);
}

#define JYPPX_TRT11_ENGINE_TENSOR_PROFILE_INT_QUERY(FUNCTION_NAME, OUTPUT_NAME, FEATURE_NAME, EXPR) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        if (profile_index < 0) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile index must be greater than or equal to zero."); return JYPPX_STATUS_INVALID_ARGUMENT; } \
        status = validate_named_tensor(tensor_name); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUTPUT_NAME = 0; \
        nvinfer1::ICudaEngine* engine_payload = nullptr; \
        status = get_engine_payload_ext(engine, &engine_payload, FEATURE_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUTPUT_NAME = static_cast<int32_t>(EXPR); \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_TRT11_ENGINE_TENSOR_PROFILE_INT_QUERY(jyppx_trt11_engine_get_tensor_bytes_per_component_for_profile, out_bytes, "engine tensor profile bytes-per-component query", engine_payload->getTensorBytesPerComponent(tensor_name, profile_index))
JYPPX_TRT11_ENGINE_TENSOR_PROFILE_INT_QUERY(jyppx_trt11_engine_get_tensor_components_per_element_for_profile, out_components, "engine tensor profile components-per-element query", engine_payload->getTensorComponentsPerElement(tensor_name, profile_index))
JYPPX_TRT11_ENGINE_TENSOR_PROFILE_INT_QUERY(jyppx_trt11_engine_get_tensor_format_for_profile, out_format, "engine tensor profile format query", engine_payload->getTensorFormat(tensor_name, profile_index))
JYPPX_TRT11_ENGINE_TENSOR_PROFILE_INT_QUERY(jyppx_trt11_engine_get_tensor_vectorized_dim_for_profile, out_dim, "engine tensor profile vectorized dimension query", engine_payload->getTensorVectorizedDim(tensor_name, profile_index))
#endif
#undef JYPPX_TRT11_ENGINE_TENSOR_PROFILE_INT_QUERY

JYPPX_StatusCode jyppx_trt11_engine_get_tensor_format_desc_for_profile(
    JYPPX_TensorRtCudaEngine* engine,
    const char* tensor_name,
    int32_t profile_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (profile_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine tensor profile format description query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(engine_payload->getTensorFormatDesc(tensor_name, profile_index), output_buffer, output_buffer_size, out_required_size);
#else
    (void)engine;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tensor profile format description query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_create_refitter(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtLogger* logger, JYPPX_TensorRtRefitter** out_refitter)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_refitter, "out_refitter");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_refitter = nullptr;
    status = jyppx::tensorrt::validate_handle(logger, kLine, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, "logger");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine refitter creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* logger_payload = get_payload<ManagedLogger>(logger);
    if (logger_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Logger handle does not carry the expected TensorRT logger payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    auto* refitter = nvinfer1::createInferRefitter(*engine_payload, *logger_payload);
    if (refitter == nullptr) { return report_null_vendor_object("createInferRefitter"); }
    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_REFITTER, refitter, &destroy_payload<nvinfer1::IRefitter>);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_refitter = reinterpret_cast<JYPPX_TensorRtRefitter*>(handle);
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    (void)logger;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine refitter creation");
#endif
}

#if JYPPX_HAS_TENSORRT
static JYPPX_StatusCode get_refitter_payload_ext(JYPPX_TensorRtRefitter* refitter, nvinfer1::IRefitter** out_refitter, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(refitter, kLine, JYPPX_TENSORRT_OBJECT_KIND_REFITTER, "refitter");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_refitter = get_payload<nvinfer1::IRefitter>(refitter);
    if (*out_refitter == nullptr)
    {
        std::ostringstream builder;
        builder << "Refitter handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }
    return JYPPX_STATUS_OK;
}
#endif

JYPPX_StatusCode jyppx_trt11_refitter_get_missing_count(JYPPX_TensorRtRefitter* refitter, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter missing weight count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = payload->getMissing(0, nullptr, nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter missing weight count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_get_all_count(JYPPX_TensorRtRefitter* refitter, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter all weight count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = payload->getAll(0, nullptr, nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter all weight count query");
#endif
}

static JYPPX_StatusCode copy_refitter_entries(
    JYPPX_TensorRtRefitter* refitter,
    const bool missing_only,
    JYPPX_TensorRtRefitEntryInfo* output_entries,
    const int32_t output_count,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
    if (output_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter output count must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (output_count > 0 && output_entries == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter output entries buffer must not be null when output_count is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, missing_only ? "refitter missing entries query" : "refitter all entries query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int32_t required_count = missing_only ? payload->getMissing(0, nullptr, nullptr) : payload->getAll(0, nullptr, nullptr);
    *out_count = required_count;
    if (output_entries == nullptr || output_count == 0 || required_count == 0)
    {
        return JYPPX_STATUS_OK;
    }

    const int32_t copy_count = output_count < required_count ? output_count : required_count;
    std::vector<char const*> layer_names(static_cast<size_t>(copy_count));
    std::vector<nvinfer1::WeightsRole> roles(static_cast<size_t>(copy_count));
    if (missing_only)
    {
        payload->getMissing(copy_count, layer_names.data(), roles.data());
    }
    else
    {
        payload->getAll(copy_count, layer_names.data(), roles.data());
    }

    for (int32_t i = 0; i < copy_count; ++i)
    {
        copy_c_string(layer_names[static_cast<size_t>(i)], output_entries[i].layer_name, sizeof(output_entries[i].layer_name));
        output_entries[i].role = static_cast<int32_t>(roles[static_cast<size_t>(i)]);
    }

    if (copy_count < required_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter output entries buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    (void)missing_only;
    (void)output_entries;
    (void)output_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter entries query");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_get_missing_entries(JYPPX_TensorRtRefitter* refitter, JYPPX_TensorRtRefitEntryInfo* output_entries, int32_t output_count, int32_t* out_count)
{
    return copy_refitter_entries(refitter, true, output_entries, output_count, out_count);
}

JYPPX_StatusCode jyppx_trt11_refitter_get_all_entries(JYPPX_TensorRtRefitter* refitter, JYPPX_TensorRtRefitEntryInfo* output_entries, int32_t output_count, int32_t* out_count)
{
    return copy_refitter_entries(refitter, false, output_entries, output_count, out_count);
}

JYPPX_StatusCode jyppx_trt11_refitter_set_weights(JYPPX_TensorRtRefitter* refitter, const char* layer_name, int32_t role, int32_t data_type, const void* values, int64_t value_count, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    status = validate_c_string(layer_name, "layer_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (role < 0 || role >= 6)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT refit weights role must be between 0 and 5.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (values == nullptr || value_count <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT refit weights values must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (get_data_type_size(data_type) == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT refit weights data type is not supported by the active TensorRT line.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter weights set");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::Weights weights{static_cast<nvinfer1::DataType>(data_type), values, value_count};
    *out_set = payload->setWeights(layer_name, static_cast<nvinfer1::WeightsRole>(role), weights) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter weights set");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_refit_cuda_engine(JYPPX_TensorRtRefitter* refitter, JYPPX_Boolean* out_refitted)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_refitted, "out_refitted");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_refitted = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter refit CUDA engine");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_refitted = payload->refitCudaEngine() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter refit CUDA engine");
#endif
}

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
static JYPPX_StatusCode get_parser_error_for_diagnostic(JYPPX_TensorRtOnnxParser* parser, const int32_t index, const nvonnxparser::IParserError** out_error)
{
    *out_error = nullptr;
    auto status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser error index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    const int32_t count = parser_payload->getNbErrors();
    if (index >= count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser error index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    const nvonnxparser::IParserError* parser_error = parser_payload->getError(index);
    if (parser_error == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT returned a null parser error object.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    *out_error = parser_error;
    return JYPPX_STATUS_OK;
}

static const char* get_parser_error_string_field(const nvonnxparser::IParserError* parser_error, const int32_t field)
{
    switch (field)
    {
    case 0: return parser_error->desc();
    case 1: return parser_error->file();
    case 2: return parser_error->func();
    case 3: return parser_error->nodeName();
    case 4: return parser_error->nodeOperator();
    default: return "";
    }
}
#endif

static JYPPX_StatusCode get_parser_diagnostic_int(JYPPX_TensorRtOnnxParser* parser, const int32_t index, const int32_t field, int32_t* out_value)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_value, "out_value");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_value = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    const nvonnxparser::IParserError* parser_error = nullptr;
    status = get_parser_error_for_diagnostic(parser, index, &parser_error);
    if (status != JYPPX_STATUS_OK) { return status; }
    switch (field)
    {
    case 0: *out_value = static_cast<int32_t>(parser_error->code()); break;
    case 1: *out_value = parser_error->line(); break;
    case 2: *out_value = parser_error->node(); break;
    default:
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unknown parser diagnostic integer field.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    (void)index;
    (void)field;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)index;
    (void)field;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser diagnostic integer query");
#endif
}

static JYPPX_StatusCode get_parser_diagnostic_string(JYPPX_TensorRtOnnxParser* parser, const int32_t index, const int32_t field, char* output_buffer, const size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    const nvonnxparser::IParserError* parser_error = nullptr;
    status = get_parser_error_for_diagnostic(parser, index, &parser_error);
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(get_parser_error_string_field(parser_error, field), output_buffer, output_buffer_size, out_required_size);
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    (void)index;
    (void)field;
    (void)output_buffer;
    (void)output_buffer_size;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)index;
    (void)field;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser diagnostic string query");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_get_error(JYPPX_TensorRtOnnxParser* parser, int32_t index, JYPPX_TensorRtParserErrorInfo* out_error)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_error, "out_error");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_error, 0, sizeof(*out_error));
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    const nvonnxparser::IParserError* parser_error = nullptr;
    status = get_parser_error_for_diagnostic(parser, index, &parser_error);
    if (status != JYPPX_STATUS_OK) { return status; }
    out_error->index = index;
    out_error->code = static_cast<int32_t>(parser_error->code());
    out_error->line = parser_error->line();
    out_error->node = parser_error->node();
    copy_c_string(parser_error->desc(), out_error->description, sizeof(out_error->description));
    copy_c_string(parser_error->file(), out_error->file, sizeof(out_error->file));
    copy_c_string(parser_error->func(), out_error->function_name, sizeof(out_error->function_name));
    copy_c_string(parser_error->nodeName(), out_error->node_name, sizeof(out_error->node_name));
    copy_c_string(parser_error->nodeOperator(), out_error->node_operator, sizeof(out_error->node_operator));
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    (void)index;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser error query");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_error_get_code(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_code)
{
    return get_parser_diagnostic_int(parser, index, 0, out_code);
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_error_get_line(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_line)
{
    return get_parser_diagnostic_int(parser, index, 1, out_line);
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_error_get_node(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_node)
{
    return get_parser_diagnostic_int(parser, index, 2, out_node);
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_error_get_description(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return get_parser_diagnostic_string(parser, index, 0, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_error_get_file(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return get_parser_diagnostic_string(parser, index, 1, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_error_get_function(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return get_parser_diagnostic_string(parser, index, 2, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_error_get_node_name(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return get_parser_diagnostic_string(parser, index, 3, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_error_get_node_operator(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return get_parser_diagnostic_string(parser, index, 4, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_error_get_local_function_stack_size(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    const nvonnxparser::IParserError* parser_error = nullptr;
    status = get_parser_error_for_diagnostic(parser, index, &parser_error);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = parser_error->localFunctionStackSize();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    (void)index;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser local function stack size query");
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_error_get_local_function_stack_entry(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t stack_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    if (stack_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser local function stack index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    const nvonnxparser::IParserError* parser_error = nullptr;
    status = get_parser_error_for_diagnostic(parser, index, &parser_error);
    if (status != JYPPX_STATUS_OK) { return status; }
    const int32_t stack_size = parser_error->localFunctionStackSize();
    if (stack_index >= stack_size)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser local function stack index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    const char* const* stack = parser_error->localFunctionStack();
    const char* value = stack != nullptr ? stack[stack_index] : "";
    return copy_string_to_buffer(value, output_buffer, output_buffer_size, out_required_size);
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser local function stack entry query");
#endif
}

#if JYPPX_HAS_TENSORRT
static JYPPX_StatusCode get_network_payload_for_api(JYPPX_TensorRtNetworkDefinition* network, nvinfer1::INetworkDefinition** out_network, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_network = get_payload<nvinfer1::INetworkDefinition>(network);
    if (*out_network == nullptr)
    {
        std::ostringstream builder;
        builder << "Network handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_tensor_payload_for_api(JYPPX_TensorRtTensor* tensor, nvinfer1::ITensor** out_tensor, const char* parameter_name)
{
    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, parameter_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = get_payload<nvinfer1::ITensor>(tensor);
    if (*out_tensor == nullptr)
    {
        std::ostringstream builder;
        builder << parameter_name << " does not carry the expected TensorRT tensor payload.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_attention_payload_for_api(JYPPX_TensorRtAttention* attention, nvinfer1::IAttention** out_attention, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(attention, kLine, JYPPX_TENSORRT_OBJECT_KIND_ATTENTION, "attention");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_attention = get_attention_payload(attention);
    if (*out_attention == nullptr)
    {
        std::ostringstream builder;
        builder << "Attention handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_layer_payload_for_api(JYPPX_TensorRtLayer* layer, nvinfer1::ILayer** out_layer, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = get_layer_payload(layer);
    if (*out_layer == nullptr)
    {
        std::ostringstream builder;
        builder << "Layer handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }
    return JYPPX_STATUS_OK;
}
#endif

JYPPX_StatusCode jyppx_trt11_network_add_identity(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network identity layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addIdentity(*input), out_layer);
#else
    (void)network;
    (void)input_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network identity layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_constant(JYPPX_TensorRtNetworkDefinition* network, const JYPPX_TensorRtDims* dims, int32_t data_type, const void* values, size_t value_count, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::Dims native_dims{};
    status = make_dims(dims, &native_dims, "dims");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::vector<uint8_t> owned_weights;
    status = copy_weights(data_type, values, value_count, &owned_weights);
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network constant layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::Weights weights{static_cast<nvinfer1::DataType>(data_type), owned_weights.data(), static_cast<int64_t>(value_count)};
    return create_layer_reference_handle(network_payload->addConstant(native_dims, weights), out_layer, std::move(owned_weights));
#else
    (void)network;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network constant layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_elementwise(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* left_tensor, JYPPX_TensorRtTensor* right_tensor, int32_t operation, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* left = nullptr;
    nvinfer1::ITensor* right = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network elementwise layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(left_tensor, &left, "left_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(right_tensor, &right, "right_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addElementWise(*left, *right, static_cast<nvinfer1::ElementWiseOperation>(operation)), out_layer);
#else
    (void)network;
    (void)left_tensor;
    (void)right_tensor;
    (void)operation;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network elementwise layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_matrix_multiply(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* left_tensor, int32_t left_operation, JYPPX_TensorRtTensor* right_tensor, int32_t right_operation, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* left = nullptr;
    nvinfer1::ITensor* right = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network matrix multiply layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(left_tensor, &left, "left_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(right_tensor, &right, "right_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addMatrixMultiply(*left, static_cast<nvinfer1::MatrixOperation>(left_operation), *right, static_cast<nvinfer1::MatrixOperation>(right_operation)), out_layer);
#else
    (void)network;
    (void)left_tensor;
    (void)left_operation;
    (void)right_tensor;
    (void)right_operation;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network matrix multiply layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_shuffle(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network shuffle layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addShuffle(*input), out_layer);
#else
    (void)network;
    (void)input_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network shuffle layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_reduce(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t operation, uint32_t axes, int32_t keep_dimensions, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network reduce layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = create_layer_reference_handle(network_payload->addReduce(*input, static_cast<nvinfer1::ReduceOperation>(operation), axes, keep_dimensions != 0), out_layer);
    if (status == JYPPX_STATUS_OK)
    {
        auto* payload = get_layer_reference_payload(*out_layer);
        if (payload != nullptr)
        {
            payload->reduce_operation = operation;
            payload->reduce_axes = axes;
            payload->reduce_keep_dimensions = keep_dimensions;
        }
    }
    return status;
#else
    (void)network;
    (void)input_tensor;
    (void)operation;
    (void)axes;
    (void)keep_dimensions;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network reduce layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_concatenation(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor** input_tensors, int32_t input_count, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (input_count <= 0 || input_tensors == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Concatenation input tensor list must not be empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network concatenation layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::vector<nvinfer1::ITensor*> inputs(static_cast<size_t>(input_count));
    for (int32_t i = 0; i < input_count; ++i)
    {
        status = get_tensor_payload_for_api(input_tensors[i], &inputs[static_cast<size_t>(i)], "input_tensors");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
    return create_layer_reference_handle(network_payload->addConcatenation(inputs.data(), input_count), out_layer);
#else
    (void)network;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network concatenation layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_slice(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, const JYPPX_TensorRtDims* start, const JYPPX_TensorRtDims* size, const JYPPX_TensorRtDims* stride, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::Dims native_start{}, native_size{}, native_stride{};
    status = make_dims(start, &native_start, "start");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = make_dims(size, &native_size, "size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = make_dims(stride, &native_stride, "stride");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network slice layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addSlice(*input, native_start, native_size, native_stride), out_layer);
#else
    (void)network;
    (void)input_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network slice layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_softmax(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network softmax layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addSoftMax(*input), out_layer);
#else
    (void)network;
    (void)input_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network softmax layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_unary(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t operation, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network unary layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = create_layer_reference_handle(network_payload->addUnary(*input, static_cast<nvinfer1::UnaryOperation>(operation)), out_layer);
    if (status == JYPPX_STATUS_OK)
    {
        auto* payload = get_layer_reference_payload(*out_layer);
        if (payload != nullptr) { payload->unary_operation = operation; }
    }
    return status;
#else
    (void)network;
    (void)input_tensor;
    (void)operation;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network unary layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_topk(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t operation, int32_t k, uint32_t axes, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network topk layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = create_layer_reference_handle(network_payload->addTopK(*input, static_cast<nvinfer1::TopKOperation>(operation), k, axes), out_layer);
    if (status == JYPPX_STATUS_OK)
    {
        auto* payload = get_layer_reference_payload(*out_layer);
        if (payload != nullptr)
        {
            payload->topk_operation = operation;
            payload->topk_k = k;
            payload->topk_axes = axes;
        }
    }
    return status;
#else
    (void)network;
    (void)input_tensor;
    (void)operation;
    (void)k;
    (void)axes;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network topk layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_gather(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* data_tensor, JYPPX_TensorRtTensor* indices_tensor, int32_t axis, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* data = nullptr;
    nvinfer1::ITensor* indices = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network gather layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(data_tensor, &data, "data_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(indices_tensor, &indices, "indices_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = create_layer_reference_handle(network_payload->addGather(*data, *indices, axis), out_layer);
    if (status == JYPPX_STATUS_OK)
    {
        auto* payload = get_layer_reference_payload(*out_layer);
        if (payload != nullptr) { payload->gather_axis = axis; }
    }
    return status;
#else
    (void)network;
    (void)data_tensor;
    (void)indices_tensor;
    (void)axis;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network gather layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_gather_v2(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* data_tensor, JYPPX_TensorRtTensor* indices_tensor, int32_t mode, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* data = nullptr;
    nvinfer1::ITensor* indices = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network gather v2 layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(data_tensor, &data, "data_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(indices_tensor, &indices, "indices_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = create_layer_reference_handle(network_payload->addGatherV2(*data, *indices, static_cast<nvinfer1::GatherMode>(mode)), out_layer);
    if (status == JYPPX_STATUS_OK)
    {
        auto* payload = get_layer_reference_payload(*out_layer);
        if (payload != nullptr) { payload->gather_axis = 0; }
    }
    return status;
#else
    (void)network;
    (void)data_tensor;
    (void)indices_tensor;
    (void)mode;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network gather v2 layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_scatter(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* data_tensor, JYPPX_TensorRtTensor* indices_tensor, JYPPX_TensorRtTensor* updates_tensor, int32_t mode, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* data = nullptr;
    nvinfer1::ITensor* indices = nullptr;
    nvinfer1::ITensor* updates = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network scatter layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(data_tensor, &data, "data_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(indices_tensor, &indices, "indices_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(updates_tensor, &updates, "updates_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addScatter(*data, *indices, *updates, static_cast<nvinfer1::ScatterMode>(mode)), out_layer);
#else
    (void)network;
    (void)data_tensor;
    (void)indices_tensor;
    (void)updates_tensor;
    (void)mode;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network scatter layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_one_hot(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* indices_tensor, JYPPX_TensorRtTensor* values_tensor, JYPPX_TensorRtTensor* depth_tensor, int32_t axis, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* indices = nullptr;
    nvinfer1::ITensor* values = nullptr;
    nvinfer1::ITensor* depth = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network one-hot layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(indices_tensor, &indices, "indices_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(values_tensor, &values, "values_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(depth_tensor, &depth, "depth_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addOneHot(*indices, *values, *depth, axis), out_layer);
#else
    (void)network;
    (void)indices_tensor;
    (void)values_tensor;
    (void)depth_tensor;
    (void)axis;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network one-hot layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_cumulative(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* axis_tensor, int32_t operation, JYPPX_Boolean exclusive, JYPPX_Boolean reverse, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* axis = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network cumulative layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(axis_tensor, &axis, "axis_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(
        network_payload->addCumulative(*input, *axis, static_cast<nvinfer1::CumulativeOperation>(operation), exclusive != JYPPX_FALSE, reverse != JYPPX_FALSE),
        out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)axis_tensor;
    (void)operation;
    (void)exclusive;
    (void)reverse;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network cumulative layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_activation(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t activation_type, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network activation layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addActivation(*input, static_cast<nvinfer1::ActivationType>(activation_type)), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)activation_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network activation layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_pooling_nd(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t pooling_type, const JYPPX_TensorRtDims* window_size, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::Dims native_window{};
    status = make_dims(window_size, &native_window, "window_size");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network pooling layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addPoolingNd(*input, static_cast<nvinfer1::PoolingType>(pooling_type), native_window), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)pooling_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network pooling layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_resize(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network resize layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addResize(*input), out_layer);
#else
    (void)network;
    (void)input_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network resize layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_shape(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network shape layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addShape(*input), out_layer);
#else
    (void)network;
    (void)input_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network shape layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_select(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* condition_tensor, JYPPX_TensorRtTensor* then_tensor, JYPPX_TensorRtTensor* else_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* condition = nullptr;
    nvinfer1::ITensor* then_payload = nullptr;
    nvinfer1::ITensor* else_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network select layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(condition_tensor, &condition, "condition_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(then_tensor, &then_payload, "then_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(else_tensor, &else_payload, "else_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addSelect(*condition, *then_payload, *else_payload), out_layer);
#else
    (void)network;
    (void)condition_tensor;
    (void)then_tensor;
    (void)else_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network select layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_fill(JYPPX_TensorRtNetworkDefinition* network, const JYPPX_TensorRtDims* dimensions, int32_t operation, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::Dims native_dims{};
    status = make_dims(dimensions, &native_dims, "dimensions");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network fill layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addFill(native_dims, static_cast<nvinfer1::FillOperation>(operation), nvinfer1::DataType::kFLOAT), out_layer);
#else
    (void)network;
    (void)operation;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network fill layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_fill_v2(JYPPX_TensorRtNetworkDefinition* network, const JYPPX_TensorRtDims* dimensions, int32_t operation, int32_t output_type, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::Dims native_dims{};
    status = make_dims(dimensions, &native_dims, "dimensions");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network fill v2 layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(
        network_payload->addFill(native_dims, static_cast<nvinfer1::FillOperation>(operation), static_cast<nvinfer1::DataType>(output_type)),
        out_layer);
#else
    (void)network;
    (void)operation;
    (void)output_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network fill v2 layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_assertion(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* condition_tensor, const char* message, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_c_string(message, "message");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* condition = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network assertion layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(condition_tensor, &condition, "condition_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addAssertion(*condition, message), out_layer);
#else
    (void)network;
    (void)condition_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network assertion layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_grid_sample(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* grid_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* grid = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network grid sample layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(grid_tensor, &grid, "grid_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addGridSample(*input, *grid), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)grid_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network grid sample layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_normalization_v2(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* scale_tensor, JYPPX_TensorRtTensor* bias_tensor, uint32_t axes, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* scale = nullptr;
    nvinfer1::ITensor* bias = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network normalization v2 layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(scale_tensor, &scale, "scale_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(bias_tensor, &bias, "bias_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addNormalizationV2(*input, *scale, *bias, axes), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)scale_tensor;
    (void)bias_tensor;
    (void)axes;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network normalization v2 layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_squeeze(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* axes_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* axes = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network squeeze layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(axes_tensor, &axes, "axes_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addSqueeze(*input, *axes), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)axes_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network squeeze layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_unsqueeze(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* axes_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* axes = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network unsqueeze layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(axes_tensor, &axes, "axes_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addUnsqueeze(*input, *axes), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)axes_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network unsqueeze layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_dynamic_quantize_v2(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, const JYPPX_TensorRtDims* block_shape, int32_t output_type, int32_t scale_type, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::Dims native_block_shape{};
    status = make_dims(block_shape, &native_block_shape, "block_shape");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network dynamic quantize v2 layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(
        network_payload->addDynamicQuantizeV2(*input, native_block_shape, static_cast<nvinfer1::DataType>(output_type), static_cast<nvinfer1::DataType>(scale_type)),
        out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)output_type;
    (void)scale_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network dynamic quantize v2 layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_parametric_relu(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* slope_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* slope = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network parametric ReLU layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(slope_tensor, &slope, "slope_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addParametricReLU(*input, *slope), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)slope_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network parametric ReLU layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_dequantize_v2(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* scale_tensor, int32_t output_type, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* scale = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network dequantize v2 layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(scale_tensor, &scale, "scale_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addDequantize(*input, *scale, static_cast<nvinfer1::DataType>(output_type)), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)scale_tensor;
    (void)output_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network dequantize v2 layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_dist_collective(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    int32_t collective_operation,
    int32_t reduce_operation,
    int64_t root,
    const int64_t* groups,
    int32_t group_count,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (group_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "DistCollective group count must not be negative.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (group_count > 0 && groups == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "DistCollective groups must not be null when group count is positive.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network DistCollective layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }

    std::vector<int64_t> owned_groups;
    if (group_count > 0)
    {
        owned_groups.assign(groups, groups + group_count);
    }

    int64_t* native_groups = owned_groups.empty() ? nullptr : owned_groups.data();
    status = create_layer_reference_handle(
        network_payload->addDistCollective(
            *input,
            static_cast<nvinfer1::CollectiveOperation>(collective_operation),
            static_cast<nvinfer1::ReduceOperation>(reduce_operation),
            root,
            native_groups,
            static_cast<int64_t>(group_count)),
        out_layer);
    if (status == JYPPX_STATUS_OK)
    {
        auto* payload = get_layer_reference_payload(*out_layer);
        if (payload != nullptr)
        {
            payload->owned_dist_collective_groups = std::move(owned_groups);
        }
    }
    return status;
#else
    (void)network;
    (void)input_tensor;
    (void)collective_operation;
    (void)reduce_operation;
    (void)root;
    (void)groups;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network DistCollective layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_rotary_embedding(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    JYPPX_TensorRtTensor* cos_cache_tensor,
    JYPPX_TensorRtTensor* sin_cache_tensor,
    JYPPX_Boolean interleaved,
    int32_t rotary_embedding_dim,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (rotary_embedding_dim < 0 || (rotary_embedding_dim % 2) != 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Rotary embedding dimension must be a non-negative even value.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* cos_cache = nullptr;
    nvinfer1::ITensor* sin_cache = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network rotary embedding layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(cos_cache_tensor, &cos_cache, "cos_cache_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(sin_cache_tensor, &sin_cache, "sin_cache_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(
        network_payload->addRotaryEmbedding(*input, *cos_cache, *sin_cache, interleaved != JYPPX_FALSE, rotary_embedding_dim),
        out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)cos_cache_tensor;
    (void)sin_cache_tensor;
    (void)interleaved;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network rotary embedding layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_kv_cache_update(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* cache_tensor,
    JYPPX_TensorRtTensor* update_tensor,
    JYPPX_TensorRtTensor* write_indices_tensor,
    int32_t cache_mode,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (cache_mode < 0)
    {
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* cache = nullptr;
    nvinfer1::ITensor* update = nullptr;
    nvinfer1::ITensor* write_indices = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network KV cache update layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(cache_tensor, &cache, "cache_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(update_tensor, &update, "update_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(write_indices_tensor, &write_indices, "write_indices_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(
        network_payload->addKVCacheUpdate(*cache, *update, *write_indices, static_cast<nvinfer1::KVCacheMode>(cache_mode)),
        out_layer);
#else
    (void)network;
    (void)cache_tensor;
    (void)update_tensor;
    (void)write_indices_tensor;
    (void)cache_mode;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network KV cache update layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_moe(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* hidden_states_tensor,
    JYPPX_TensorRtTensor* selected_experts_tensor,
    JYPPX_TensorRtTensor* scores_tensor,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* hidden_states = nullptr;
    nvinfer1::ITensor* selected_experts = nullptr;
    nvinfer1::ITensor* scores = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network MoE layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(hidden_states_tensor, &hidden_states, "hidden_states_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(selected_experts_tensor, &selected_experts, "selected_experts_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(scores_tensor, &scores, "scores_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addMoE(*hidden_states, *selected_experts, *scores), out_layer);
#else
    (void)network;
    (void)hidden_states_tensor;
    (void)selected_experts_tensor;
    (void)scores_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network MoE layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_attention_v2(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* query_tensor,
    JYPPX_TensorRtTensor* key_tensor,
    JYPPX_TensorRtTensor* value_tensor,
    int32_t normalization_operation,
    int32_t causal_kind,
    JYPPX_TensorRtAttention** out_attention)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_attention, "out_attention");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_attention = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* query = nullptr;
    nvinfer1::ITensor* key = nullptr;
    nvinfer1::ITensor* value = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network attention creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(query_tensor, &query, "query_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(key_tensor, &key, "key_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(value_tensor, &value, "value_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_attention_reference_handle(
        network_payload->addAttentionV2(
            *query,
            *key,
            *value,
            static_cast<nvinfer1::AttentionNormalizationOp>(normalization_operation),
            static_cast<nvinfer1::CausalMaskKind>(causal_kind)),
        out_attention);
#else
    (void)network;
    (void)query_tensor;
    (void)key_tensor;
    (void)value_tensor;
    (void)normalization_operation;
    (void)causal_kind;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network attention creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_normalization_operation(JYPPX_TensorRtAttention* attention, int32_t operation, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention normalization operation update");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = payload->setNormalizationOperation(static_cast<nvinfer1::AttentionNormalizationOp>(operation)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)operation;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention normalization operation update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_normalization_operation(JYPPX_TensorRtAttention* attention, int32_t* out_operation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_operation, "out_operation");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_operation = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention normalization operation query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_operation = static_cast<int32_t>(payload->getNormalizationOperation());
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention normalization operation query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_mask(JYPPX_TensorRtAttention* attention, JYPPX_TensorRtTensor* mask_tensor, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    nvinfer1::ITensor* mask = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention mask update");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(mask_tensor, &mask, "mask_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = payload->setMask(*mask) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)mask_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention mask update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_mask(JYPPX_TensorRtAttention* attention, JYPPX_TensorRtTensor** out_tensor, JYPPX_Boolean* out_has_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_has_tensor, "out_has_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
    *out_has_tensor = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention mask query");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITensor* tensor = payload->getMask();
    if (tensor == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    status = create_tensor_reference_handle(tensor, out_tensor);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_tensor = JYPPX_TRUE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention mask query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_causal_kind(JYPPX_TensorRtAttention* attention, int32_t causal_kind, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention causal kind update");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = payload->setCausalKind(static_cast<nvinfer1::CausalMaskKind>(causal_kind)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)causal_kind;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention causal kind update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_causal_kind(JYPPX_TensorRtAttention* attention, int32_t* out_causal_kind)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_causal_kind, "out_causal_kind");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_causal_kind = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention causal kind query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_causal_kind = static_cast<int32_t>(payload->getCausalKind());
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention causal kind query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_decomposable(JYPPX_TensorRtAttention* attention, JYPPX_Boolean decomposable, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention decomposable update");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = payload->setDecomposable(decomposable == JYPPX_TRUE) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)decomposable;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention decomposable update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_decomposable(JYPPX_TensorRtAttention* attention, JYPPX_Boolean* out_decomposable)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_decomposable, "out_decomposable");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_decomposable = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention decomposable query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_decomposable = payload->getDecomposable() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention decomposable query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_input(JYPPX_TensorRtAttention* attention, int32_t index, JYPPX_TensorRtTensor* input_tensor, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention input update");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_index(index, payload->getNbInputs(), "input");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = payload->setInput(index, *input) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)index;
    (void)input_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention input update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_input_count(JYPPX_TensorRtAttention* attention, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention input count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = payload->getNbInputs();
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention input count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_input(JYPPX_TensorRtAttention* attention, int32_t index, JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention input query");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_index(index, payload->getNbInputs(), "input");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_tensor_reference_handle(payload->getInput(index), out_tensor);
#else
    (void)attention;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention input query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_output_count(JYPPX_TensorRtAttention* attention, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention output count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = payload->getNbOutputs();
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention output count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_output(JYPPX_TensorRtAttention* attention, int32_t index, JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention output query");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_index(index, payload->getNbOutputs(), "output");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_tensor_reference_handle(payload->getOutput(index), out_tensor);
#else
    (void)attention;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention output query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_name(JYPPX_TensorRtAttention* attention, const char* name, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    if (name == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "attention name must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention name update");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = payload->setName(name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)name;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention name update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_name(JYPPX_TensorRtAttention* attention, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention name query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(payload->getName(), output_buffer, output_buffer_size, out_required_size);
#else
    (void)attention;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention name query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_normalization_quantize_scale(JYPPX_TensorRtAttention* attention, JYPPX_TensorRtTensor* scale_tensor, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    nvinfer1::ITensor* scale = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention normalization quantize scale update");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(scale_tensor, &scale, "scale_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = payload->setNormalizationQuantizeScale(*scale) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)scale_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention normalization quantize scale update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_normalization_quantize_scale(JYPPX_TensorRtAttention* attention, JYPPX_TensorRtTensor** out_tensor, JYPPX_Boolean* out_has_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_has_tensor, "out_has_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
    *out_has_tensor = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention normalization quantize scale query");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITensor* tensor = payload->getNormalizationQuantizeScale();
    if (tensor == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    status = create_tensor_reference_handle(tensor, out_tensor);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_tensor = JYPPX_TRUE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention normalization quantize scale query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_normalization_quantize_to_type(JYPPX_TensorRtAttention* attention, int32_t data_type, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention normalization quantize type update");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = payload->setNormalizationQuantizeToType(static_cast<nvinfer1::DataType>(data_type)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)data_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention normalization quantize type update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_normalization_quantize_to_type(JYPPX_TensorRtAttention* attention, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention normalization quantize type query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = static_cast<int32_t>(payload->getNormalizationQuantizeToType());
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention normalization quantize type query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_metadata(JYPPX_TensorRtAttention* attention, const char* metadata, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    if (metadata == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "attention metadata must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention metadata update");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = payload->setMetadata(metadata) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)metadata;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention metadata update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_metadata(JYPPX_TensorRtAttention* attention, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention metadata query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(payload->getMetadata(), output_buffer, output_buffer_size, out_required_size);
#else
    (void)attention;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention metadata query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_nb_ranks(JYPPX_TensorRtAttention* attention, int32_t rank_count, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention rank count update");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = payload->setNbRanks(rank_count) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)rank_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention rank count update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_nb_ranks(JYPPX_TensorRtAttention* attention, int32_t* out_rank_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_rank_count, "out_rank_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_rank_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention rank count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_rank_count = payload->getNbRanks();
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention rank count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_query_form(JYPPX_TensorRtAttention* attention, int32_t form, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention query form update");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = payload->setQueryForm(static_cast<nvinfer1::AttentionIOForm>(form)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)form;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention query form update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_query_form(JYPPX_TensorRtAttention* attention, int32_t* out_form)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_form, "out_form");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_form = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention query form query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_form = static_cast<int32_t>(payload->getQueryForm());
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention query form query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_key_value_form(JYPPX_TensorRtAttention* attention, int32_t form, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention key-value form update");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = payload->setKeyValueForm(static_cast<nvinfer1::AttentionIOForm>(form)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)form;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention key-value form update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_key_value_form(JYPPX_TensorRtAttention* attention, int32_t* out_form)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_form, "out_form");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_form = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention key-value form query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_form = static_cast<int32_t>(payload->getKeyValueForm());
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention key-value form query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_query_lengths(JYPPX_TensorRtAttention* attention, JYPPX_TensorRtTensor* lengths_tensor, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    nvinfer1::ITensor* lengths = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention query lengths update");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (lengths_tensor != nullptr)
    {
        status = get_tensor_payload_for_api(lengths_tensor, &lengths, "lengths_tensor");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
    *out_success = payload->setQueryLengths(lengths) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)lengths_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention query lengths update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_query_lengths(JYPPX_TensorRtAttention* attention, JYPPX_TensorRtTensor** out_tensor, JYPPX_Boolean* out_has_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_has_tensor, "out_has_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
    *out_has_tensor = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention query lengths query");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITensor* tensor = payload->getQueryLengths();
    if (tensor == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    status = create_tensor_reference_handle(tensor, out_tensor);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_tensor = JYPPX_TRUE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention query lengths query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_set_key_value_lengths(JYPPX_TensorRtAttention* attention, JYPPX_TensorRtTensor* lengths_tensor, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    nvinfer1::ITensor* lengths = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention key-value lengths update");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (lengths_tensor != nullptr)
    {
        status = get_tensor_payload_for_api(lengths_tensor, &lengths, "lengths_tensor");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
    *out_success = payload->setKeyValueLengths(lengths) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    (void)lengths_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention key-value lengths update");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_get_key_value_lengths(JYPPX_TensorRtAttention* attention, JYPPX_TensorRtTensor** out_tensor, JYPPX_Boolean* out_has_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_has_tensor, "out_has_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
    *out_has_tensor = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IAttention* payload = nullptr;
    status = get_attention_payload_for_api(attention, &payload, "attention key-value lengths query");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITensor* tensor = payload->getKeyValueLengths();
    if (tensor == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    status = create_tensor_reference_handle(tensor, out_tensor);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_tensor = JYPPX_TRUE;
    return JYPPX_STATUS_OK;
#else
    (void)attention;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention key-value lengths query");
#endif
}

JYPPX_StatusCode jyppx_trt11_attention_boundary_layer_get_attention(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtAttention** out_attention)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_attention, "out_attention");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_attention = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* layer_payload = nullptr;
    status = get_layer_payload_for_api(layer, &layer_payload, "attention boundary getAttention");
    if (status != JYPPX_STATUS_OK) { return status; }
    const nvinfer1::LayerType layer_type = layer_payload->getType();
    if (layer_type != nvinfer1::LayerType::kATTENTION_INPUT && layer_type != nvinfer1::LayerType::kATTENTION_OUTPUT)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer is not a TensorRT attention boundary layer.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* boundary_layer = static_cast<nvinfer1::IAttentionBoundaryLayer*>(layer_payload);
    return create_attention_reference_handle(boundary_layer->getAttention(), out_attention);
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "attention boundary getAttention");
#endif
}

JYPPX_StatusCode jyppx_trt11_layer_get_output(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* payload = nullptr;
    status = get_layer_payload_for_api(layer, &payload, "layer output query");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_index(index, payload->getNbOutputs(), "output");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_tensor_reference_handle(payload->getOutput(index), out_tensor);
#else
    (void)layer;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer output query");
#endif
}

JYPPX_StatusCode jyppx_trt11_layer_get_input(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* payload = nullptr;
    status = get_layer_payload_for_api(layer, &payload, "layer input query");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_index(index, payload->getNbInputs(), "input");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_tensor_reference_handle(payload->getInput(index), out_tensor);
#else
    (void)layer;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer input query");
#endif
}

JYPPX_StatusCode jyppx_trt11_layer_get_input_count(JYPPX_TensorRtLayer* layer, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* payload = nullptr;
    status = get_layer_payload_for_api(layer, &payload, "layer input count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = payload->getNbInputs();
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer input count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_layer_get_output_count(JYPPX_TensorRtLayer* layer, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* payload = nullptr;
    status = get_layer_payload_for_api(layer, &payload, "layer output count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = payload->getNbOutputs();
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer output count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_layer_get_type(JYPPX_TensorRtLayer* layer, int32_t* out_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_type, "out_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_type = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* payload = nullptr;
    status = get_layer_payload_for_api(layer, &payload, "layer type query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_type = static_cast<int32_t>(payload->getType());
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer type query");
#endif
}

JYPPX_StatusCode jyppx_trt11_layer_get_name(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* payload = nullptr;
    status = get_layer_payload_for_api(layer, &payload, "layer name query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(payload->getName(), output_buffer, output_buffer_size, out_required_size);
#else
    (void)layer;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer name query");
#endif
}

JYPPX_StatusCode jyppx_trt11_layer_set_name(JYPPX_TensorRtLayer* layer, const char* name)
{
    auto status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* payload = nullptr;
    status = get_layer_payload_for_api(layer, &payload, "layer name set");
    if (status != JYPPX_STATUS_OK) { return status; }
    payload->setName(name);
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer name set");
#endif
}

static JYPPX_StatusCode trt11_validate_layer_for_removed_type_api(JYPPX_TensorRtLayer* layer, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::ostringstream message;
    message << "TensorRT 11 removed " << feature_name << " from ILayer; use strongly typed network flows or layer-specific APIs.";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, message.str().c_str());
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt11_layer_set_precision(JYPPX_TensorRtLayer* layer, int32_t data_type)
{
    (void)data_type;
    return trt11_validate_layer_for_removed_type_api(layer, "setPrecision");
}

JYPPX_StatusCode jyppx_trt11_layer_get_precision(JYPPX_TensorRtLayer* layer, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = -1;
    return trt11_validate_layer_for_removed_type_api(layer, "getPrecision");
}

JYPPX_StatusCode jyppx_trt11_layer_precision_is_set(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_is_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_set, "out_is_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_set = JYPPX_FALSE;
    return trt11_validate_layer_for_removed_type_api(layer, "precisionIsSet");
}

JYPPX_StatusCode jyppx_trt11_layer_reset_precision(JYPPX_TensorRtLayer* layer)
{
    return trt11_validate_layer_for_removed_type_api(layer, "resetPrecision");
}

JYPPX_StatusCode jyppx_trt11_layer_set_output_type(JYPPX_TensorRtLayer* layer, int32_t index, int32_t data_type)
{
    if (index < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    (void)data_type;
    return trt11_validate_layer_for_removed_type_api(layer, "setOutputType");
}

JYPPX_StatusCode jyppx_trt11_layer_get_output_type(JYPPX_TensorRtLayer* layer, int32_t index, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* payload = nullptr;
    status = get_layer_payload_for_api(layer, &payload, "layer output type query");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_index(index, payload->getNbOutputs(), "output");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = static_cast<int32_t>(payload->getOutputType(index));
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer output type query");
#endif
}

JYPPX_StatusCode jyppx_trt11_layer_output_type_is_set(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_Boolean* out_is_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_set, "out_is_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (index < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    *out_is_set = JYPPX_FALSE;
    return trt11_validate_layer_for_removed_type_api(layer, "outputTypeIsSet");
}

JYPPX_StatusCode jyppx_trt11_layer_reset_output_type(JYPPX_TensorRtLayer* layer, int32_t index)
{
    if (index < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    return trt11_validate_layer_for_removed_type_api(layer, "resetOutputType");
}

JYPPX_StatusCode jyppx_trt11_tensor_get_name(JYPPX_TensorRtTensor* tensor, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(payload->getName(), output_buffer, output_buffer_size, out_required_size);
#else
    (void)tensor;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor name query");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_set_name(JYPPX_TensorRtTensor* tensor, const char* name)
{
    auto status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    payload->setName(name);
    return JYPPX_STATUS_OK;
#else
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor name set");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_get_data_type(JYPPX_TensorRtTensor* tensor, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = static_cast<int32_t>(payload->getType());
    return JYPPX_STATUS_OK;
#else
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor data type query");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_get_shape(JYPPX_TensorRtTensor* tensor, JYPPX_TensorRtDims* out_dims)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dims, "out_dims");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_dims, 0, sizeof(*out_dims));
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_dims(payload->getDimensions(), out_dims);
    return JYPPX_STATUS_OK;
#else
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor shape query");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_get_dimensions(JYPPX_TensorRtTensor* tensor, JYPPX_TensorRtDims* out_dims)
{
    return jyppx_trt11_tensor_get_shape(tensor, out_dims);
}

JYPPX_StatusCode jyppx_trt11_tensor_get_location(JYPPX_TensorRtTensor* tensor, int32_t* out_location)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_location, "out_location");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_location = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_location = static_cast<int32_t>(payload->getLocation());
    return JYPPX_STATUS_OK;
#else
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor location query");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_set_location(JYPPX_TensorRtTensor* tensor, int32_t location)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITensor* payload = nullptr;
    auto status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    payload->setLocation(static_cast<nvinfer1::TensorLocation>(location));
    return JYPPX_STATUS_OK;
#else
    (void)tensor;
    (void)location;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor location set");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_get_allowed_formats(JYPPX_TensorRtTensor* tensor, uint32_t* out_formats)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_formats, "out_formats");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_formats = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_formats = static_cast<uint32_t>(payload->getAllowedFormats());
    return JYPPX_STATUS_OK;
#else
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor allowed formats query");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_set_allowed_formats(JYPPX_TensorRtTensor* tensor, uint32_t formats)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITensor* payload = nullptr;
    auto status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    payload->setAllowedFormats(static_cast<nvinfer1::TensorFormats>(formats));
    return JYPPX_STATUS_OK;
#else
    (void)tensor;
    (void)formats;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor allowed formats set");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_get_broadcast_across_batch(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_broadcast_across_batch)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_broadcast_across_batch, "out_broadcast_across_batch");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_broadcast_across_batch = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_broadcast_across_batch = payload->getBroadcastAcrossBatch() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor broadcast-across-batch query");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_set_broadcast_across_batch(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean broadcast_across_batch)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITensor* payload = nullptr;
    auto status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    payload->setBroadcastAcrossBatch(broadcast_across_batch != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#else
    (void)tensor;
    (void)broadcast_across_batch;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor broadcast-across-batch set");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_is_shape_tensor(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_shape_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_shape_tensor, "out_is_shape_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_shape_tensor = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_shape_tensor = payload->isShapeTensor() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor shape-tensor query");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_is_execution_tensor(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_execution_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_execution_tensor, "out_is_execution_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_execution_tensor = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_execution_tensor = payload->isExecutionTensor() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor execution-tensor query");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_get_dimension_name(JYPPX_TensorRtTensor* tensor, int32_t dimension_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    if (dimension_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor dimension index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    const char* name = payload->getDimensionName(dimension_index);
    return copy_string_to_buffer(name == nullptr ? "" : name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)tensor;
    (void)dimension_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor dimension name query");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_set_dimension_name(JYPPX_TensorRtTensor* tensor, int32_t dimension_index, const char* name)
{
#if JYPPX_HAS_TENSORRT
    if (dimension_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor dimension index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (name == nullptr || name[0] == '\0')
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor dimension name must not be null or empty. Use clear_dimension_name to remove a name.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    nvinfer1::ITensor* payload = nullptr;
    auto status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    payload->setDimensionName(dimension_index, name);
    return JYPPX_STATUS_OK;
#else
    (void)tensor;
    (void)dimension_index;
    (void)name;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor dimension name set");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_clear_dimension_name(JYPPX_TensorRtTensor* tensor, int32_t dimension_index)
{
#if JYPPX_HAS_TENSORRT
    if (dimension_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor dimension index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    nvinfer1::ITensor* payload = nullptr;
    auto status = get_tensor_payload_for_api(tensor, &payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    payload->setDimensionName(dimension_index, nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)tensor;
    (void)dimension_index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor dimension name clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_tensor_get_dynamic_range_min(JYPPX_TensorRtTensor*, float* out_value)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_value, "out_value");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_value = 0.0F;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 11 no longer exposes ITensor dynamic range accessors.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt11_tensor_get_dynamic_range_max(JYPPX_TensorRtTensor*, float* out_value)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_value, "out_value");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_value = 0.0F;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 11 no longer exposes ITensor dynamic range accessors.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt11_tensor_set_dynamic_range(JYPPX_TensorRtTensor*, float, float)
{
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 11 no longer exposes ITensor dynamic range accessors.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt11_tensor_reset_dynamic_range(JYPPX_TensorRtTensor*)
{
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 11 no longer exposes ITensor dynamic range accessors.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

#include "modules/layers/layer_tensor_metadata.inc"
#include "modules/layers/deployment_metadata.inc"
#include "modules/layers/weight_info_metadata.inc"
#include "modules/deployment/dims64_metadata.inc"
#include "modules/layers/control_flow.inc"
#include "modules/layers/advanced_layers.inc"
#include "modules/deployment/runtime_serialization_refit.inc"
#include "modules/deployment/refitter_logger_presence.inc"
#include "modules/deployment/onnx_parser_builder_config_attachment.inc"
#include "modules/deployment/runtime_controls.inc"
#define JYPPX_TRT_SAFE_PLUGIN_PREFIX jyppx_trt11_
#define JYPPX_TRT_SAFE_PLUGIN_EXPECTED_MAJOR 11
#include "../common/safe_deferred_plugin_initialization.inc"
#undef JYPPX_TRT_SAFE_PLUGIN_EXPECTED_MAJOR
#undef JYPPX_TRT_SAFE_PLUGIN_PREFIX
#include "modules/deployment/diagnostics.inc"
#include "modules/deployment/boundary_controls.inc"
#define JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_INTERFACE_INFO_API(name) jyppx_trt11_##name
#include "../common/execution_context_callback_interface_info.inc"
#undef JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_INTERFACE_INFO_API
#define JYPPX_TRT_OWNER_SCOPED_VERSIONED_METADATA_API(name) jyppx_trt11_##name
#include "../common/owner_scoped_versioned_interface_metadata.inc"
#undef JYPPX_TRT_OWNER_SCOPED_VERSIONED_METADATA_API
#define JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_STATE_API(name) jyppx_trt11_##name
#include "../common/execution_context_callback_state_snapshot.inc"
#undef JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_STATE_API
#include "modules/deployment/fourteenth_batch_controls.inc"
#include "modules/deployment/fifteenth_batch_controls.inc"

#define JYPPX_TRT_EXECUTION_CONTEXT_SET_AUX_STREAMS_API jyppx_trt11_execution_context_set_aux_streams
#define JYPPX_TRT_EXECUTION_CONTEXT_CLEAR_AUX_STREAMS_API jyppx_trt11_execution_context_clear_aux_streams
#include "../common/execution_context_auxiliary_streams.inc"
#undef JYPPX_TRT_EXECUTION_CONTEXT_CLEAR_AUX_STREAMS_API
#undef JYPPX_TRT_EXECUTION_CONTEXT_SET_AUX_STREAMS_API

#define JYPPX_TRT_EXECUTION_CONTEXT_EXECUTE_V2_API jyppx_trt11_execution_context_execute_v2_safe
#define JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR 11
#include "../common/execution_context_synchronous_inference.inc"
#undef JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR
#undef JYPPX_TRT_EXECUTION_CONTEXT_EXECUTE_V2_API

#define JYPPX_TRT_PLUGIN_PREFIX jyppx_trt11_
#include "../common/plugin_registry_inventory.inc"
#undef JYPPX_TRT_PLUGIN_PREFIX

#define JYPPX_TRT_PLUGIN_V2_LAYER_PREFIX jyppx_trt11_
#include "../common/plugin_v2_layer_metadata_snapshot.inc"
#undef JYPPX_TRT_PLUGIN_V2_LAYER_PREFIX
#define JYPPX_TRT_PLUGIN_V3_LAYER_PREFIX jyppx_trt11_
#include "../common/plugin_v3_layer_metadata_snapshot.inc"
#undef JYPPX_TRT_PLUGIN_V3_LAYER_PREFIX
#define JYPPX_TRT_PLUGIN_OWNER_QUERY_PREFIX jyppx_trt11_
#define JYPPX_TRT_PLUGIN_OWNER_QUERY_ENABLE_V3 1
#include "../common/plugin_layer_owner_scoped_query_snapshots.inc"
#undef JYPPX_TRT_PLUGIN_OWNER_QUERY_ENABLE_V3
#undef JYPPX_TRT_PLUGIN_OWNER_QUERY_PREFIX
#define JYPPX_TRT_GLOBAL_PLUGIN_PREFIX jyppx_trt11_
#include "../common/global_runtime_plugin_probe.inc"
#undef JYPPX_TRT_GLOBAL_PLUGIN_PREFIX

#define JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API(name) jyppx_trt11_##name
#include "../common/allocator_owner_dry_run.inc"
#undef JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API

#define JYPPX_TRT_ONNX_CONFIG_API(name) jyppx_trt11_onnx_config_##name
#define JYPPX_TRT_EXPECTED_MAJOR 11
#include "../common/onnx_config_controls.inc"

#include "../common/debug_listener_native_owner_noncopyable_storage.inc"
#include "../common/debug_listener_native_nothrow_destructor.inc"
#include "../common/debug_listener_native_owner_lifecycle_gate.inc"
#include "../common/debug_listener_native_attach_entry_minimal_safety.inc"
#include "../common/debug_listener_native_attach_bridge_shape_gate.inc"
#include "../common/debug_listener_native_nothrow_vtable_scaffold_gate.inc"
#include "../common/debug_listener_exception_status_mapping_gate.inc"
#include "../common/debug_listener_inflight_accounting_gate.inc"
#include "../common/debug_listener_nothrow_vtable_callback_stub.inc"
#include "../common/debug_listener_borrowed_debug_tensor_metadata_runtime_gate.inc"
#include "../common/debug_listener_native_vtable_install_preflight.inc"
#include "../common/debug_listener_native_owner_vtable_install_experiment.inc"
#include "../common/debug_listener_real_non_null_attach_runtime_smoke.inc"
#include "../common/debug_listener_process_debug_tensor_callback_trampoline.inc"
#include "../common/debug_listener_real_callback_runtime_proof.inc"
#define JYPPX_TRT_DEBUG_LISTENER_OWNER_API(name) jyppx_trt11_##name
#include "../common/debug_listener_callback_owner.inc"
#undef JYPPX_TRT_DEBUG_LISTENER_OWNER_API

#define jyppx_trt10_network_add_convolution_nd jyppx_trt11_network_add_convolution_nd
#define jyppx_trt10_network_add_scale_nd jyppx_trt11_network_add_scale_nd
#define jyppx_trt10_network_add_padding_nd jyppx_trt11_network_add_padding_nd
#define jyppx_trt10_convolution_layer_get_nb_output_maps jyppx_trt11_convolution_layer_get_nb_output_maps
#define jyppx_trt10_convolution_layer_set_nb_output_maps jyppx_trt11_convolution_layer_set_nb_output_maps
#define jyppx_trt10_convolution_layer_get_nb_groups jyppx_trt11_convolution_layer_get_nb_groups
#define jyppx_trt10_convolution_layer_set_nb_groups jyppx_trt11_convolution_layer_set_nb_groups
#define jyppx_trt10_convolution_layer_get_stride_nd jyppx_trt11_convolution_layer_get_stride_nd
#define jyppx_trt10_convolution_layer_set_stride_nd jyppx_trt11_convolution_layer_set_stride_nd
#define jyppx_trt10_convolution_layer_get_padding_nd jyppx_trt11_convolution_layer_get_padding_nd
#define jyppx_trt10_convolution_layer_set_padding_nd jyppx_trt11_convolution_layer_set_padding_nd
#define jyppx_trt10_convolution_layer_get_pre_padding jyppx_trt11_convolution_layer_get_pre_padding
#define jyppx_trt10_convolution_layer_set_pre_padding jyppx_trt11_convolution_layer_set_pre_padding
#define jyppx_trt10_convolution_layer_get_post_padding jyppx_trt11_convolution_layer_get_post_padding
#define jyppx_trt10_convolution_layer_set_post_padding jyppx_trt11_convolution_layer_set_post_padding
#define jyppx_trt10_convolution_layer_get_dilation_nd jyppx_trt11_convolution_layer_get_dilation_nd
#define jyppx_trt10_convolution_layer_set_dilation_nd jyppx_trt11_convolution_layer_set_dilation_nd
#define jyppx_trt10_convolution_layer_get_padding_mode jyppx_trt11_convolution_layer_get_padding_mode
#define jyppx_trt10_convolution_layer_set_padding_mode jyppx_trt11_convolution_layer_set_padding_mode
#define jyppx_trt10_scale_layer_get_mode jyppx_trt11_scale_layer_get_mode
#define jyppx_trt10_scale_layer_set_mode jyppx_trt11_scale_layer_set_mode
#define jyppx_trt10_scale_layer_get_channel_axis jyppx_trt11_scale_layer_get_channel_axis
#define jyppx_trt10_scale_layer_set_channel_axis jyppx_trt11_scale_layer_set_channel_axis
#define jyppx_trt10_padding_layer_get_pre_padding_nd jyppx_trt11_padding_layer_get_pre_padding_nd
#define jyppx_trt10_padding_layer_set_pre_padding_nd jyppx_trt11_padding_layer_set_pre_padding_nd
#define jyppx_trt10_padding_layer_get_post_padding_nd jyppx_trt11_padding_layer_get_post_padding_nd
#define jyppx_trt10_padding_layer_set_post_padding_nd jyppx_trt11_padding_layer_set_post_padding_nd
#include "../v10/modules/layers/convolution_scale_padding.inc"
#undef jyppx_trt10_network_add_convolution_nd
#undef jyppx_trt10_network_add_scale_nd
#undef jyppx_trt10_network_add_padding_nd
#undef jyppx_trt10_convolution_layer_get_nb_output_maps
#undef jyppx_trt10_convolution_layer_set_nb_output_maps
#undef jyppx_trt10_convolution_layer_get_nb_groups
#undef jyppx_trt10_convolution_layer_set_nb_groups
#undef jyppx_trt10_convolution_layer_get_stride_nd
#undef jyppx_trt10_convolution_layer_set_stride_nd
#undef jyppx_trt10_convolution_layer_get_padding_nd
#undef jyppx_trt10_convolution_layer_set_padding_nd
#undef jyppx_trt10_convolution_layer_get_pre_padding
#undef jyppx_trt10_convolution_layer_set_pre_padding
#undef jyppx_trt10_convolution_layer_get_post_padding
#undef jyppx_trt10_convolution_layer_set_post_padding
#undef jyppx_trt10_convolution_layer_get_dilation_nd
#undef jyppx_trt10_convolution_layer_set_dilation_nd
#undef jyppx_trt10_convolution_layer_get_padding_mode
#undef jyppx_trt10_convolution_layer_set_padding_mode
#undef jyppx_trt10_scale_layer_get_mode
#undef jyppx_trt10_scale_layer_set_mode
#undef jyppx_trt10_scale_layer_get_channel_axis
#undef jyppx_trt10_scale_layer_set_channel_axis
#undef jyppx_trt10_padding_layer_get_pre_padding_nd
#undef jyppx_trt10_padding_layer_set_pre_padding_nd
#undef jyppx_trt10_padding_layer_get_post_padding_nd
#undef jyppx_trt10_padding_layer_set_post_padding_nd

#define jyppx_trt10_network_add_deconvolution_nd jyppx_trt11_network_add_deconvolution_nd
#define jyppx_trt10_deconvolution_layer_get_nb_output_maps jyppx_trt11_deconvolution_layer_get_nb_output_maps
#define jyppx_trt10_deconvolution_layer_set_nb_output_maps jyppx_trt11_deconvolution_layer_set_nb_output_maps
#define jyppx_trt10_deconvolution_layer_get_nb_groups jyppx_trt11_deconvolution_layer_get_nb_groups
#define jyppx_trt10_deconvolution_layer_set_nb_groups jyppx_trt11_deconvolution_layer_set_nb_groups
#define jyppx_trt10_deconvolution_layer_get_kernel_size_nd jyppx_trt11_deconvolution_layer_get_kernel_size_nd
#define jyppx_trt10_deconvolution_layer_set_kernel_size_nd jyppx_trt11_deconvolution_layer_set_kernel_size_nd
#define jyppx_trt10_deconvolution_layer_get_stride_nd jyppx_trt11_deconvolution_layer_get_stride_nd
#define jyppx_trt10_deconvolution_layer_set_stride_nd jyppx_trt11_deconvolution_layer_set_stride_nd
#define jyppx_trt10_deconvolution_layer_get_padding_nd jyppx_trt11_deconvolution_layer_get_padding_nd
#define jyppx_trt10_deconvolution_layer_set_padding_nd jyppx_trt11_deconvolution_layer_set_padding_nd
#define jyppx_trt10_deconvolution_layer_get_pre_padding jyppx_trt11_deconvolution_layer_get_pre_padding
#define jyppx_trt10_deconvolution_layer_set_pre_padding jyppx_trt11_deconvolution_layer_set_pre_padding
#define jyppx_trt10_deconvolution_layer_get_post_padding jyppx_trt11_deconvolution_layer_get_post_padding
#define jyppx_trt10_deconvolution_layer_set_post_padding jyppx_trt11_deconvolution_layer_set_post_padding
#define jyppx_trt10_deconvolution_layer_get_dilation_nd jyppx_trt11_deconvolution_layer_get_dilation_nd
#define jyppx_trt10_deconvolution_layer_set_dilation_nd jyppx_trt11_deconvolution_layer_set_dilation_nd
#define jyppx_trt10_deconvolution_layer_get_padding_mode jyppx_trt11_deconvolution_layer_get_padding_mode
#define jyppx_trt10_deconvolution_layer_set_padding_mode jyppx_trt11_deconvolution_layer_set_padding_mode
#include "../v10/modules/layers/deconvolution.inc"
#undef jyppx_trt10_network_add_deconvolution_nd
#undef jyppx_trt10_deconvolution_layer_get_nb_output_maps
#undef jyppx_trt10_deconvolution_layer_set_nb_output_maps
#undef jyppx_trt10_deconvolution_layer_get_nb_groups
#undef jyppx_trt10_deconvolution_layer_set_nb_groups
#undef jyppx_trt10_deconvolution_layer_get_kernel_size_nd
#undef jyppx_trt10_deconvolution_layer_set_kernel_size_nd
#undef jyppx_trt10_deconvolution_layer_get_stride_nd
#undef jyppx_trt10_deconvolution_layer_set_stride_nd
#undef jyppx_trt10_deconvolution_layer_get_padding_nd
#undef jyppx_trt10_deconvolution_layer_set_padding_nd
#undef jyppx_trt10_deconvolution_layer_get_pre_padding
#undef jyppx_trt10_deconvolution_layer_set_pre_padding
#undef jyppx_trt10_deconvolution_layer_get_post_padding
#undef jyppx_trt10_deconvolution_layer_set_post_padding
#undef jyppx_trt10_deconvolution_layer_get_dilation_nd
#undef jyppx_trt10_deconvolution_layer_set_dilation_nd
#undef jyppx_trt10_deconvolution_layer_get_padding_mode
#undef jyppx_trt10_deconvolution_layer_set_padding_mode

#define jyppx_trt10_network_add_lrn jyppx_trt11_network_add_lrn
#define jyppx_trt10_lrn_layer_get_window_size jyppx_trt11_lrn_layer_get_window_size
#define jyppx_trt10_lrn_layer_set_window_size jyppx_trt11_lrn_layer_set_window_size
#define jyppx_trt10_lrn_layer_get_alpha jyppx_trt11_lrn_layer_get_alpha
#define jyppx_trt10_lrn_layer_set_alpha jyppx_trt11_lrn_layer_set_alpha
#define jyppx_trt10_lrn_layer_get_beta jyppx_trt11_lrn_layer_get_beta
#define jyppx_trt10_lrn_layer_set_beta jyppx_trt11_lrn_layer_set_beta
#define jyppx_trt10_lrn_layer_get_k jyppx_trt11_lrn_layer_get_k
#define jyppx_trt10_lrn_layer_set_k jyppx_trt11_lrn_layer_set_k
#include "../v10/modules/layers/lrn.inc"
#undef jyppx_trt10_network_add_lrn
#undef jyppx_trt10_lrn_layer_get_window_size
#undef jyppx_trt10_lrn_layer_set_window_size
#undef jyppx_trt10_lrn_layer_get_alpha
#undef jyppx_trt10_lrn_layer_set_alpha
#undef jyppx_trt10_lrn_layer_get_beta
#undef jyppx_trt10_lrn_layer_set_beta
#undef jyppx_trt10_lrn_layer_get_k
#undef jyppx_trt10_lrn_layer_set_k

#define jyppx_trt10_network_add_quantize jyppx_trt11_network_add_quantize
#define jyppx_trt10_network_add_dequantize jyppx_trt11_network_add_dequantize
#define jyppx_trt10_quantize_layer_get_axis jyppx_trt11_quantize_layer_get_axis
#define jyppx_trt10_quantize_layer_set_axis jyppx_trt11_quantize_layer_set_axis
#define jyppx_trt10_dequantize_layer_get_axis jyppx_trt11_dequantize_layer_get_axis
#define jyppx_trt10_dequantize_layer_set_axis jyppx_trt11_dequantize_layer_set_axis
#include "../v10/modules/layers/quantization.inc"
#undef jyppx_trt10_network_add_quantize
#undef jyppx_trt10_network_add_dequantize
#undef jyppx_trt10_quantize_layer_get_axis
#undef jyppx_trt10_quantize_layer_set_axis
#undef jyppx_trt10_dequantize_layer_get_axis
#undef jyppx_trt10_dequantize_layer_set_axis
#include "modules/deployment/twenty_fourth_batch_aliases.inc"

#define JYPPX_TRT_ERROR_CODE_METADATA_API jyppx_trt11_error_code_get_exclusive_upper_bound
#include "../common/error_code_metadata.inc"
#undef JYPPX_TRT_ERROR_CODE_METADATA_API

#include "modules/deferred/twenty_third_batch_deferred.inc"
