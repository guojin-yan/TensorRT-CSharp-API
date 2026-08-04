#include "jyppx/tensorrt/trt10.h"

#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../common/object.hpp"
#include "../../cuda/object.hpp"
#include "../../common/error_state.hpp"

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

#ifndef JYPPX_HAS_TENSORRT_ONNXPARSER
#define JYPPX_HAS_TENSORRT_ONNXPARSER 0
#endif

#ifndef JYPPX_HAS_TENSORRT_ONNX_CONFIG
#define JYPPX_HAS_TENSORRT_ONNX_CONFIG 0
#endif

namespace
{
constexpr JYPPX_TensorRtLine kLine = JYPPX_TENSORRT_LINE_10;

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

private:
    JYPPX_TensorRtLoggerCallback callback_{nullptr};
    void* user_state_{nullptr};
    int32_t minimum_severity_{static_cast<int32_t>(Severity::kWARNING)};
    bool last_callback_failed_{false};
};

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
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
#endif

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
};

struct LoopReferencePayload
{
    nvinfer1::ILoop* loop{};
};

struct IfConditionalReferencePayload
{
    nvinfer1::IIfConditional* conditional{};
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

JYPPX_StatusCode get_engine_payload_ext(JYPPX_TensorRtCudaEngine* engine, nvinfer1::ICudaEngine** out_engine, const char* feature_name);

nvinfer1::ILayer* get_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* payload = get_payload<LayerReferencePayload>(object);
    return payload != nullptr ? payload->layer : nullptr;
}

LayerReferencePayload* get_layer_reference_payload(const JYPPX_TensorRtObjectBase* object)
{
    return get_payload<LayerReferencePayload>(object);
}

nvinfer1::IShuffleLayer* get_shuffle_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kSHUFFLE)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IShuffleLayer*>(layer);
}

nvinfer1::IReverseSequenceLayer* get_reverse_sequence_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kREVERSE_SEQUENCE)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IReverseSequenceLayer*>(layer);
}

nvinfer1::IMatrixMultiplyLayer* get_matrix_multiply_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kMATRIX_MULTIPLY)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IMatrixMultiplyLayer*>(layer);
}

nvinfer1::IFillLayer* get_fill_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kFILL)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IFillLayer*>(layer);
}

nvinfer1::IReduceLayer* get_reduce_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kREDUCE)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IReduceLayer*>(layer);
}

nvinfer1::IConcatenationLayer* get_concatenation_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kCONCATENATION)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IConcatenationLayer*>(layer);
}

nvinfer1::ISliceLayer* get_slice_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kSLICE)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::ISliceLayer*>(layer);
}

nvinfer1::ISoftMaxLayer* get_softmax_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kSOFTMAX)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::ISoftMaxLayer*>(layer);
}

nvinfer1::IUnaryLayer* get_unary_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kUNARY)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IUnaryLayer*>(layer);
}

nvinfer1::ITopKLayer* get_topk_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kTOPK)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::ITopKLayer*>(layer);
}

nvinfer1::IGatherLayer* get_gather_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kGATHER)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IGatherLayer*>(layer);
}

nvinfer1::IActivationLayer* get_activation_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kACTIVATION)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IActivationLayer*>(layer);
}

nvinfer1::IPoolingLayer* get_pooling_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kPOOLING)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IPoolingLayer*>(layer);
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

nvinfer1::IResizeLayer* get_resize_layer_payload(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kRESIZE)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IResizeLayer*>(layer);
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

void copy_dims(const nvinfer1::Dims& source, JYPPX_TensorRtDims* destination)
{
    destination->nb_dims = source.nbDims;
    for (int32_t i = 0; i < 8; ++i)
    {
        destination->d[i] = i < source.nbDims ? static_cast<int32_t>(source.d[i]) : 0;
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

size_t get_data_type_size(const int32_t data_type)
{
    switch (static_cast<nvinfer1::DataType>(data_type))
    {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kBOOL:
        return 1;
    case nvinfer1::DataType::kUINT8:
        return 1;
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    case nvinfer1::DataType::kFP8:
        return 1;
    case nvinfer1::DataType::kBF16:
        return 2;
    case nvinfer1::DataType::kINT64:
        return 8;
#endif
    default:
        return 0;
    }
}

JYPPX_StatusCode copy_weights(const int32_t data_type, const void* values, const size_t value_count, std::vector<uint8_t>* out_owned_data)
{
    if (values == nullptr || value_count == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Constant weights must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const size_t element_size = get_data_type_size(data_type);
    if (element_size == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unsupported TensorRT constant weight data type.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const size_t byte_count = value_count * element_size;
    if (byte_count / element_size != value_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Constant weight byte count overflow.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    out_owned_data->resize(byte_count);
    std::memcpy(out_owned_data->data(), values, byte_count);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode copy_optional_weights(const int32_t data_type, const void* values, const size_t value_count, const char* name, const bool required, std::vector<uint8_t>* out_owned_data)
{
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

    const size_t element_size = get_data_type_size(data_type);
    if (element_size == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unsupported TensorRT weight data type.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const size_t byte_count = value_count * element_size;
    if (byte_count / element_size != value_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT weight byte count overflow.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    out_owned_data->resize(byte_count);
    std::memcpy(out_owned_data->data(), values, byte_count);
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

    auto* payload = new (std::nothrow) LayerReferencePayload{};
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

    copy_c_string(safe_value, output_buffer, output_buffer_size);
    return JYPPX_STATUS_OK;
}

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
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

#if JYPPX_HAS_TENSORRT_ONNXPARSER
void fill_parser_error(const nvonnxparser::IParserError* parser_error, const int32_t index, JYPPX_TensorRtParserErrorInfo* out_error)
{
    std::memset(out_error, 0, sizeof(*out_error));
    out_error->index = index;
    if (parser_error == nullptr)
    {
        return;
    }

    out_error->code = static_cast<int32_t>(parser_error->code());
    out_error->line = parser_error->line();
    out_error->node = parser_error->node();
    copy_c_string(parser_error->desc(), out_error->description, sizeof(out_error->description));
    copy_c_string(parser_error->file(), out_error->file, sizeof(out_error->file));
    copy_c_string(parser_error->func(), out_error->function_name, sizeof(out_error->function_name));
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    copy_c_string(parser_error->nodeName(), out_error->node_name, sizeof(out_error->node_name));
    copy_c_string(parser_error->nodeOperator(), out_error->node_operator, sizeof(out_error->node_operator));
#endif
}

void set_parser_failure_message(const nvonnxparser::IParser& parser)
{
    const int32_t error_count = parser.getNbErrors();
    if (error_count <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser failed without returning parser errors.");
        return;
    }

    const nvonnxparser::IParserError* parser_error = parser.getError(0);
    if (parser_error == nullptr || parser_error->desc() == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser failed and returned an empty first error.");
        return;
    }

    std::ostringstream builder;
    builder << "ONNX parser failed: " << parser_error->desc();
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
}
#endif

JYPPX_StatusCode copy_engine_information(nvinfer1::IEngineInspector* inspector, const int32_t format, char* output_buffer, const size_t output_buffer_size, size_t* out_required_size)
{
    if (out_required_size == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "out_required_size must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const auto layer_format = static_cast<nvinfer1::LayerInformationFormat>(format);
    const char* text = inspector->getEngineInformation(layer_format);
    if (text == nullptr)
    {
        return report_null_vendor_object("getEngineInformation");
    }

    const size_t required_size = std::strlen(text) + 1;
    *out_required_size = required_size;
    if (output_buffer == nullptr || output_buffer_size == 0)
    {
        return JYPPX_STATUS_OK;
    }

    const size_t copy_length = required_size <= output_buffer_size ? required_size - 1 : output_buffer_size - 1;
    std::memcpy(output_buffer, text, copy_length);
    output_buffer[copy_length] = '\0';
    return required_size <= output_buffer_size ? JYPPX_STATUS_OK : JYPPX_STATUS_BUFFER_TOO_SMALL;
}
#endif

#if !JYPPX_HAS_TENSORRT
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

#define JYPPX_TRT10_VALIDATE_NAMED_TENSOR_DEFINED 1
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

    copy_c_string(safe_value, output_buffer, output_buffer_size);
    return JYPPX_STATUS_OK;
}

size_t get_data_type_size(const int32_t data_type)
{
    switch (data_type)
    {
    case 0:
        return 4;
    case 1:
        return 2;
    case 2:
        return 1;
    case 3:
        return 4;
    case 4:
        return 1;
    case 5:
        return 1;
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    case 6:
        return 1;
    case 7:
        return 2;
    case 8:
        return 8;
#endif
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 11
    case 9:
        return 1;
    case 10:
        return 2;
#endif
    default:
        return 0;
    }
}
#endif

bool trt10_vendor_available()
{
#if JYPPX_HAS_TENSORRT
    return JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10;
#else
    return false;
#endif
}
}

JYPPX_StatusCode jyppx_trt10_query_adapter_info(JYPPX_TensorRtAdapterInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    jyppx::tensorrt::fill_adapter_info(out_info, kLine);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_logger_create(JYPPX_TensorRtLogger** out_logger)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_logger, "out_logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_logger = nullptr;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "logger creation");
    }

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
    *out_logger = reinterpret_cast<JYPPX_TensorRtLogger*>(jyppx::tensorrt::create_object(kLine, JYPPX_TENSORRT_OBJECT_KIND_LOGGER));
    return *out_logger != nullptr ? JYPPX_STATUS_OK : JYPPX_STATUS_OUT_OF_MEMORY;
#endif
}

JYPPX_StatusCode jyppx_trt10_logger_create_with_callback(
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
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed logger callback creation");
    }

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

JYPPX_StatusCode jyppx_trt10_logger_emit_diagnostic(
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
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed logger diagnostic emission");
    }

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

JYPPX_StatusCode jyppx_trt10_progress_monitor_create_with_callback(
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

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
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
#elif JYPPX_HAS_TENSORRT
    (void)user_state;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed progress monitor callback creation");
#else
    (void)user_state;
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed progress monitor callback creation");
#endif
}

JYPPX_StatusCode jyppx_trt10_progress_monitor_emit_diagnostic(
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

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
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
#elif JYPPX_HAS_TENSORRT
    (void)event_kind;
    (void)parent_phase;
    (void)step;
    (void)nb_steps;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "managed progress monitor diagnostic emission");
#else
    (void)event_kind;
    (void)parent_phase;
    (void)step;
    (void)nb_steps;
    return jyppx::tensorrt::report_vendor_missing(kLine, "managed progress monitor diagnostic emission");
#endif
}

JYPPX_StatusCode jyppx_trt10_progress_monitor_get_interface_info(
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

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
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

JYPPX_StatusCode jyppx_trt10_progress_monitor_get_api_language(
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

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
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

JYPPX_StatusCode jyppx_trt10_profiler_create_with_callback(
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
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
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

JYPPX_StatusCode jyppx_trt10_profiler_emit_diagnostic(
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
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
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

JYPPX_StatusCode jyppx_trt10_runtime_create(JYPPX_TensorRtLogger* logger, JYPPX_TensorRtRuntime** out_runtime)
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
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "runtime creation");
    }

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

JYPPX_StatusCode jyppx_trt10_builder_create(JYPPX_TensorRtLogger* logger, JYPPX_TensorRtBuilder** out_builder)
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
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder creation");
    }

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

JYPPX_StatusCode jyppx_trt10_builder_platform_has_fast_fp16(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_supported)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_supported, "out_supported");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_supported = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder FP16 platform capability query");
    }

    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_supported = builder_payload->platformHasFastFp16() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder FP16 platform capability query");
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_platform_has_fast_int8(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_supported)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_supported, "out_supported");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_supported = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder INT8 platform capability query");
    }

    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_supported = builder_payload->platformHasFastInt8() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder INT8 platform capability query");
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_platform_has_tf32(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_supported)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_supported, "out_supported");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_supported = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder TF32 platform capability query");
    }

    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_supported = builder_payload->platformHasTf32() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder TF32 platform capability query");
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_get_dla_core_count(JYPPX_TensorRtBuilder* builder, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder DLA core count query");
    }

    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = builder_payload->getNbDLACores();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder DLA core count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_get_max_dla_batch_size(JYPPX_TensorRtBuilder* builder, int32_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_size = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder maximum DLA batch size query");
    }

    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_size = builder_payload->getMaxDLABatchSize();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder maximum DLA batch size query");
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_set_max_threads(JYPPX_TensorRtBuilder* builder, int32_t max_threads, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = JYPPX_FALSE;
    if (max_threads < 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder max_threads must be greater than or equal to 1.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder max threads set");
    }

    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_set = builder_payload->setMaxThreads(max_threads) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)max_threads;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder max threads set");
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_get_max_threads(JYPPX_TensorRtBuilder* builder, int32_t* out_max_threads)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_max_threads, "out_max_threads");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_max_threads = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder max threads query");
    }

    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_max_threads = builder_payload->getMaxThreads();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder max threads query");
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_is_network_supported(
    JYPPX_TensorRtBuilder* builder,
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtBuilderConfig* config,
    JYPPX_Boolean* out_supported)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_supported, "out_supported");
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

    *out_supported = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder network support query");
    }

    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (builder_payload == nullptr || network_payload == nullptr || config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder, network, or builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        *out_supported = builder_payload->isNetworkSupported(*network_payload, *config_payload) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        *out_supported = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(kLine, "builder network support query", exception.what());
    }
    catch (...)
    {
        *out_supported = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(kLine, "builder network support query", "unknown native exception");
    }
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder network support query");
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_create_config(JYPPX_TensorRtBuilder* builder, JYPPX_TensorRtBuilderConfig** out_config)
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
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config creation");
    }

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

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
JYPPX_StatusCode jyppx_trt10_builder_config_progress_monitor_exception(const char* feature_name, const std::exception& exception)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
}

JYPPX_StatusCode jyppx_trt10_builder_config_progress_monitor_unknown_exception(const char* feature_name)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
}

JYPPX_StatusCode jyppx_trt10_builder_config_set_progress_monitor_with_seh_guard(
    nvinfer1::IBuilderConfig* config_payload,
    ManagedProgressMonitor* monitor_payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        config_payload->setProgressMonitor(monitor_payload);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    config_payload->setProgressMonitor(monitor_payload);
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_config_has_progress_monitor_with_seh_guard(
    nvinfer1::IBuilderConfig* config_payload,
    JYPPX_Boolean* out_has_monitor,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_monitor = config_payload->getProgressMonitor() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_monitor = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_monitor = config_payload->getProgressMonitor() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_config_clear_progress_monitor_with_seh_guard(
    nvinfer1::IBuilderConfig* config_payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        config_payload->setProgressMonitor(nullptr);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    config_payload->setProgressMonitor(nullptr);
    return JYPPX_STATUS_OK;
#endif
}
#endif

JYPPX_StatusCode jyppx_trt10_builder_config_set_progress_monitor(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtProgressMonitor* monitor)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(monitor, kLine, JYPPX_TENSORRT_OBJECT_KIND_PROGRESS_MONITOR, "monitor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* monitor_payload = get_payload<ManagedProgressMonitor>(monitor);
    if (config_payload == nullptr || monitor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config or progress monitor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        return jyppx_trt10_builder_config_set_progress_monitor_with_seh_guard(
            config_payload,
            monitor_payload,
            "builder config progress monitor attach");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt10_builder_config_progress_monitor_exception("builder config progress monitor attach", exception);
    }
    catch (...)
    {
        return jyppx_trt10_builder_config_progress_monitor_unknown_exception("builder config progress monitor attach");
    }
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config progress monitor attach");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config progress monitor attach");
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_config_has_progress_monitor(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_has_monitor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_monitor, "out_has_monitor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_monitor = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry a TensorRT builder config payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        return jyppx_trt10_builder_config_has_progress_monitor_with_seh_guard(
            config_payload,
            out_has_monitor,
            "builder config progress monitor query");
    }
    catch (const std::exception& exception)
    {
        *out_has_monitor = JYPPX_FALSE;
        return jyppx_trt10_builder_config_progress_monitor_exception("builder config progress monitor query", exception);
    }
    catch (...)
    {
        *out_has_monitor = JYPPX_FALSE;
        return jyppx_trt10_builder_config_progress_monitor_unknown_exception("builder config progress monitor query");
    }
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config progress monitor query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config progress monitor query");
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_config_clear_progress_monitor(JYPPX_TensorRtBuilderConfig* config)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry a TensorRT builder config payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        return jyppx_trt10_builder_config_clear_progress_monitor_with_seh_guard(
            config_payload,
            "builder config progress monitor clear");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt10_builder_config_progress_monitor_exception("builder config progress monitor clear", exception);
    }
    catch (...)
    {
        return jyppx_trt10_builder_config_progress_monitor_unknown_exception("builder config progress monitor clear");
    }
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config progress monitor clear");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config progress monitor clear");
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_create_network(JYPPX_TensorRtBuilder* builder, uint32_t creation_flags, JYPPX_TensorRtNetworkDefinition** out_network)
{
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
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "network definition creation");
    }

    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::NetworkDefinitionCreationFlags flags = static_cast<nvinfer1::NetworkDefinitionCreationFlags>(creation_flags);
    nvinfer1::INetworkDefinition* network = builder_payload->createNetworkV2(flags);
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

JYPPX_StatusCode jyppx_trt10_builder_build_serialized_network(JYPPX_TensorRtBuilder* builder, JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtHostMemory** out_host_memory)
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
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "serialized network build");
    }

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

JYPPX_StatusCode jyppx_trt10_host_memory_get_size(JYPPX_TensorRtHostMemory* host_memory, size_t* out_size)
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
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "host memory size query");
    }

    auto* host_memory_payload = get_payload<nvinfer1::IHostMemory>(host_memory);
    if (host_memory_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Host memory handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_size = host_memory_payload->size();
    return JYPPX_STATUS_OK;
#else
    *out_size = 0;
    return jyppx::tensorrt::report_vendor_missing(kLine, "host memory size query");
#endif
}

JYPPX_StatusCode jyppx_trt10_host_memory_copy_to_buffer(JYPPX_TensorRtHostMemory* host_memory, void* destination, size_t destination_size, size_t* out_bytes_written)
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

    if (destination == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Destination buffer for host memory copy must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "host memory data copy");
    }

    auto* host_memory_payload = get_payload<nvinfer1::IHostMemory>(host_memory);
    if (host_memory_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Host memory handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const size_t size = host_memory_payload->size();
    if (destination_size < size)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Destination buffer is too small for TensorRT host memory.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    std::memcpy(destination, host_memory_payload->data(), size);
    *out_bytes_written = size;
    return JYPPX_STATUS_OK;
#else
    (void)destination;
    (void)destination_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "host memory data copy");
#endif
}

JYPPX_StatusCode jyppx_trt10_runtime_deserialize_engine(JYPPX_TensorRtRuntime* runtime, const void* engine_data, size_t engine_size, JYPPX_TensorRtCudaEngine** out_engine)
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
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine data buffer must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_engine = nullptr;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine deserialization");
    }

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

JYPPX_StatusCode jyppx_trt10_runtime_deserialize_host_memory(JYPPX_TensorRtRuntime* runtime, JYPPX_TensorRtHostMemory* host_memory, JYPPX_TensorRtCudaEngine** out_engine)
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

    status = jyppx::tensorrt::validate_handle(host_memory, kLine, JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY, "host_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_engine = nullptr;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine deserialization from host memory");
    }

    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    auto* host_memory_payload = get_payload<nvinfer1::IHostMemory>(host_memory);
    if (runtime_payload == nullptr || host_memory_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime or host memory handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return jyppx_trt10_runtime_deserialize_engine(
        runtime,
        host_memory_payload->data(),
        host_memory_payload->size(),
        out_engine);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine deserialization from host memory");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_serialize(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtHostMemory** out_host_memory)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_host_memory, "out_host_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_host_memory = nullptr;

#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine serialize");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::IHostMemory* host_memory = engine_payload->serialize();
    if (host_memory == nullptr)
    {
        return report_null_vendor_object("ICudaEngine::serialize");
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
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine serialize");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_create_serialization_config(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtSerializationConfig** out_config)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_config, "out_config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_config = nullptr;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine serialization config creation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::ISerializationConfig* config = engine_payload->createSerializationConfig();
    if (config == nullptr)
    {
        return report_null_vendor_object("ICudaEngine::createSerializationConfig");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, config, &destroy_payload<nvinfer1::ISerializationConfig>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_config = reinterpret_cast<JYPPX_TensorRtSerializationConfig*>(handle);
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine serialization config creation");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_serialize_with_config(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtSerializationConfig* config, JYPPX_TensorRtHostMemory** out_host_memory)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_host_memory, "out_host_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_host_memory = nullptr;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine serialize with config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* config_payload = get_payload<nvinfer1::ISerializationConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Serialization config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IHostMemory* host_memory = engine_payload->serializeWithConfig(*config_payload);
    if (host_memory == nullptr)
    {
        return report_null_vendor_object("ICudaEngine::serializeWithConfig");
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
    (void)engine;
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine serialize with config");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_create_runtime_config(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtRuntimeConfig** out_config)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_config, "out_config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_config = nullptr;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine runtime config creation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::IRuntimeConfig* config = engine_payload->createRuntimeConfig();
    if (config == nullptr)
    {
        return report_null_vendor_object("ICudaEngine::createRuntimeConfig");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME_CONFIG, config, &destroy_payload<nvinfer1::IRuntimeConfig>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_config = reinterpret_cast<JYPPX_TensorRtRuntimeConfig*>(handle);
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine runtime config creation");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_create_execution_context_with_runtime_config(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtRuntimeConfig* runtime_config, JYPPX_TensorRtExecutionContext** out_context)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_context, "out_context");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_context = nullptr;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "execution context creation with runtime config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(runtime_config, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME_CONFIG, "runtime_config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* config_payload = get_payload<nvinfer1::IRuntimeConfig>(runtime_config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IExecutionContext* context = engine_payload->createExecutionContext(config_payload);
    if (context == nullptr)
    {
        return report_null_vendor_object("ICudaEngine::createExecutionContext(runtimeConfig)");
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
    (void)engine;
    (void)runtime_config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context creation with runtime config");
#endif
}

JYPPX_StatusCode jyppx_trt10_serialization_config_set_flags(JYPPX_TensorRtSerializationConfig* config, uint32_t flags, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* config_payload = get_payload<nvinfer1::ISerializationConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Serialization config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_set = config_payload->setFlags(static_cast<nvinfer1::SerializationFlags>(flags)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)flags;
    return jyppx::tensorrt::report_vendor_missing(kLine, "serialization config flags set");
#endif
}

JYPPX_StatusCode jyppx_trt10_serialization_config_get_flags(JYPPX_TensorRtSerializationConfig* config, uint32_t* out_flags)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = 0;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* config_payload = get_payload<nvinfer1::ISerializationConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Serialization config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_flags = static_cast<uint32_t>(config_payload->getFlags());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "serialization config flags query");
#endif
}

JYPPX_StatusCode jyppx_trt10_serialization_config_set_flag(JYPPX_TensorRtSerializationConfig* config, int32_t flag, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = JYPPX_FALSE;
    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Serialization flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* config_payload = get_payload<nvinfer1::ISerializationConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Serialization config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_set = config_payload->setFlag(static_cast<nvinfer1::SerializationFlag>(flag)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "serialization config flag set");
#endif
}

JYPPX_StatusCode jyppx_trt10_serialization_config_clear_flag(JYPPX_TensorRtSerializationConfig* config, int32_t flag, JYPPX_Boolean* out_cleared)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cleared, "out_cleared");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_cleared = JYPPX_FALSE;
    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Serialization flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* config_payload = get_payload<nvinfer1::ISerializationConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Serialization config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_cleared = config_payload->clearFlag(static_cast<nvinfer1::SerializationFlag>(flag)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "serialization config flag clear");
#endif
}

JYPPX_StatusCode jyppx_trt10_serialization_config_get_flag(JYPPX_TensorRtSerializationConfig* config, int32_t flag, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_enabled = JYPPX_FALSE;
    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Serialization flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* config_payload = get_payload<nvinfer1::ISerializationConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Serialization config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_enabled = config_payload->getFlag(static_cast<nvinfer1::SerializationFlag>(flag)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "serialization config flag query");
#endif
}

JYPPX_StatusCode jyppx_trt10_runtime_config_set_execution_context_allocation_strategy(JYPPX_TensorRtRuntimeConfig* config, int32_t strategy)
{
    if (strategy < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context allocation strategy must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* config_payload = get_payload<nvinfer1::IRuntimeConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setExecutionContextAllocationStrategy(static_cast<nvinfer1::ExecutionContextAllocationStrategy>(strategy));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime config allocation strategy set");
#endif
}

JYPPX_StatusCode jyppx_trt10_runtime_config_get_execution_context_allocation_strategy(JYPPX_TensorRtRuntimeConfig* config, int32_t* out_strategy)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_strategy, "out_strategy");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_strategy = 0;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* config_payload = get_payload<nvinfer1::IRuntimeConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_strategy = static_cast<int32_t>(config_payload->getExecutionContextAllocationStrategy());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime config allocation strategy query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_streamable_weights_size(JYPPX_TensorRtCudaEngine* engine, int64_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_size = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine streamable weights size query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_size = engine_payload->getStreamableWeightsSize();
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine streamable weights size query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_minimum_weight_streaming_budget(JYPPX_TensorRtCudaEngine* engine, int64_t* out_budget)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_budget, "out_budget");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_budget = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine minimum weight streaming budget query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    *out_budget = engine_payload->getMinimumWeightStreamingBudget();
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine minimum weight streaming budget query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_set_weight_streaming_budget_v2(JYPPX_TensorRtCudaEngine* engine, int64_t budget, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine weight streaming budget set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = engine_payload->setWeightStreamingBudgetV2(budget) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    (void)budget;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine weight streaming budget set");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_weight_streaming_budget_v2(JYPPX_TensorRtCudaEngine* engine, int64_t* out_budget)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_budget, "out_budget");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_budget = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine weight streaming budget query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_budget = engine_payload->getWeightStreamingBudgetV2();
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine weight streaming budget query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_weight_streaming_automatic_budget(JYPPX_TensorRtCudaEngine* engine, int64_t* out_budget)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_budget, "out_budget");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_budget = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine weight streaming automatic budget query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_budget = engine_payload->getWeightStreamingAutomaticBudget();
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine weight streaming automatic budget query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_weight_streaming_scratch_memory_size(JYPPX_TensorRtCudaEngine* engine, int64_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_size = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine weight streaming scratch memory query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_size = engine_payload->getWeightStreamingScratchMemorySize();
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine weight streaming scratch memory query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_hardware_compatibility_level(JYPPX_TensorRtCudaEngine* engine, int32_t* out_level)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_level, "out_level");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_level = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine hardware compatibility level query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_level = static_cast<int32_t>(engine_payload->getHardwareCompatibilityLevel());
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine hardware compatibility level query");
#endif
}

JYPPX_StatusCode jyppx_trt10_cuda_engine_has_implicit_batch_dimension(JYPPX_TensorRtCudaEngine* engine, JYPPX_Boolean* out_has_implicit_batch)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_implicit_batch, "out_has_implicit_batch");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_implicit_batch = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine implicit batch compatibility query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_implicit_batch = engine_payload->hasImplicitBatchDimension() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine implicit batch compatibility query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_io_tensor_count(JYPPX_TensorRtCudaEngine* engine, int32_t* out_count)
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

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine handle does not carry a TensorRT engine payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = engine_payload->getNbIOTensors();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    *out_count = 0;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine I/O tensor count query");
#else
    *out_count = 0;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine I/O tensor count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_io_tensor_info(JYPPX_TensorRtCudaEngine* engine, int32_t index, JYPPX_TensorRtTensorInfo* out_info)
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

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
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
#elif JYPPX_HAS_TENSORRT
    std::memset(out_info, 0, sizeof(*out_info));
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine I/O tensor info query");
#else
    std::memset(out_info, 0, sizeof(*out_info));
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine I/O tensor info query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_device_memory_size(JYPPX_TensorRtCudaEngine* engine, size_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_size = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine handle does not carry a TensorRT engine payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int64_t size = engine_payload->getDeviceMemorySizeV2();
    if (size < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT reported a negative device memory size.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    *out_size = static_cast<size_t>(size);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine device memory size query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine device memory size query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_device_memory_size_for_profile(JYPPX_TensorRtCudaEngine* engine, int32_t profile_index, size_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_size = 0;
    if (profile_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine profile device memory size query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (profile_index >= engine_payload->getNbOptimizationProfiles())
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_size = engine_payload->getDeviceMemorySizeForProfile(profile_index);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine profile device memory size query");
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine profile device memory size query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_device_memory_size_v2(JYPPX_TensorRtCudaEngine* engine, size_t* out_size)
{
    return jyppx_trt10_engine_get_device_memory_size(engine, out_size);
}

JYPPX_StatusCode jyppx_trt10_engine_get_device_memory_size_for_profile_v2(JYPPX_TensorRtCudaEngine* engine, int32_t profile_index, size_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_size = 0;
    if (profile_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine profile V2 device memory size query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (profile_index >= engine_payload->getNbOptimizationProfiles())
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const int64_t size = engine_payload->getDeviceMemorySizeForProfileV2(profile_index);
    if (size < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT reported a negative profile device memory size.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    *out_size = static_cast<size_t>(size);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine profile V2 device memory size query");
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine profile V2 device memory size query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_nb_aux_streams(JYPPX_TensorRtCudaEngine* engine, int32_t* out_stream_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_stream_count, "out_stream_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_stream_count = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine auxiliary stream count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_stream_count = engine_payload->getNbAuxStreams();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine auxiliary stream count query");
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine auxiliary stream count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_is_debug_tensor(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, JYPPX_Boolean* out_is_debug_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_debug_tensor, "out_is_debug_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_debug_tensor = JYPPX_FALSE;
    status = validate_c_string(tensor_name, "tensor_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine debug tensor query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_debug_tensor = engine_payload->isDebugTensor(tensor_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)engine;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine debug tensor query");
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine debug tensor query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_optimization_profile_count(JYPPX_TensorRtCudaEngine* engine, int32_t* out_count)
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

    *out_count = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine handle does not carry a TensorRT engine payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = engine_payload->getNbOptimizationProfiles();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine optimization profile count query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine optimization profile count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_io_tensor_name(
    JYPPX_TensorRtCudaEngine* engine,
    int32_t index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
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

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
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

    return copy_string_to_buffer(engine_payload->getIOTensorName(index), output_buffer, output_buffer_size, out_required_size);
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine I/O tensor name query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine I/O tensor name query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_tensor_index(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_index)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_index, "out_index");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(tensor_name, "tensor_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_index = -1;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine handle does not carry a TensorRT engine payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int32_t count = engine_payload->getNbIOTensors();
    for (int32_t index = 0; index < count; ++index)
    {
        const char* name = engine_payload->getIOTensorName(index);
        if (name != nullptr && std::strcmp(name, tensor_name) == 0)
        {
            *out_index = index;
            return JYPPX_STATUS_OK;
        }
    }

    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor name was not found in the engine I/O tensor list.");
    return JYPPX_STATUS_INVALID_ARGUMENT;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine tensor index query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tensor index query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_tensor_data_type(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(tensor_name, "tensor_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_data_type = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine handle does not carry a TensorRT engine payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_data_type = static_cast<int32_t>(engine_payload->getTensorDataType(tensor_name));
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine tensor data type query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tensor data type query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_tensor_shape(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, JYPPX_TensorRtDims* out_shape)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_shape, "out_shape");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(tensor_name, "tensor_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_shape, 0, sizeof(*out_shape));

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine handle does not carry a TensorRT engine payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_dims(engine_payload->getTensorShape(tensor_name), out_shape);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine tensor shape query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tensor shape query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_tensor_io_mode(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* out_io_mode)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_io_mode, "out_io_mode");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(tensor_name, "tensor_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_io_mode = JYPPX_TENSORRT_IO_MODE_UNKNOWN;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine handle does not carry a TensorRT engine payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const nvinfer1::TensorIOMode mode = engine_payload->getTensorIOMode(tensor_name);
    *out_io_mode = mode == nvinfer1::TensorIOMode::kINPUT
        ? JYPPX_TENSORRT_IO_MODE_INPUT
        : (mode == nvinfer1::TensorIOMode::kOUTPUT ? JYPPX_TENSORRT_IO_MODE_OUTPUT : JYPPX_TENSORRT_IO_MODE_UNKNOWN);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine tensor I/O mode query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tensor I/O mode query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_create_execution_context(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtExecutionContext** out_context)
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
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context creation");
    }

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

JYPPX_StatusCode jyppx_trt10_engine_create_execution_context_without_device_memory(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtExecutionContext** out_context)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_context, "out_context");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_context = nullptr;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "execution context without device memory creation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::IExecutionContext* context = engine_payload->createExecutionContextWithoutDeviceMemory();
    if (context == nullptr)
    {
        return report_null_vendor_object("createExecutionContextWithoutDeviceMemory");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, context, &destroy_payload<nvinfer1::IExecutionContext>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_context = reinterpret_cast<JYPPX_TensorRtExecutionContext*>(handle);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context without device memory creation");
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context without device memory creation");
#endif
}

JYPPX_StatusCode jyppx_trt10_builder_create_optimization_profile(JYPPX_TensorRtBuilder* builder, JYPPX_TensorRtOptimizationProfile** out_profile)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_profile, "out_profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_profile = nullptr;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "optimization profile creation");
    }

    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IOptimizationProfile* profile = builder_payload->createOptimizationProfile();
    if (profile == nullptr)
    {
        return report_null_vendor_object("createOptimizationProfile");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, profile, nullptr);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_profile = reinterpret_cast<JYPPX_TensorRtOptimizationProfile*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile creation");
#endif
}

JYPPX_StatusCode jyppx_trt10_optimization_profile_set_shape(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, const JYPPX_TensorRtDims* dims)
{
    auto status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "optimization profile shape set");
    }

    if (selector < 0 || selector > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile selector must be 0 (min), 1 (opt), or 2 (max).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    nvinfer1::Dims native_dims{};
    status = make_dims(dims, &native_dims, "dims");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const bool result = profile_payload->setDimensions(input_name, static_cast<nvinfer1::OptProfileSelector>(selector), native_dims);
    if (!result)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IOptimizationProfile::setDimensions returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape set");
#endif
}

JYPPX_StatusCode jyppx_trt10_optimization_profile_get_shape(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, JYPPX_TensorRtDims* out_dims)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dims, "out_dims");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (selector < 0 || selector > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile selector must be 0 (min), 1 (opt), or 2 (max).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    out_dims->nb_dims = 0;
    for (int32_t i = 0; i < 8; ++i)
    {
        out_dims->d[i] = 0;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "optimization profile shape query");
    }

    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_dims(profile_payload->getDimensions(input_name, static_cast<nvinfer1::OptProfileSelector>(selector)), out_dims);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape query");
#endif
}

#include "modules/builder/optimization_profile.inc"
#include "modules/builder/builder_config.inc"

#include "modules/network/network_tensor.inc"

JYPPX_StatusCode jyppx_trt10_network_add_identity(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IIdentityLayer* layer = network_payload->addIdentity(*tensor_payload);
    return create_layer_reference_handle(layer, out_layer);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add identity");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_constant(JYPPX_TensorRtNetworkDefinition* network, const JYPPX_TensorRtDims* dims, int32_t data_type, const void* values, size_t value_count, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_dims{};
    status = make_dims(dims, &native_dims, "dims");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    size_t expected_count = 1;
    for (int32_t index = 0; index < native_dims.nbDims; ++index)
    {
        if (native_dims.d[index] < 0)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Constant layer dimensions must be static.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        expected_count *= static_cast<size_t>(native_dims.d[index]);
    }

    if (expected_count != value_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Constant weight count must match the product of constant dimensions.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    std::vector<uint8_t> owned_weights;
    status = copy_weights(data_type, values, value_count, &owned_weights);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::Weights weights{static_cast<nvinfer1::DataType>(data_type), owned_weights.data(), static_cast<int64_t>(value_count)};
    nvinfer1::IConstantLayer* layer = network_payload->addConstant(native_dims, weights);
    return create_layer_reference_handle(layer, out_layer, std::move(owned_weights));
#else
    (void)dims;
    (void)data_type;
    (void)values;
    (void)value_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add constant");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_elementwise(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* left_tensor, JYPPX_TensorRtTensor* right_tensor, int32_t operation, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(left_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "left_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(right_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "right_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (operation < 0 || operation > 13)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ElementWise operation must be in the TensorRT enum range [0, 13].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* left_payload = get_payload<nvinfer1::ITensor>(left_tensor);
    auto* right_payload = get_payload<nvinfer1::ITensor>(right_tensor);
    if (network_payload == nullptr || left_payload == nullptr || right_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IElementWiseLayer* layer = network_payload->addElementWise(*left_payload, *right_payload, static_cast<nvinfer1::ElementWiseOperation>(operation));
    return create_layer_reference_handle(layer, out_layer);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add elementwise");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_matrix_multiply(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* left_tensor,
    int32_t left_operation,
    JYPPX_TensorRtTensor* right_tensor,
    int32_t right_operation,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(left_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "left_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(right_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "right_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (left_operation < 0 || left_operation > 2 || right_operation < 0 || right_operation > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Matrix operation must be in the TensorRT enum range [0, 2].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* left_payload = get_payload<nvinfer1::ITensor>(left_tensor);
    auto* right_payload = get_payload<nvinfer1::ITensor>(right_tensor);
    if (network_payload == nullptr || left_payload == nullptr || right_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    auto* layer = network_payload->addMatrixMultiply(
        *left_payload,
        static_cast<nvinfer1::MatrixOperation>(left_operation),
        *right_payload,
        static_cast<nvinfer1::MatrixOperation>(right_operation));
    return create_layer_reference_handle(layer, out_layer);
#else
    (void)left_operation;
    (void)right_operation;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add matrix multiply");
#endif
}

JYPPX_StatusCode jyppx_trt10_matrix_multiply_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t input_index, int32_t operation)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (input_index < 0 || input_index > 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Matrix multiply input index must be 0 or 1.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (operation < 0 || operation > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Matrix operation must be in the TensorRT enum range [0, 2].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto* matrix_layer = get_matrix_multiply_layer_payload(layer);
    if (matrix_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT matrix multiply layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    matrix_layer->setOperation(input_index, static_cast<nvinfer1::MatrixOperation>(operation));
    return JYPPX_STATUS_OK;
#else
    (void)input_index;
    (void)operation;
    return jyppx::tensorrt::report_vendor_missing(kLine, "matrix multiply layer operation set");
#endif
}

JYPPX_StatusCode jyppx_trt10_matrix_multiply_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t input_index, int32_t* out_operation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_operation, "out_operation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (input_index < 0 || input_index > 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Matrix multiply input index must be 0 or 1.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_operation = 0;

#if JYPPX_HAS_TENSORRT
    auto* matrix_layer = get_matrix_multiply_layer_payload(layer);
    if (matrix_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT matrix multiply layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_operation = static_cast<int32_t>(matrix_layer->getOperation(input_index));
    return JYPPX_STATUS_OK;
#else
    (void)input_index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "matrix multiply layer operation query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_shuffle(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IShuffleLayer* layer = network_payload->addShuffle(*tensor_payload);
    return create_layer_reference_handle(layer, out_layer);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add shuffle");
#endif
}

JYPPX_StatusCode jyppx_trt10_shuffle_layer_set_reshape_dimensions(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* dims)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_dims{};
    status = make_dims(dims, &native_dims, "dims");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* shuffle_layer = get_shuffle_layer_payload(layer);
    if (shuffle_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT shuffle layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    shuffle_layer->setReshapeDimensions(native_dims);
    return JYPPX_STATUS_OK;
#else
    (void)dims;
    return jyppx::tensorrt::report_vendor_missing(kLine, "shuffle layer reshape set");
#endif
}

JYPPX_StatusCode jyppx_trt10_shuffle_layer_get_reshape_dimensions(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_dims)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dims, "out_dims");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_dims, 0, sizeof(*out_dims));

#if JYPPX_HAS_TENSORRT
    auto* shuffle_layer = get_shuffle_layer_payload(layer);
    if (shuffle_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT shuffle layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_dims(shuffle_layer->getReshapeDimensions(), out_dims);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "shuffle layer reshape query");
#endif
}

JYPPX_StatusCode jyppx_trt10_shuffle_layer_set_first_transpose(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* permutation)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Permutation native_permutation{};
    status = make_permutation(permutation, &native_permutation, "permutation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* shuffle_layer = get_shuffle_layer_payload(layer);
    if (shuffle_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT shuffle layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    shuffle_layer->setFirstTranspose(native_permutation);
    return JYPPX_STATUS_OK;
#else
    (void)permutation;
    return jyppx::tensorrt::report_vendor_missing(kLine, "shuffle layer first transpose set");
#endif
}

JYPPX_StatusCode jyppx_trt10_shuffle_layer_get_first_transpose(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_permutation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_permutation, "out_permutation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_permutation, 0, sizeof(*out_permutation));

#if JYPPX_HAS_TENSORRT
    auto* shuffle_layer = get_shuffle_layer_payload(layer);
    if (shuffle_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT shuffle layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_permutation(shuffle_layer->getFirstTranspose(), get_layer_input_rank(shuffle_layer), out_permutation);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "shuffle layer first transpose query");
#endif
}

JYPPX_StatusCode jyppx_trt10_shuffle_layer_set_second_transpose(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* permutation)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Permutation native_permutation{};
    status = make_permutation(permutation, &native_permutation, "permutation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* shuffle_layer = get_shuffle_layer_payload(layer);
    if (shuffle_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT shuffle layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    shuffle_layer->setSecondTranspose(native_permutation);
    return JYPPX_STATUS_OK;
#else
    (void)permutation;
    return jyppx::tensorrt::report_vendor_missing(kLine, "shuffle layer second transpose set");
#endif
}

JYPPX_StatusCode jyppx_trt10_shuffle_layer_get_second_transpose(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_permutation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_permutation, "out_permutation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_permutation, 0, sizeof(*out_permutation));

#if JYPPX_HAS_TENSORRT
    auto* shuffle_layer = get_shuffle_layer_payload(layer);
    if (shuffle_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT shuffle layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_permutation(shuffle_layer->getSecondTranspose(), get_shuffle_second_transpose_rank(shuffle_layer), out_permutation);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "shuffle layer second transpose query");
#endif
}

JYPPX_StatusCode jyppx_trt10_shuffle_layer_set_zero_is_placeholder(JYPPX_TensorRtLayer* layer, JYPPX_Boolean zero_is_placeholder)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* shuffle_layer = get_shuffle_layer_payload(layer);
    if (shuffle_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT shuffle layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    shuffle_layer->setZeroIsPlaceholder(zero_is_placeholder != 0);
    return JYPPX_STATUS_OK;
#else
    (void)zero_is_placeholder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "shuffle layer zero-is-placeholder set");
#endif
}

JYPPX_StatusCode jyppx_trt10_shuffle_layer_get_zero_is_placeholder(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_zero_is_placeholder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_zero_is_placeholder, "out_zero_is_placeholder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_zero_is_placeholder = 0;

#if JYPPX_HAS_TENSORRT
    auto* shuffle_layer = get_shuffle_layer_payload(layer);
    if (shuffle_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT shuffle layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_zero_is_placeholder = shuffle_layer->getZeroIsPlaceholder() ? 1 : 0;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "shuffle layer zero-is-placeholder query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_reduce(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t operation, uint32_t axes, int32_t keep_dimensions, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (operation < 0 || operation > 4)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Reduce operation must be in the TensorRT enum range [0, 4].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (axes == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Reduce axes bitmask must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IReduceLayer* layer = network_payload->addReduce(*tensor_payload, static_cast<nvinfer1::ReduceOperation>(operation), axes, keep_dimensions != 0);
    return create_layer_reference_handle(layer, out_layer, {}, operation, axes, keep_dimensions != 0 ? 1 : 0);
#else
    (void)operation;
    (void)axes;
    (void)keep_dimensions;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add reduce");
#endif
}

JYPPX_StatusCode jyppx_trt10_reduce_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t* out_operation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_operation, "out_operation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_operation = 0;

#if JYPPX_HAS_TENSORRT
    auto* payload = get_layer_reference_payload(layer);
    if (payload == nullptr || payload->layer == nullptr || payload->layer->getType() != nvinfer1::LayerType::kREDUCE)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT reduce layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_operation = payload->reduce_operation;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "reduce layer operation query");
#endif
}

JYPPX_StatusCode jyppx_trt10_reduce_layer_get_axes(JYPPX_TensorRtLayer* layer, uint32_t* out_axes)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_axes, "out_axes");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_axes = 0;

#if JYPPX_HAS_TENSORRT
    auto* payload = get_layer_reference_payload(layer);
    if (payload == nullptr || payload->layer == nullptr || payload->layer->getType() != nvinfer1::LayerType::kREDUCE)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT reduce layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_axes = payload->reduce_axes;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "reduce layer axes query");
#endif
}

JYPPX_StatusCode jyppx_trt10_reduce_layer_get_keep_dimensions(JYPPX_TensorRtLayer* layer, int32_t* out_keep_dimensions)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_keep_dimensions, "out_keep_dimensions");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_keep_dimensions = 0;

#if JYPPX_HAS_TENSORRT
    auto* payload = get_layer_reference_payload(layer);
    if (payload == nullptr || payload->layer == nullptr || payload->layer->getType() != nvinfer1::LayerType::kREDUCE)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT reduce layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_keep_dimensions = payload->reduce_keep_dimensions;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "reduce layer keep dimensions query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_concatenation(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor** input_tensors, int32_t input_count, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (input_tensors == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Concatenation input tensor array must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (input_count < 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Concatenation requires at least two input tensors.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    for (int32_t index = 0; index < input_count; ++index)
    {
        status = jyppx::tensorrt::validate_handle(input_tensors[index], kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensors[]");
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    std::vector<nvinfer1::ITensor*> native_tensors(static_cast<size_t>(input_count));
    for (int32_t index = 0; index < input_count; ++index)
    {
        native_tensors[static_cast<size_t>(index)] = get_payload<nvinfer1::ITensor>(input_tensors[index]);
        if (native_tensors[static_cast<size_t>(index)] == nullptr)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Input tensor handle does not carry the expected TensorRT payload.");
            return JYPPX_STATUS_INVALID_STATE;
        }
    }

    nvinfer1::IConcatenationLayer* layer = network_payload->addConcatenation(native_tensors.data(), input_count);
    return create_layer_reference_handle(layer, out_layer);
#else
    (void)input_tensors;
    (void)input_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add concatenation");
#endif
}

JYPPX_StatusCode jyppx_trt10_concatenation_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (axis < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Concatenation axis must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto* concatenation_layer = get_concatenation_layer_payload(layer);
    if (concatenation_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT concatenation layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    concatenation_layer->setAxis(axis);
    return JYPPX_STATUS_OK;
#else
    (void)axis;
    return jyppx::tensorrt::report_vendor_missing(kLine, "concatenation layer axis set");
#endif
}

JYPPX_StatusCode jyppx_trt10_concatenation_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_axis, "out_axis");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_axis = 0;

#if JYPPX_HAS_TENSORRT
    auto* concatenation_layer = get_concatenation_layer_payload(layer);
    if (concatenation_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT concatenation layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_axis = concatenation_layer->getAxis();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "concatenation layer axis query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_slice(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    const JYPPX_TensorRtDims* start,
    const JYPPX_TensorRtDims* size,
    const JYPPX_TensorRtDims* stride,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_start{};
    nvinfer1::Dims native_size{};
    nvinfer1::Dims native_stride{};
    status = make_dims(start, &native_start, "start");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = make_dims(size, &native_size, "size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = make_dims(stride, &native_stride, "stride");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (native_start.nbDims != native_size.nbDims || native_start.nbDims != native_stride.nbDims)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Slice start, size, and stride dimensions must have the same rank.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::ISliceLayer* layer = network_payload->addSlice(*tensor_payload, native_start, native_size, native_stride);
    return create_layer_reference_handle(layer, out_layer);
#else
    (void)start;
    (void)size;
    (void)stride;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add slice");
#endif
}

JYPPX_StatusCode jyppx_trt10_slice_layer_set_start(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* start)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_start{};
    status = make_dims(start, &native_start, "start");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* slice_layer = get_slice_layer_payload(layer);
    if (slice_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT slice layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    slice_layer->setStart(native_start);
    return JYPPX_STATUS_OK;
#else
    (void)start;
    return jyppx::tensorrt::report_vendor_missing(kLine, "slice layer start set");
#endif
}

JYPPX_StatusCode jyppx_trt10_slice_layer_get_start(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_start)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_start, "out_start");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_start, 0, sizeof(*out_start));

#if JYPPX_HAS_TENSORRT
    auto* slice_layer = get_slice_layer_payload(layer);
    if (slice_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT slice layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_dims(slice_layer->getStart(), out_start);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "slice layer start query");
#endif
}

JYPPX_StatusCode jyppx_trt10_slice_layer_set_size(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* size)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_size{};
    status = make_dims(size, &native_size, "size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* slice_layer = get_slice_layer_payload(layer);
    if (slice_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT slice layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    slice_layer->setSize(native_size);
    return JYPPX_STATUS_OK;
#else
    (void)size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "slice layer size set");
#endif
}

JYPPX_StatusCode jyppx_trt10_slice_layer_get_size(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_size, 0, sizeof(*out_size));

#if JYPPX_HAS_TENSORRT
    auto* slice_layer = get_slice_layer_payload(layer);
    if (slice_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT slice layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_dims(slice_layer->getSize(), out_size);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "slice layer size query");
#endif
}

JYPPX_StatusCode jyppx_trt10_slice_layer_set_stride(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* stride)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_stride{};
    status = make_dims(stride, &native_stride, "stride");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* slice_layer = get_slice_layer_payload(layer);
    if (slice_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT slice layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    slice_layer->setStride(native_stride);
    return JYPPX_STATUS_OK;
#else
    (void)stride;
    return jyppx::tensorrt::report_vendor_missing(kLine, "slice layer stride set");
#endif
}

JYPPX_StatusCode jyppx_trt10_slice_layer_get_stride(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_stride)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_stride, "out_stride");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_stride, 0, sizeof(*out_stride));

#if JYPPX_HAS_TENSORRT
    auto* slice_layer = get_slice_layer_payload(layer);
    if (slice_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT slice layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_dims(slice_layer->getStride(), out_stride);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "slice layer stride query");
#endif
}

JYPPX_StatusCode jyppx_trt10_slice_layer_set_mode(JYPPX_TensorRtLayer* layer, int32_t mode)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (mode < 0 || mode > 4)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Slice sample mode must be in the TensorRT enum range [0, 4].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto* slice_layer = get_slice_layer_payload(layer);
    if (slice_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT slice layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    slice_layer->setMode(static_cast<nvinfer1::SampleMode>(mode));
    return JYPPX_STATUS_OK;
#else
    (void)mode;
    return jyppx::tensorrt::report_vendor_missing(kLine, "slice layer mode set");
#endif
}

JYPPX_StatusCode jyppx_trt10_slice_layer_get_mode(JYPPX_TensorRtLayer* layer, int32_t* out_mode)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_mode, "out_mode");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_mode = 0;

#if JYPPX_HAS_TENSORRT
    auto* slice_layer = get_slice_layer_payload(layer);
    if (slice_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT slice layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_mode = static_cast<int32_t>(slice_layer->getMode());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "slice layer mode query");
#endif
}

JYPPX_StatusCode jyppx_trt10_slice_layer_set_axes(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* axes)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::Dims native_axes{};
    status = make_dims(axes, &native_axes, "axes");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* slice_layer = get_slice_layer_payload(layer);
    if (slice_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT slice layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    slice_layer->setAxes(native_axes);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)axes;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "slice layer axes set");
#else
    (void)axes;
    return jyppx::tensorrt::report_vendor_missing(kLine, "slice layer axes set");
#endif
}

JYPPX_StatusCode jyppx_trt10_slice_layer_get_axes(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_axes)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_axes, "out_axes");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_axes, 0, sizeof(*out_axes));
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* slice_layer = get_slice_layer_payload(layer);
    if (slice_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT slice layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_dims(slice_layer->getAxes(), out_axes);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "slice layer axes query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "slice layer axes query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_softmax(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::ISoftMaxLayer* layer = network_payload->addSoftMax(*tensor_payload);
    return create_layer_reference_handle(layer, out_layer, {}, -1, 0, 0, layer != nullptr ? layer->getAxes() : 0);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add softmax");
#endif
}

JYPPX_StatusCode jyppx_trt10_softmax_layer_set_axes(JYPPX_TensorRtLayer* layer, uint32_t axes)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (axes == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "SoftMax axes bitmask must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto* softmax_layer = get_softmax_layer_payload(layer);
    if (softmax_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT softmax layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    softmax_layer->setAxes(axes);
    if (auto* payload = get_layer_reference_payload(layer))
    {
        payload->softmax_axes = axes;
    }

    return JYPPX_STATUS_OK;
#else
    (void)axes;
    return jyppx::tensorrt::report_vendor_missing(kLine, "softmax layer axes set");
#endif
}

JYPPX_StatusCode jyppx_trt10_softmax_layer_get_axes(JYPPX_TensorRtLayer* layer, uint32_t* out_axes)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_axes, "out_axes");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_axes = 0;

#if JYPPX_HAS_TENSORRT
    auto* payload = get_layer_reference_payload(layer);
    if (payload == nullptr || payload->layer == nullptr || payload->layer->getType() != nvinfer1::LayerType::kSOFTMAX)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT softmax layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_axes = payload->softmax_axes != 0 ? payload->softmax_axes : static_cast<nvinfer1::ISoftMaxLayer*>(payload->layer)->getAxes();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "softmax layer axes query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_unary(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t operation, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (operation < 0 || operation > 24)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unary operation must be in the TensorRT enum range [0, 24].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IUnaryLayer* layer = network_payload->addUnary(*tensor_payload, static_cast<nvinfer1::UnaryOperation>(operation));
    return create_layer_reference_handle(layer, out_layer, {}, -1, 0, 0, 0, operation);
#else
    (void)operation;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add unary");
#endif
}

JYPPX_StatusCode jyppx_trt10_unary_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t* out_operation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_operation, "out_operation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_operation = 0;

#if JYPPX_HAS_TENSORRT
    auto* payload = get_layer_reference_payload(layer);
    if (payload == nullptr || payload->layer == nullptr || payload->layer->getType() != nvinfer1::LayerType::kUNARY)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT unary layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_operation = payload->unary_operation;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "unary layer operation query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_topk(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t operation, int32_t k, uint32_t axes, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (operation < 0 || operation > 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TopK operation must be in the TensorRT enum range [0, 1].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (k <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TopK k must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (axes == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TopK axes bitmask must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::ITopKLayer* layer = network_payload->addTopK(*tensor_payload, static_cast<nvinfer1::TopKOperation>(operation), k, axes);
    return create_layer_reference_handle(layer, out_layer, {}, -1, 0, 0, 0, -1, operation, k, axes);
#else
    (void)operation;
    (void)k;
    (void)axes;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add topk");
#endif
}

JYPPX_StatusCode jyppx_trt10_topk_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t* out_operation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_operation, "out_operation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_operation = 0;

#if JYPPX_HAS_TENSORRT
    auto* payload = get_layer_reference_payload(layer);
    if (payload == nullptr || payload->layer == nullptr || payload->layer->getType() != nvinfer1::LayerType::kTOPK)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT TopK layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_operation = payload->topk_operation;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "topk layer operation query");
#endif
}

JYPPX_StatusCode jyppx_trt10_topk_layer_get_k(JYPPX_TensorRtLayer* layer, int32_t* out_k)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_k, "out_k");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_k = 0;

#if JYPPX_HAS_TENSORRT
    auto* payload = get_layer_reference_payload(layer);
    if (payload == nullptr || payload->layer == nullptr || payload->layer->getType() != nvinfer1::LayerType::kTOPK)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT TopK layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_k = payload->topk_k;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "topk layer k query");
#endif
}

JYPPX_StatusCode jyppx_trt10_topk_layer_get_axes(JYPPX_TensorRtLayer* layer, uint32_t* out_axes)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_axes, "out_axes");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_axes = 0;

#if JYPPX_HAS_TENSORRT
    auto* payload = get_layer_reference_payload(layer);
    if (payload == nullptr || payload->layer == nullptr || payload->layer->getType() != nvinfer1::LayerType::kTOPK)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT TopK layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_axes = payload->topk_axes;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "topk layer axes query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_gather(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* data_tensor, JYPPX_TensorRtTensor* indices_tensor, int32_t axis, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(data_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "data_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(indices_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "indices_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (axis < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Gather axis must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* data_payload = get_payload<nvinfer1::ITensor>(data_tensor);
    auto* indices_payload = get_payload<nvinfer1::ITensor>(indices_tensor);
    if (network_payload == nullptr || data_payload == nullptr || indices_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IGatherLayer* layer = network_payload->addGather(*data_payload, *indices_payload, axis);
    return create_layer_reference_handle(layer, out_layer, {}, -1, 0, 0, 0, -1, -1, 0, 0, axis);
#else
    (void)axis;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add gather");
#endif
}

JYPPX_StatusCode jyppx_trt10_gather_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_axis, "out_axis");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_axis = 0;

#if JYPPX_HAS_TENSORRT
    auto* payload = get_layer_reference_payload(layer);
    if (payload == nullptr || payload->layer == nullptr || payload->layer->getType() != nvinfer1::LayerType::kGATHER)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT gather layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_axis = payload->gather_axis;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "gather layer axis query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_activation(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t activation_type, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (activation_type < 0 || activation_type > 13)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 10 activation type must be between 0 and 13.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    auto* layer = network_payload->addActivation(*tensor_payload, static_cast<nvinfer1::ActivationType>(activation_type));
    return create_layer_reference_handle(layer, out_layer);
#else
    (void)activation_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add activation");
#endif
}

JYPPX_StatusCode jyppx_trt10_activation_layer_get_type(JYPPX_TensorRtLayer* layer, int32_t* out_activation_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_activation_type, "out_activation_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_activation_type = 0;

#if JYPPX_HAS_TENSORRT
    auto* activation_layer = get_activation_layer_payload(layer);
    if (activation_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT activation layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_activation_type = static_cast<int32_t>(activation_layer->getActivationType());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "activation layer type query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_pooling_nd(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t pooling_type, const JYPPX_TensorRtDims* window_size, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (pooling_type < 0 || pooling_type > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Pooling type must be between 0 and 2.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_window_size{};
    status = make_dims(window_size, &native_window_size, "window_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    auto* layer = network_payload->addPoolingNd(*tensor_payload, static_cast<nvinfer1::PoolingType>(pooling_type), native_window_size);
    return create_layer_reference_handle(layer, out_layer);
#else
    (void)pooling_type;
    (void)window_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add pooling");
#endif
}

JYPPX_StatusCode jyppx_trt10_pooling_layer_get_type(JYPPX_TensorRtLayer* layer, int32_t* out_pooling_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_pooling_type, "out_pooling_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pooling_type = 0;

#if JYPPX_HAS_TENSORRT
    auto* pooling_layer = get_pooling_layer_payload(layer);
    if (pooling_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_pooling_type = static_cast<int32_t>(pooling_layer->getPoolingType());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer type query");
#endif
}

JYPPX_StatusCode jyppx_trt10_pooling_layer_set_window_size_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* window_size)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_window_size{};
    status = make_dims(window_size, &native_window_size, "window_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* pooling_layer = get_pooling_layer_payload(layer);
    if (pooling_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    pooling_layer->setWindowSizeNd(native_window_size);
    return JYPPX_STATUS_OK;
#else
    (void)window_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer window size set");
#endif
}

JYPPX_StatusCode jyppx_trt10_pooling_layer_get_window_size_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_window_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_window_size, "out_window_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* pooling_layer = get_pooling_layer_payload(layer);
    if (pooling_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_dims(pooling_layer->getWindowSizeNd(), out_window_size);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer window size query");
#endif
}

JYPPX_StatusCode jyppx_trt10_pooling_layer_set_stride_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* stride)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_stride{};
    status = make_dims(stride, &native_stride, "stride");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* pooling_layer = get_pooling_layer_payload(layer);
    if (pooling_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    pooling_layer->setStrideNd(native_stride);
    return JYPPX_STATUS_OK;
#else
    (void)stride;
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer stride set");
#endif
}

JYPPX_StatusCode jyppx_trt10_pooling_layer_get_stride_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_stride)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_stride, "out_stride");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* pooling_layer = get_pooling_layer_payload(layer);
    if (pooling_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_dims(pooling_layer->getStrideNd(), out_stride);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer stride query");
#endif
}

JYPPX_StatusCode jyppx_trt10_pooling_layer_set_padding_nd(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* padding)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_padding{};
    status = make_dims(padding, &native_padding, "padding");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* pooling_layer = get_pooling_layer_payload(layer);
    if (pooling_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    pooling_layer->setPaddingNd(native_padding);
    return JYPPX_STATUS_OK;
#else
    (void)padding;
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer padding set");
#endif
}

JYPPX_StatusCode jyppx_trt10_pooling_layer_get_padding_nd(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_padding)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_padding, "out_padding");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* pooling_layer = get_pooling_layer_payload(layer);
    if (pooling_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_dims(pooling_layer->getPaddingNd(), out_padding);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer padding query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_resize(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    auto* layer = network_payload->addResize(*tensor_payload);
    return create_layer_reference_handle(layer, out_layer);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add resize");
#endif
}

JYPPX_StatusCode jyppx_trt10_resize_layer_set_output_dimensions(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* dimensions)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_dimensions{};
    status = make_dims(dimensions, &native_dimensions, "dimensions");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* resize_layer = get_resize_layer_payload(layer);
    if (resize_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT resize layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    resize_layer->setOutputDimensions(native_dimensions);
    return JYPPX_STATUS_OK;
#else
    (void)dimensions;
    return jyppx::tensorrt::report_vendor_missing(kLine, "resize layer output dimensions set");
#endif
}

JYPPX_StatusCode jyppx_trt10_resize_layer_get_output_dimensions(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_dimensions)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dimensions, "out_dimensions");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* resize_layer = get_resize_layer_payload(layer);
    if (resize_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT resize layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_dims(resize_layer->getOutputDimensions(), out_dimensions);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "resize layer output dimensions query");
#endif
}

JYPPX_StatusCode jyppx_trt10_resize_layer_set_mode(JYPPX_TensorRtLayer* layer, int32_t resize_mode)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (resize_mode < 0 || resize_mode > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Resize mode must be between 0 and 2.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto* resize_layer = get_resize_layer_payload(layer);
    if (resize_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT resize layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    resize_layer->setResizeMode(static_cast<nvinfer1::InterpolationMode>(resize_mode));
    return JYPPX_STATUS_OK;
#else
    (void)resize_mode;
    return jyppx::tensorrt::report_vendor_missing(kLine, "resize layer mode set");
#endif
}

JYPPX_StatusCode jyppx_trt10_resize_layer_get_mode(JYPPX_TensorRtLayer* layer, int32_t* out_resize_mode)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_resize_mode, "out_resize_mode");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_resize_mode = 0;

#if JYPPX_HAS_TENSORRT
    auto* resize_layer = get_resize_layer_payload(layer);
    if (resize_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT resize layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_resize_mode = static_cast<int32_t>(resize_layer->getResizeMode());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "resize layer mode query");
#endif
}

JYPPX_StatusCode jyppx_trt10_resize_layer_set_scales(JYPPX_TensorRtLayer* layer, const float* scales, int32_t scale_count)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (scales == nullptr || scale_count <= 0 || scale_count > 8)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Resize scales must contain between 1 and 8 values.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto* resize_layer = get_resize_layer_payload(layer);
    if (resize_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT resize layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    resize_layer->setScales(scales, scale_count);
    return JYPPX_STATUS_OK;
#else
    (void)scales;
    (void)scale_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "resize layer scales set");
#endif
}

JYPPX_StatusCode jyppx_trt10_resize_layer_get_scales(JYPPX_TensorRtLayer* layer, float* scales, int32_t scale_capacity, int32_t* out_scale_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_scale_count, "out_scale_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (scale_capacity < 0 || scale_capacity > 8)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Resize scales output capacity must be between 0 and 8.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (scale_capacity > 0 && scales == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Resize scales output buffer must not be null when capacity is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_scale_count = 0;

#if JYPPX_HAS_TENSORRT
    auto* resize_layer = get_resize_layer_payload(layer);
    if (resize_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT resize layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_scale_count = resize_layer->getScales(scale_capacity, scales);
    return JYPPX_STATUS_OK;
#else
    (void)scales;
    (void)scale_capacity;
    return jyppx::tensorrt::report_vendor_missing(kLine, "resize layer scales query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_shape(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    auto* layer = network_payload->addShape(*tensor_payload);
    return create_layer_reference_handle(layer, out_layer);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add shape");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_select(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* condition_tensor,
    JYPPX_TensorRtTensor* then_tensor,
    JYPPX_TensorRtTensor* else_tensor,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(condition_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "condition_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(then_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "then_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(else_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "else_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* condition_payload = get_payload<nvinfer1::ITensor>(condition_tensor);
    auto* then_payload = get_payload<nvinfer1::ITensor>(then_tensor);
    auto* else_payload = get_payload<nvinfer1::ITensor>(else_tensor);
    if (network_payload == nullptr || condition_payload == nullptr || then_payload == nullptr || else_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    auto* layer = network_payload->addSelect(*condition_payload, *then_payload, *else_payload);
    return create_layer_reference_handle(layer, out_layer);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add select");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_fill(JYPPX_TensorRtNetworkDefinition* network, const JYPPX_TensorRtDims* dimensions, int32_t operation, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (operation < 0 || operation > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Fill operation must be in the TensorRT enum range [0, 2].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_dimensions{};
    status = make_dims(dimensions, &native_dimensions, "dimensions");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    auto* layer = network_payload->addFill(native_dimensions, static_cast<nvinfer1::FillOperation>(operation));
    return create_layer_reference_handle(layer, out_layer);
#else
    (void)dimensions;
    (void)operation;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add fill");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_add_reverse_sequence(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    JYPPX_TensorRtTensor* sequence_lengths_tensor,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(sequence_lengths_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "sequence_lengths_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* input_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    auto* sequence_lengths_payload = get_payload<nvinfer1::ITensor>(sequence_lengths_tensor);
    if (network_payload == nullptr || input_payload == nullptr || sequence_lengths_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    auto* layer = network_payload->addReverseSequence(*input_payload, *sequence_lengths_payload);
    return create_layer_reference_handle(layer, out_layer);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add reverse sequence");
#endif
}

JYPPX_StatusCode jyppx_trt10_fill_layer_set_dimensions(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* dimensions)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_dimensions{};
    status = make_dims(dimensions, &native_dimensions, "dimensions");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* fill_layer = get_fill_layer_payload(layer);
    if (fill_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT fill layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    fill_layer->setDimensions(native_dimensions);
    return JYPPX_STATUS_OK;
#else
    (void)dimensions;
    return jyppx::tensorrt::report_vendor_missing(kLine, "fill layer dimensions set");
#endif
}

JYPPX_StatusCode jyppx_trt10_fill_layer_get_dimensions(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_dimensions)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dimensions, "out_dimensions");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_dimensions, 0, sizeof(*out_dimensions));

#if JYPPX_HAS_TENSORRT
    auto* fill_layer = get_fill_layer_payload(layer);
    if (fill_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT fill layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_dims(fill_layer->getDimensions(), out_dimensions);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "fill layer dimensions query");
#endif
}

JYPPX_StatusCode jyppx_trt10_fill_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t operation)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (operation < 0 || operation > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Fill operation must be in the TensorRT enum range [0, 2].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto* fill_layer = get_fill_layer_payload(layer);
    if (fill_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT fill layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    fill_layer->setOperation(static_cast<nvinfer1::FillOperation>(operation));
    return JYPPX_STATUS_OK;
#else
    (void)operation;
    return jyppx::tensorrt::report_vendor_missing(kLine, "fill layer operation set");
#endif
}

JYPPX_StatusCode jyppx_trt10_fill_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t* out_operation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_operation, "out_operation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_operation = 0;

#if JYPPX_HAS_TENSORRT
    auto* fill_layer = get_fill_layer_payload(layer);
    if (fill_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT fill layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_operation = static_cast<int32_t>(fill_layer->getOperation());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "fill layer operation query");
#endif
}

JYPPX_StatusCode jyppx_trt10_fill_layer_set_alpha(JYPPX_TensorRtLayer* layer, double alpha)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* fill_layer = get_fill_layer_payload(layer);
    if (fill_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT fill layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    fill_layer->setAlpha(alpha);
    return JYPPX_STATUS_OK;
#else
    (void)alpha;
    return jyppx::tensorrt::report_vendor_missing(kLine, "fill layer alpha set");
#endif
}

JYPPX_StatusCode jyppx_trt10_fill_layer_get_alpha(JYPPX_TensorRtLayer* layer, double* out_alpha)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_alpha, "out_alpha");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_alpha = 0.0;

#if JYPPX_HAS_TENSORRT
    auto* fill_layer = get_fill_layer_payload(layer);
    if (fill_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT fill layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_alpha = fill_layer->getAlpha();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "fill layer alpha query");
#endif
}

JYPPX_StatusCode jyppx_trt10_fill_layer_set_beta(JYPPX_TensorRtLayer* layer, double beta)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* fill_layer = get_fill_layer_payload(layer);
    if (fill_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT fill layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    fill_layer->setBeta(beta);
    return JYPPX_STATUS_OK;
#else
    (void)beta;
    return jyppx::tensorrt::report_vendor_missing(kLine, "fill layer beta set");
#endif
}

JYPPX_StatusCode jyppx_trt10_fill_layer_get_beta(JYPPX_TensorRtLayer* layer, double* out_beta)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_beta, "out_beta");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_beta = 0.0;

#if JYPPX_HAS_TENSORRT
    auto* fill_layer = get_fill_layer_payload(layer);
    if (fill_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT fill layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_beta = fill_layer->getBeta();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "fill layer beta query");
#endif
}

JYPPX_StatusCode jyppx_trt10_reverse_sequence_layer_set_batch_axis(JYPPX_TensorRtLayer* layer, int32_t axis)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (axis < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ReverseSequence batch axis must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto* reverse_sequence_layer = get_reverse_sequence_layer_payload(layer);
    if (reverse_sequence_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT reverse sequence layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    reverse_sequence_layer->setBatchAxis(axis);
    return JYPPX_STATUS_OK;
#else
    (void)axis;
    return jyppx::tensorrt::report_vendor_missing(kLine, "reverse sequence layer batch axis set");
#endif
}

JYPPX_StatusCode jyppx_trt10_reverse_sequence_layer_get_batch_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_axis, "out_axis");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_axis = 0;

#if JYPPX_HAS_TENSORRT
    auto* reverse_sequence_layer = get_reverse_sequence_layer_payload(layer);
    if (reverse_sequence_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT reverse sequence layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_axis = reverse_sequence_layer->getBatchAxis();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "reverse sequence layer batch axis query");
#endif
}

JYPPX_StatusCode jyppx_trt10_reverse_sequence_layer_set_sequence_axis(JYPPX_TensorRtLayer* layer, int32_t axis)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (axis < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ReverseSequence sequence axis must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto* reverse_sequence_layer = get_reverse_sequence_layer_payload(layer);
    if (reverse_sequence_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT reverse sequence layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    reverse_sequence_layer->setSequenceAxis(axis);
    return JYPPX_STATUS_OK;
#else
    (void)axis;
    return jyppx::tensorrt::report_vendor_missing(kLine, "reverse sequence layer sequence axis set");
#endif
}

JYPPX_StatusCode jyppx_trt10_reverse_sequence_layer_get_sequence_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_axis, "out_axis");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_axis = 0;

#if JYPPX_HAS_TENSORRT
    auto* reverse_sequence_layer = get_reverse_sequence_layer_payload(layer);
    if (reverse_sequence_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT reverse sequence layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_axis = reverse_sequence_layer->getSequenceAxis();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "reverse sequence layer sequence axis query");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_get_output(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_tensor = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    status = validate_index(index, layer_payload->getNbOutputs(), "layer output");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return create_tensor_reference_handle(layer_payload->getOutput(index), out_tensor);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer output query");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_get_input(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_tensor = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    status = validate_index(index, layer_payload->getNbInputs(), "layer input");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return create_tensor_reference_handle(layer_payload->getInput(index), out_tensor);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer input query");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_get_input_count(JYPPX_TensorRtLayer* layer, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = layer_payload->getNbInputs();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer input count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_get_output_count(JYPPX_TensorRtLayer* layer, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = layer_payload->getNbOutputs();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer output count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_get_type(JYPPX_TensorRtLayer* layer, int32_t* out_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_type, "out_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_type = 0;

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_type = static_cast<int32_t>(layer_payload->getType());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer type query");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_get_name(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return copy_string_to_buffer(layer_payload->getName(), output_buffer, output_buffer_size, out_required_size);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer name query");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_set_name(JYPPX_TensorRtLayer* layer, const char* name)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    layer_payload->setName(name);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer name set");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_get_metadata(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return copy_string_to_buffer(layer_payload->getMetadata(), output_buffer, output_buffer_size, out_required_size);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer metadata query");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_set_metadata(JYPPX_TensorRtLayer* layer, const char* metadata)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (metadata == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer metadata must not be null. Use an empty string to clear metadata.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    layer_payload->setMetadata(metadata);
    return JYPPX_STATUS_OK;
#else
    (void)metadata;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer metadata set");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_set_precision(JYPPX_TensorRtLayer* layer, int32_t data_type)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    layer_payload->setPrecision(static_cast<nvinfer1::DataType>(data_type));
    return JYPPX_STATUS_OK;
#else
    (void)data_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer precision set");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_get_precision(JYPPX_TensorRtLayer* layer, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_data_type = -1;

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_data_type = static_cast<int32_t>(layer_payload->getPrecision());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer precision query");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_precision_is_set(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_is_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_set, "out_is_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_set = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_is_set = layer_payload->precisionIsSet() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer precision set-state query");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_reset_precision(JYPPX_TensorRtLayer* layer)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    layer_payload->resetPrecision();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer precision reset");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_set_output_type(JYPPX_TensorRtLayer* layer, int32_t index, int32_t data_type)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    status = validate_index(index, layer_payload->getNbOutputs(), "output");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    layer_payload->setOutputType(index, static_cast<nvinfer1::DataType>(data_type));
    return JYPPX_STATUS_OK;
#else
    (void)index;
    (void)data_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer output type set");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_get_output_type(JYPPX_TensorRtLayer* layer, int32_t index, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_data_type = -1;

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    status = validate_index(index, layer_payload->getNbOutputs(), "output");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_data_type = static_cast<int32_t>(layer_payload->getOutputType(index));
    return JYPPX_STATUS_OK;
#else
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer output type query");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_output_type_is_set(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_Boolean* out_is_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_set, "out_is_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_set = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    status = validate_index(index, layer_payload->getNbOutputs(), "output");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_set = layer_payload->outputTypeIsSet(index) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer output type set-state query");
#endif
}

JYPPX_StatusCode jyppx_trt10_layer_reset_output_type(JYPPX_TensorRtLayer* layer, int32_t index)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    status = validate_index(index, layer_payload->getNbOutputs(), "output");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    layer_payload->resetOutputType(index);
    return JYPPX_STATUS_OK;
#else
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer output type reset");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_get_name(JYPPX_TensorRtTensor* tensor, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return copy_string_to_buffer(tensor_payload->getName(), output_buffer, output_buffer_size, out_required_size);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor name query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_set_name(JYPPX_TensorRtTensor* tensor, const char* name)
{
    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    tensor_payload->setName(name);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor name set");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_get_data_type(JYPPX_TensorRtTensor* tensor, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_data_type = -1;

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_data_type = static_cast<int32_t>(tensor_payload->getType());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor data type query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_set_data_type(JYPPX_TensorRtTensor* tensor, int32_t data_type)
{
    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    tensor_payload->setType(static_cast<nvinfer1::DataType>(data_type));
    return JYPPX_STATUS_OK;
#else
    (void)data_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor data type set");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_get_shape(JYPPX_TensorRtTensor* tensor, JYPPX_TensorRtDims* out_dims)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dims, "out_dims");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_dims, 0, sizeof(*out_dims));

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    copy_dims(tensor_payload->getDimensions(), out_dims);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor shape query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_set_shape(JYPPX_TensorRtTensor* tensor, const JYPPX_TensorRtDims* dims)
{
    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_dims{};
    status = make_dims(dims, &native_dims, "dims");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    tensor_payload->setDimensions(native_dims);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor shape set");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_get_dimensions(JYPPX_TensorRtTensor* tensor, JYPPX_TensorRtDims* out_dims)
{
    return jyppx_trt10_tensor_get_shape(tensor, out_dims);
}

JYPPX_StatusCode jyppx_trt10_tensor_set_dimensions(JYPPX_TensorRtTensor* tensor, const JYPPX_TensorRtDims* dims)
{
    return jyppx_trt10_tensor_set_shape(tensor, dims);
}

JYPPX_StatusCode jyppx_trt10_tensor_get_location(JYPPX_TensorRtTensor* tensor, int32_t* out_location)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_location, "out_location");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_location = 0;

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_location = static_cast<int32_t>(tensor_payload->getLocation());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor location query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_set_location(JYPPX_TensorRtTensor* tensor, int32_t location)
{
    if (location < 0 || location > 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor location must be 0 (device) or 1 (host).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    tensor_payload->setLocation(static_cast<nvinfer1::TensorLocation>(location));
    return JYPPX_STATUS_OK;
#else
    (void)location;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor location set");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_get_allowed_formats(JYPPX_TensorRtTensor* tensor, uint32_t* out_formats)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_formats, "out_formats");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_formats = 0;

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_formats = tensor_payload->getAllowedFormats();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor allowed formats query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_set_allowed_formats(JYPPX_TensorRtTensor* tensor, uint32_t formats)
{
    if (formats == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor allowed formats bitmask must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    tensor_payload->setAllowedFormats(formats);
    return JYPPX_STATUS_OK;
#else
    (void)formats;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor allowed formats set");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_set_dynamic_range(JYPPX_TensorRtTensor* tensor, float min, float max)
{
    if (min > max)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor dynamic range min must be less than or equal to max.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (!tensor_payload->setDynamicRange(min, max))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT rejected the dynamic range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
#else
    (void)min;
    (void)max;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor dynamic range set");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_dynamic_range_is_set(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_set, "out_is_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_set = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_is_set = tensor_payload->dynamicRangeIsSet() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor dynamic range set-state query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_get_dynamic_range_min(JYPPX_TensorRtTensor* tensor, float* out_min)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_min, "out_min");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_min = 0.0F;

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_min = tensor_payload->getDynamicRangeMin();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor dynamic range min query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_get_dynamic_range_max(JYPPX_TensorRtTensor* tensor, float* out_max)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_max, "out_max");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_max = 0.0F;

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_max = tensor_payload->getDynamicRangeMax();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor dynamic range max query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_reset_dynamic_range(JYPPX_TensorRtTensor* tensor)
{
    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    tensor_payload->resetDynamicRange();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor dynamic range reset");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_get_broadcast_across_batch(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_broadcast_across_batch)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_broadcast_across_batch, "out_broadcast_across_batch");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_broadcast_across_batch = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_broadcast_across_batch = tensor_payload->getBroadcastAcrossBatch() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor broadcast-across-batch query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_set_broadcast_across_batch(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean broadcast_across_batch)
{
    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    tensor_payload->setBroadcastAcrossBatch(broadcast_across_batch != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#else
    (void)broadcast_across_batch;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor broadcast-across-batch set");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_is_network_input(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_network_input)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_network_input, "out_is_network_input");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_network_input = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_is_network_input = tensor_payload->isNetworkInput() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor network-input query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_is_network_output(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_network_output)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_network_output, "out_is_network_output");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_network_output = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_is_network_output = tensor_payload->isNetworkOutput() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor network-output query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_is_shape_tensor(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_shape_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_shape_tensor, "out_is_shape_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_shape_tensor = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_is_shape_tensor = tensor_payload->isShapeTensor() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor shape-tensor query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_is_execution_tensor(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_execution_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_execution_tensor, "out_is_execution_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_execution_tensor = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_is_execution_tensor = tensor_payload->isExecutionTensor() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor execution-tensor query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_get_dimension_name(JYPPX_TensorRtTensor* tensor, int32_t dimension_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (dimension_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor dimension index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const char* name = tensor_payload->getDimensionName(dimension_index);
    return copy_string_to_buffer(name == nullptr ? "" : name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)dimension_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor dimension name query");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_set_dimension_name(JYPPX_TensorRtTensor* tensor, int32_t dimension_index, const char* name)
{
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

    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    tensor_payload->setDimensionName(dimension_index, name);
    return JYPPX_STATUS_OK;
#else
    (void)dimension_index;
    (void)name;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor dimension name set");
#endif
}

JYPPX_StatusCode jyppx_trt10_tensor_clear_dimension_name(JYPPX_TensorRtTensor* tensor, int32_t dimension_index)
{
    if (dimension_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor dimension index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    tensor_payload->setDimensionName(dimension_index, nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)dimension_index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tensor dimension name clear");
#endif
}

#include "modules/parser/parser_inspector.inc"

#define JYPPX_TRT_EXPECTED_MAJOR 10
#define JYPPX_TRT_ONNX_PARSER_API(name) jyppx_trt10_onnx_parser_##name
#include "../common/onnx_parser_support.inc"

#define JYPPX_TRT_EXPECTED_MAJOR 10
#define JYPPX_TRT_PARSER_REFITTER_API(name) jyppx_trt10_parser_refitter_##name
#include "../common/parser_refitter_diagnostics.inc"

JYPPX_StatusCode jyppx_trt10_engine_get_layer_count(JYPPX_TensorRtCudaEngine* engine, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine layer count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = engine_payload->getNbLayers();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine layer count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_is_refittable(JYPPX_TensorRtCudaEngine* engine, JYPPX_Boolean* out_refittable)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_refittable, "out_refittable");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_refittable = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine refittable query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_refittable = engine_payload->isRefittable() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine refittable query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_create_refitter(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtLogger* logger, JYPPX_TensorRtRefitter** out_refitter)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_refitter, "out_refitter");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_refitter = nullptr;
    status = jyppx::tensorrt::validate_handle(logger, kLine, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, "logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine refitter creation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* logger_payload = get_payload<ManagedLogger>(logger);
    if (logger_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Logger handle does not carry the expected TensorRT logger payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IRefitter* refitter = nvinfer1::createInferRefitter(*engine_payload, *logger_payload);
    if (refitter == nullptr)
    {
        return report_null_vendor_object("createInferRefitter");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_REFITTER, refitter, &destroy_payload<nvinfer1::IRefitter>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_refitter = reinterpret_cast<JYPPX_TensorRtRefitter*>(handle);
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    (void)logger;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine refitter creation");
#endif
}

JYPPX_StatusCode jyppx_trt10_refitter_get_missing_count(JYPPX_TensorRtRefitter* refitter, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
    status = jyppx::tensorrt::validate_handle(refitter, kLine, JYPPX_TENSORRT_OBJECT_KIND_REFITTER, "refitter");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* refitter_payload = get_payload<nvinfer1::IRefitter>(refitter);
    if (refitter_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter handle does not carry the expected TensorRT refitter payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = refitter_payload->getMissing(0, nullptr, nullptr);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter missing weight count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_refitter_get_all_count(JYPPX_TensorRtRefitter* refitter, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
    status = jyppx::tensorrt::validate_handle(refitter, kLine, JYPPX_TENSORRT_OBJECT_KIND_REFITTER, "refitter");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* refitter_payload = get_payload<nvinfer1::IRefitter>(refitter);
    if (refitter_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter handle does not carry the expected TensorRT refitter payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = refitter_payload->getAll(0, nullptr, nullptr);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter all weight count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_refitter_get_missing_entries(JYPPX_TensorRtRefitter* refitter, JYPPX_TensorRtRefitEntryInfo* output_entries, int32_t output_count, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

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

    status = jyppx::tensorrt::validate_handle(refitter, kLine, JYPPX_TENSORRT_OBJECT_KIND_REFITTER, "refitter");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* refitter_payload = get_payload<nvinfer1::IRefitter>(refitter);
    if (refitter_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter handle does not carry the expected TensorRT refitter payload.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const int32_t required_count = refitter_payload->getMissing(0, nullptr, nullptr);
    *out_count = required_count;
    if (output_entries == nullptr || output_count == 0 || required_count == 0)
    {
        return JYPPX_STATUS_OK;
    }

    const int32_t copy_count = output_count < required_count ? output_count : required_count;
    std::vector<char const*> layer_names(static_cast<size_t>(copy_count));
    std::vector<nvinfer1::WeightsRole> roles(static_cast<size_t>(copy_count));
    refitter_payload->getMissing(copy_count, layer_names.data(), roles.data());

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
    (void)output_entries;
    (void)output_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter missing entries query");
#endif
}

JYPPX_StatusCode jyppx_trt10_refitter_get_all_entries(JYPPX_TensorRtRefitter* refitter, JYPPX_TensorRtRefitEntryInfo* output_entries, int32_t output_count, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

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

    status = jyppx::tensorrt::validate_handle(refitter, kLine, JYPPX_TENSORRT_OBJECT_KIND_REFITTER, "refitter");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* refitter_payload = get_payload<nvinfer1::IRefitter>(refitter);
    if (refitter_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter handle does not carry the expected TensorRT refitter payload.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const int32_t required_count = refitter_payload->getAll(0, nullptr, nullptr);
    *out_count = required_count;
    if (output_entries == nullptr || output_count == 0 || required_count == 0)
    {
        return JYPPX_STATUS_OK;
    }

    const int32_t copy_count = output_count < required_count ? output_count : required_count;
    std::vector<char const*> layer_names(static_cast<size_t>(copy_count));
    std::vector<nvinfer1::WeightsRole> roles(static_cast<size_t>(copy_count));
    refitter_payload->getAll(copy_count, layer_names.data(), roles.data());

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
    (void)output_entries;
    (void)output_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter all entries query");
#endif
}

JYPPX_StatusCode jyppx_trt10_refitter_set_weights(JYPPX_TensorRtRefitter* refitter, const char* layer_name, int32_t role, int32_t data_type, const void* values, int64_t value_count, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = JYPPX_FALSE;
    status = validate_c_string(layer_name, "layer_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

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

    status = jyppx::tensorrt::validate_handle(refitter, kLine, JYPPX_TENSORRT_OBJECT_KIND_REFITTER, "refitter");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* refitter_payload = get_payload<nvinfer1::IRefitter>(refitter);
    if (refitter_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter handle does not carry the expected TensorRT refitter payload.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    nvinfer1::Weights weights{static_cast<nvinfer1::DataType>(data_type), values, value_count};
    *out_set = refitter_payload->setWeights(layer_name, static_cast<nvinfer1::WeightsRole>(role), weights) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)values;
    (void)value_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter weights set");
#endif
}

JYPPX_StatusCode jyppx_trt10_refitter_refit_cuda_engine(JYPPX_TensorRtRefitter* refitter, JYPPX_Boolean* out_refitted)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_refitted, "out_refitted");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_refitted = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(refitter, kLine, JYPPX_TENSORRT_OBJECT_KIND_REFITTER, "refitter");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* refitter_payload = get_payload<nvinfer1::IRefitter>(refitter);
    if (refitter_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter handle does not carry the expected TensorRT refitter payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const bool refitted = refitter_payload->refitCudaEngine();
    *out_refitted = refitted ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter refit CUDA engine");
#endif
}

#define JYPPX_TRT_REFITTER_API(name) jyppx_trt10_##name
#include "../common/refitter_controls.inc"
#define JYPPX_TRT_REFITTER_API(name) jyppx_trt10_##name
#include "../common/refitter_trt10_extended.inc"

JYPPX_StatusCode jyppx_trt10_engine_get_name(JYPPX_TensorRtCudaEngine* engine, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(engine_payload->getName(), output_buffer, output_buffer_size, out_required_size);
#else
    (void)engine;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine name query");
#endif
}

#define JYPPX_TRT10_ENGINE_TENSOR_INT_QUERY(FUNCTION_NAME, OUTPUT_NAME, FEATURE_NAME, EXPR) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        status = validate_named_tensor(tensor_name); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        *OUTPUT_NAME = 0; \
        nvinfer1::ICudaEngine* engine_payload = nullptr; \
        status = get_engine_payload_ext(engine, &engine_payload, FEATURE_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        *OUTPUT_NAME = static_cast<int32_t>(EXPR); \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_TRT10_ENGINE_TENSOR_INT_QUERY(jyppx_trt10_engine_get_tensor_location, out_location, "engine tensor location query", engine_payload->getTensorLocation(tensor_name))
JYPPX_TRT10_ENGINE_TENSOR_INT_QUERY(jyppx_trt10_engine_get_tensor_bytes_per_component, out_bytes, "engine tensor bytes-per-component query", engine_payload->getTensorBytesPerComponent(tensor_name))
JYPPX_TRT10_ENGINE_TENSOR_INT_QUERY(jyppx_trt10_engine_get_tensor_components_per_element, out_components, "engine tensor components-per-element query", engine_payload->getTensorComponentsPerElement(tensor_name))
JYPPX_TRT10_ENGINE_TENSOR_INT_QUERY(jyppx_trt10_engine_get_tensor_format, out_format, "engine tensor format query", engine_payload->getTensorFormat(tensor_name))
JYPPX_TRT10_ENGINE_TENSOR_INT_QUERY(jyppx_trt10_engine_get_tensor_vectorized_dim, out_dim, "engine tensor vectorized dimension query", engine_payload->getTensorVectorizedDim(tensor_name))
#else
#define JYPPX_TRT10_ENGINE_TENSOR_INT_MISSING(FUNCTION_NAME, OUTPUT_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtCudaEngine*, const char*, int32_t* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        *OUTPUT_NAME = 0; \
        return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); \
    }
JYPPX_TRT10_ENGINE_TENSOR_INT_MISSING(jyppx_trt10_engine_get_tensor_location, out_location, "engine tensor location query")
JYPPX_TRT10_ENGINE_TENSOR_INT_MISSING(jyppx_trt10_engine_get_tensor_bytes_per_component, out_bytes, "engine tensor bytes-per-component query")
JYPPX_TRT10_ENGINE_TENSOR_INT_MISSING(jyppx_trt10_engine_get_tensor_components_per_element, out_components, "engine tensor components-per-element query")
JYPPX_TRT10_ENGINE_TENSOR_INT_MISSING(jyppx_trt10_engine_get_tensor_format, out_format, "engine tensor format query")
JYPPX_TRT10_ENGINE_TENSOR_INT_MISSING(jyppx_trt10_engine_get_tensor_vectorized_dim, out_dim, "engine tensor vectorized dimension query")
#undef JYPPX_TRT10_ENGINE_TENSOR_INT_MISSING
#endif

#undef JYPPX_TRT10_ENGINE_TENSOR_INT_QUERY

JYPPX_StatusCode jyppx_trt10_engine_get_tensor_format_desc(
    JYPPX_TensorRtCudaEngine* engine,
    const char* tensor_name,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine tensor format description query");
    if (status != JYPPX_STATUS_OK) { return status; }

    return copy_string_to_buffer(engine_payload->getTensorFormatDesc(tensor_name), output_buffer, output_buffer_size, out_required_size);
#else
    (void)engine;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tensor format description query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_is_shape_inference_io(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, JYPPX_Boolean* out_is_shape_inference_io)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_shape_inference_io, "out_is_shape_inference_io");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_shape_inference_io = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine shape-inference I/O query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_shape_inference_io = engine_payload->isShapeInferenceIO(tensor_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    (void)tensor_name;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine shape-inference I/O query");
#endif
}

#define JYPPX_TRT10_ENGINE_TENSOR_PROFILE_INT_QUERY(FUNCTION_NAME, OUTPUT_NAME, FEATURE_NAME, EXPR) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        if (profile_index < 0) \
        { \
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile index must be greater than or equal to zero."); \
            return JYPPX_STATUS_INVALID_ARGUMENT; \
        } \
        status = validate_named_tensor(tensor_name); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        *OUTPUT_NAME = 0; \
        nvinfer1::ICudaEngine* engine_payload = nullptr; \
        status = get_engine_payload_ext(engine, &engine_payload, FEATURE_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        *OUTPUT_NAME = static_cast<int32_t>(EXPR); \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_TRT10_ENGINE_TENSOR_PROFILE_INT_QUERY(jyppx_trt10_engine_get_tensor_bytes_per_component_for_profile, out_bytes, "engine tensor profile bytes-per-component query", engine_payload->getTensorBytesPerComponent(tensor_name, profile_index))
JYPPX_TRT10_ENGINE_TENSOR_PROFILE_INT_QUERY(jyppx_trt10_engine_get_tensor_components_per_element_for_profile, out_components, "engine tensor profile components-per-element query", engine_payload->getTensorComponentsPerElement(tensor_name, profile_index))
JYPPX_TRT10_ENGINE_TENSOR_PROFILE_INT_QUERY(jyppx_trt10_engine_get_tensor_format_for_profile, out_format, "engine tensor profile format query", engine_payload->getTensorFormat(tensor_name, profile_index))
JYPPX_TRT10_ENGINE_TENSOR_PROFILE_INT_QUERY(jyppx_trt10_engine_get_tensor_vectorized_dim_for_profile, out_dim, "engine tensor profile vectorized dimension query", engine_payload->getTensorVectorizedDim(tensor_name, profile_index))
#else
#define JYPPX_TRT10_ENGINE_TENSOR_PROFILE_INT_MISSING(FUNCTION_NAME, OUTPUT_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtCudaEngine*, const char*, int32_t, int32_t* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        *OUTPUT_NAME = 0; \
        return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); \
    }
JYPPX_TRT10_ENGINE_TENSOR_PROFILE_INT_MISSING(jyppx_trt10_engine_get_tensor_bytes_per_component_for_profile, out_bytes, "engine tensor profile bytes-per-component query")
JYPPX_TRT10_ENGINE_TENSOR_PROFILE_INT_MISSING(jyppx_trt10_engine_get_tensor_components_per_element_for_profile, out_components, "engine tensor profile components-per-element query")
JYPPX_TRT10_ENGINE_TENSOR_PROFILE_INT_MISSING(jyppx_trt10_engine_get_tensor_format_for_profile, out_format, "engine tensor profile format query")
JYPPX_TRT10_ENGINE_TENSOR_PROFILE_INT_MISSING(jyppx_trt10_engine_get_tensor_vectorized_dim_for_profile, out_dim, "engine tensor profile vectorized dimension query")
#undef JYPPX_TRT10_ENGINE_TENSOR_PROFILE_INT_MISSING
#endif
#undef JYPPX_TRT10_ENGINE_TENSOR_PROFILE_INT_QUERY

JYPPX_StatusCode jyppx_trt10_engine_get_tensor_format_desc_for_profile(
    JYPPX_TensorRtCudaEngine* engine,
    const char* tensor_name,
    int32_t profile_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
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

JYPPX_StatusCode jyppx_trt10_engine_get_profile_shape(
    JYPPX_TensorRtCudaEngine* engine,
    const char* tensor_name,
    int32_t profile_index,
    int32_t selector,
    JYPPX_TensorRtDims* out_shape)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_shape, "out_shape");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_shape, 0, sizeof(*out_shape));
#if JYPPX_HAS_TENSORRT
    if (profile_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = validate_profile_selector(selector);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine profile shape query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    copy_dims(engine_payload->getProfileShape(tensor_name, profile_index, static_cast<nvinfer1::OptProfileSelector>(selector)), out_shape);
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    (void)tensor_name;
    (void)profile_index;
    (void)selector;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine profile shape query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_engine_capability(JYPPX_TensorRtCudaEngine* engine, int32_t* out_capability)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_capability, "out_capability");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_capability = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine capability query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_capability = static_cast<int32_t>(engine_payload->getEngineCapability());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine capability query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_tactic_sources(JYPPX_TensorRtCudaEngine* engine, uint32_t* out_tactic_sources)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tactic_sources, "out_tactic_sources");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_tactic_sources = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine tactic sources query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_tactic_sources = engine_payload->getTacticSources();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine tactic sources query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_profiling_verbosity(JYPPX_TensorRtCudaEngine* engine, int32_t* out_verbosity)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_verbosity, "out_verbosity");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_verbosity = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine profiling verbosity query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_verbosity = static_cast<int32_t>(engine_payload->getProfilingVerbosity());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine profiling verbosity query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_max_batch_size(JYPPX_TensorRtCudaEngine* engine, int32_t* out_max_batch_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_max_batch_size, "out_max_batch_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_max_batch_size = 0;
    (void)engine;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ICudaEngine::getMaxBatchSize is a TensorRT 8 compatibility API and is not supported by the TensorRT 10 adapter.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

#define JYPPX_TRT10_CONTEXT_DIMS_QUERY(FUNCTION_NAME, OUTPUT_NAME, FEATURE_NAME, EXPR) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_TensorRtDims* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        std::memset(OUTPUT_NAME, 0, sizeof(*OUTPUT_NAME)); \
        status = validate_named_tensor(tensor_name); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        nvinfer1::IExecutionContext* context_payload = nullptr; \
        status = get_context_payload_ext(context, &context_payload, FEATURE_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        copy_dims(EXPR, OUTPUT_NAME); \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_TRT10_CONTEXT_DIMS_QUERY(jyppx_trt10_execution_context_get_tensor_shape, out_shape, "execution context tensor shape query", context_payload->getTensorShape(tensor_name))
JYPPX_TRT10_CONTEXT_DIMS_QUERY(jyppx_trt10_execution_context_get_tensor_strides, out_strides, "execution context tensor strides query", context_payload->getTensorStrides(tensor_name))
#else
#define JYPPX_TRT10_CONTEXT_DIMS_MISSING(FUNCTION_NAME, OUTPUT_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtExecutionContext*, const char*, JYPPX_TensorRtDims* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        std::memset(OUTPUT_NAME, 0, sizeof(*OUTPUT_NAME)); \
        return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); \
    }
JYPPX_TRT10_CONTEXT_DIMS_MISSING(jyppx_trt10_execution_context_get_tensor_shape, out_shape, "execution context tensor shape query")
JYPPX_TRT10_CONTEXT_DIMS_MISSING(jyppx_trt10_execution_context_get_tensor_strides, out_strides, "execution context tensor strides query")
#undef JYPPX_TRT10_CONTEXT_DIMS_MISSING
#endif
#undef JYPPX_TRT10_CONTEXT_DIMS_QUERY

JYPPX_StatusCode jyppx_trt10_execution_context_is_tensor_address_bound(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean* out_bound)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_bound, "out_bound");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_bound = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor address query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_bound = context_payload->getTensorAddress(tensor_name) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    (void)tensor_name;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context tensor address query");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_set_debug_sync(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean debug_sync)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context debug sync set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    context_payload->setDebugSync(debug_sync != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#else
    (void)context;
    (void)debug_sync;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context debug sync set");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_get_debug_sync(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_debug_sync)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_debug_sync, "out_debug_sync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_debug_sync = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context debug sync query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_debug_sync = context_payload->getDebugSync() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context debug sync query");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_set_nvtx_verbosity(JYPPX_TensorRtExecutionContext* context, int32_t verbosity, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = JYPPX_FALSE;
    if (verbosity < 0)
    {
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context NVTX verbosity set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = context_payload->setNvtxVerbosity(static_cast<nvinfer1::ProfilingVerbosity>(verbosity)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)context;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context NVTX verbosity set");
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context NVTX verbosity set");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_get_nvtx_verbosity(JYPPX_TensorRtExecutionContext* context, int32_t* out_verbosity)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_verbosity, "out_verbosity");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_verbosity = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context NVTX verbosity query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_verbosity = static_cast<int32_t>(context_payload->getNvtxVerbosity());
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)context;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context NVTX verbosity query");
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context NVTX verbosity query");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_set_name(JYPPX_TensorRtExecutionContext* context, const char* name)
{
    auto status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context name set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    context_payload->setName(name);
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context name set");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_get_name(JYPPX_TensorRtExecutionContext* context, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(context_payload->getName(), output_buffer, output_buffer_size, out_required_size);
#else
    (void)context;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context name query");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_get_optimization_profile(JYPPX_TensorRtExecutionContext* context, int32_t* out_profile_index)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_profile_index, "out_profile_index");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_profile_index = -1;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context optimization profile query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_profile_index = context_payload->getOptimizationProfile();
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context optimization profile query");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_set_optimization_profile_async(JYPPX_TensorRtExecutionContext* context, int32_t profile_index, JYPPX_CudaStream* stream)
{
    if (profile_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context async profile selection");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* stream_payload = reinterpret_cast<jyppx::cuda::StreamObject*>(stream);
    const bool result = context_payload->setOptimizationProfileAsync(profile_index, stream_payload->handle);
    if (!result)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setOptimizationProfileAsync returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)context;
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT execution context async profile selection");
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context async profile selection");
#endif
}

#define JYPPX_TRT10_CONTEXT_BOOL_QUERY(FUNCTION_NAME, OUTPUT_NAME, FEATURE_NAME, EXPR) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        *OUTPUT_NAME = JYPPX_FALSE; \
        nvinfer1::IExecutionContext* context_payload = nullptr; \
        status = get_context_payload_ext(context, &context_payload, FEATURE_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        *OUTPUT_NAME = (EXPR) ? JYPPX_TRUE : JYPPX_FALSE; \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_TRT10_CONTEXT_BOOL_QUERY(jyppx_trt10_execution_context_all_input_dimensions_specified, out_specified, "execution context input dimensions specified query", context_payload->allInputDimensionsSpecified())
JYPPX_TRT10_CONTEXT_BOOL_QUERY(jyppx_trt10_execution_context_all_input_shapes_specified, out_specified, "execution context input shapes specified query", context_payload->allInputShapesSpecified())
JYPPX_TRT10_CONTEXT_BOOL_QUERY(jyppx_trt10_execution_context_get_enqueue_emits_profile, out_enqueue_emits_profile, "execution context enqueue emits profile query", context_payload->getEnqueueEmitsProfile())
#else
#define JYPPX_TRT10_CONTEXT_BOOL_MISSING(FUNCTION_NAME, OUTPUT_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtExecutionContext*, JYPPX_Boolean* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        *OUTPUT_NAME = JYPPX_FALSE; \
        return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); \
    }
JYPPX_TRT10_CONTEXT_BOOL_MISSING(jyppx_trt10_execution_context_all_input_dimensions_specified, out_specified, "execution context input dimensions specified query")
JYPPX_TRT10_CONTEXT_BOOL_MISSING(jyppx_trt10_execution_context_all_input_shapes_specified, out_specified, "execution context input shapes specified query")
JYPPX_TRT10_CONTEXT_BOOL_MISSING(jyppx_trt10_execution_context_get_enqueue_emits_profile, out_enqueue_emits_profile, "execution context enqueue emits profile query")
#undef JYPPX_TRT10_CONTEXT_BOOL_MISSING
#endif
#undef JYPPX_TRT10_CONTEXT_BOOL_QUERY

JYPPX_StatusCode jyppx_trt10_execution_context_infer_shapes(JYPPX_TensorRtExecutionContext* context, int32_t* out_missing_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_missing_count, "out_missing_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_missing_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context shape inference");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t missing_count = context_payload->inferShapes(0, nullptr);
    if (missing_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT execution context inferShapes failed.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    *out_missing_count = missing_count;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context shape inference");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_set_enqueue_emits_profile(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean enqueue_emits_profile)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context enqueue emits profile set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    context_payload->setEnqueueEmitsProfile(enqueue_emits_profile != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#else
    (void)context;
    (void)enqueue_emits_profile;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context enqueue emits profile set");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_set_profiler(JYPPX_TensorRtExecutionContext* context, JYPPX_TensorRtProfiler* profiler)
{
#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
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

JYPPX_StatusCode jyppx_trt10_execution_context_clear_profiler(JYPPX_TensorRtExecutionContext* context)
{
#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context profiler clear");
    }

    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context profiler clear");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        context_payload->setProfiler(nullptr);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "execution context profiler clear", exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "execution context profiler clear", "unknown native exception");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context profiler clear");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_has_profiler(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_has_profiler)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_profiler, "out_has_profiler");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_profiler = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context profiler query");
    }

    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context profiler query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        *out_has_profiler = context_payload->getProfiler() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        *out_has_profiler = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(kLine, "execution context profiler query", exception.what());
    }
    catch (...)
    {
        *out_has_profiler = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(kLine, "execution context profiler query", "unknown native exception");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context profiler query");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_report_to_profiler(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_reported)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_reported, "out_reported");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_reported = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context profiler report");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_reported = context_payload->reportToProfiler() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context profiler report");
#endif
}

JYPPX_StatusCode jyppx_trt10_reduce_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t operation)
{
    if (operation < 0 || operation > 4)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Reduce operation must be in the TensorRT enum range [0, 4].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    LayerReferencePayload* payload = nullptr;
    auto status = get_layer_reference_ext(layer, &payload, "reduce");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* reduce_layer = get_reduce_layer_payload(layer);
    if (reduce_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT reduce layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    reduce_layer->setOperation(static_cast<nvinfer1::ReduceOperation>(operation));
    payload->reduce_operation = operation;
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "reduce layer operation set");
#endif
}

JYPPX_StatusCode jyppx_trt10_reduce_layer_set_axes(JYPPX_TensorRtLayer* layer, uint32_t axes)
{
    if (axes == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Reduce axes bitmask must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    (void)layer;
    (void)axes;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT does not expose a mutable reduce axes setter after layer creation; create the reduce layer with the desired axes.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt10_reduce_layer_set_keep_dimensions(JYPPX_TensorRtLayer* layer, JYPPX_Boolean keep_dimensions)
{
#if JYPPX_HAS_TENSORRT
    LayerReferencePayload* payload = nullptr;
    auto status = get_layer_reference_ext(layer, &payload, "reduce");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* reduce_layer = get_reduce_layer_payload(layer);
    if (reduce_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT reduce layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    reduce_layer->setKeepDimensions(keep_dimensions != JYPPX_FALSE);
    payload->reduce_keep_dimensions = keep_dimensions != JYPPX_FALSE ? 1 : 0;
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)keep_dimensions;
    return jyppx::tensorrt::report_vendor_missing(kLine, "reduce layer keep dimensions set");
#endif
}

JYPPX_StatusCode jyppx_trt10_unary_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t operation)
{
    if (operation < 0 || operation > 24)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unary operation must be in the TensorRT enum range [0, 24].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    LayerReferencePayload* payload = nullptr;
    auto status = get_layer_reference_ext(layer, &payload, "unary");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* unary_layer = get_unary_layer_payload(layer);
    if (unary_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT unary layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    unary_layer->setOperation(static_cast<nvinfer1::UnaryOperation>(operation));
    payload->unary_operation = operation;
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "unary layer operation set");
#endif
}

JYPPX_StatusCode jyppx_trt10_topk_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t operation)
{
    if (operation < 0 || operation > 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TopK operation must be in the TensorRT enum range [0, 1].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    LayerReferencePayload* payload = nullptr;
    auto status = get_layer_reference_ext(layer, &payload, "topk");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* topk_layer = get_topk_layer_payload(layer);
    if (topk_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT TopK layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    topk_layer->setOperation(static_cast<nvinfer1::TopKOperation>(operation));
    payload->topk_operation = operation;
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "topk layer operation set");
#endif
}

JYPPX_StatusCode jyppx_trt10_topk_layer_set_k(JYPPX_TensorRtLayer* layer, int32_t k)
{
    if (k <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TopK k must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    LayerReferencePayload* payload = nullptr;
    auto status = get_layer_reference_ext(layer, &payload, "topk");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* topk_layer = get_topk_layer_payload(layer);
    if (topk_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT TopK layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    topk_layer->setK(k);
    payload->topk_k = k;
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "topk layer k set");
#endif
}

JYPPX_StatusCode jyppx_trt10_topk_layer_set_axes(JYPPX_TensorRtLayer* layer, uint32_t axes)
{
    if (axes == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TopK axes bitmask must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    (void)layer;
    (void)axes;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT does not expose a mutable TopK axes setter after layer creation; create the TopK layer with the desired axes.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt10_gather_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis)
{
    if (axis < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Gather axis must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    LayerReferencePayload* payload = nullptr;
    auto status = get_layer_reference_ext(layer, &payload, "gather");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* gather_layer = get_gather_layer_payload(layer);
    if (gather_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT gather layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    gather_layer->setGatherAxis(axis);
    payload->gather_axis = axis;
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "gather layer axis set");
#endif
}

JYPPX_StatusCode jyppx_trt10_elementwise_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t* out_operation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_operation, "out_operation");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_operation = 0;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* elementwise_layer = get_elementwise_layer_payload_ext(layer);
    if (elementwise_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT elementwise layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_operation = static_cast<int32_t>(elementwise_layer->getOperation());
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "elementwise layer operation query");
#endif
}

JYPPX_StatusCode jyppx_trt10_elementwise_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t operation)
{
    if (operation < 0 || operation > 13)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ElementWise operation must be in the TensorRT enum range [0, 13].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* elementwise_layer = get_elementwise_layer_payload_ext(layer);
    if (elementwise_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT elementwise layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    elementwise_layer->setOperation(static_cast<nvinfer1::ElementWiseOperation>(operation));
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "elementwise layer operation set");
#endif
}

JYPPX_StatusCode jyppx_trt10_activation_layer_set_type(JYPPX_TensorRtLayer* layer, int32_t activation_type)
{
    if (activation_type < 0 || activation_type > 13)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 10 activation type must be between 0 and 13.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* activation_layer = get_activation_layer_payload(layer);
    if (activation_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT activation layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    activation_layer->setActivationType(static_cast<nvinfer1::ActivationType>(activation_type));
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "activation layer type set");
#endif
}

#define JYPPX_TRT10_ACTIVATION_DOUBLE_ACCESSOR(FUNCTION_GET, FUNCTION_SET, GETTER, SETTER, OUTPUT_NAME, FIELD_NAME) \
    JYPPX_StatusCode FUNCTION_GET(JYPPX_TensorRtLayer* layer, double* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        *OUTPUT_NAME = 0.0; \
        status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        auto* activation_layer = get_activation_layer_payload(layer); \
        if (activation_layer == nullptr) \
        { \
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT activation layer."); \
            return JYPPX_STATUS_INVALID_STATE; \
        } \
        *OUTPUT_NAME = activation_layer->GETTER(); \
        return JYPPX_STATUS_OK; \
    } \
    JYPPX_StatusCode FUNCTION_SET(JYPPX_TensorRtLayer* layer, double FIELD_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        auto* activation_layer = get_activation_layer_payload(layer); \
        if (activation_layer == nullptr) \
        { \
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT activation layer."); \
            return JYPPX_STATUS_INVALID_STATE; \
        } \
        activation_layer->SETTER(FIELD_NAME); \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_TRT10_ACTIVATION_DOUBLE_ACCESSOR(jyppx_trt10_activation_layer_get_alpha, jyppx_trt10_activation_layer_set_alpha, getAlpha, setAlpha, out_alpha, alpha)
JYPPX_TRT10_ACTIVATION_DOUBLE_ACCESSOR(jyppx_trt10_activation_layer_get_beta, jyppx_trt10_activation_layer_set_beta, getBeta, setBeta, out_beta, beta)
#else
JYPPX_StatusCode jyppx_trt10_activation_layer_get_alpha(JYPPX_TensorRtLayer*, double* out_alpha)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_alpha, "out_alpha");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_alpha = 0.0;
    return jyppx::tensorrt::report_vendor_missing(kLine, "activation layer alpha query");
}
JYPPX_StatusCode jyppx_trt10_activation_layer_set_alpha(JYPPX_TensorRtLayer*, double)
{
    return jyppx::tensorrt::report_vendor_missing(kLine, "activation layer alpha set");
}
JYPPX_StatusCode jyppx_trt10_activation_layer_get_beta(JYPPX_TensorRtLayer*, double* out_beta)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_beta, "out_beta");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_beta = 0.0;
    return jyppx::tensorrt::report_vendor_missing(kLine, "activation layer beta query");
}
JYPPX_StatusCode jyppx_trt10_activation_layer_set_beta(JYPPX_TensorRtLayer*, double)
{
    return jyppx::tensorrt::report_vendor_missing(kLine, "activation layer beta set");
}
#endif
#undef JYPPX_TRT10_ACTIVATION_DOUBLE_ACCESSOR

JYPPX_StatusCode jyppx_trt10_pooling_layer_set_type(JYPPX_TensorRtLayer* layer, int32_t pooling_type)
{
    if (pooling_type < 0 || pooling_type > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Pooling type must be between 0 and 2.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* pooling_layer = get_pooling_layer_payload(layer);
    if (pooling_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    pooling_layer->setPoolingType(static_cast<nvinfer1::PoolingType>(pooling_type));
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer type set");
#endif
}

#define JYPPX_TRT10_POOLING_DOUBLE_ACCESSOR(FUNCTION_GET, FUNCTION_SET, GETTER, SETTER, OUTPUT_NAME, FIELD_NAME, FEATURE) \
    JYPPX_StatusCode FUNCTION_GET(JYPPX_TensorRtLayer* layer, double* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        *OUTPUT_NAME = 0.0; \
        status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        auto* pooling_layer = get_pooling_layer_payload(layer); \
        if (pooling_layer == nullptr) \
        { \
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer."); \
            return JYPPX_STATUS_INVALID_STATE; \
        } \
        *OUTPUT_NAME = pooling_layer->GETTER(); \
        return JYPPX_STATUS_OK; \
    } \
    JYPPX_StatusCode FUNCTION_SET(JYPPX_TensorRtLayer* layer, double FIELD_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        auto* pooling_layer = get_pooling_layer_payload(layer); \
        if (pooling_layer == nullptr) \
        { \
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer."); \
            return JYPPX_STATUS_INVALID_STATE; \
        } \
        pooling_layer->SETTER(static_cast<float>(FIELD_NAME)); \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_TRT10_POOLING_DOUBLE_ACCESSOR(jyppx_trt10_pooling_layer_get_blend_factor, jyppx_trt10_pooling_layer_set_blend_factor, getBlendFactor, setBlendFactor, out_blend_factor, blend_factor, "pooling blend factor")
#else
JYPPX_StatusCode jyppx_trt10_pooling_layer_get_blend_factor(JYPPX_TensorRtLayer*, double* out_blend_factor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_blend_factor, "out_blend_factor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_blend_factor = 0.0;
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer blend factor query");
}
JYPPX_StatusCode jyppx_trt10_pooling_layer_set_blend_factor(JYPPX_TensorRtLayer*, double)
{
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer blend factor set");
}
#endif
#undef JYPPX_TRT10_POOLING_DOUBLE_ACCESSOR

JYPPX_StatusCode jyppx_trt10_pooling_layer_get_average_count_excludes_padding(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_excludes_padding)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_excludes_padding, "out_excludes_padding");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_excludes_padding = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* pooling_layer = get_pooling_layer_payload(layer);
    if (pooling_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_excludes_padding = pooling_layer->getAverageCountExcludesPadding() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer average count excludes padding query");
#endif
}

JYPPX_StatusCode jyppx_trt10_pooling_layer_set_average_count_excludes_padding(JYPPX_TensorRtLayer* layer, JYPPX_Boolean excludes_padding)
{
#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* pooling_layer = get_pooling_layer_payload(layer);
    if (pooling_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    pooling_layer->setAverageCountExcludesPadding(excludes_padding != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)excludes_padding;
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer average count excludes padding set");
#endif
}

#define JYPPX_TRT10_POOLING_DIMS_ACCESSOR(FUNCTION_GET, FUNCTION_SET, GETTER, SETTER, OUTPUT_NAME, INPUT_NAME, FEATURE) \
    JYPPX_StatusCode FUNCTION_GET(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        std::memset(OUTPUT_NAME, 0, sizeof(*OUTPUT_NAME)); \
        status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        auto* pooling_layer = get_pooling_layer_payload(layer); \
        if (pooling_layer == nullptr) \
        { \
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer."); \
            return JYPPX_STATUS_INVALID_STATE; \
        } \
        copy_dims(pooling_layer->GETTER(), OUTPUT_NAME); \
        return JYPPX_STATUS_OK; \
    } \
    JYPPX_StatusCode FUNCTION_SET(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* INPUT_NAME) \
    { \
        nvinfer1::Dims native_dims{}; \
        auto status = make_dims(INPUT_NAME, &native_dims, #INPUT_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        auto* pooling_layer = get_pooling_layer_payload(layer); \
        if (pooling_layer == nullptr) \
        { \
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer."); \
            return JYPPX_STATUS_INVALID_STATE; \
        } \
        pooling_layer->SETTER(native_dims); \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_TRT10_POOLING_DIMS_ACCESSOR(jyppx_trt10_pooling_layer_get_pre_padding, jyppx_trt10_pooling_layer_set_pre_padding, getPrePadding, setPrePadding, out_pre_padding, pre_padding, "pooling pre-padding")
JYPPX_TRT10_POOLING_DIMS_ACCESSOR(jyppx_trt10_pooling_layer_get_post_padding, jyppx_trt10_pooling_layer_set_post_padding, getPostPadding, setPostPadding, out_post_padding, post_padding, "pooling post-padding")
#else
JYPPX_StatusCode jyppx_trt10_pooling_layer_get_pre_padding(JYPPX_TensorRtLayer*, JYPPX_TensorRtDims* out_pre_padding)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_pre_padding, "out_pre_padding");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_pre_padding, 0, sizeof(*out_pre_padding));
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer pre-padding query");
}
JYPPX_StatusCode jyppx_trt10_pooling_layer_set_pre_padding(JYPPX_TensorRtLayer*, const JYPPX_TensorRtDims*)
{
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer pre-padding set");
}
JYPPX_StatusCode jyppx_trt10_pooling_layer_get_post_padding(JYPPX_TensorRtLayer*, JYPPX_TensorRtDims* out_post_padding)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_post_padding, "out_post_padding");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_post_padding, 0, sizeof(*out_post_padding));
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer post-padding query");
}
JYPPX_StatusCode jyppx_trt10_pooling_layer_set_post_padding(JYPPX_TensorRtLayer*, const JYPPX_TensorRtDims*)
{
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer post-padding set");
}
#endif
#undef JYPPX_TRT10_POOLING_DIMS_ACCESSOR

JYPPX_StatusCode jyppx_trt10_pooling_layer_get_padding_mode(JYPPX_TensorRtLayer* layer, int32_t* out_padding_mode)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_padding_mode, "out_padding_mode");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_padding_mode = 0;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* pooling_layer = get_pooling_layer_payload(layer);
    if (pooling_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_padding_mode = static_cast<int32_t>(pooling_layer->getPaddingMode());
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer padding mode query");
#endif
}

JYPPX_StatusCode jyppx_trt10_pooling_layer_set_padding_mode(JYPPX_TensorRtLayer* layer, int32_t padding_mode)
{
    if (padding_mode < 0 || padding_mode > 3)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Padding mode must be in the TensorRT enum range [0, 3].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* pooling_layer = get_pooling_layer_payload(layer);
    if (pooling_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT pooling layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    pooling_layer->setPaddingMode(static_cast<nvinfer1::PaddingMode>(padding_mode));
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "pooling layer padding mode set");
#endif
}

#define JYPPX_TRT10_RESIZE_ENUM_ACCESSOR(FUNCTION_GET, FUNCTION_SET, GETTER, SETTER, OUTPUT_NAME, INPUT_NAME, MIN_VALUE, MAX_VALUE, ENUM_TYPE, FEATURE) \
    JYPPX_StatusCode FUNCTION_GET(JYPPX_TensorRtLayer* layer, int32_t* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        *OUTPUT_NAME = 0; \
        status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        auto* resize_layer = get_resize_layer_payload(layer); \
        if (resize_layer == nullptr) \
        { \
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT resize layer."); \
            return JYPPX_STATUS_INVALID_STATE; \
        } \
        *OUTPUT_NAME = static_cast<int32_t>(resize_layer->GETTER()); \
        return JYPPX_STATUS_OK; \
    } \
    JYPPX_StatusCode FUNCTION_SET(JYPPX_TensorRtLayer* layer, int32_t INPUT_NAME) \
    { \
        if (INPUT_NAME < MIN_VALUE || INPUT_NAME > MAX_VALUE) \
        { \
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, FEATURE " value is outside the supported TensorRT enum range."); \
            return JYPPX_STATUS_INVALID_ARGUMENT; \
        } \
        auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) \
        { \
            return status; \
        } \
        auto* resize_layer = get_resize_layer_payload(layer); \
        if (resize_layer == nullptr) \
        { \
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT resize layer."); \
            return JYPPX_STATUS_INVALID_STATE; \
        } \
        resize_layer->SETTER(static_cast<ENUM_TYPE>(INPUT_NAME)); \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_TRT10_RESIZE_ENUM_ACCESSOR(jyppx_trt10_resize_layer_get_coordinate_transformation, jyppx_trt10_resize_layer_set_coordinate_transformation, getCoordinateTransformation, setCoordinateTransformation, out_coordinate_transformation, coordinate_transformation, 0, 2, nvinfer1::ResizeCoordinateTransformation, "Resize coordinate transformation")
JYPPX_TRT10_RESIZE_ENUM_ACCESSOR(jyppx_trt10_resize_layer_get_selector_for_single_pixel, jyppx_trt10_resize_layer_set_selector_for_single_pixel, getSelectorForSinglePixel, setSelectorForSinglePixel, out_selector, selector, 0, 1, nvinfer1::ResizeSelector, "Resize selector")
JYPPX_TRT10_RESIZE_ENUM_ACCESSOR(jyppx_trt10_resize_layer_get_nearest_rounding, jyppx_trt10_resize_layer_set_nearest_rounding, getNearestRounding, setNearestRounding, out_rounding, rounding, 0, 3, nvinfer1::ResizeRoundMode, "Resize nearest rounding")
#else
#define JYPPX_TRT10_RESIZE_ENUM_MISSING(FUNCTION_GET, FUNCTION_SET, OUTPUT_NAME, INPUT_NAME, FEATURE) \
    JYPPX_StatusCode FUNCTION_GET(JYPPX_TensorRtLayer*, int32_t* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUTPUT_NAME = 0; \
        return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE " query"); \
    } \
    JYPPX_StatusCode FUNCTION_SET(JYPPX_TensorRtLayer*, int32_t INPUT_NAME) \
    { \
        (void)INPUT_NAME; \
        return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE " set"); \
    }
JYPPX_TRT10_RESIZE_ENUM_MISSING(jyppx_trt10_resize_layer_get_coordinate_transformation, jyppx_trt10_resize_layer_set_coordinate_transformation, out_coordinate_transformation, coordinate_transformation, "resize coordinate transformation")
JYPPX_TRT10_RESIZE_ENUM_MISSING(jyppx_trt10_resize_layer_get_selector_for_single_pixel, jyppx_trt10_resize_layer_set_selector_for_single_pixel, out_selector, selector, "resize selector")
JYPPX_TRT10_RESIZE_ENUM_MISSING(jyppx_trt10_resize_layer_get_nearest_rounding, jyppx_trt10_resize_layer_set_nearest_rounding, out_rounding, rounding, "resize nearest rounding")
#undef JYPPX_TRT10_RESIZE_ENUM_MISSING
#endif
#undef JYPPX_TRT10_RESIZE_ENUM_ACCESSOR

JYPPX_StatusCode jyppx_trt10_resize_layer_get_cubic_coeff(JYPPX_TensorRtLayer* layer, double* out_cubic_coeff)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cubic_coeff, "out_cubic_coeff");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_cubic_coeff = 0.0;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* resize_layer = get_resize_layer_payload(layer);
    if (resize_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT resize layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_cubic_coeff = resize_layer->getCubicCoeff();
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "resize layer cubic coefficient query");
#endif
}

JYPPX_StatusCode jyppx_trt10_resize_layer_set_cubic_coeff(JYPPX_TensorRtLayer* layer, double cubic_coeff)
{
#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* resize_layer = get_resize_layer_payload(layer);
    if (resize_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT resize layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    resize_layer->setCubicCoeff(static_cast<float>(cubic_coeff));
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)cubic_coeff;
    return jyppx::tensorrt::report_vendor_missing(kLine, "resize layer cubic coefficient set");
#endif
}

JYPPX_StatusCode jyppx_trt10_resize_layer_get_exclude_outside(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_exclude_outside)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_exclude_outside, "out_exclude_outside");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_exclude_outside = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* resize_layer = get_resize_layer_payload(layer);
    if (resize_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT resize layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_exclude_outside = resize_layer->getExcludeOutside() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "resize layer exclude outside query");
#endif
}

JYPPX_StatusCode jyppx_trt10_resize_layer_set_exclude_outside(JYPPX_TensorRtLayer* layer, JYPPX_Boolean exclude_outside)
{
#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* resize_layer = get_resize_layer_payload(layer);
    if (resize_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT resize layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    resize_layer->setExcludeOutside(exclude_outside != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)exclude_outside;
    return jyppx::tensorrt::report_vendor_missing(kLine, "resize layer exclude outside set");
#endif
}

JYPPX_StatusCode jyppx_trt10_resize_layer_get_resize_mode(JYPPX_TensorRtLayer* layer, int32_t* out_mode)
{
    return jyppx_trt10_resize_layer_get_mode(layer, out_mode);
}

JYPPX_StatusCode jyppx_trt10_resize_layer_set_resize_mode(JYPPX_TensorRtLayer* layer, int32_t mode)
{
    return jyppx_trt10_resize_layer_set_mode(layer, mode);
}

#define JYPPX_TRT_RUNTIME_API(name) jyppx_trt10_##name
#include "../common/runtime_controls.inc"
#undef JYPPX_TRT_RUNTIME_API

#define JYPPX_TRT_BUILDER_CALLBACK_BOUNDARY_API(name) jyppx_trt10_##name
#include "../common/builder_callback_boundary_controls.inc"
#undef JYPPX_TRT_BUILDER_CALLBACK_BOUNDARY_API

#define JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(name) jyppx_trt10_##name
#include "../common/error_recorder_boundary_controls.inc"
#undef JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API

#define JYPPX_TRT_EXECUTION_CONTEXT_ALLOCATOR_API(name) jyppx_trt10_##name
#include "../common/execution_context_allocator_controls.inc"
#undef JYPPX_TRT_EXECUTION_CONTEXT_ALLOCATOR_API

#define JYPPX_TRT_EXECUTION_CONTEXT_DEBUG_LISTENER_API(name) jyppx_trt10_##name
#include "../common/execution_context_debug_listener_controls.inc"
#undef JYPPX_TRT_EXECUTION_CONTEXT_DEBUG_LISTENER_API

#define JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_STATE_API(name) jyppx_trt10_##name
#include "../common/execution_context_callback_state_snapshot.inc"
#undef JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_STATE_API

#define JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_INTERFACE_INFO_API(name) jyppx_trt10_##name
#include "../common/execution_context_callback_interface_info.inc"
#undef JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_INTERFACE_INFO_API

#define JYPPX_TRT_OWNER_SCOPED_VERSIONED_METADATA_API(name) jyppx_trt10_##name
#include "../common/owner_scoped_versioned_interface_metadata.inc"
#undef JYPPX_TRT_OWNER_SCOPED_VERSIONED_METADATA_API

#define JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API(name) jyppx_trt10_##name
#include "../common/allocator_owner_dry_run.inc"
#undef JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API

#define JYPPX_TRT_ONNX_CONFIG_API(name) jyppx_trt10_onnx_config_##name
#define JYPPX_TRT_EXPECTED_MAJOR 10
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
#define JYPPX_TRT_DEBUG_LISTENER_OWNER_API(name) jyppx_trt10_##name
#include "../common/debug_listener_callback_owner.inc"
#undef JYPPX_TRT_DEBUG_LISTENER_OWNER_API
#define JYPPX_TRT_OUTPUT_ALLOCATOR_OWNER_API(name) jyppx_trt10_##name
#include "../common/output_allocator_callback_owner.inc"
#undef JYPPX_TRT_OUTPUT_ALLOCATOR_OWNER_API

#include "modules/layers/convolution_scale_padding.inc"
#include "modules/layers/deconvolution.inc"
#include "modules/layers/lrn.inc"
#include "modules/layers/quantization.inc"
#include "modules/layers/deployment_metadata_compat.inc"
#include "modules/layers/control_flow_compat.inc"
#include "modules/layers/weight_info_metadata.inc"
#include "modules/layers/advanced_layer_compat.inc"
#include "modules/layers/cross_version_network_compat.inc"

#include "modules/builder/timing_cache_controls.inc"
#include "modules/deployment/host_memory_metadata.inc"
#include "modules/deployment/engine_profile_tensor_values.inc"
#include "modules/context/deployment_context.inc"
#define JYPPX_TRT_SAFE_PLUGIN_PREFIX jyppx_trt10_
#define JYPPX_TRT_SAFE_PLUGIN_EXPECTED_MAJOR 10
#include "../common/safe_deferred_plugin_initialization.inc"
#undef JYPPX_TRT_SAFE_PLUGIN_EXPECTED_MAJOR
#undef JYPPX_TRT_SAFE_PLUGIN_PREFIX
#define JYPPX_TRT_SAFE_ONNX_PREFIX jyppx_trt10_
#define JYPPX_TRT_SAFE_ONNX_EXPECTED_MAJOR 10
#include "../common/safe_deferred_onnx_parse.inc"
#undef JYPPX_TRT_SAFE_ONNX_EXPECTED_MAJOR
#undef JYPPX_TRT_SAFE_ONNX_PREFIX

#define JYPPX_TRT_EXECUTION_CONTEXT_SET_AUX_STREAMS_API jyppx_trt10_execution_context_set_aux_streams
#define JYPPX_TRT_EXECUTION_CONTEXT_CLEAR_AUX_STREAMS_API jyppx_trt10_execution_context_clear_aux_streams
#include "../common/execution_context_auxiliary_streams.inc"
#undef JYPPX_TRT_EXECUTION_CONTEXT_CLEAR_AUX_STREAMS_API
#undef JYPPX_TRT_EXECUTION_CONTEXT_SET_AUX_STREAMS_API

#define JYPPX_TRT_EXECUTION_CONTEXT_EXECUTE_V2_API jyppx_trt10_execution_context_execute_v2_safe
#define JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR 10
#include "../common/execution_context_synchronous_inference.inc"
#undef JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR
#undef JYPPX_TRT_EXECUTION_CONTEXT_EXECUTE_V2_API

#define JYPPX_TRT_PLUGIN_PREFIX jyppx_trt10_
#include "../common/plugin_registry_inventory.inc"
#undef JYPPX_TRT_PLUGIN_PREFIX

#define JYPPX_TRT_DIRECT_ENGINE_BUILD_API jyppx_trt10_builder_build_engine_with_config
#define JYPPX_TRT_DIRECT_ENGINE_DESTROY destroy_payload<nvinfer1::ICudaEngine>
#include "../common/direct_engine_build.inc"
#undef JYPPX_TRT_DIRECT_ENGINE_DESTROY
#undef JYPPX_TRT_DIRECT_ENGINE_BUILD_API

#define JYPPX_TRT_ERROR_CODE_METADATA_API jyppx_trt10_error_code_get_exclusive_upper_bound
#include "../common/error_code_metadata.inc"
#undef JYPPX_TRT_ERROR_CODE_METADATA_API

#define JYPPX_TRT_PLUGIN_V2_LAYER_PREFIX jyppx_trt10_
#define JYPPX_TRT_PLUGIN_V2_LAYER_ENABLE_BROADCAST_EXPORTS 1
#include "../common/plugin_v2_layer_metadata_snapshot.inc"
#undef JYPPX_TRT_PLUGIN_V2_LAYER_ENABLE_BROADCAST_EXPORTS
#undef JYPPX_TRT_PLUGIN_V2_LAYER_PREFIX
#define JYPPX_TRT_PLUGIN_V3_LAYER_PREFIX jyppx_trt10_
#include "../common/plugin_v3_layer_metadata_snapshot.inc"
#undef JYPPX_TRT_PLUGIN_V3_LAYER_PREFIX
#define JYPPX_TRT_PLUGIN_OWNER_QUERY_PREFIX jyppx_trt10_
#define JYPPX_TRT_PLUGIN_OWNER_QUERY_ENABLE_V3 1
#include "../common/plugin_layer_owner_scoped_query_snapshots.inc"
#undef JYPPX_TRT_PLUGIN_OWNER_QUERY_ENABLE_V3
#undef JYPPX_TRT_PLUGIN_OWNER_QUERY_PREFIX
#define JYPPX_TRT_GLOBAL_PLUGIN_PREFIX jyppx_trt10_
#include "../common/global_runtime_plugin_probe.inc"
#undef JYPPX_TRT_GLOBAL_PLUGIN_PREFIX
#include "modules/deferred/twenty_third_batch_deferred.inc"
#include "modules/deferred/cross_version_plugin_deferred.inc"
#include "modules/deferred/cross_version_other_deferred.inc"
#include "modules/deferred/cross_version_onnx_parser_global_deferred.inc"
#include "modules/deferred/cross_version_runtime_serialization_deferred.inc"
#include "modules/deferred/cross_version_diagnostics_refitter_deferred.inc"

