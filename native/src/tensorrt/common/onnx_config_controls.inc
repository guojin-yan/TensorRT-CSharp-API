#ifndef JYPPX_TRT_ONNX_CONFIG_API
#error "JYPPX_TRT_ONNX_CONFIG_API must be defined before including onnx_config_controls.inc."
#endif

#ifndef JYPPX_TRT_EXPECTED_MAJOR
#error "JYPPX_TRT_EXPECTED_MAJOR must be defined before including onnx_config_controls.inc."
#endif

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
class BridgeOwnedOnnxConfig final : public nvonnxparser::IOnnxConfig
{
public:
    BridgeOwnedOnnxConfig() = default;
    ~BridgeOwnedOnnxConfig() noexcept override = default;

    void setModelDtype(const nvinfer1::DataType data_type) noexcept override
    {
        data_type_ = data_type;
    }

    nvinfer1::DataType getModelDtype() const noexcept override
    {
        return data_type_;
    }

    char const* getModelFileName() const noexcept override
    {
        return model_file_name_.c_str();
    }

    void setModelFileName(char const* onnx_filename) noexcept override
    {
        model_file_name_ = onnx_filename != nullptr ? onnx_filename : "";
    }

    Verbosity getVerbosityLevel() const noexcept override
    {
        return verbosity_;
    }

    void addVerbosity() noexcept override
    {
        if (verbosity_ < INT32_MAX)
        {
            ++verbosity_;
        }
    }

    void reduceVerbosity() noexcept override
    {
        if (verbosity_ > 0)
        {
            --verbosity_;
        }
    }

    void setVerbosityLevel(const Verbosity verbosity) noexcept override
    {
        verbosity_ = verbosity >= 0 ? verbosity : 0;
    }

    char const* getTextFileName() const noexcept override
    {
        return text_file_name_.c_str();
    }

    void setTextFileName(char const* text_file_name) noexcept override
    {
        text_file_name_ = text_file_name != nullptr ? text_file_name : "";
    }

    char const* getFullTextFileName() const noexcept override
    {
        return full_text_file_name_.c_str();
    }

    void setFullTextFileName(char const* full_text_file_name) noexcept override
    {
        full_text_file_name_ = full_text_file_name != nullptr ? full_text_file_name : "";
    }

    bool getPrintLayerInfo() const noexcept override
    {
        return print_layer_info_;
    }

    void setPrintLayerInfo(const bool enabled) noexcept override
    {
        print_layer_info_ = enabled;
    }

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    void destroy() noexcept override
    {
        delete this;
    }
#endif

private:
    nvinfer1::DataType data_type_{nvinfer1::DataType::kFLOAT};
    Verbosity verbosity_{0};
    bool print_layer_info_{false};
    std::string model_file_name_;
    std::string text_file_name_;
    std::string full_text_file_name_;
};

static JYPPX_StatusCode get_onnx_config_payload_for_controls(
    JYPPX_TensorRtOnnxConfig* config,
    nvonnxparser::IOnnxConfig** out_config,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != JYPPX_TRT_EXPECTED_MAJOR)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, feature_name);
    }

    *out_config = get_payload<nvonnxparser::IOnnxConfig>(config);
    if (*out_config == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX config handle does not carry the expected TensorRT ONNX config payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode create_onnx_config_with_seh_guard(nvonnxparser::IOnnxConfig** out_config, const char* feature_name)
{
    *out_config = nullptr;
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_config = new (std::nothrow) BridgeOwnedOnnxConfig();
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_config = nullptr;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_config = new (std::nothrow) BridgeOwnedOnnxConfig();
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode get_onnx_config_model_dtype_with_seh_guard(nvonnxparser::IOnnxConfig* config, int32_t* out_data_type, const char* feature_name)
{
    *out_data_type = 0;
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_data_type = static_cast<int32_t>(config->getModelDtype());
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_data_type = 0;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_data_type = static_cast<int32_t>(config->getModelDtype());
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode set_onnx_config_model_dtype_with_seh_guard(nvonnxparser::IOnnxConfig* config, const int32_t data_type, const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        config->setModelDtype(static_cast<nvinfer1::DataType>(data_type));
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    config->setModelDtype(static_cast<nvinfer1::DataType>(data_type));
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode get_onnx_config_verbosity_with_seh_guard(nvonnxparser::IOnnxConfig* config, int32_t* out_verbosity, const char* feature_name)
{
    *out_verbosity = 0;
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_verbosity = static_cast<int32_t>(config->getVerbosityLevel());
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_verbosity = 0;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_verbosity = static_cast<int32_t>(config->getVerbosityLevel());
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode set_onnx_config_verbosity_with_seh_guard(nvonnxparser::IOnnxConfig* config, const int32_t verbosity, const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        config->setVerbosityLevel(static_cast<nvonnxparser::IOnnxConfig::Verbosity>(verbosity));
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    config->setVerbosityLevel(static_cast<nvonnxparser::IOnnxConfig::Verbosity>(verbosity));
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode add_onnx_config_verbosity_with_seh_guard(nvonnxparser::IOnnxConfig* config, const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        config->addVerbosity();
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    config->addVerbosity();
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode reduce_onnx_config_verbosity_with_seh_guard(nvonnxparser::IOnnxConfig* config, const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        config->reduceVerbosity();
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    config->reduceVerbosity();
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode get_onnx_config_string_with_seh_guard(
    nvonnxparser::IOnnxConfig* config,
    const int32_t selector,
    const char** out_value,
    const char* feature_name)
{
    *out_value = "";
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        switch (selector)
        {
        case 0:
            *out_value = config->getModelFileName();
            break;
        case 1:
            *out_value = config->getTextFileName();
            break;
        case 2:
            *out_value = config->getFullTextFileName();
            break;
        default:
            *out_value = "";
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unsupported ONNX config string selector.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        if (*out_value == nullptr)
        {
            *out_value = "";
        }

        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_value = "";
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    switch (selector)
    {
    case 0:
        *out_value = config->getModelFileName();
        break;
    case 1:
        *out_value = config->getTextFileName();
        break;
    case 2:
        *out_value = config->getFullTextFileName();
        break;
    default:
        *out_value = "";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unsupported ONNX config string selector.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (*out_value == nullptr)
    {
        *out_value = "";
    }

    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode set_onnx_config_string_with_seh_guard(
    nvonnxparser::IOnnxConfig* config,
    const int32_t selector,
    const char* value,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        switch (selector)
        {
        case 0:
            config->setModelFileName(value);
            break;
        case 1:
            config->setTextFileName(value);
            break;
        case 2:
            config->setFullTextFileName(value);
            break;
        default:
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unsupported ONNX config string selector.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    switch (selector)
    {
    case 0:
        config->setModelFileName(value);
        break;
    case 1:
        config->setTextFileName(value);
        break;
    case 2:
        config->setFullTextFileName(value);
        break;
    default:
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unsupported ONNX config string selector.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode get_onnx_config_print_layer_info_with_seh_guard(nvonnxparser::IOnnxConfig* config, JYPPX_Boolean* out_enabled, const char* feature_name)
{
    *out_enabled = JYPPX_FALSE;
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_enabled = config->getPrintLayerInfo() ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_enabled = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_enabled = config->getPrintLayerInfo() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode set_onnx_config_print_layer_info_with_seh_guard(nvonnxparser::IOnnxConfig* config, const JYPPX_Boolean enabled, const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        config->setPrintLayerInfo(enabled != JYPPX_FALSE);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    config->setPrintLayerInfo(enabled != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#endif
}
#endif

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(create)(JYPPX_TensorRtOnnxConfig** out_config)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_config, "out_config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_config = nullptr;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != JYPPX_TRT_EXPECTED_MAJOR)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX config creation");
    }

    try
    {
        nvonnxparser::IOnnxConfig* config = nullptr;
        status = create_onnx_config_with_seh_guard(&config, "ONNX config creation");
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        if (config == nullptr)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT returned a null ONNX config.");
            return JYPPX_STATUS_INVALID_STATE;
        }

        JYPPX_TensorRtObjectBase* handle = nullptr;
        status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_ONNX_CONFIG, config, &destroy_onnx_config_payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        *out_config = reinterpret_cast<JYPPX_TensorRtOnnxConfig*>(handle);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        *out_config = nullptr;
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config creation", exception.what());
    }
    catch (...)
    {
        *out_config = nullptr;
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config creation", "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX config creation");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(get_model_dtype)(JYPPX_TensorRtOnnxConfig* config, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_data_type = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IOnnxConfig* config_payload = nullptr;
    status = get_onnx_config_payload_for_controls(config, &config_payload, "ONNX config model dtype query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return get_onnx_config_model_dtype_with_seh_guard(config_payload, out_data_type, "ONNX config model dtype query");
    }
    catch (const std::exception& exception)
    {
        *out_data_type = 0;
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config model dtype query", exception.what());
    }
    catch (...)
    {
        *out_data_type = 0;
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config model dtype query", "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)config;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX config model dtype query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(set_model_dtype)(JYPPX_TensorRtOnnxConfig* config, int32_t data_type)
{
    if (data_type != 0 && data_type != 1 && data_type != 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX config model dtype must be Float, Half, or Int8.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IOnnxConfig* config_payload = nullptr;
    auto status = get_onnx_config_payload_for_controls(config, &config_payload, "ONNX config model dtype set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return set_onnx_config_model_dtype_with_seh_guard(config_payload, data_type, "ONNX config model dtype set");
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config model dtype set", exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config model dtype set", "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)config;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX config model dtype set");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(get_verbosity_level)(JYPPX_TensorRtOnnxConfig* config, int32_t* out_verbosity)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_verbosity, "out_verbosity");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_verbosity = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IOnnxConfig* config_payload = nullptr;
    status = get_onnx_config_payload_for_controls(config, &config_payload, "ONNX config verbosity query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return get_onnx_config_verbosity_with_seh_guard(config_payload, out_verbosity, "ONNX config verbosity query");
    }
    catch (const std::exception& exception)
    {
        *out_verbosity = 0;
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config verbosity query", exception.what());
    }
    catch (...)
    {
        *out_verbosity = 0;
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config verbosity query", "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)config;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX config verbosity query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(set_verbosity_level)(JYPPX_TensorRtOnnxConfig* config, int32_t verbosity)
{
    if (verbosity < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX config verbosity must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IOnnxConfig* config_payload = nullptr;
    auto status = get_onnx_config_payload_for_controls(config, &config_payload, "ONNX config verbosity set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return set_onnx_config_verbosity_with_seh_guard(config_payload, verbosity, "ONNX config verbosity set");
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config verbosity set", exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config verbosity set", "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)config;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX config verbosity set");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(add_verbosity)(JYPPX_TensorRtOnnxConfig* config)
{
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IOnnxConfig* config_payload = nullptr;
    auto status = get_onnx_config_payload_for_controls(config, &config_payload, "ONNX config add verbosity");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return add_onnx_config_verbosity_with_seh_guard(config_payload, "ONNX config add verbosity");
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config add verbosity", exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config add verbosity", "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)config;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX config add verbosity");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(reduce_verbosity)(JYPPX_TensorRtOnnxConfig* config)
{
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IOnnxConfig* config_payload = nullptr;
    auto status = get_onnx_config_payload_for_controls(config, &config_payload, "ONNX config reduce verbosity");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return reduce_onnx_config_verbosity_with_seh_guard(config_payload, "ONNX config reduce verbosity");
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config reduce verbosity", exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config reduce verbosity", "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)config;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX config reduce verbosity");
#endif
}

static JYPPX_StatusCode get_onnx_config_string(
    JYPPX_TensorRtOnnxConfig* config,
    const int32_t selector,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IOnnxConfig* config_payload = nullptr;
    status = get_onnx_config_payload_for_controls(config, &config_payload, feature_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        const char* value = "";
        status = get_onnx_config_string_with_seh_guard(config_payload, selector, &value, feature_name);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        return copy_string_to_buffer(value, output_buffer, output_buffer_size, out_required_size);
    }
    catch (const std::exception& exception)
    {
        *out_required_size = 0;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_required_size = 0;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)config;
    (void)selector;
    (void)output_buffer;
    (void)output_buffer_size;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)config;
    (void)selector;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
#endif
}

static JYPPX_StatusCode set_onnx_config_string(
    JYPPX_TensorRtOnnxConfig* config,
    const int32_t selector,
    const char* value,
    const char* feature_name)
{
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IOnnxConfig* config_payload = nullptr;
    auto status = get_onnx_config_payload_for_controls(config, &config_payload, feature_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return set_onnx_config_string_with_seh_guard(config_payload, selector, value != nullptr ? value : "", feature_name);
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)config;
    (void)selector;
    (void)value;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)config;
    (void)selector;
    (void)value;
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(get_model_file_name)(JYPPX_TensorRtOnnxConfig* config, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return get_onnx_config_string(config, 0, output_buffer, output_buffer_size, out_required_size, "ONNX config model file name query");
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(set_model_file_name)(JYPPX_TensorRtOnnxConfig* config, const char* value)
{
    return set_onnx_config_string(config, 0, value, "ONNX config model file name set");
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(get_text_file_name)(JYPPX_TensorRtOnnxConfig* config, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return get_onnx_config_string(config, 1, output_buffer, output_buffer_size, out_required_size, "ONNX config text file name query");
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(set_text_file_name)(JYPPX_TensorRtOnnxConfig* config, const char* value)
{
    return set_onnx_config_string(config, 1, value, "ONNX config text file name set");
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(get_full_text_file_name)(JYPPX_TensorRtOnnxConfig* config, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return get_onnx_config_string(config, 2, output_buffer, output_buffer_size, out_required_size, "ONNX config full text file name query");
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(set_full_text_file_name)(JYPPX_TensorRtOnnxConfig* config, const char* value)
{
    return set_onnx_config_string(config, 2, value, "ONNX config full text file name set");
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(get_print_layer_info)(JYPPX_TensorRtOnnxConfig* config, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_enabled = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IOnnxConfig* config_payload = nullptr;
    status = get_onnx_config_payload_for_controls(config, &config_payload, "ONNX config print layer info query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return get_onnx_config_print_layer_info_with_seh_guard(config_payload, out_enabled, "ONNX config print layer info query");
    }
    catch (const std::exception& exception)
    {
        *out_enabled = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config print layer info query", exception.what());
    }
    catch (...)
    {
        *out_enabled = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config print layer info query", "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)config;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX config print layer info query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_CONFIG_API(set_print_layer_info)(JYPPX_TensorRtOnnxConfig* config, JYPPX_Boolean enabled)
{
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IOnnxConfig* config_payload = nullptr;
    auto status = get_onnx_config_payload_for_controls(config, &config_payload, "ONNX config print layer info set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return set_onnx_config_print_layer_info_with_seh_guard(config_payload, enabled, "ONNX config print layer info set");
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config print layer info set", exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX config print layer info set", "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)config;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX config print layer info set");
#endif
}

#undef JYPPX_TRT_ONNX_CONFIG_API
#undef JYPPX_TRT_EXPECTED_MAJOR
