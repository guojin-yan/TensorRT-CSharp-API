#include "jyppx/cuda/runtime.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "../common/error_state.hpp"

namespace
{
constexpr size_t kMaximumSourceSize = 16U * 1024U * 1024U;
constexpr size_t kMaximumHeaderSize = 4U * 1024U * 1024U;
constexpr size_t kMaximumCombinedHeaderSize = 16U * 1024U * 1024U;
constexpr size_t kMaximumNameSize = 4096U;
constexpr size_t kMaximumOptionSize = 16384U;
constexpr size_t kMaximumOutputSize = 512U * 1024U * 1024U;
constexpr uint32_t kMaximumHeaderCount = 256U;
constexpr uint32_t kMaximumOptionCount = 256U;
constexpr uint32_t kMaximumNameExpressionCount = 256U;
constexpr int kNvrtcSuccess = 0;
constexpr int kNvrtcInvalidProgram = 4;

using NvrtcProgram = void*;
using NvrtcVersion = int (*)(int*, int*);
using NvrtcGetErrorString = const char* (*)(int);
using NvrtcCreateProgram = int (*)(NvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
using NvrtcDestroyProgram = int (*)(NvrtcProgram*);
using NvrtcCompileProgram = int (*)(NvrtcProgram, int, const char* const*);
using NvrtcGetProgramLogSize = int (*)(NvrtcProgram, size_t*);
using NvrtcGetProgramLog = int (*)(NvrtcProgram, char*);
using NvrtcGetArtifactSize = int (*)(NvrtcProgram, size_t*);
using NvrtcGetArtifact = int (*)(NvrtcProgram, char*);
using NvrtcAddNameExpression = int (*)(NvrtcProgram, const char*);
using NvrtcGetLoweredName = int (*)(NvrtcProgram, const char*, const char**);

#if defined(_WIN32)
using DynamicLibrary = HMODULE;
#else
using DynamicLibrary = void*;
#endif

template <typename TFunction>
TFunction function_pointer_from_address(void* address) noexcept
{
    static_assert(sizeof(TFunction) == sizeof(address), "Dynamic function pointer size mismatch.");
    TFunction function{};
    std::memcpy(&function, &address, sizeof(function));
    return function;
}

void* load_symbol(const DynamicLibrary library, const char* name) noexcept
{
#if defined(_WIN32)
    FARPROC address = GetProcAddress(library, name);
    void* result = nullptr;
    static_assert(sizeof(address) == sizeof(result), "Windows function pointer size mismatch.");
    std::memcpy(&result, &address, sizeof(result));
    return result;
#else
    return dlsym(library, name);
#endif
}

DynamicLibrary load_library(const char* name) noexcept
{
#if defined(_WIN32)
    return LoadLibraryA(name);
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void unload_library(const DynamicLibrary library) noexcept
{
    if (library == nullptr)
    {
        return;
    }
#if defined(_WIN32)
    FreeLibrary(library);
#else
    dlclose(library);
#endif
}

template <typename TFunction>
TFunction resolve(const DynamicLibrary library, const char* name) noexcept
{
    return function_pointer_from_address<TFunction>(load_symbol(library, name));
}

struct RtcApi final
{
    DynamicLibrary library{};
    std::string loaded_library_name;
    std::string dependency_diagnostic;
    int version_major{};
    int version_minor{};
    NvrtcVersion version{};
    NvrtcGetErrorString get_error_string{};
    NvrtcCreateProgram create_program{};
    NvrtcDestroyProgram destroy_program{};
    NvrtcCompileProgram compile_program{};
    NvrtcGetProgramLogSize get_log_size{};
    NvrtcGetProgramLog get_log{};
    NvrtcGetArtifactSize get_ptx_size{};
    NvrtcGetArtifact get_ptx{};
    NvrtcGetArtifactSize get_cubin_size{};
    NvrtcGetArtifact get_cubin{};
    NvrtcGetArtifactSize get_lto_ir_size{};
    NvrtcGetArtifact get_lto_ir{};
    NvrtcGetArtifactSize get_nvvm_size{};
    NvrtcGetArtifact get_nvvm{};
    NvrtcAddNameExpression add_name_expression{};
    NvrtcGetLoweredName get_lowered_name{};

    RtcApi()
    {
        std::vector<std::string> candidates;
#if defined(_WIN32)
        char* explicit_library = nullptr;
        size_t explicit_library_size = 0;
        if (_dupenv_s(&explicit_library, &explicit_library_size, "JYPPX_NVRTC_LIBRARY") == 0 &&
            explicit_library != nullptr && explicit_library[0] != '\0')
        {
            candidates.emplace_back(explicit_library);
        }
        std::free(explicit_library);
#else
        const char* explicit_library = std::getenv("JYPPX_NVRTC_LIBRARY");
        if (explicit_library != nullptr && explicit_library[0] != '\0')
        {
            candidates.emplace_back(explicit_library);
        }
#endif

#if defined(_WIN32)
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        candidates.emplace_back("nvrtc64_130_0.dll");
#elif JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION < 12000
        candidates.emplace_back("nvrtc64_112_0.dll");
#else
        candidates.emplace_back("nvrtc64_120_0.dll");
#endif
        candidates.emplace_back("nvrtc64_130_0.dll");
        candidates.emplace_back("nvrtc64_120_0.dll");
        candidates.emplace_back("nvrtc64_112_0.dll");
#else
        candidates.emplace_back("libnvrtc.so");
        candidates.emplace_back("libnvrtc.so.13");
        candidates.emplace_back("libnvrtc.so.12");
        candidates.emplace_back("libnvrtc.so.11.2");
#endif

        std::vector<std::string> unique_candidates;
        for (const std::string& candidate : candidates)
        {
            if (std::find(unique_candidates.begin(), unique_candidates.end(), candidate) == unique_candidates.end())
            {
                unique_candidates.push_back(candidate);
            }
        }

        for (const std::string& candidate : unique_candidates)
        {
            DynamicLibrary candidate_library = load_library(candidate.c_str());
            if (candidate_library == nullptr)
            {
                continue;
            }

            NvrtcVersion candidate_version = resolve<NvrtcVersion>(candidate_library, "nvrtcVersion");
            NvrtcGetErrorString candidate_get_error_string = resolve<NvrtcGetErrorString>(candidate_library, "nvrtcGetErrorString");
            NvrtcCreateProgram candidate_create_program = resolve<NvrtcCreateProgram>(candidate_library, "nvrtcCreateProgram");
            NvrtcDestroyProgram candidate_destroy_program = resolve<NvrtcDestroyProgram>(candidate_library, "nvrtcDestroyProgram");
            NvrtcCompileProgram candidate_compile_program = resolve<NvrtcCompileProgram>(candidate_library, "nvrtcCompileProgram");
            NvrtcGetProgramLogSize candidate_get_log_size = resolve<NvrtcGetProgramLogSize>(candidate_library, "nvrtcGetProgramLogSize");
            NvrtcGetProgramLog candidate_get_log = resolve<NvrtcGetProgramLog>(candidate_library, "nvrtcGetProgramLog");
            NvrtcGetArtifactSize candidate_get_ptx_size = resolve<NvrtcGetArtifactSize>(candidate_library, "nvrtcGetPTXSize");
            NvrtcGetArtifact candidate_get_ptx = resolve<NvrtcGetArtifact>(candidate_library, "nvrtcGetPTX");
            NvrtcGetArtifactSize candidate_get_cubin_size = resolve<NvrtcGetArtifactSize>(candidate_library, "nvrtcGetCUBINSize");
            NvrtcGetArtifact candidate_get_cubin = resolve<NvrtcGetArtifact>(candidate_library, "nvrtcGetCUBIN");
            NvrtcAddNameExpression candidate_add_name_expression = resolve<NvrtcAddNameExpression>(candidate_library, "nvrtcAddNameExpression");
            NvrtcGetLoweredName candidate_get_lowered_name = resolve<NvrtcGetLoweredName>(candidate_library, "nvrtcGetLoweredName");

            const bool has_required_symbols = candidate_version != nullptr &&
                candidate_get_error_string != nullptr && candidate_create_program != nullptr &&
                candidate_destroy_program != nullptr && candidate_compile_program != nullptr &&
                candidate_get_log_size != nullptr && candidate_get_log != nullptr &&
                candidate_get_ptx_size != nullptr && candidate_get_ptx != nullptr &&
                candidate_get_cubin_size != nullptr && candidate_get_cubin != nullptr &&
                candidate_add_name_expression != nullptr && candidate_get_lowered_name != nullptr;
            int candidate_major = 0;
            int candidate_minor = 0;
            if (!has_required_symbols || candidate_version(&candidate_major, &candidate_minor) != kNvrtcSuccess)
            {
                unload_library(candidate_library);
                continue;
            }

            library = candidate_library;
            loaded_library_name = candidate;
            version_major = candidate_major;
            version_minor = candidate_minor;
            version = candidate_version;
            get_error_string = candidate_get_error_string;
            create_program = candidate_create_program;
            destroy_program = candidate_destroy_program;
            compile_program = candidate_compile_program;
            get_log_size = candidate_get_log_size;
            get_log = candidate_get_log;
            get_ptx_size = candidate_get_ptx_size;
            get_ptx = candidate_get_ptx;
            get_cubin_size = candidate_get_cubin_size;
            get_cubin = candidate_get_cubin;
            add_name_expression = candidate_add_name_expression;
            get_lowered_name = candidate_get_lowered_name;
            get_lto_ir_size = resolve<NvrtcGetArtifactSize>(library, "nvrtcGetLTOIRSize");
            get_lto_ir = resolve<NvrtcGetArtifact>(library, "nvrtcGetLTOIR");
            get_nvvm_size = resolve<NvrtcGetArtifactSize>(library, "nvrtcGetNVVMSize");
            get_nvvm = resolve<NvrtcGetArtifact>(library, "nvrtcGetNVVM");
            dependency_diagnostic.clear();
            return;
        }

        std::ostringstream message;
        message << "NVRTC is an optional dependency and could not be loaded. Set JYPPX_NVRTC_LIBRARY to an exact NVRTC library path or make a matching Toolkit runtime visible. Tried:";
        for (const std::string& candidate : unique_candidates)
        {
            message << ' ' << candidate;
        }
        dependency_diagnostic = message.str();
    }

    ~RtcApi()
    {
        unload_library(library);
    }

    RtcApi(const RtcApi&) = delete;
    RtcApi& operator=(const RtcApi&) = delete;

    bool available() const noexcept
    {
        return library != nullptr;
    }
};

RtcApi& rtc_api()
{
    static RtcApi api;
    return api;
}

struct RtcHeader final
{
    std::string source;
    std::string include_name;
};

struct RtcProgramObject final
{
    std::string source;
    std::string program_name;
    std::vector<RtcHeader> headers;
    std::vector<std::string> name_expressions;
    std::vector<std::string> compile_options;
    NvrtcProgram program{};
    int compiler_result{kNvrtcInvalidProgram};
    bool compile_attempted{};
};

std::mutex g_program_mutex;
std::unordered_set<RtcProgramObject*> g_programs;

void set_rtc_error(const std::string& message)
{
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_CUDA, message.c_str());
}

JYPPX_StatusCode invalid_argument(const char* operation, const char* message)
{
    std::ostringstream text;
    text << operation << ": " << message;
    set_rtc_error(text.str());
    return JYPPX_STATUS_INVALID_ARGUMENT;
}

JYPPX_StatusCode invalid_state(const char* operation, const char* message)
{
    std::ostringstream text;
    text << operation << ": " << message;
    set_rtc_error(text.str());
    return JYPPX_STATUS_INVALID_STATE;
}

JYPPX_StatusCode dependency_missing(const char* operation)
{
    std::ostringstream text;
    text << operation << ": " << rtc_api().dependency_diagnostic;
    set_rtc_error(text.str());
    return JYPPX_STATUS_DEPENDENCY_MISSING;
}

JYPPX_StatusCode report_nvrtc_error(const char* operation, const int result)
{
    const char* description = rtc_api().get_error_string != nullptr ? rtc_api().get_error_string(result) : nullptr;
    std::ostringstream text;
    text << operation << " failed with NVRTC result " << result;
    if (description != nullptr && description[0] != '\0')
    {
        text << " (" << description << ')';
    }
    set_rtc_error(text.str());
    return result == 1 ? JYPPX_STATUS_OUT_OF_MEMORY : JYPPX_STATUS_RUNTIME_ERROR;
}

template <typename TCallable>
JYPPX_StatusCode run_rtc_noexcept(const char* operation, TCallable&& callable) noexcept
{
    try
    {
        return callable();
    }
    catch (const std::bad_alloc&)
    {
        std::ostringstream text;
        text << operation << ": native allocation failed inside the CUDA RTC bridge.";
        set_rtc_error(text.str());
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }
    catch (...)
    {
        std::ostringstream text;
        text << operation << ": a C++ exception was contained by the CUDA RTC bridge.";
        set_rtc_error(text.str());
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
}

bool is_valid_utf8(const uint8_t* bytes, const size_t size) noexcept
{
    size_t index = 0;
    while (index < size)
    {
        const uint8_t first = bytes[index++];
        if (first <= 0x7FU)
        {
            continue;
        }

        uint32_t code_point = 0;
        size_t trailing = 0;
        if ((first & 0xE0U) == 0xC0U)
        {
            code_point = first & 0x1FU;
            trailing = 1;
            if (code_point == 0)
            {
                return false;
            }
        }
        else if ((first & 0xF0U) == 0xE0U)
        {
            code_point = first & 0x0FU;
            trailing = 2;
        }
        else if ((first & 0xF8U) == 0xF0U)
        {
            code_point = first & 0x07U;
            trailing = 3;
        }
        else
        {
            return false;
        }

        if (index + trailing > size)
        {
            return false;
        }
        for (size_t offset = 0; offset < trailing; ++offset)
        {
            const uint8_t next = bytes[index++];
            if ((next & 0xC0U) != 0x80U)
            {
                return false;
            }
            code_point = (code_point << 6U) | (next & 0x3FU);
        }

        if ((trailing == 1 && code_point < 0x80U) ||
            (trailing == 2 && code_point < 0x800U) ||
            (trailing == 3 && code_point < 0x10000U) ||
            code_point > 0x10FFFFU || (code_point >= 0xD800U && code_point <= 0xDFFFU))
        {
            return false;
        }
    }
    return true;
}

JYPPX_StatusCode copy_utf8_input(
    const char* operation,
    const char* input_name,
    const uint8_t* input,
    const size_t input_size,
    const size_t maximum_size,
    const bool allow_empty,
    std::string& output)
{
    if (input_size == 0 && !allow_empty)
    {
        std::ostringstream message;
        message << input_name << " must not be empty.";
        return invalid_argument(operation, message.str().c_str());
    }
    if (input_size > maximum_size)
    {
        std::ostringstream message;
        message << input_name << " exceeds the bounded RTC input limit.";
        return invalid_argument(operation, message.str().c_str());
    }
    if (input_size != 0 && input == nullptr)
    {
        std::ostringstream message;
        message << input_name << " pointer must not be null when its size is non-zero.";
        return invalid_argument(operation, message.str().c_str());
    }
    if (input_size != 0 && std::memchr(input, 0, input_size) != nullptr)
    {
        std::ostringstream message;
        message << input_name << " must not contain an embedded NUL.";
        return invalid_argument(operation, message.str().c_str());
    }
    if (input_size != 0 && !is_valid_utf8(input, input_size))
    {
        std::ostringstream message;
        message << input_name << " must contain valid UTF-8.";
        return invalid_argument(operation, message.str().c_str());
    }

    output.assign(reinterpret_cast<const char*>(input), input_size);
    return JYPPX_STATUS_OK;
}

size_t bounded_c_string_length(const char* value, const size_t maximum) noexcept
{
    if (value == nullptr)
    {
        return maximum + 1U;
    }
    size_t length = 0;
    while (length <= maximum && value[length] != '\0')
    {
        ++length;
    }
    return length;
}

JYPPX_StatusCode lookup_program_locked(JYPPX_CudaRtcProgram* handle, RtcProgramObject** out_program)
{
    if (handle == nullptr)
    {
        return invalid_argument("CUDA RTC program", "program handle must not be null.");
    }
    auto* program = reinterpret_cast<RtcProgramObject*>(handle);
    if (g_programs.find(program) == g_programs.end())
    {
        return invalid_state("CUDA RTC program", "program handle is disposed or does not belong to this bridge.");
    }
    *out_program = program;
    return JYPPX_STATUS_OK;
}

void destroy_nvrtc_program(RtcProgramObject& program) noexcept
{
    if (program.program != nullptr && rtc_api().destroy_program != nullptr)
    {
        NvrtcProgram value = program.program;
        program.program = nullptr;
        (void)rtc_api().destroy_program(&value);
    }
}

JYPPX_StatusCode copy_text_to_buffer(
    const std::string& value,
    uint8_t* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size)
{
    if (out_required_size == nullptr)
    {
        return invalid_argument("CUDA RTC buffer copy", "out_required_size must not be null.");
    }
    if (value.size() >= kMaximumOutputSize)
    {
        return invalid_state("CUDA RTC buffer copy", "reported output exceeds the bounded RTC output limit.");
    }

    const size_t required_size = value.size() + 1U;
    *out_required_size = required_size;
    if (output_buffer == nullptr || output_buffer_size == 0)
    {
        return JYPPX_STATUS_OK;
    }
    if (output_buffer_size < required_size)
    {
        output_buffer[0] = 0;
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }
    if (!value.empty())
    {
        std::memcpy(output_buffer, value.data(), value.size());
    }
    output_buffer[value.size()] = 0;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode query_artifact(
    RtcProgramObject& program,
    const int32_t artifact_kind,
    NvrtcGetArtifactSize* out_size_function,
    NvrtcGetArtifact* out_copy_function,
    JYPPX_Boolean* out_available,
    size_t* out_size)
{
    if (!program.compile_attempted || program.compiler_result != kNvrtcSuccess || program.program == nullptr)
    {
        return invalid_state("CUDA RTC artifact query", "a successful compilation is required before artifacts can be queried.");
    }

    NvrtcGetArtifactSize size_function = nullptr;
    NvrtcGetArtifact copy_function = nullptr;
    switch (artifact_kind)
    {
    case JYPPX_CUDA_RTC_ARTIFACT_PTX:
        size_function = rtc_api().get_ptx_size;
        copy_function = rtc_api().get_ptx;
        break;
    case JYPPX_CUDA_RTC_ARTIFACT_CUBIN:
        size_function = rtc_api().get_cubin_size;
        copy_function = rtc_api().get_cubin;
        break;
    case JYPPX_CUDA_RTC_ARTIFACT_LTO_IR:
        size_function = rtc_api().get_lto_ir_size;
        copy_function = rtc_api().get_lto_ir;
        break;
    default:
        return invalid_argument("CUDA RTC artifact query", "artifact_kind is not a supported copied artifact kind.");
    }

    *out_available = JYPPX_FALSE;
    *out_size = 0;
    if (out_size_function != nullptr)
    {
        *out_size_function = size_function;
    }
    if (out_copy_function != nullptr)
    {
        *out_copy_function = copy_function;
    }
    if (size_function == nullptr || copy_function == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    size_t size = 0;
    const int result = size_function(program.program, &size);
    if (result == kNvrtcInvalidProgram)
    {
        return JYPPX_STATUS_OK;
    }
    if (result != kNvrtcSuccess)
    {
        return report_nvrtc_error("NVRTC artifact size query", result);
    }
    if (size == 0)
    {
        return JYPPX_STATUS_OK;
    }
    if (size > kMaximumOutputSize)
    {
        return invalid_state("CUDA RTC artifact query", "NVRTC reported an oversized artifact.");
    }

    *out_available = JYPPX_TRUE;
    *out_size = size;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode query_capability_impl(JYPPX_CudaRtcCapabilityInfo* out_info) noexcept
{
    return run_rtc_noexcept("CUDA RTC capability query", [&]() {
        if (out_info == nullptr)
        {
            return invalid_argument("CUDA RTC capability query", "out_info must not be null.");
        }
        std::memset(out_info, 0, sizeof(*out_info));
        RtcApi& api = rtc_api();
        out_info->dependency_available = api.available() ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->version_major = api.version_major;
        out_info->version_minor = api.version_minor;
        out_info->supports_ptx = api.get_ptx_size != nullptr && api.get_ptx != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->supports_cubin = api.get_cubin_size != nullptr && api.get_cubin != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->supports_lto_ir = api.get_lto_ir_size != nullptr && api.get_lto_ir != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->supports_deprecated_nvvm = api.get_nvvm_size != nullptr && api.get_nvvm != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->supports_name_expressions = api.add_name_expression != nullptr && api.get_lowered_name != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    });
}

JYPPX_StatusCode get_loader_text_impl(
    const bool library_name,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size) noexcept
{
    return run_rtc_noexcept("CUDA RTC loader diagnostic", [&]() {
        RtcApi& api = rtc_api();
        const std::string& value = library_name ? api.loaded_library_name : api.dependency_diagnostic;
        return copy_text_to_buffer(value, reinterpret_cast<uint8_t*>(output_buffer), output_buffer_size, out_required_size);
    });
}

JYPPX_StatusCode create_program_impl(
    const uint8_t* source,
    const size_t source_size,
    const uint8_t* program_name,
    const size_t program_name_size,
    JYPPX_CudaRtcProgram** out_program) noexcept
{
    return run_rtc_noexcept("CUDA RTC program create", [&]() {
        if (out_program == nullptr)
        {
            return invalid_argument("CUDA RTC program create", "out_program must not be null.");
        }
        *out_program = nullptr;
        if (!rtc_api().available())
        {
            return dependency_missing("CUDA RTC program create");
        }

        std::unique_ptr<RtcProgramObject> program(new RtcProgramObject());
        JYPPX_StatusCode status = copy_utf8_input("CUDA RTC program create", "source", source, source_size, kMaximumSourceSize, false, program->source);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
        status = copy_utf8_input("CUDA RTC program create", "program_name", program_name, program_name_size, kMaximumNameSize, false, program->program_name);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        std::lock_guard<std::mutex> lock(g_program_mutex);
        RtcProgramObject* raw_program = program.release();
        g_programs.insert(raw_program);
        *out_program = reinterpret_cast<JYPPX_CudaRtcProgram*>(raw_program);
        return JYPPX_STATUS_OK;
    });
}

JYPPX_StatusCode add_header_impl(
    JYPPX_CudaRtcProgram* handle,
    const uint8_t* header_source,
    const size_t header_source_size,
    const uint8_t* include_name,
    const size_t include_name_size) noexcept
{
    return run_rtc_noexcept("CUDA RTC add header", [&]() {
        std::lock_guard<std::mutex> lock(g_program_mutex);
        RtcProgramObject* program = nullptr;
        JYPPX_StatusCode status = lookup_program_locked(handle, &program);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
        if (program->headers.size() >= kMaximumHeaderCount)
        {
            return invalid_argument("CUDA RTC add header", "header count exceeds the bounded RTC limit.");
        }

        size_t combined_size = header_source_size;
        for (const RtcHeader& existing : program->headers)
        {
            if (existing.source.size() > kMaximumCombinedHeaderSize - combined_size)
            {
                return invalid_argument("CUDA RTC add header", "combined header bytes exceed the bounded RTC limit.");
            }
            combined_size += existing.source.size();
        }
        if (combined_size > kMaximumCombinedHeaderSize)
        {
            return invalid_argument("CUDA RTC add header", "combined header bytes exceed the bounded RTC limit.");
        }

        RtcHeader header;
        status = copy_utf8_input("CUDA RTC add header", "header_source", header_source, header_source_size, kMaximumHeaderSize, true, header.source);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
        status = copy_utf8_input("CUDA RTC add header", "include_name", include_name, include_name_size, kMaximumNameSize, false, header.include_name);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
        for (const RtcHeader& existing : program->headers)
        {
            if (existing.include_name == header.include_name)
            {
                return invalid_argument("CUDA RTC add header", "duplicate include_name values are not allowed.");
            }
        }
        program->headers.push_back(std::move(header));
        return JYPPX_STATUS_OK;
    });
}

JYPPX_StatusCode add_name_expression_impl(
    JYPPX_CudaRtcProgram* handle,
    const uint8_t* expression,
    const size_t expression_size) noexcept
{
    return run_rtc_noexcept("CUDA RTC add name expression", [&]() {
        std::lock_guard<std::mutex> lock(g_program_mutex);
        RtcProgramObject* program = nullptr;
        JYPPX_StatusCode status = lookup_program_locked(handle, &program);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
        if (program->name_expressions.size() >= kMaximumNameExpressionCount)
        {
            return invalid_argument("CUDA RTC add name expression", "name expression count exceeds the bounded RTC limit.");
        }

        std::string retained_expression;
        status = copy_utf8_input("CUDA RTC add name expression", "expression", expression, expression_size, kMaximumNameSize, false, retained_expression);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
        if (std::find(program->name_expressions.begin(), program->name_expressions.end(), retained_expression) != program->name_expressions.end())
        {
            return invalid_argument("CUDA RTC add name expression", "duplicate name expressions are not allowed.");
        }
        program->name_expressions.push_back(std::move(retained_expression));
        return JYPPX_STATUS_OK;
    });
}

JYPPX_StatusCode compile_program_impl(
    JYPPX_CudaRtcProgram* handle,
    const char** options,
    const uint32_t option_count,
    int32_t* out_compiler_result) noexcept
{
    return run_rtc_noexcept("CUDA RTC compile", [&]() {
        if (out_compiler_result == nullptr)
        {
            return invalid_argument("CUDA RTC compile", "out_compiler_result must not be null.");
        }
        *out_compiler_result = kNvrtcInvalidProgram;
        if (option_count > kMaximumOptionCount || (option_count != 0 && options == nullptr))
        {
            return invalid_argument("CUDA RTC compile", "options pointer/count exceeds the bounded RTC contract.");
        }

        std::lock_guard<std::mutex> lock(g_program_mutex);
        RtcProgramObject* program = nullptr;
        JYPPX_StatusCode status = lookup_program_locked(handle, &program);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        std::vector<std::string> retained_options;
        retained_options.reserve(option_count);
        for (uint32_t index = 0; index < option_count; ++index)
        {
            const size_t length = bounded_c_string_length(options[index], kMaximumOptionSize);
            if (length == 0 || length > kMaximumOptionSize)
            {
                return invalid_argument("CUDA RTC compile", "each option must be a bounded non-empty NUL-terminated string.");
            }
            const auto* bytes = reinterpret_cast<const uint8_t*>(options[index]);
            if (!is_valid_utf8(bytes, length))
            {
                return invalid_argument("CUDA RTC compile", "each option must contain valid UTF-8.");
            }
            std::string option(options[index], length);
            if (std::find(retained_options.begin(), retained_options.end(), option) != retained_options.end())
            {
                return invalid_argument("CUDA RTC compile", "duplicate compile options are not allowed.");
            }
            retained_options.push_back(std::move(option));
        }

        destroy_nvrtc_program(*program);
        program->compile_options = std::move(retained_options);
        program->compile_attempted = true;
        program->compiler_result = kNvrtcInvalidProgram;

        std::vector<const char*> header_sources;
        std::vector<const char*> header_names;
        header_sources.reserve(program->headers.size());
        header_names.reserve(program->headers.size());
        for (const RtcHeader& header : program->headers)
        {
            header_sources.push_back(header.source.c_str());
            header_names.push_back(header.include_name.c_str());
        }

        int result = rtc_api().create_program(
            &program->program,
            program->source.c_str(),
            program->program_name.c_str(),
            static_cast<int>(program->headers.size()),
            header_sources.empty() ? nullptr : header_sources.data(),
            header_names.empty() ? nullptr : header_names.data());
        if (result != kNvrtcSuccess)
        {
            program->compiler_result = result;
            *out_compiler_result = result;
            return JYPPX_STATUS_OK;
        }

        for (const std::string& expression : program->name_expressions)
        {
            result = rtc_api().add_name_expression(program->program, expression.c_str());
            if (result != kNvrtcSuccess)
            {
                program->compiler_result = result;
                *out_compiler_result = result;
                return JYPPX_STATUS_OK;
            }
        }

        std::vector<const char*> option_pointers;
        option_pointers.reserve(program->compile_options.size());
        for (const std::string& option : program->compile_options)
        {
            option_pointers.push_back(option.c_str());
        }
        result = rtc_api().compile_program(
            program->program,
            static_cast<int>(option_pointers.size()),
            option_pointers.empty() ? nullptr : option_pointers.data());
        program->compiler_result = result;
        *out_compiler_result = result;
        return JYPPX_STATUS_OK;
    });
}

JYPPX_StatusCode get_log_impl(
    JYPPX_CudaRtcProgram* handle,
    uint8_t* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size) noexcept
{
    return run_rtc_noexcept("CUDA RTC log copy", [&]() {
        if (out_required_size == nullptr)
        {
            return invalid_argument("CUDA RTC log copy", "out_required_size must not be null.");
        }
        *out_required_size = 0;
        std::lock_guard<std::mutex> lock(g_program_mutex);
        RtcProgramObject* program = nullptr;
        JYPPX_StatusCode status = lookup_program_locked(handle, &program);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
        if (!program->compile_attempted)
        {
            return invalid_state("CUDA RTC log copy", "compile must be attempted before the program log can be copied.");
        }
        if (program->program == nullptr)
        {
            return copy_text_to_buffer(std::string(), output_buffer, output_buffer_size, out_required_size);
        }

        size_t required_size = 0;
        const int size_result = rtc_api().get_log_size(program->program, &required_size);
        if (size_result != kNvrtcSuccess)
        {
            return report_nvrtc_error("nvrtcGetProgramLogSize", size_result);
        }
        if (required_size == 0 || required_size > kMaximumOutputSize)
        {
            return invalid_state("CUDA RTC log copy", "NVRTC reported an empty or oversized log buffer size.");
        }
        *out_required_size = required_size;
        if (output_buffer == nullptr || output_buffer_size == 0)
        {
            return JYPPX_STATUS_OK;
        }
        if (output_buffer_size < required_size)
        {
            output_buffer[0] = 0;
            return JYPPX_STATUS_BUFFER_TOO_SMALL;
        }
        const int copy_result = rtc_api().get_log(program->program, reinterpret_cast<char*>(output_buffer));
        return copy_result == kNvrtcSuccess
            ? JYPPX_STATUS_OK
            : report_nvrtc_error("nvrtcGetProgramLog", copy_result);
    });
}

JYPPX_StatusCode try_get_artifact_size_impl(
    JYPPX_CudaRtcProgram* handle,
    const int32_t artifact_kind,
    JYPPX_Boolean* out_available,
    size_t* out_size) noexcept
{
    return run_rtc_noexcept("CUDA RTC artifact size", [&]() {
        if (out_available == nullptr || out_size == nullptr)
        {
            return invalid_argument("CUDA RTC artifact size", "out_available and out_size must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_program_mutex);
        RtcProgramObject* program = nullptr;
        JYPPX_StatusCode status = lookup_program_locked(handle, &program);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
        return query_artifact(*program, artifact_kind, nullptr, nullptr, out_available, out_size);
    });
}

JYPPX_StatusCode copy_artifact_impl(
    JYPPX_CudaRtcProgram* handle,
    const int32_t artifact_kind,
    uint8_t* output_buffer,
    const size_t output_buffer_size,
    size_t* out_written_size) noexcept
{
    return run_rtc_noexcept("CUDA RTC artifact copy", [&]() {
        if (out_written_size == nullptr)
        {
            return invalid_argument("CUDA RTC artifact copy", "out_written_size must not be null.");
        }
        *out_written_size = 0;
        std::lock_guard<std::mutex> lock(g_program_mutex);
        RtcProgramObject* program = nullptr;
        JYPPX_StatusCode status = lookup_program_locked(handle, &program);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        NvrtcGetArtifact copy_function = nullptr;
        JYPPX_Boolean available = JYPPX_FALSE;
        size_t required_size = 0;
        status = query_artifact(*program, artifact_kind, nullptr, &copy_function, &available, &required_size);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
        if (available == JYPPX_FALSE || copy_function == nullptr)
        {
            return JYPPX_STATUS_NOT_SUPPORTED;
        }
        *out_written_size = required_size;
        if (output_buffer == nullptr || output_buffer_size < required_size)
        {
            return output_buffer == nullptr && output_buffer_size == 0 ? JYPPX_STATUS_OK : JYPPX_STATUS_BUFFER_TOO_SMALL;
        }
        const int result = copy_function(program->program, reinterpret_cast<char*>(output_buffer));
        return result == kNvrtcSuccess ? JYPPX_STATUS_OK : report_nvrtc_error("NVRTC artifact copy", result);
    });
}

JYPPX_StatusCode get_lowered_name_impl(
    JYPPX_CudaRtcProgram* handle,
    const uint32_t expression_index,
    uint8_t* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size) noexcept
{
    return run_rtc_noexcept("CUDA RTC lowered name copy", [&]() {
        std::lock_guard<std::mutex> lock(g_program_mutex);
        RtcProgramObject* program = nullptr;
        JYPPX_StatusCode status = lookup_program_locked(handle, &program);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
        if (!program->compile_attempted || program->compiler_result != kNvrtcSuccess || program->program == nullptr)
        {
            return invalid_state("CUDA RTC lowered name copy", "a successful compilation is required before lowered names can be copied.");
        }
        if (expression_index >= program->name_expressions.size())
        {
            return invalid_argument("CUDA RTC lowered name copy", "expression_index is outside the retained name-expression list.");
        }

        const char* lowered_name = nullptr;
        const int result = rtc_api().get_lowered_name(
            program->program,
            program->name_expressions[expression_index].c_str(),
            &lowered_name);
        if (result != kNvrtcSuccess)
        {
            return report_nvrtc_error("nvrtcGetLoweredName", result);
        }
        if (lowered_name == nullptr)
        {
            return invalid_state("CUDA RTC lowered name copy", "NVRTC returned a null lowered name.");
        }
        const size_t length = bounded_c_string_length(lowered_name, kMaximumNameSize);
        if (length > kMaximumNameSize)
        {
            return invalid_state("CUDA RTC lowered name copy", "NVRTC returned an unterminated or oversized lowered name.");
        }
        return copy_text_to_buffer(std::string(lowered_name, length), output_buffer, output_buffer_size, out_required_size);
    });
}

JYPPX_StatusCode destroy_program_impl(JYPPX_CudaRtcProgram* handle) noexcept
{
    return run_rtc_noexcept("CUDA RTC program destroy", [&]() {
        if (handle == nullptr)
        {
            return JYPPX_STATUS_OK;
        }

        std::unique_ptr<RtcProgramObject> program;
        {
            std::lock_guard<std::mutex> lock(g_program_mutex);
            auto* candidate = reinterpret_cast<RtcProgramObject*>(handle);
            const auto found = g_programs.find(candidate);
            if (found == g_programs.end())
            {
                return JYPPX_STATUS_OK;
            }
            g_programs.erase(found);
            program.reset(candidate);
        }
        destroy_nvrtc_program(*program);
        return JYPPX_STATUS_OK;
    });
}

JYPPX_StatusCode report_rtc_seh(const char* operation) noexcept
{
    return run_rtc_noexcept("CUDA RTC SEH report", [&]() {
        std::ostringstream message;
        message << operation << ": a Windows structured exception was contained by the CUDA RTC bridge.";
        set_rtc_error(message.str());
        return JYPPX_STATUS_RUNTIME_ERROR;
    });
}

#if defined(_WIN32) && defined(_MSC_VER)
#define JYPPX_CUDA_RTC_GUARD(operation, expression) \
    __try { return (expression); } \
    __except (EXCEPTION_EXECUTE_HANDLER) { return report_rtc_seh(operation); }
#else
#define JYPPX_CUDA_RTC_GUARD(operation, expression) return (expression)
#endif
}

JYPPX_StatusCode jyppx_cuda_rtc_query_capability_safe(JYPPX_CudaRtcCapabilityInfo* out_info)
{
    JYPPX_CUDA_RTC_GUARD("CUDA RTC capability query", query_capability_impl(out_info));
}

JYPPX_StatusCode jyppx_cuda_rtc_get_loaded_library_name_safe(char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    JYPPX_CUDA_RTC_GUARD("CUDA RTC loaded library name", get_loader_text_impl(true, output_buffer, output_buffer_size, out_required_size));
}

JYPPX_StatusCode jyppx_cuda_rtc_get_dependency_diagnostic_safe(char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    JYPPX_CUDA_RTC_GUARD("CUDA RTC dependency diagnostic", get_loader_text_impl(false, output_buffer, output_buffer_size, out_required_size));
}

JYPPX_StatusCode jyppx_cuda_rtc_program_create_safe(const uint8_t* source, size_t source_size, const uint8_t* program_name, size_t program_name_size, JYPPX_CudaRtcProgram** out_program)
{
    JYPPX_CUDA_RTC_GUARD("CUDA RTC program create", create_program_impl(source, source_size, program_name, program_name_size, out_program));
}

JYPPX_StatusCode jyppx_cuda_rtc_program_add_header_safe(JYPPX_CudaRtcProgram* program, const uint8_t* header_source, size_t header_source_size, const uint8_t* include_name, size_t include_name_size)
{
    JYPPX_CUDA_RTC_GUARD("CUDA RTC add header", add_header_impl(program, header_source, header_source_size, include_name, include_name_size));
}

JYPPX_StatusCode jyppx_cuda_rtc_program_add_name_expression_safe(JYPPX_CudaRtcProgram* program, const uint8_t* expression, size_t expression_size)
{
    JYPPX_CUDA_RTC_GUARD("CUDA RTC add name expression", add_name_expression_impl(program, expression, expression_size));
}

JYPPX_StatusCode jyppx_cuda_rtc_program_compile_safe(JYPPX_CudaRtcProgram* program, const char** options, uint32_t option_count, int32_t* out_compiler_result)
{
    JYPPX_CUDA_RTC_GUARD("CUDA RTC compile", compile_program_impl(program, options, option_count, out_compiler_result));
}

JYPPX_StatusCode jyppx_cuda_rtc_program_get_log_safe(JYPPX_CudaRtcProgram* program, uint8_t* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    JYPPX_CUDA_RTC_GUARD("CUDA RTC log copy", get_log_impl(program, output_buffer, output_buffer_size, out_required_size));
}

JYPPX_StatusCode jyppx_cuda_rtc_program_try_get_artifact_size_safe(JYPPX_CudaRtcProgram* program, int32_t artifact_kind, JYPPX_Boolean* out_available, size_t* out_size)
{
    JYPPX_CUDA_RTC_GUARD("CUDA RTC artifact size", try_get_artifact_size_impl(program, artifact_kind, out_available, out_size));
}

JYPPX_StatusCode jyppx_cuda_rtc_program_copy_artifact_safe(JYPPX_CudaRtcProgram* program, int32_t artifact_kind, uint8_t* output_buffer, size_t output_buffer_size, size_t* out_written_size)
{
    JYPPX_CUDA_RTC_GUARD("CUDA RTC artifact copy", copy_artifact_impl(program, artifact_kind, output_buffer, output_buffer_size, out_written_size));
}

JYPPX_StatusCode jyppx_cuda_rtc_program_get_lowered_name_safe(JYPPX_CudaRtcProgram* program, uint32_t expression_index, uint8_t* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    JYPPX_CUDA_RTC_GUARD("CUDA RTC lowered name copy", get_lowered_name_impl(program, expression_index, output_buffer, output_buffer_size, out_required_size));
}

JYPPX_StatusCode jyppx_cuda_rtc_program_destroy_safe(JYPPX_CudaRtcProgram* program)
{
    JYPPX_CUDA_RTC_GUARD("CUDA RTC program destroy", destroy_program_impl(program));
}
