#include <cstdint>
#include <iostream>
#include <string>

#include <NvInfer.h>

#if defined(_WIN32)
#include <windows.h>
#endif

class ProbeLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* msg) noexcept override
    {
        std::cerr << "[TensorRT-" << static_cast<int>(severity) << "] " << (msg != nullptr ? msg : "<null>") << std::endl;
    }
};

#if defined(_WIN32)
std::string get_module_path(wchar_t const* module_name)
{
    HMODULE module = GetModuleHandleW(module_name);
    if (module == nullptr)
    {
        return "<not-loaded>";
    }

    wchar_t buffer[MAX_PATH];
    DWORD length = GetModuleFileNameW(module, buffer, MAX_PATH);
    if (length == 0)
    {
        return "<path-unavailable>";
    }

    int size_needed = WideCharToMultiByte(CP_UTF8, 0, buffer, static_cast<int>(length), nullptr, 0, nullptr, nullptr);
    std::string path(size_needed, '\0');
    WideCharToMultiByte(CP_UTF8, 0, buffer, static_cast<int>(length), path.data(), size_needed, nullptr, nullptr);
    return path;
}
#endif

int main()
{
    ProbeLogger logger;

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    if (builder == nullptr)
    {
        std::cerr << "createInferBuilder returned null" << std::endl;
        return 1;
    }

#if defined(_WIN32)
    std::cout << "Loaded nvinfer.dll: " << get_module_path(L"nvinfer.dll") << std::endl;
    std::cout << "Loaded nvinfer_builder_resource.dll: " << get_module_path(L"nvinfer_builder_resource.dll") << std::endl;
    std::cout << "Loaded nvinfer_builder_resource_10.dll: " << get_module_path(L"nvinfer_builder_resource_10.dll") << std::endl;
#else
    std::cout << "Loaded TensorRT through the platform dynamic loader." << std::endl;
#endif

    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    const uint32_t explicitPrecision = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_PRECISION);

    const uint32_t candidates[] = {
        explicitBatch,
        0U,
        explicitPrecision,
        explicitBatch | explicitPrecision
    };

    nvinfer1::INetworkDefinition* network = nullptr;
    uint32_t winningFlags = 0;
    for (uint32_t flags : candidates)
    {
        network = builder->createNetworkV2(flags);
        std::cout << "createNetworkV2(" << flags << ") => " << (network != nullptr ? "success" : "null") << std::endl;
        if (network != nullptr)
        {
            winningFlags = flags;
            break;
        }
    }

    if (network == nullptr)
    {
        builder->destroy();
        return 2;
    }

    std::cout << "createNetworkV2 succeeded with flags=" << winningFlags << std::endl;
    network->destroy();
    builder->destroy();
    return 0;
}
