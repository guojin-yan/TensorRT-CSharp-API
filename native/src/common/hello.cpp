#include "jyppx/common/hello.h"

namespace
{
constexpr const char* kBridgeName = "jyppxtrtbridge";
constexpr const char* kBridgeBanner = "TensorRtSharp4.0 native bridge skeleton";
}

int jyppx_common_get_abi_version(void)
{
    return JYPPX_BRIDGE_ABI_VERSION;
}

const char* jyppx_common_get_bridge_name(void)
{
    return kBridgeName;
}

const char* jyppx_common_get_bridge_banner(void)
{
    return kBridgeBanner;
}

