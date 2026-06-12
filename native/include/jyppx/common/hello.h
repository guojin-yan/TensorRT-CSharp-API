#pragma once

#include "jyppx/common/bridge_exports.h"

#define JYPPX_BRIDGE_ABI_VERSION 1

JYPPX_C_API(int) jyppx_common_get_abi_version(void);
JYPPX_C_API(const char*) jyppx_common_get_bridge_name(void);
JYPPX_C_API(const char*) jyppx_common_get_bridge_banner(void);

