#pragma once

#if defined(_WIN32)
  #if defined(JYPPX_BRIDGE_BUILD)
    #define JYPPX_BRIDGE_API __declspec(dllexport)
  #else
    #define JYPPX_BRIDGE_API __declspec(dllimport)
  #endif
#else
  #define JYPPX_BRIDGE_API __attribute__((visibility("default")))
#endif

#if defined(__cplusplus)
  #define JYPPX_EXTERN_C extern "C"
#else
  #define JYPPX_EXTERN_C
#endif

#define JYPPX_C_API(return_type) JYPPX_EXTERN_C JYPPX_BRIDGE_API return_type

