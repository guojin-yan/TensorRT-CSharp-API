#ifndef COMMON_H
#define COMMON_H

#include "NvInfer.h"
#include <iostream>



# if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#	define TRT_EXPORTS __declspec(dllexport)
# endif


#if defined WIN32 || defined _WIN32
#  define TRT_CDECL __cdecl
#  define TRT_STDCALL __stdcall
#else
#  define TRT_CDECL
#  define TRT_STDCALL
#endif

#ifndef TRT_EXTERN_C
#  define TRT_EXTERN_C extern "C"
#endif // !TRT_EXTERN_C


#ifndef TRTAPI
#  define TRTAPI(rettype) TRT_EXTERN_C TRT_EXPORTS rettype TRT_CDECL
#endif



#if defined WIN32 || defined _WIN32
#define BEGIN_WRAP_TRTAPI
#define END_WRAP_TRTAPI return ExceptionStatus::NotOccurred;
#else
#define BEGIN_WRAP_TRTAPI try{
#define END_WRAP_TRTAPI return ExceptionStatus::NotOccurred;}catch(std::exception){return ExceptionStatus::Occurred;}
#endif


#define CHECKTRT(op) op; if (!gRecorder.empty()){ dup_last_err_msg(gRecorder.getErrorDesc(0)); return ExceptionStatus::OccurredTRT;}

#define CHECKCUDA(op) {cudaError_t code = op; if ( code != cudaSuccess) {dup_last_err_msg(cudaGetErrorString(code)); return ExceptionStatus::OccurredCuda;}}

char* str_to_char_array(const std::string& str);

void dup_last_err_msg(const char* msg);


TRTAPI(const char*) trt_get_last_err_msg();



#endif // !COMMON_H

