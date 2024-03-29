#ifndef TENSORRT_EXTERN_H
#define TENSORRT_EXTERN_H


#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include "assert.h"
#include "cuda.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"


// @brief 用于创建IBuilder、IRuntime或IRefitter实例的记录器用于通过该接口创建的所有对象。
// 在释放所有创建的对象之前，记录器应一直有效。
// 主要是实例化ILogger类下的log()方法。
class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* message)  noexcept {
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << message << std::endl;
	}
};

// @brief 推理核心结构体
typedef struct tensorRT_nvinfer {
	Logger logger;
	// 反序列化引擎
	nvinfer1::IRuntime* runtime;
	// 推理引擎
	// 保存模型的模型结构、模型参数以及最优计算kernel配置；
	// 不能跨平台和跨TensorRT版本移植
	nvinfer1::ICudaEngine* engine;
	// 上下文
	// 储存中间值，实际进行推理的对象
	// 由engine创建，可创建多个对象，进行多推理任务
	nvinfer1::IExecutionContext* context;
	// GPU显存输入/输出缓冲
	void** data_buffer;
} NvinferStruct;


// @brief 将本地onnx模型转为tensorrt中的engine格式，并保存到本地
extern "C" __declspec(dllexport) void __stdcall onnx_to_engine(const char* onnx_file);

// @brief 读取本地engine模型，并初始化NvinferStruct
extern "C" __declspec(dllexport) void __stdcall nvinfer_init(const char* engine_file, NvinferStruct**nvinfer_ptr);


extern "C" __declspec(dllexport) void __stdcall copy_float_host_to_device_byname(NvinferStruct * nvinfer_ptr, const char* node_name, float* data);

extern "C" __declspec(dllexport) void __stdcall tensorRT_infer(NvinferStruct * nvinfer_ptr);

extern "C" __declspec(dllexport) void __stdcall copy_float_device_to_host_byname(NvinferStruct * nvinfer_ptr, const char* node_name, float* data);

extern "C" __declspec(dllexport) void __stdcall nvinfer_delete(NvinferStruct* p)

#endif // !TENSORRT_EXTERN_H


