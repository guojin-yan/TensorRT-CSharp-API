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

#include "common.h"
#include "exception_status.h"
#include "logging.h"
#include "logger.h"





// @brief 推理核心结构体
typedef struct tensorRT_nvinfer {
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
	void** dataBuffer;
	cudaStream_t stream;
} NvinferStruct;


// @brief 将本地onnx模型转为tensorrt中的engine格式，并保存到本地
TRTAPI(ExceptionStatus) onnxToEngine(const char* onnxFile, int memorySize);

TRTAPI(ExceptionStatus) onnxToEngineDynamicShape(const char* onnxFile, int memorySize, const char* nodeName, 
	int* minShapes, int* optShapes, int* maxShapes);

// @brief 读取本地engine模型，并初始化NvinferStruct，分配缓存空间
TRTAPI(ExceptionStatus) nvinferInit(const char* engineFile, NvinferStruct **ptr);
TRTAPI(ExceptionStatus) nvinferInitDynamicShape(const char* engineFile, int maxBatahSize, NvinferStruct** ptr);
// @brief 通过指定节点名称，将内存上的数据拷贝到设备上
TRTAPI(ExceptionStatus) copyFloatHostToDeviceByName(NvinferStruct *ptr, const char* nodeName, float* data);

// @brief 通过指定节点编号，将内存上的数据拷贝到设备上
TRTAPI(ExceptionStatus) copyFloatHostToDeviceByIndex(NvinferStruct *ptr, int nodeIndex, float* data);

TRTAPI(ExceptionStatus) setBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int nbDims, int* dims);
TRTAPI(ExceptionStatus) setBindingDimensionsByIndex(NvinferStruct* ptr, int nodeIndex, int nbDims, int* dims);

// @brief 推理设备上的数据
TRTAPI(ExceptionStatus) tensorRtInfer(NvinferStruct * ptr);


// @brief 通过指定节点名称，将设备上的数据拷贝到内存上
TRTAPI(ExceptionStatus) copyFloatDeviceToHostByName(NvinferStruct *ptr, const char* nodeName, float* data);

// @brief 通过指定节点编号，将设备上的数据拷贝到内存上
TRTAPI(ExceptionStatus) copyFloatDeviceToHostByIndex(NvinferStruct *ptr, int nodeIndex, float* data);

// @brief 删除分配的内存
TRTAPI(ExceptionStatus) nvinferDelete(NvinferStruct *ptr);

// @brief 通过节点名称获取绑定节点的形状信息
TRTAPI(ExceptionStatus) getBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int* dimLength, int* dims);

// @brief 通过节点编号获取绑定节点的形状信息
TRTAPI(ExceptionStatus) getBindingDimensionsByIndex(NvinferStruct* ptr, int nodeIndex, int* dimLength, int* dims);


#endif // !TENSORRT_EXTERN_H


