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


// @brief ���ڴ���IBuilder��IRuntime��IRefitterʵ���ļ�¼������ͨ���ýӿڴ��������ж���
// ���ͷ����д����Ķ���֮ǰ����¼��Ӧһֱ��Ч��
// ��Ҫ��ʵ����ILogger���µ�log()������
class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* message)  noexcept {
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << message << std::endl;
	}
};

// @brief ������Ľṹ��
typedef struct tensorRT_nvinfer {
	Logger logger;
	// �����л�����
	nvinfer1::IRuntime* runtime;
	// ��������
	// ����ģ�͵�ģ�ͽṹ��ģ�Ͳ����Լ����ż���kernel���ã�
	// ���ܿ�ƽ̨�Ϳ�TensorRT�汾��ֲ
	nvinfer1::ICudaEngine* engine;
	// ������
	// �����м�ֵ��ʵ�ʽ�������Ķ���
	// ��engine�������ɴ���������󣬽��ж���������
	nvinfer1::IExecutionContext* context;
	// GPU�Դ�����/�������
	void** data_buffer;
} NvinferStruct;


// @brief ������onnxģ��תΪtensorrt�е�engine��ʽ�������浽����
extern "C" __declspec(dllexport) void __stdcall onnx_to_engine(const char* onnx_file);

// @brief ��ȡ����engineģ�ͣ�����ʼ��NvinferStruct
extern "C" __declspec(dllexport) void __stdcall nvinfer_init(const char* engine_file, NvinferStruct**nvinfer_ptr);


extern "C" __declspec(dllexport) void __stdcall copy_float_host_to_device_byname(NvinferStruct * nvinfer_ptr, const char* node_name, float* data);

extern "C" __declspec(dllexport) void __stdcall tensorRT_infer(NvinferStruct * nvinfer_ptr);

extern "C" __declspec(dllexport) void __stdcall copy_float_device_to_host_byname(NvinferStruct * nvinfer_ptr, const char* node_name, float* data);

extern "C" __declspec(dllexport) void __stdcall nvinfer_delete(NvinferStruct* p)

#endif // !TENSORRT_EXTERN_H


