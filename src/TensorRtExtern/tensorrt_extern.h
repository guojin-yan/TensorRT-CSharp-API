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





// @brief ������Ľṹ��
typedef struct tensorRT_nvinfer {
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
	void** dataBuffer;
} NvinferStruct;


// @brief ������onnxģ��תΪtensorrt�е�engine��ʽ�������浽����
TRTAPI(ExceptionStatus) onnxToEngine(const char* onnxFile);

// @brief ��ȡ����engineģ�ͣ�����ʼ��NvinferStruct�����仺��ռ�
TRTAPI(ExceptionStatus) nvinferInit(const char* engineFile, NvinferStruct **ptr);

// @brief ͨ��ָ���ڵ����ƣ����ڴ��ϵ����ݿ������豸��
TRTAPI(ExceptionStatus) copyFloatHostToDeviceByName(NvinferStruct *ptr, const char* nodeName, float* data);

// @brief ͨ��ָ���ڵ��ţ����ڴ��ϵ����ݿ������豸��
TRTAPI(ExceptionStatus) copyFloatHostToDeviceByIndex(NvinferStruct *ptr, int nodeIndex, float* data);

// @brief �����豸�ϵ�����
TRTAPI(ExceptionStatus) tensorRtInfer(NvinferStruct * ptr);

// @brief ͨ��ָ���ڵ����ƣ����豸�ϵ����ݿ������ڴ���
TRTAPI(ExceptionStatus) copyFloatDeviceToHostByName(NvinferStruct *ptr, const char* nodeName, float* data);

// @brief ͨ��ָ���ڵ��ţ����豸�ϵ����ݿ������ڴ���
TRTAPI(ExceptionStatus) copyFloatDeviceToHostByIndex(NvinferStruct *ptr, int nodeIndex, float* data);

// @brief ɾ��������ڴ�
TRTAPI(ExceptionStatus) nvinferDelete(NvinferStruct *ptr);

// @brief ͨ���ڵ����ƻ�ȡ�󶨽ڵ����״��Ϣ
TRTAPI(ExceptionStatus) getBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int* dimLength, int* dims);

// @brief ͨ���ڵ��Ż�ȡ�󶨽ڵ����״��Ϣ
TRTAPI(ExceptionStatus) getBindingDimensionsByIndex(NvinferStruct* ptr, int nodeIndex, int* dimLength, int* dims);


#endif // !TENSORRT_EXTERN_H


