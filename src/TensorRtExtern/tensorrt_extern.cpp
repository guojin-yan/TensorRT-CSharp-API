#include "tensorrt_extern.h"
#include <string>

#include "ErrorRecorder.h"

#include "common.h"

// @brief ������onnxģ��תΪtensorrt�е�engine��ʽ�������浽����
// @param onnx_file_path_wchar onnxģ�ͱ��ص�ַ
// @param engine_file_path_wchar engineģ�ͱ��ص�ַ
// @param type ���ģ�;��ȣ�
ExceptionStatus onnxToEngine(const char* onnxFile, int memorySize) {
	BEGIN_WRAP_TRTAPI
	// ��·����Ϊ�������ݸ�����
	std::string path(onnxFile);
	std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
	std::string modelPath = path.substr(0, iPos);//��ȡ�ļ�·��
	std::string modelName = path.substr(iPos, path.length() - iPos);//��ȡ����׺���ļ���
	std::string modelName_ = modelName.substr(0, modelName.rfind("."));//��ȡ������׺���ļ�����
	std::string engineFile = modelPath + modelName_ + ".engine";
	//std::cout << model_name << std::endl;
	//std::cout << model_name_ << std::endl;
	//std::cout << model_path << std::endl;
	//std::cout << engine_file << std::endl;

	// ����������ȡcuda�ں�Ŀ¼�Ի�ȡ����ʵ��
	// ���ڴ���config��network��engine����������ĺ�����
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
	// ������������
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	// ����onnx�����ļ�
	// tensorRTģ����
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	// onnx�ļ�������
	// ��onnx�ļ������������rensorRT����ṹ
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
	// ����onnx�ļ�
	parser->parseFromFile(onnxFile, 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");

	// ������������
	// �������������ö���
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// ����������ռ��С��
	config->setMaxWorkspaceSize(1024 * 1024 * memorySize);
	// ����ģ���������
	config->setFlag(nvinfer1::BuilderFlag::kFP16);
	// ������������
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	// ��������ǹ���浽����
	std::cout << "try to save engine file now~~~" << std::endl;
	std::ofstream filePtr(engineFile, std::ios::binary);
	if (!filePtr) {
		std::cerr << "could not open plan output file" << std::endl;
		return ExceptionStatus::Occurred;
	}
	// ��ģ��ת��Ϊ�ļ�������
	nvinfer1::IHostMemory* modelStream = engine->serialize();
	// ���ļ����浽����
	filePtr.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	// ���ٴ����Ķ���
	modelStream->destroy();
	engine->destroy();
	network->destroy();
	parser->destroy();
	std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
	END_WRAP_TRTAPI
}

// @brief ��ȡ����engineģ�ͣ�����ʼ��NvinferStruct�����仺��ռ�
ExceptionStatus nvinferInit(const char* engineFile, NvinferStruct** ptr){
	BEGIN_WRAP_TRTAPI
	// �Զ����Ʒ�ʽ��ȡ�ʼ�
	std::ifstream filePtr(engineFile, std::ios::binary);
	if (!filePtr.good()) {
		std::cerr << "�ļ��޷��򿪣���ȷ���ļ��Ƿ���ã�" << std::endl;
		dup_last_err_msg("Model file reading error, please confirm if the file exists or if the format is correct.");
		return ExceptionStatus::Occurred;
	}

	size_t size = 0;
	filePtr.seekg(0, filePtr.end);	// ����ָ����ļ�ĩβ��ʼ�ƶ�0���ֽ�
	size = filePtr.tellg();	// ���ض�ָ���λ�ã���ʱ��ָ���λ�þ����ļ����ֽ���
	filePtr.seekg(0, filePtr.beg);	// ����ָ����ļ���ͷ��ʼ�ƶ�0���ֽ�
	char* modelStream = new char[size];
	filePtr.read(modelStream, size);
	// �ر��ļ�
	filePtr.close();

	// ����������Ľṹ�壬��ʼ������
	NvinferStruct* p = new NvinferStruct();
	// ��ʼ�������л�����
	p->runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
	p->runtime->setErrorRecorder(&gRecorder);
	// ��ʼ����������
	p->engine = p->runtime->deserializeCudaEngine(modelStream, size);
	// ����������
	p->context = p->engine->createExecutionContext();
	int numNode = p->engine->getNbBindings();
	// ����gpu���ݻ�����
	p->dataBuffer = new void* [numNode];
	delete[] modelStream;

	for (int i = 0; i < numNode; i++) {
		CHECKTRT(nvinfer1::Dims dims = p->engine->getBindingDimensions(i));
		std::vector<int> shape(dims.d, dims.d + dims.nbDims);
		size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
		CHECKCUDA(cudaMalloc(&(p->dataBuffer[i]), size * sizeof(float)));
	}
	*ptr = p;
	END_WRAP_TRTAPI
}

ExceptionStatus copyFloatHostToDeviceByName(NvinferStruct* ptr, const char* nodeName, float* data)
{
	BEGIN_WRAP_TRTAPI
	CHECKTRT(int nodeIndex = ptr->engine->getBindingIndex(nodeName));
	// ��ȡ����ڵ�δ����Ϣ
	CHECKTRT(nvinfer1::Dims dims = ptr->engine->getBindingDimensions(nodeIndex));
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpy(ptr->dataBuffer[nodeIndex], data, size * sizeof(float), cudaMemcpyHostToDevice));
	END_WRAP_TRTAPI
}

ExceptionStatus copyFloatHostToDeviceByIndex(NvinferStruct* ptr, int nodeIndex, float* data)
{
	BEGIN_WRAP_TRTAPI
	// ��ȡ����ڵ�δ����Ϣ
	CHECKTRT(nvinfer1::Dims dims = ptr->engine->getBindingDimensions(nodeIndex));
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpy(ptr->dataBuffer[nodeIndex], data, size * sizeof(float), cudaMemcpyHostToDevice));
	END_WRAP_TRTAPI
}
ExceptionStatus tensorRtInfer(NvinferStruct* ptr)
{
	BEGIN_WRAP_TRTAPI
	CHECKTRT(ptr->context->executeV2((void**)ptr->dataBuffer));
	END_WRAP_TRTAPI
}

ExceptionStatus copyFloatDeviceToHostByName(NvinferStruct* ptr, const char* nodeName, float* data)
{
	BEGIN_WRAP_TRTAPI
	CHECKTRT(int nodeIndex = ptr->engine->getBindingIndex(nodeName));
	// ��ȡ����ڵ�δ����Ϣ
	CHECKTRT(nvinfer1::Dims dims = ptr->engine->getBindingDimensions(nodeIndex));
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpy(data, ptr->dataBuffer[nodeIndex], size * sizeof(float), cudaMemcpyDeviceToHost));
	END_WRAP_TRTAPI
}



ExceptionStatus copyFloatDeviceToHostByIndex(NvinferStruct* ptr, int nodeIndex, float* data)
{
	BEGIN_WRAP_TRTAPI
	// ��ȡ����ڵ�δ����Ϣ
	CHECKTRT(nvinfer1::Dims dims = ptr->engine->getBindingDimensions(nodeIndex));
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpy(data, ptr->dataBuffer[nodeIndex], size * sizeof(float), cudaMemcpyDeviceToHost));
	END_WRAP_TRTAPI
}

ExceptionStatus nvinferDelete(NvinferStruct* ptr)
{
	BEGIN_WRAP_TRTAPI
	delete ptr->dataBuffer;
	CHECKTRT(ptr->context->destroy();)
	CHECKTRT(ptr->engine->destroy();)
	CHECKTRT(ptr->runtime->destroy();)
	delete ptr;
	END_WRAP_TRTAPI
}

ExceptionStatus getBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int* dimLength, int* dims)
{
	BEGIN_WRAP_TRTAPI
	CHECKTRT(int nodeIndex = ptr->engine->getBindingIndex(nodeName));
	// ��ȡ����ڵ�δ����Ϣ
	CHECKTRT(nvinfer1::Dims shape_d = ptr->engine->getBindingDimensions(nodeIndex));
	*dimLength = shape_d.nbDims;
	for (int i = 0; i < *dimLength; ++i)
	{
		*dims++ = shape_d.d[i];
	}
	END_WRAP_TRTAPI
}

ExceptionStatus getBindingDimensionsByIndex(NvinferStruct* ptr, int nodeIndex, int* dimLength, int* dims)
{
	BEGIN_WRAP_TRTAPI
	// ��ȡ����ڵ�δ����Ϣ
	CHECKTRT(nvinfer1::Dims shape_d = ptr->engine->getBindingDimensions(nodeIndex));
	*dimLength = shape_d.nbDims;
	for (int i = 0; i < *dimLength; ++i)
	{
		*dims++= shape_d.d[i];
	}
	END_WRAP_TRTAPI
}


