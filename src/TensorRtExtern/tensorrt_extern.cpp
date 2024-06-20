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

ExceptionStatus onnxToEngineDynamicShape(const char* onnxFile, int memorySize, const char* nodeName,
	int* minShapes, int* optShapes, int* maxShapes) 
{
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

	// ������������
	// �������������ö���
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// ����������ռ��С��
	config->setMaxWorkspaceSize(1024 * 1024 * memorySize);
	// ����ģ���������
	config->setFlag(nvinfer1::BuilderFlag::kFP16);

	nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

	profile->setDimensions(nodeName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(minShapes[0], minShapes[1], minShapes[2], minShapes[3]));
	profile->setDimensions(nodeName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(optShapes[0], optShapes[1], optShapes[2], optShapes[3]));
	profile->setDimensions(nodeName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxShapes[0], maxShapes[1], maxShapes[2], maxShapes[3]));

	config->addOptimizationProfile(profile);

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
	CHECKTRT(p->runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
	CHECKTRT(p->runtime->setErrorRecorder(&gRecorder));
	// ��ʼ����������
	CHECKTRT(p->engine = p->runtime->deserializeCudaEngine(modelStream, size));
	// ����������
	CHECKTRT(p->context = p->engine->createExecutionContext());
	CHECKCUDA(cudaStreamCreate(&(p->stream)));
	CHECKTRT(int numNode = p->engine->getNbBindings());
	// ����gpu���ݻ�����
	p->dataBuffer = new void* [numNode];
	delete[] modelStream;

	for (int i = 0; i < numNode; i++) {
		CHECKTRT(nvinfer1::Dims dims = p->engine->getBindingDimensions(i));
		CHECKTRT(nvinfer1::DataType type = p->engine->getBindingDataType(i));
		std::vector<int> shape(dims.d, dims.d + dims.nbDims);
		size_t size  = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
		switch (type)
		{
		case nvinfer1::DataType::kINT32:
		case nvinfer1::DataType::kFLOAT: size *= 4; break;  // ��ȷΪ���� float
		case nvinfer1::DataType::kHALF: size *= 2; break;
		case nvinfer1::DataType::kBOOL:
		case nvinfer1::DataType::kINT8:
		default:break;
		}
		CHECKCUDA(cudaMalloc(&(p->dataBuffer[i]), size));
	}
	*ptr = p;
	END_WRAP_TRTAPI
}

ExceptionStatus nvinferInitDynamicShape(const char* engineFile, int maxBatahSize, NvinferStruct** ptr) {
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
	CHECKTRT(p->runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
	CHECKTRT(p->runtime->setErrorRecorder(&gRecorder));
	// ��ʼ����������
	CHECKTRT(p->engine = p->runtime->deserializeCudaEngine(modelStream, size));
	// ����������
	CHECKTRT(p->context = p->engine->createExecutionContext());
	CHECKCUDA(cudaStreamCreate(&(p->stream)));
	CHECKTRT(int numNode = p->engine->getNbBindings());
	// ����gpu���ݻ�����
	p->dataBuffer = new void* [numNode];
	delete[] modelStream;

	for (int i = 0; i < numNode; i++) {
		CHECKTRT(nvinfer1::Dims dims = p->engine->getBindingDimensions(i));
		CHECKTRT(nvinfer1::DataType type = p->engine->getBindingDataType(i));
		size_t size = std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
		switch (type)
		{
		case nvinfer1::DataType::kINT32:
		case nvinfer1::DataType::kFLOAT: size *= 4; break;  // ��ȷΪ���� float
		case nvinfer1::DataType::kHALF: size *= 2; break;
		case nvinfer1::DataType::kBOOL:
		case nvinfer1::DataType::kINT8:
		default:break;
		}
		CHECKCUDA(cudaMalloc(&(p->dataBuffer[i]), size * maxBatahSize));
	}
	*ptr = p;
	END_WRAP_TRTAPI
}

ExceptionStatus copyFloatHostToDeviceByName(NvinferStruct* ptr, const char* nodeName, float* data)
{
	BEGIN_WRAP_TRTAPI
	CHECKTRT(int nodeIndex = ptr->engine->getBindingIndex(nodeName));
	// ��ȡ����ڵ�δ����Ϣ
	CHECKTRT(nvinfer1::Dims dims = ptr->context->getBindingDimensions(nodeIndex));
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpyAsync(ptr->dataBuffer[nodeIndex], data, size * sizeof(float), cudaMemcpyHostToDevice, ptr->stream));
	END_WRAP_TRTAPI
}

ExceptionStatus copyFloatHostToDeviceByIndex(NvinferStruct* ptr, int nodeIndex, float* data)
{
	BEGIN_WRAP_TRTAPI
	// ��ȡ����ڵ�δ����Ϣ
	CHECKTRT(nvinfer1::Dims dims = ptr->context->getBindingDimensions(nodeIndex));
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpyAsync(ptr->dataBuffer[nodeIndex], data, size * sizeof(float), cudaMemcpyHostToDevice, ptr->stream));
	END_WRAP_TRTAPI
}
ExceptionStatus setBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int nbDims, int* dims)
{
	CHECKTRT(int nodeIndex = ptr->engine->getBindingIndex(nodeName));
	switch (nbDims)
	{
		case 2:
			CHECKTRT(ptr->context->setBindingDimensions(nodeIndex, nvinfer1::Dims2(dims[0], dims[1])));
			break;
		case 3:
			CHECKTRT(ptr->context->setBindingDimensions(nodeIndex, nvinfer1::Dims3(dims[0], dims[1], dims[2])));
			break;
		case 4:
			CHECKTRT(ptr->context->setBindingDimensions(nodeIndex, nvinfer1::Dims4(dims[0], dims[1], dims[2], dims[3])));
			break;
		default:break;
	}
	END_WRAP_TRTAPI
}
ExceptionStatus setBindingDimensionsByIndex(NvinferStruct* ptr, int nodeIndex, int nbDims, int* dims)
{
	switch (nbDims)
	{
		case 2:
			CHECKTRT(ptr->context->setBindingDimensions(nodeIndex, nvinfer1::Dims2(dims[0], dims[1])));
			break;
		case 3:
			CHECKTRT(ptr->context->setBindingDimensions(nodeIndex, nvinfer1::Dims3(dims[0], dims[1], dims[2])));
			break;
		case 4:
			CHECKTRT(ptr->context->setBindingDimensions(nodeIndex, nvinfer1::Dims4(dims[0], dims[1], dims[2], dims[3])));
			break;
		default:break;
	}
	END_WRAP_TRTAPI
}

ExceptionStatus tensorRtInfer(NvinferStruct* ptr)
{
	BEGIN_WRAP_TRTAPI
	CHECKTRT(ptr->context->enqueueV2((void**)ptr->dataBuffer, ptr->stream, nullptr));
	END_WRAP_TRTAPI
}

ExceptionStatus copyFloatDeviceToHostByName(NvinferStruct* ptr, const char* nodeName, float* data)
{
	BEGIN_WRAP_TRTAPI
	CHECKTRT(int nodeIndex = ptr->engine->getBindingIndex(nodeName));
	// ��ȡ����ڵ�δ����Ϣ
	CHECKTRT(nvinfer1::Dims dims = ptr->context->getBindingDimensions(nodeIndex));
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpyAsync(data, ptr->dataBuffer[nodeIndex], size * sizeof(float), cudaMemcpyDeviceToHost, ptr->stream));
	CHECKCUDA(cudaStreamSynchronize(ptr->stream));
	END_WRAP_TRTAPI
}



ExceptionStatus copyFloatDeviceToHostByIndex(NvinferStruct* ptr, int nodeIndex, float* data)
{
	BEGIN_WRAP_TRTAPI
	// ��ȡ����ڵ�δ����Ϣ
	CHECKTRT(nvinfer1::Dims dims = ptr->context->getBindingDimensions(nodeIndex));
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpyAsync(data, ptr->dataBuffer[nodeIndex], size * sizeof(float), cudaMemcpyDeviceToHost, ptr->stream));
	CHECKCUDA(cudaStreamSynchronize(ptr->stream));
	END_WRAP_TRTAPI
}

ExceptionStatus nvinferDelete(NvinferStruct* ptr)
{
	BEGIN_WRAP_TRTAPI
	CHECKTRT(int numNode = ptr->engine->getNbBindings());
	for (int i = 0; i < numNode; ++i) 
	{
		CHECKCUDA(cudaFree(ptr->dataBuffer[i]);)
		ptr->dataBuffer[i] = nullptr;
	}
	delete ptr->dataBuffer;
	ptr->dataBuffer = nullptr;
	CHECKTRT(ptr->context->destroy();)
	CHECKTRT(ptr->engine->destroy();)
	CHECKTRT(ptr->runtime->destroy();)
	CHECKCUDA(cudaStreamDestroy(ptr->stream));
	delete ptr;
	END_WRAP_TRTAPI
}

ExceptionStatus getBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int* dimLength, int* dims)
{
	BEGIN_WRAP_TRTAPI
	CHECKTRT(int nodeIndex = ptr->engine->getBindingIndex(nodeName));
	// ��ȡ����ڵ�δ����Ϣ
	CHECKTRT(nvinfer1::Dims shape_d = ptr->context->getBindingDimensions(nodeIndex));
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
	CHECKTRT(nvinfer1::Dims shape_d = ptr->context->getBindingDimensions(nodeIndex));
	*dimLength = shape_d.nbDims;
	for (int i = 0; i < *dimLength; ++i)
	{
		*dims++= shape_d.d[i];
	}
	END_WRAP_TRTAPI
}


