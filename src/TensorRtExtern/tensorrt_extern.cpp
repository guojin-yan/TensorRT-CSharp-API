#include "tensorrt_extern.h"
#include <string>

#include "ErrorRecorder.h"

#include "common.h"

// @brief 将本地onnx模型转为tensorrt中的engine格式，并保存到本地
// @param onnx_file_path_wchar onnx模型本地地址
// @param engine_file_path_wchar engine模型本地地址
// @param type 输出模型精度，
ExceptionStatus onnxToEngine(const char* onnxFile, int memorySize) {
	BEGIN_WRAP_TRTAPI
	// 将路径作为参数传递给函数
	std::string path(onnxFile);
	std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
	std::string modelPath = path.substr(0, iPos);//获取文件路径
	std::string modelName = path.substr(iPos, path.length() - iPos);//获取带后缀的文件名
	std::string modelName_ = modelName.substr(0, modelName.rfind("."));//获取不带后缀的文件名名
	std::string engineFile = modelPath + modelName_ + ".engine";
	//std::cout << model_name << std::endl;
	//std::cout << model_name_ << std::endl;
	//std::cout << model_path << std::endl;
	//std::cout << engine_file << std::endl;

	// 构建器，获取cuda内核目录以获取最快的实现
	// 用于创建config、network、engine的其他对象的核心类
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
	// 定义网络属性
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	// 解析onnx网络文件
	// tensorRT模型类
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	// onnx文件解析类
	// 将onnx文件解析，并填充rensorRT网络结构
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
	// 解析onnx文件
	parser->parseFromFile(onnxFile, 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");

	// 创建推理引擎
	// 创建生成器配置对象。
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// 设置最大工作空间大小。
	config->setMaxWorkspaceSize(1024 * 1024 * memorySize);
	// 设置模型输出精度
	config->setFlag(nvinfer1::BuilderFlag::kFP16);
	// 创建推理引擎
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	// 将推理银枪保存到本地
	std::cout << "try to save engine file now~~~" << std::endl;
	std::ofstream filePtr(engineFile, std::ios::binary);
	if (!filePtr) {
		std::cerr << "could not open plan output file" << std::endl;
		return ExceptionStatus::Occurred;
	}
	// 将模型转化为文件流数据
	nvinfer1::IHostMemory* modelStream = engine->serialize();
	// 将文件保存到本地
	filePtr.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	// 销毁创建的对象
	modelStream->destroy();
	engine->destroy();
	network->destroy();
	parser->destroy();
	std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
	END_WRAP_TRTAPI
}

// @brief 读取本地engine模型，并初始化NvinferStruct，分配缓存空间
ExceptionStatus nvinferInit(const char* engineFile, NvinferStruct** ptr){
	BEGIN_WRAP_TRTAPI
	// 以二进制方式读取问价
	std::ifstream filePtr(engineFile, std::ios::binary);
	if (!filePtr.good()) {
		std::cerr << "文件无法打开，请确定文件是否可用！" << std::endl;
		dup_last_err_msg("Model file reading error, please confirm if the file exists or if the format is correct.");
		return ExceptionStatus::Occurred;
	}

	size_t size = 0;
	filePtr.seekg(0, filePtr.end);	// 将读指针从文件末尾开始移动0个字节
	size = filePtr.tellg();	// 返回读指针的位置，此时读指针的位置就是文件的字节数
	filePtr.seekg(0, filePtr.beg);	// 将读指针从文件开头开始移动0个字节
	char* modelStream = new char[size];
	filePtr.read(modelStream, size);
	// 关闭文件
	filePtr.close();

	// 创建推理核心结构体，初始化变量
	NvinferStruct* p = new NvinferStruct();
	// 初始化反序列化引擎
	p->runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
	p->runtime->setErrorRecorder(&gRecorder);
	// 初始化推理引擎
	p->engine = p->runtime->deserializeCudaEngine(modelStream, size);
	// 创建上下文
	p->context = p->engine->createExecutionContext();
	int numNode = p->engine->getNbBindings();
	// 创建gpu数据缓冲区
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
	// 获取输入节点未读信息
	CHECKTRT(nvinfer1::Dims dims = ptr->engine->getBindingDimensions(nodeIndex));
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpy(ptr->dataBuffer[nodeIndex], data, size * sizeof(float), cudaMemcpyHostToDevice));
	END_WRAP_TRTAPI
}

ExceptionStatus copyFloatHostToDeviceByIndex(NvinferStruct* ptr, int nodeIndex, float* data)
{
	BEGIN_WRAP_TRTAPI
	// 获取输入节点未读信息
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
	// 获取输入节点未读信息
	CHECKTRT(nvinfer1::Dims dims = ptr->engine->getBindingDimensions(nodeIndex));
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpy(data, ptr->dataBuffer[nodeIndex], size * sizeof(float), cudaMemcpyDeviceToHost));
	END_WRAP_TRTAPI
}



ExceptionStatus copyFloatDeviceToHostByIndex(NvinferStruct* ptr, int nodeIndex, float* data)
{
	BEGIN_WRAP_TRTAPI
	// 获取输入节点未读信息
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
	// 获取输入节点未读信息
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
	// 获取输入节点未读信息
	CHECKTRT(nvinfer1::Dims shape_d = ptr->engine->getBindingDimensions(nodeIndex));
	*dimLength = shape_d.nbDims;
	for (int i = 0; i < *dimLength; ++i)
	{
		*dims++= shape_d.d[i];
	}
	END_WRAP_TRTAPI
}


