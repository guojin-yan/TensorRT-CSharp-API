#include "tensorrt_extern.h"
#include <string>

// @brief 将本地onnx模型转为tensorrt中的engine格式，并保存到本地
// @param onnx_file_path_wchar onnx模型本地地址
// @param engine_file_path_wchar engine模型本地地址
// @param type 输出模型精度，
void  onnx_to_engine(const char* onnx_file) {
	Logger logger;
	// 将路径作为参数传递给函数

	std::string path(onnx_file);
	std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
	std::string model_path = path.substr(0, iPos);//获取文件路径
	std::string model_name = path.substr(iPos, path.length() - iPos);//获取带后缀的文件名
	std::string model_name_ = model_name.substr(0, model_name.rfind("."));//获取不带后缀的文件名名
	std::string engine_file = model_path + model_name_ + ".engine";
	//std::cout << model_name << std::endl;
	//std::cout << model_name_ << std::endl;
	//std::cout << model_path << std::endl;
	//std::cout << engine_file << std::endl;

	// 构建器，获取cuda内核目录以获取最快的实现
	// 用于创建config、network、engine的其他对象的核心类
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	// 定义网络属性
	const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	// 解析onnx网络文件
	// tensorRT模型类
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicit_batch);
	// onnx文件解析类
	// 将onnx文件解析，并填充rensorRT网络结构
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
	// 解析onnx文件
	parser->parseFromFile(onnx_file, 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");

	// 创建推理引擎
	// 创建生成器配置对象。
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// 设置最大工作空间大小。
	config->setMaxWorkspaceSize(16 * (1 << 20));
	// 设置模型输出精度
	config->setFlag(nvinfer1::BuilderFlag::kFP16);
	// 创建推理引擎
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	// 将推理银枪保存到本地
	std::cout << "try to save engine file now~~~" << std::endl;
	std::ofstream file_ptr(engine_file, std::ios::binary);
	if (!file_ptr) {
		std::cerr << "could not open plan output file" << std::endl;
		return;
	}
	// 将模型转化为文件流数据
	nvinfer1::IHostMemory* model_stream = engine->serialize();
	// 将文件保存到本地
	file_ptr.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());
	// 销毁创建的对象
	model_stream->destroy();
	engine->destroy();
	network->destroy();
	parser->destroy();
	std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
}

void nvinfer_init(const char* engine_file, NvinferStruct** nvinfer_ptr)
{
	// 以二进制方式读取问价
	std::ifstream file_ptr(engine_file, std::ios::binary);
	if (!file_ptr.good()) {
		std::cerr << "文件无法打开，请确定文件是否可用！" << std::endl;
	}

	size_t size = 0;
	file_ptr.seekg(0, file_ptr.end);	// 将读指针从文件末尾开始移动0个字节
	size = file_ptr.tellg();	// 返回读指针的位置，此时读指针的位置就是文件的字节数
	file_ptr.seekg(0, file_ptr.beg);	// 将读指针从文件开头开始移动0个字节
	char* model_stream = new char[size];
	file_ptr.read(model_stream, size);
	// 关闭文件
	file_ptr.close();

	// 创建推理核心结构体，初始化变量
	NvinferStruct* p = new NvinferStruct();
	// 初始化反序列化引擎
	p->runtime = nvinfer1::createInferRuntime(p->logger);
	// 初始化推理引擎
	p->engine = p->runtime->deserializeCudaEngine(model_stream, size);
	// 创建上下文
	p->context = p->engine->createExecutionContext();
	int num_ionode = p->engine->getNbBindings();
	// 创建gpu数据缓冲区
	p->data_buffer = new void* [num_ionode];
	delete[] model_stream;

	for (int i = 0; i < num_ionode; i++) {
		nvinfer1::Dims shape_d = p->engine->getBindingDimensions(i);
		std::vector<int> shape(shape_d.d, shape_d.d + shape_d.nbDims);
		size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
		cudaMalloc(&(p->data_buffer[i]), size * sizeof(float));
	}
	*nvinfer_ptr = p;
}

void copy_float_host_to_device_byname(NvinferStruct* p, const char* node_name, float* data)
{
	int node_index = p->engine->getBindingIndex(node_name);
	// 获取输入节点未读信息
	nvinfer1::Dims shape_d = p->engine->getBindingDimensions(node_index);
	std::vector<int> shape(shape_d.d, shape_d.d + shape_d.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	cudaMemcpy(p->data_buffer[node_index], data, size * sizeof(float), cudaMemcpyHostToDevice);
}

void tensorRT_infer(NvinferStruct* p) 
{
	bool context = p->context->executeV2((void**)p->data_buffer);
}

void copy_float_device_to_host_byname(NvinferStruct* p, const char* node_name, float* data)
{
	int node_index = p->engine->getBindingIndex(node_name);
	// 获取输入节点未读信息
	nvinfer1::Dims shape_d = p->engine->getBindingDimensions(node_index);
	std::vector<int> shape(shape_d.d, shape_d.d + shape_d.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	cudaMemcpy(data, p->data_buffer[node_index], size * sizeof(float), cudaMemcpyDeviceToHost);
}

void  nvinfer_delete(NvinferStruct* p) 
{
	delete p->data_buffer;
	p->context->destroy();
	p->engine->destroy();
	p->runtime->destroy();
	delete p;
}