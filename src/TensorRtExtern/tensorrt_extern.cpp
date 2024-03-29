#include "tensorrt_extern.h"
#include <string>

// @brief ������onnxģ��תΪtensorrt�е�engine��ʽ�������浽����
// @param onnx_file_path_wchar onnxģ�ͱ��ص�ַ
// @param engine_file_path_wchar engineģ�ͱ��ص�ַ
// @param type ���ģ�;��ȣ�
void  onnx_to_engine(const char* onnx_file) {
	Logger logger;
	// ��·����Ϊ�������ݸ�����

	std::string path(onnx_file);
	std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
	std::string model_path = path.substr(0, iPos);//��ȡ�ļ�·��
	std::string model_name = path.substr(iPos, path.length() - iPos);//��ȡ����׺���ļ���
	std::string model_name_ = model_name.substr(0, model_name.rfind("."));//��ȡ������׺���ļ�����
	std::string engine_file = model_path + model_name_ + ".engine";
	//std::cout << model_name << std::endl;
	//std::cout << model_name_ << std::endl;
	//std::cout << model_path << std::endl;
	//std::cout << engine_file << std::endl;

	// ����������ȡcuda�ں�Ŀ¼�Ի�ȡ����ʵ��
	// ���ڴ���config��network��engine����������ĺ�����
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	// ������������
	const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	// ����onnx�����ļ�
	// tensorRTģ����
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicit_batch);
	// onnx�ļ�������
	// ��onnx�ļ������������rensorRT����ṹ
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
	// ����onnx�ļ�
	parser->parseFromFile(onnx_file, 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");

	// ������������
	// �������������ö���
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// ����������ռ��С��
	config->setMaxWorkspaceSize(16 * (1 << 20));
	// ����ģ���������
	config->setFlag(nvinfer1::BuilderFlag::kFP16);
	// ������������
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	// ��������ǹ���浽����
	std::cout << "try to save engine file now~~~" << std::endl;
	std::ofstream file_ptr(engine_file, std::ios::binary);
	if (!file_ptr) {
		std::cerr << "could not open plan output file" << std::endl;
		return;
	}
	// ��ģ��ת��Ϊ�ļ�������
	nvinfer1::IHostMemory* model_stream = engine->serialize();
	// ���ļ����浽����
	file_ptr.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());
	// ���ٴ����Ķ���
	model_stream->destroy();
	engine->destroy();
	network->destroy();
	parser->destroy();
	std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
}

void nvinfer_init(const char* engine_file, NvinferStruct** nvinfer_ptr)
{
	// �Զ����Ʒ�ʽ��ȡ�ʼ�
	std::ifstream file_ptr(engine_file, std::ios::binary);
	if (!file_ptr.good()) {
		std::cerr << "�ļ��޷��򿪣���ȷ���ļ��Ƿ���ã�" << std::endl;
	}

	size_t size = 0;
	file_ptr.seekg(0, file_ptr.end);	// ����ָ����ļ�ĩβ��ʼ�ƶ�0���ֽ�
	size = file_ptr.tellg();	// ���ض�ָ���λ�ã���ʱ��ָ���λ�þ����ļ����ֽ���
	file_ptr.seekg(0, file_ptr.beg);	// ����ָ����ļ���ͷ��ʼ�ƶ�0���ֽ�
	char* model_stream = new char[size];
	file_ptr.read(model_stream, size);
	// �ر��ļ�
	file_ptr.close();

	// ����������Ľṹ�壬��ʼ������
	NvinferStruct* p = new NvinferStruct();
	// ��ʼ�������л�����
	p->runtime = nvinfer1::createInferRuntime(p->logger);
	// ��ʼ����������
	p->engine = p->runtime->deserializeCudaEngine(model_stream, size);
	// ����������
	p->context = p->engine->createExecutionContext();
	int num_ionode = p->engine->getNbBindings();
	// ����gpu���ݻ�����
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
	// ��ȡ����ڵ�δ����Ϣ
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
	// ��ȡ����ڵ�δ����Ϣ
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