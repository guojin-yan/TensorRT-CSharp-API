#include "tensorrt_extern.h"

int main1() 
{
	//const char* onnx = "E:\\Model\\yolov8\\yolov8s.onnx";
	const char* engine = "E:\\Model\\yolov8\\yolov8s.engine";
	//onnx_to_engine(onnx, engine, 1);

	NvinferStruct* p = new NvinferStruct();

	nvinfer_init(engine, &p);
	std::vector<float> input_data(1 * 3 * 640 * 640);
	srand((unsigned)time(NULL));
	for (int i = 0; i < input_data.size(); ++i)
	{
		float B = (rand() % 10) / 1000;//根据需要设置除数
		input_data[i] = B;
	}

	copy_float_host_to_device_byname(p, "images", input_data.data());

	tensorRT_infer(p);

	std::vector<float> output_data(84*8400);
	copy_float_device_to_host_byname(p, "output0", output_data.data());

	//Logger logger;
	//// 以二进制方式读取问价
	//std::ifstream file_ptr(engine, std::ios::binary);
	//if (!file_ptr.good()) {
	//	std::cerr << "文件无法打开，请确定文件是否可用！" << std::endl;
	//}

	//size_t size = 0;
	//file_ptr.seekg(0, file_ptr.end);	// 将读指针从文件末尾开始移动0个字节
	//size = file_ptr.tellg();	// 返回读指针的位置，此时读指针的位置就是文件的字节数
	//file_ptr.seekg(0, file_ptr.beg);	// 将读指针从文件开头开始移动0个字节
	//char* model_stream = new char[size];
	//file_ptr.read(model_stream, size);
	//// 关闭文件
	//file_ptr.close();

	//// 创建推理核心结构体，初始化变量
	//NvinferStruct* p = new NvinferStruct();
	//// 初始化反序列化引擎
	//p->runtime = nvinfer1::createInferRuntime(logger);;
	//// 初始化推理引擎
	//p->engine = p->runtime->deserializeCudaEngine(model_stream, size);
	//// 创建上下文
	//p->context = p->engine->createExecutionContext();
	//int num_ionode = p->engine->getNbBindings();
	//std::cout << "num_ionode: " << num_ionode << std::endl;
	//nvinfer1::Dims node_dim = p->engine->getBindingDimensions(0);

	//std::cout << "input shape: " << node_dim.d[0] << ", " << node_dim.d[1] << ", " << node_dim.d[2] << ", " << std::endl;

	//p->context = p->engine->createExecutionContext();

	//

	//p->data_buffer = new void* [num_ionode];
	//num_ionode = p->engine->getNbBindings();
	//for (int i = 0; i < num_ionode; i++) {
	//	nvinfer1::Dims shape_d = p->engine->getBindingDimensions(i);
	//	std::vector<int> shape(shape_d.d+1, shape_d.d + shape_d.nbDims);
	//	size_t size = 10*std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	//	cudaMalloc(&(p->data_buffer[i]), size * sizeof(float));
	//}

	//// 创建cuda流
	//cudaStreamCreate(&p->stream);
	//std::vector<float> input_data(2*3*640*640);
	//srand((unsigned)time(NULL));
	//for (int i = 0; i < input_data.size(); ++i) 
	//{
	//	float B = (rand() % 10) / 1000;//根据需要设置除数
	//	input_data[i] = B;
	//}
	//// 将图片数据copy到输入流中

	//// 将输入数据由内存到GPU显存
	//cudaMemcpyAsync(p->data_buffer[0], input_data.data(), 640*640*2 * 3 * sizeof(float), cudaMemcpyHostToDevice, p->stream);

	//p->context->setInputShape("images", nvinfer1::Dims4{ 2, 3 , 640, 640 });
	//p->context->setBindingDimensions(0, nvinfer1::Dims4{ 2,3,640,640 });
	//node_dim = p->context->getBindingDimensions(0);
	//std::cout << "input shape1111111111: " << node_dim.d[0] << ", " << node_dim.d[1] << ", " << node_dim.d[2] << ", " << std::endl;
	////node_dim = p->engine->getBindingDimensions(0);
	//std::cout << "input shape: " << p->context->getOptimizationProfile() << std::endl;
	////std::cout << "input shape: " << node_dim.d[0] << ", " << node_dim.d[1] << ", " << node_dim.d[2] << ", " << std::endl;
	//p->context->enqueueV2(p->data_buffer, p->stream, nullptr);

	//// 读取输出数据
	//nvinfer1::Dims shape_d = p->engine->getBindingDimensions(1);
	//std::vector<int> shape(shape_d.d+1, shape_d.d + shape_d.nbDims);
	// size =10* std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	//// 创建输出数据
	//std::vector<float> output_data(size);
	//// 将输出数据由GPU显存到内存
	//cudaMemcpyAsync(output_data.data(), p->data_buffer[1], size * sizeof(float), cudaMemcpyDeviceToHost, p->stream);

	////for (int i = 0; i < size; i++) {
	////	*output_result = output_data[i];
	////	output_result++;
	////}

	return 0;
}


int main2()
{
	const char* onnx = "E:\\Model\\yolov8\\yolov8s_2.onnx";
	onnx_to_engine(onnx);
	return 0;
}