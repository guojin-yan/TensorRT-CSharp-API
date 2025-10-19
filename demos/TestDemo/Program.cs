using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Cuda.Memory;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Nvinfer;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Diagnostics;
using System.Net;
using System.Runtime.InteropServices;
namespace TestDemo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!" + CudaRuntime.getRuntimeVersion());

            //buildModel();
            inferModel();
        }

        public static void inferModel() 
        {
            // 文件路径
            string filePath = "yolo11s-obb.engine";
            FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            byte[] data = new byte[fileStream.Length];

            BinaryReader binaryReader = new BinaryReader(fileStream);
           
            data = binaryReader.ReadBytes((int)fileStream.Length); // 读取整个文件到byte数组
        

            Runtime runtime = new Runtime();

            CudaEngine cudaEngine = runtime.deserializeCudaEngineByBlob(data, (ulong)fileStream.Length);
            JYPPX.TensorRtSharp.Nvinfer.ExecutionContext executionContext = cudaEngine.createExecutionContext(TrtExecutionContextAllocationStrategy.kSTATIC);

            Dims dims = executionContext.getTensorShape("images");

            Cuda1DMemory<float> input = new Cuda1DMemory<float>(3*1024*1024);

            Cuda1DMemory<float> output = new Cuda1DMemory<float>(1 * 20 * 21504);

            float[] outputHost = new float[1 * 84 * 8400];
            executionContext.setInputTensorAddress("images", input.get());

            executionContext.setOutputTensorAddress("output0", output.get());

            Mat img = Cv2.ImRead("plane.png");
            float[] inputHost = preProcess(img, out float scales);



            input.copyFromHost(inputHost);
            CudaStream cudaStream = new CudaStream();
            executionContext.executeV3(cudaStream);
            output.copyToHost(outputHost);

            Stopwatch sw = new Stopwatch();
            sw.Start();
            for(int i = 0; i < 1; ++i) 
            {
                input.copyFromHost(inputHost);
                executionContext.executeV3(cudaStream);
                output.copyToHost(outputHost);
            }

            sw.Stop();
            Console.WriteLine($"inference time: {sw.ElapsedMilliseconds/1f} ms");

            List<ObbData> result = postprocess(outputHost, scales);
            Mat re = drawObbResult(result, img);
            Cv2.ImShow("www", re);
            Cv2.WaitKey(0); 
            Console.ReadLine();

        }

        public static void buildModel() 
        {
            Build build = new Build();
            Console.WriteLine($"platformHasFastFp16: {build.platformHasFastFp16()}");
            Console.WriteLine($"platformHasFastInt8: {build.platformHasFastInt8()}");
            Console.WriteLine($"maxDLABatchSize: {build.maxDLABatchSize()}");


            NetworkDefinition networkDefinition = build.createNetworkV2(TrtNetworkDefinitionCreationFlag.kEXPLICIT_BATCH);

            BuilderConfig builderConfig = build.createBuilderConfig();

            OnnxParser onnxParser = new OnnxParser(networkDefinition);
            string modelpath = "yolo11s-obb.onnx";
            if (onnxParser.parseFromFile(modelpath, 2) == false)
            {
                Console.WriteLine($"parse onnx model failed");
                return;
            }


            builderConfig.setFlag(TrtBuilderFlag.kFP16);

            CudaStream cudaStream = new CudaStream();

            builderConfig.setProfileStream(cudaStream);

            HostMemory hostMemory = build.buildSerializedNetwork(networkDefinition, builderConfig);

            // 文件路径
            string filePath = "yolo11s-obb.engine";

            // 创建或打开文件
            using (FileStream fs = new FileStream(filePath, FileMode.Create, FileAccess.Write))
            {
                // 写入数据，例如写入一些字节数据
                fs.Write(hostMemory.getByteData(), 0, (int)hostMemory.Size);
            }

            Console.ReadLine();
        }



        /// <summary>
        /// 模型输入尺寸(如果模型输入形状有变化，此处要修改，使用ONNX模型查看工具Netron进行查看)
        /// </summary>
        private static int inputSize = 1024;

        /// <summary>
        /// 模型输出尺寸(如果模型输入形状有变化，此处要修改，使用ONNX模型查看工具Netron进行查看)
        /// </summary>
        private static int outputSize = 21504;

        private static int categNum = 15;

        private static float[] preProcess(Mat img, out float scales)
        {
            // 创建临时Mat对象并转换颜色空间（BGR→RGB）
            Mat mat = new Mat();
            Cv2.CvtColor(img, mat, ColorConversionCodes.BGR2RGB);

            // 根据图像长宽比计算缩放比例
            Rect roi = new Rect();
            if (img.Cols > img.Rows)  // 宽>高的情况
            {
                scales = (float)img.Cols / (float)inputSize;
                Cv2.Resize(mat, mat, new Size(inputSize, img.Rows / scales));
                roi = new Rect(0, 0, inputSize, (int)(img.Rows / scales));
            }
            else  // 高≥宽的情况
            {
                scales = (float)img.Rows / (float)inputSize;
                Cv2.Resize(mat, mat, new Size(img.Cols / scales, inputSize));
                roi = new Rect(0, 0, (int)(img.Cols / scales), inputSize);
            }

            // 创建640x640的黑色背景Mat
            Mat mat1 = Mat.Zeros(inputSize, inputSize, MatType.CV_8UC3);
            // 将缩放后的图像拷贝到中央
            mat.CopyTo(new Mat(mat1, roi));
            // 归一化到0-1范围（CV_32FC3类型）
            mat1.ConvertTo(mat1, MatType.CV_32FC3, 1.0 / 255.0);

            // 准备输出数组（CHW格式：3x640x640）
            float[] array = new float[inputSize * inputSize * 3];
            GCHandle gCHandle = default(GCHandle);
            try
            {
                // 固定内存地址用于高效数据拷贝
                gCHandle = GCHandle.Alloc(array, GCHandleType.Pinned);
                IntPtr intPtr = gCHandle.AddrOfPinnedObject();

                // 分离RGB三个通道到连续内存
                for (int i = 0; i < 3; i++)
                {
                    Mat mat2 = Mat.FromPixelData(inputSize, inputSize, MatType.CV_32FC1,
                                intPtr + (i * inputSize * inputSize * 4), 0L);
                    Cv2.ExtractChannel(mat1, mat2, i);
                }
                return array;
            }
            finally
            {
                gCHandle.Free();  // 释放固定内存
            }
        }

        public class ObbData
        {

            public int index;

            public float score;

            public RotatedRect box;
        }
        private static List<ObbData> postprocess(float[] result, float scales)
        {
            // 初始化存储容器
            List<RotatedRect> positionBoxes = new List<RotatedRect>();  // 矩形框
            List<int> classIds = new List<int>();             // 类别ID
            List<float> confidences = new List<float>();      // 置信度
            List<float> rotations = new List<float>();        // 旋转角度

            // 解析模型输出（8400个预测框）
            for (int i = 0; i < outputSize; i++)
            {
                for (int j = 4; j < (categNum + 4); j++)  // 遍历每个类别
                {
                    float conf = result[outputSize * j + i];
                    int label = j - 4;
                    if (conf > 0.2)  // 置信度阈值过滤
                    {
                        // 解析中心点坐标、宽高和旋转角度
                        float cx = result[outputSize * 0 + i];
                        float cy = result[outputSize * 1 + i];
                        float ow = result[outputSize * 2 + i];
                        float oh = result[outputSize * 3 + i];
                        //float rotation = 0;
                        float rotation = result[outputSize * (categNum + 4) + i];

                        // 创建旋转矩形框（考虑预处理时的缩放）
                        RotatedRect box = new RotatedRect(
                            new Point2f(cx * scales, cy * scales),
                            new Size2f(ow * scales, oh * scales),
                            (float)(rotation * 180.0 / Math.PI));

                        // 存储检测结果
                        positionBoxes.Add(box);
                        classIds.Add(label);
                        confidences.Add(conf);
                        rotations.Add(rotation);
                    }
                }
            }

            // 执行非极大值抑制（NMS）
            int[] indexes = new int[positionBoxes.Count];
            CvDnn.NMSBoxes(positionBoxes, confidences, 0.5f, 0.3f, out indexes);

            // 封装最终结果
            List<ObbData> boxes = new List<ObbData>();
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
                boxes.Add(new ObbData
                {
                    index = classIds[index],
                    score = confidences[index],
                    box = positionBoxes[index]
                });
            }
            return boxes;
        }

        public static Mat drawObbResult(List<ObbData> bresult, Mat image)
        {
            for (int i = 0; i < bresult.Count; i++)
            {
                // 获取旋转框的四个角点
                Point2f[] array = bresult[i].box.Points();

                // 绘制四边形边框
                for (int j = 0; j < 4; j++)
                {
                    Cv2.Line(image, (Point)array[j], (Point)array[(j + 1) % 4],
                            new Scalar(255.0, 100.0, 200.0), 2);
                }

                // 在左上角显示类别和置信度
                Cv2.PutText(image,
                            $"{bresult[i].index}-{bresult[i].score:0.00}",
                            (Point)array[0],
                            HersheyFonts.HersheySimplex,
                            0.8,
                            new Scalar(0.0, 0.0, 0.0),
                            2);
            }
            return image;
        }

    }
}
