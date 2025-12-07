using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Nvinfer;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenCvSharp.Extensions;
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;
using static System.Net.Mime.MediaTypeNames;
using static System.Runtime.InteropServices.JavaScript.JSType;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace WinFormsAppDemo
{
    public partial class Form1 : Form
    {

        private static LogCallbackFunction _callbackDelegate;
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.InitialDirectory = "E:\\Text_Model\\"; // 初始目录
                openFileDialog.Title = "选择文件"; // 对话框标题
                //openFileDialog.Filter = "文本文件 (*.txt)|*.txt|所有文件 (*.*)|*.*"; // 过滤条件
                openFileDialog.FilterIndex = 2; // 默认显示所有文件
                openFileDialog.RestoreDirectory = true; // 打开后恢复当前目录
                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    // 获取文件路径
                    textBox1.Text = openFileDialog.FileName;
                }
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Task task = Task.Run(() =>
            {
                // 在UI线程上执行操作
                this.Invoke((MethodInvoker)delegate
                {
                    this.Enabled = false;
                });
                using Builder build = new Builder();
                Console.WriteLine($"platformHasFastFp16: {build.platformHasFastFp16()}");
                Console.WriteLine($"platformHasFastInt8: {build.platformHasFastInt8()}");
                Console.WriteLine($"maxDLABatchSize: {build.maxDLABatchSize()}");


                using NetworkDefinition networkDefinition = build.createNetworkV2(TrtNetworkDefinitionCreationFlag.kEXPLICIT_BATCH);

                using BuilderConfig builderConfig = build.createBuilderConfig();

                using OnnxParser onnxParser = new OnnxParser(networkDefinition);
                string modelpath = textBox1.Text.Trim();
                if (onnxParser.parseFromFile(modelpath, 2) == false)
                {
                    Console.WriteLine($"parse onnx model failed");
                    return;
                }
                builderConfig.setFlag(TrtBuilderFlag.kFP16);
                //builderConfig.clearFlag(TrtBuilderFlag.kTF32);
                using CudaStream cudaStream = new CudaStream();

                builderConfig.setProfileStream(cudaStream);


                using HostMemory hostMemory = build.buildSerializedNetwork(networkDefinition, builderConfig);

                // 文件路径

                string directoryPath = Path.GetDirectoryName(modelpath);
                string fileNameWithoutExtension = Path.GetFileNameWithoutExtension(modelpath);
                string fullPathWithoutExtension = Path.Combine(directoryPath, fileNameWithoutExtension);
                string filePath = fullPathWithoutExtension + ".engine";

                // 创建或打开文件
                using (FileStream fs = new FileStream(filePath, FileMode.Create, FileAccess.Write))
                {
                    // 写入数据，例如写入一些字节数据
                    fs.Write(hostMemory.getByteData(), 0, (int)hostMemory.Size);
                }

                this.Invoke((MethodInvoker)delegate
                {
                    this.Enabled = true;
                    textBox2.Text = filePath;
                });
                textBox2.Text = filePath;
            });
        }

        // 线程安全地追加日志到 RichTextBox
        private static void AppendTextBox(TextBoxBase box, string text)
        {
            if (box.InvokeRequired)  // 跨线程调用检查
            {
                box.BeginInvoke(new Action<RichTextBox, string>(AppendTextBox), box, text);
            }
            else
            {
                box.AppendText(text + Environment.NewLine);  // 追加日志
                box.ScrollToCaret();                         // 自动滚动到底部
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            _callbackDelegate = (message) =>
            {
                AppendTextBox(richTextBox1, message);
            };
            Logger.Instance.SetCallback(_callbackDelegate);
            Logger.Instance.SetThreshold(LoggerSeverity.kINFO);

            Logger.Instance.ERROR("Logger initialized in Form1_Load\n");

            comboBox1.Items.Add("Yolov11-Det");
            comboBox1.Items.Add("Yolov11-Obb");
            comboBox1.SelectedIndex = 0;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.InitialDirectory = "E:\\Text_Model\\"; // 初始目录
                openFileDialog.Title = "选择文件"; // 对话框标题
                //openFileDialog.Filter = "文本文件 (*.txt)|*.txt|所有文件 (*.*)|*.*"; // 过滤条件
                openFileDialog.FilterIndex = 2; // 默认显示所有文件
                openFileDialog.RestoreDirectory = true; // 打开后恢复当前目录
                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    // 获取文件路径
                    textBox2.Text = openFileDialog.FileName;
                }
            }
        }


        string inputName = "";
        Dims inputDims = new Dims();
        string outputName = "";
        Dims outputDims = new Dims();

        Cuda1DMemory<float> input = new Cuda1DMemory<float>();
        Cuda1DMemory<float> output = new Cuda1DMemory<float>();
        Runtime runtime = new Runtime();
        CudaEngine cudaEngine = new CudaEngine();
        JYPPX.TensorRtSharp.Nvinfer.ExecutionContext executionContext = new JYPPX.TensorRtSharp.Nvinfer.ExecutionContext();
        CudaStream cudaStream = new CudaStream();
        CudaStream cudaStream1 = new CudaStream();
        CudaStream cudaStream2 = new CudaStream();
        float[] outputHost = new float[1];

        bool flag = false;
        CudaEngine cudaEngine1 = new CudaEngine();
        Cuda1DMemory<float> input1 = new Cuda1DMemory<float>();
        Cuda1DMemory<float> output1 = new Cuda1DMemory<float>();
        JYPPX.TensorRtSharp.Nvinfer.ExecutionContext executionContext1 = new JYPPX.TensorRtSharp.Nvinfer.ExecutionContext();
        CudaStream cudaStream01 = new CudaStream();
        CudaStream cudaStream11 = new CudaStream();
        CudaStream cudaStream21 = new CudaStream();
        float[] outputHost1 = new float[1];

        CudaEngine cudaEngine2 = new CudaEngine();
        Cuda1DMemory<float> input2 = new Cuda1DMemory<float>();
        Cuda1DMemory<float> output2 = new Cuda1DMemory<float>();
        JYPPX.TensorRtSharp.Nvinfer.ExecutionContext executionContext2 = new JYPPX.TensorRtSharp.Nvinfer.ExecutionContext();
        CudaStream cudaStream02 = new CudaStream();
        CudaStream cudaStream12 = new CudaStream();
        CudaStream cudaStream22 = new CudaStream();
        float[] outputHost2 = new float[1];


        CudaEngine cudaEngine3 = new CudaEngine();
        Cuda1DMemory<float> input3 = new Cuda1DMemory<float>();
        Cuda1DMemory<float> output3 = new Cuda1DMemory<float>();
        JYPPX.TensorRtSharp.Nvinfer.ExecutionContext executionContext3 = new JYPPX.TensorRtSharp.Nvinfer.ExecutionContext();
        CudaStream cudaStream03 = new CudaStream();
        CudaStream cudaStream13 = new CudaStream();
        CudaStream cudaStream23 = new CudaStream();
        float[] outputHost3 = new float[1];





        private void button4_Click(object sender, EventArgs e)
        {
            string filePath = textBox2.Text.Trim();
            FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            byte[] data = new byte[fileStream.Length];

            BinaryReader binaryReader = new BinaryReader(fileStream);

            data = binaryReader.ReadBytes((int)fileStream.Length); // 读取整个文件到byte数组

            runtime.setMaxThreads(10);

            cudaEngine = runtime.deserializeCudaEngineByBlob(data, (ulong)fileStream.Length);

            executionContext = cudaEngine.createExecutionContext(TrtExecutionContextAllocationStrategy.kSTATIC);

            int count = cudaEngine.getNbIOTensors();
            for (int i = 0; i < count; i++)
            {
                string name = cudaEngine.getIOTensorName(i);
                TrtTensorIOMode type = cudaEngine.getTensorIOMode(name);
                Dims dim = cudaEngine.getTensorShape(name);
                if (type == TrtTensorIOMode.kINPUT)
                {
                    inputName = name;
                    inputDims = dim;
                }
                if (type == TrtTensorIOMode.kOUTPUT)
                {
                    outputName = name;
                    outputDims = dim;
                }

                Logger.Instance.INFO($"TensorIO[{i}] Name:{name} Mode:{type.ToString()} Dims:{dim.ToString()}");
            }
            input = new Cuda1DMemory<float>((ulong)inputDims.GetElementProduct());
            output = new Cuda1DMemory<float>((ulong)outputDims.GetElementProduct());
            outputHost = new float[(ulong)outputDims.GetElementProduct()];
            executionContext.setInputTensorAddress(inputName, input.get());
            executionContext.setOutputTensorAddress(outputName, output.get());




        }

        private void button5_Click(object sender, EventArgs e)
        {
            Mat img = Cv2.ImRead(textBox3.Text.Trim());
            float[] inputHost = preProcess(img, out float scales);

            input.copyFromHostAsync(inputHost, cudaStream2);
            executionContext.executeV3(cudaStream);
            output.copyToHostAsync(outputHost, cudaStream1);


            Stopwatch sw = new Stopwatch();
            sw.Start();
            input.copyFromHostAsync(inputHost, cudaStream2);
            cudaStream2.Synchronize();
            sw.Stop();
            Logger.Instance.INFO($" copyFromHostAsync time: {sw.ElapsedMilliseconds} ms");
            sw.Restart();
            executionContext.executeV3(cudaStream);
            cudaStream.Synchronize();
            sw.Stop();
            Logger.Instance.INFO($"inference time: {sw.ElapsedMilliseconds} ms");
            sw.Restart();
            output.copyToHostAsync(outputHost, cudaStream1);
            cudaStream1.Synchronize();
            sw.Stop();
            Logger.Instance.INFO($" copyToHostAsync time: {sw.ElapsedMilliseconds} ms");
            sw.Stop();

            //Logger.Instance.INFO($"The inference time: {sw.ElapsedMilliseconds} ms");

            List<ObbData> result = postprocess(outputHost, scales);
            Mat re = drawObbResult(result, img);

            using (var memoryStream = new MemoryStream())
            {

                // 计算宽高比例并保持原比例
                double scale = Math.Min(
                    (double)pictureBox1.Width / img.Width,
                    (double)pictureBox1.Height / img.Height);

                var scaledSize = new OpenCvSharp.Size(
                    (int)(img.Width * scale),
                    (int)(img.Height * scale));

                Mat resized = new Mat();
                Cv2.Resize(re, resized, scaledSize, 0, 0);

                // 创建目标图像并填充黑色
                Mat outputMat = new Mat(pictureBox1.Height, pictureBox1.Width, img.Type(), Scalar.Black);

                // 计算粘贴位置（居中）
                int x = (pictureBox1.Width - resized.Width) / 2;
                int y = (pictureBox1.Height - resized.Height) / 2;

                // ROI方式复制图像
                Mat roi = new Mat(outputMat, new OpenCvSharp.Rect(x, y, resized.Width, resized.Height));
                resized.CopyTo(roi);
                resized.Dispose();


                BitmapConverter.ToBitmap(roi).Save(memoryStream, System.Drawing.Imaging.ImageFormat.Bmp);
                memoryStream.Position = 0;     // 重置流位置

                // 4. 创建Bitmap并显示在PictureBox中
                pictureBox1.Image?.Dispose();   // 释放旧图像（如果存在）
                pictureBox1.Image = new System.Drawing.Bitmap(memoryStream);
            }
        }

        private void button6_Click(object sender, EventArgs e)
        {
            Mat img = Cv2.ImRead(textBox3.Text.Trim());
            float[] inputHost = preProcess(img, out float scales);

            int.TryParse(textBox4.Text.Trim(), out int count);
            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int i = 0; i < count; ++i)
            {
                input.copyFromHostAsync(inputHost, cudaStream2);
                cudaStream2.Synchronize();
                executionContext.executeV3(cudaStream);
                cudaStream.Synchronize();
                output.copyToHostAsync(outputHost, cudaStream1);
                cudaStream1.Synchronize();
            }

            sw.Stop();
            Logger.Instance.INFO($"The task 0 average time for reasoning {count} times is: {sw.ElapsedMilliseconds / count} ms");

            List<ObbData> result = postprocess(outputHost, scales);
            Mat re = drawObbResult(result, img);

            using (var memoryStream = new MemoryStream())
            {

                // 计算宽高比例并保持原比例
                double scale = Math.Min(
                    (double)pictureBox1.Width / img.Width,
                    (double)pictureBox1.Height / img.Height);

                var scaledSize = new OpenCvSharp.Size(
                    (int)(img.Width * scale),
                    (int)(img.Height * scale));

                Mat resized = new Mat();
                Cv2.Resize(img, resized, scaledSize, 0, 0);

                // 创建目标图像并填充黑色
                Mat outputMat = new Mat(pictureBox1.Height, pictureBox1.Width, img.Type(), Scalar.Black);

                // 计算粘贴位置（居中）
                int x = (pictureBox1.Width - resized.Width) / 2;
                int y = (pictureBox1.Height - resized.Height) / 2;

                // ROI方式复制图像
                Mat roi = new Mat(outputMat, new OpenCvSharp.Rect(x, y, resized.Width, resized.Height));
                resized.CopyTo(roi);
                resized.Dispose();


                BitmapConverter.ToBitmap(roi).Save(memoryStream, System.Drawing.Imaging.ImageFormat.Bmp);
                memoryStream.Position = 0;     // 重置流位置

                // 4. 创建Bitmap并显示在PictureBox中
                pictureBox1.Image?.Dispose();   // 释放旧图像（如果存在）
                pictureBox1.Image = new System.Drawing.Bitmap(memoryStream);
            }
        }

        private void button8_Click(object sender, EventArgs e)
        {
            if (!flag)
            {
                string filePath = textBox2.Text.Trim();
                FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
                byte[] data = new byte[fileStream.Length];

                BinaryReader binaryReader = new BinaryReader(fileStream);

                data = binaryReader.ReadBytes((int)fileStream.Length); // 读取整个文件到byte数组

                cudaEngine1 = runtime.deserializeCudaEngineByBlob(data, (ulong)fileStream.Length);
                cudaEngine2 = runtime.deserializeCudaEngineByBlob(data, (ulong)fileStream.Length);
                cudaEngine3 = runtime.deserializeCudaEngineByBlob(data, (ulong)fileStream.Length);


                input1 = new Cuda1DMemory<float>((ulong)inputDims.GetElementProduct());
                output1 = new Cuda1DMemory<float>((ulong)outputDims.GetElementProduct());
                input2 = new Cuda1DMemory<float>((ulong)inputDims.GetElementProduct());
                output2 = new Cuda1DMemory<float>((ulong)outputDims.GetElementProduct());
                input3 = new Cuda1DMemory<float>((ulong)inputDims.GetElementProduct());
                output3 = new Cuda1DMemory<float>((ulong)outputDims.GetElementProduct());
                executionContext1 = cudaEngine1.createExecutionContext(TrtExecutionContextAllocationStrategy.kSTATIC);
                executionContext2 = cudaEngine2.createExecutionContext(TrtExecutionContextAllocationStrategy.kSTATIC);
                executionContext3 = cudaEngine3.createExecutionContext(TrtExecutionContextAllocationStrategy.kSTATIC);

                outputHost1 = new float[(ulong)outputDims.GetElementProduct()];
                executionContext1.setInputTensorAddress(inputName, input1.get());
                executionContext1.setOutputTensorAddress(outputName, output1.get());

                outputHost2 = new float[(ulong)outputDims.GetElementProduct()];
                executionContext2.setInputTensorAddress(inputName, input2.get());
                executionContext2.setOutputTensorAddress(outputName, output2.get());


                outputHost3 = new float[(ulong)outputDims.GetElementProduct()];
                executionContext3.setInputTensorAddress(inputName, input3.get());
                executionContext3.setOutputTensorAddress(outputName, output3.get());
                flag = true;
            }
         

            Mat img = Cv2.ImRead(textBox3.Text.Trim());
            float[] inputHost = preProcess(img, out float scales);


            int.TryParse(textBox4.Text.Trim(), out int count);

            Task task1 = new Task(() =>
            {
                Stopwatch sw = new Stopwatch();
                for (int i = 0; i < count; ++i)
                {
                    sw.Start();
                    input1.copyFromHostAsync(inputHost, cudaStream21);
                    executionContext1.executeV3(cudaStream01);
                    output1.copyToHostAsync(outputHost1, cudaStream11);
                }

                sw.Stop();
                Logger.Instance.INFO($"The task 1 average time for reasoning {count} times is: {sw.ElapsedMilliseconds / count} ms");
            });


            Task task2 = new Task(() =>
            {
                Stopwatch sw = new Stopwatch();
                for (int i = 0; i < count; ++i)
                {

                    sw.Start();
                    input2.copyFromHostAsync(inputHost, cudaStream22);
                    executionContext2.executeV3(cudaStream02);
                    output2.copyToHostAsync(outputHost2, cudaStream12);
                }

                sw.Stop();
                Logger.Instance.INFO($"The  task 2 average time for reasoning {count} times is: {sw.ElapsedMilliseconds / count} ms");
            });

            Task task3 = new Task(() =>
            {
                Stopwatch sw = new Stopwatch();
                for (int i = 0; i < count; ++i)
                {
                    sw.Start();
                    input3.copyFromHostAsync(inputHost, cudaStream23);
                    executionContext3.executeV3(cudaStream03);
                    output3.copyToHostAsync(outputHost3, cudaStream13);
                }

                sw.Stop();
                Logger.Instance.INFO($"The  task 3 average time for reasoning {count} times is: {sw.ElapsedMilliseconds / count} ms");
            });




            Task task = new Task(() =>
            {
                Stopwatch sw = new Stopwatch();


                sw.Start();
                for (int i = 0; i < count; ++i)
                {
                    input.copyFromHostAsync(inputHost, cudaStream2);
                    executionContext.executeV3(cudaStream);
                    output.copyToHostAsync(outputHost, cudaStream1);
                }


                sw.Stop();
                Logger.Instance.INFO($"The task 0 average time for reasoning {count} times is: {sw.ElapsedMilliseconds / count} ms");
            });

            Logger.Instance.INFO($"Start infer:{DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff")}");
            task.Start();
            task1.Start();
            task2.Start();

            task3.Start();

            task.Wait();
            task1.Wait();
            task2.Wait();

            task3.Wait();

            Logger.Instance.INFO($"Finsh infer:{DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff")}");
            List<ObbData> result = postprocess(outputHost, scales);
            Mat re = drawObbResult(result, img);

            using (var memoryStream = new MemoryStream())
            {

                // 计算宽高比例并保持原比例
                double scale = Math.Min(
                    (double)pictureBox1.Width / img.Width,
                    (double)pictureBox1.Height / img.Height);

                var scaledSize = new OpenCvSharp.Size(
                    (int)(img.Width * scale),
                    (int)(img.Height * scale));

                Mat resized = new Mat();
                Cv2.Resize(img, resized, scaledSize, 0, 0);

                // 创建目标图像并填充黑色
                Mat outputMat = new Mat(pictureBox1.Height, pictureBox1.Width, img.Type(), Scalar.Black);

                // 计算粘贴位置（居中）
                int x = (pictureBox1.Width - resized.Width) / 2;
                int y = (pictureBox1.Height - resized.Height) / 2;

                // ROI方式复制图像
                Mat roi = new Mat(outputMat, new OpenCvSharp.Rect(x, y, resized.Width, resized.Height));
                resized.CopyTo(roi);
                resized.Dispose();


                BitmapConverter.ToBitmap(roi).Save(memoryStream, System.Drawing.Imaging.ImageFormat.Bmp);
                memoryStream.Position = 0;     // 重置流位置

                // 4. 创建Bitmap并显示在PictureBox中
                pictureBox1.Image?.Dispose();   // 释放旧图像（如果存在）
                pictureBox1.Image = new System.Drawing.Bitmap(memoryStream);
            }
        }
        private float[] preProcess(Mat img, out float scales)
        {
            int inputSize = (int)inputDims.GetDimension(2); ;

            int outputSize = (int)outputDims.GetDimension(2);

            int categNum = (int)outputDims.GetDimension(1) - 4;


            // 创建临时Mat对象并转换颜色空间（BGR→RGB）
            Mat mat = new Mat();
            Cv2.CvtColor(img, mat, ColorConversionCodes.BGR2RGB);

            // 根据图像长宽比计算缩放比例
            Rect roi = new Rect();
            if (img.Cols > img.Rows)  // 宽>高的情况
            {
                scales = (float)img.Cols / (float)inputSize;
                Cv2.Resize(mat, mat, new OpenCvSharp.Size(inputSize, img.Rows / scales));
                roi = new Rect(0, 0, inputSize, (int)(img.Rows / scales));
            }
            else  // 高≥宽的情况
            {
                scales = (float)img.Rows / (float)inputSize;
                Cv2.Resize(mat, mat, new OpenCvSharp.Size(img.Cols / scales, inputSize));
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
        private List<ObbData> postprocess(float[] result, float scales)
        {
            int inputSize = (int)inputDims.GetDimension(2); ;

            int outputSize = (int)outputDims.GetDimension(2);

            int categNum = (int)outputDims.GetDimension(1);

            if (comboBox1.SelectedIndex == 0)
            {
                categNum = categNum - 4;
            }
            else if (comboBox1.SelectedIndex == 1)
            {
                categNum = categNum - 5;
            }


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

                        float rotation = 0;
                        if (comboBox1.SelectedIndex == 0)
                        {
                            rotation = 0;
                        }
                        else if (comboBox1.SelectedIndex == 1)
                        {
                            rotation = result[outputSize * (categNum + 4) + i];
                        }

                        //float rotation = result[outputSize * resultCount + i];

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

        public Mat drawObbResult(List<ObbData> bresult, Mat image)
        {
            for (int i = 0; i < bresult.Count; i++)
            {
                // 获取旋转框的四个角点
                Point2f[] array = bresult[i].box.Points();

                // 绘制四边形边框
                for (int j = 0; j < 4; j++)
                {
                    Cv2.Line(image, (OpenCvSharp.Point)array[j], (OpenCvSharp.Point)array[(j + 1) % 4],
                            new Scalar(255.0, 100.0, 200.0), 2);
                }

                // 在左上角显示类别和置信度
                Cv2.PutText(image,
                            $"{bresult[i].index}-{bresult[i].score:0.00}",
                            (OpenCvSharp.Point)array[0],
                            HersheyFonts.HersheySimplex,
                            0.8,
                            new Scalar(0.0, 0.0, 0.0),
                            2);
            }
            return image;
        }

        private void button7_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.InitialDirectory = "E:\\Text_Model\\"; // 初始目录
                openFileDialog.Title = "选择文件"; // 对话框标题
                //openFileDialog.Filter = "文本文件 (*.txt)|*.txt|所有文件 (*.*)|*.*"; // 过滤条件
                openFileDialog.FilterIndex = 2; // 默认显示所有文件
                openFileDialog.RestoreDirectory = true; // 打开后恢复当前目录
                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    // 获取文件路径
                    textBox3.Text = openFileDialog.FileName;
                }
            }
        }


    }
}
