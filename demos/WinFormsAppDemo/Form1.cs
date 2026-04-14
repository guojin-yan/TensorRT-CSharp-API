using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Nvinfer;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System.Diagnostics;

namespace WinFormsAppDemo
{
    public partial class Form1 : Form
    {
        private static LogCallbackFunction _callbackDelegate = default!;

        private readonly Runtime runtime = new();
        private readonly List<InferenceSession> parallelSessions = [];
        private LoadedYoloModel? loadedModel;

        public Form1()
        {
            InitializeComponent();
            FormClosed += Form1_FormClosed;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            _callbackDelegate = message => AppendTextBox(richTextBox1, message);
            Logger.Instance.SetCallback(_callbackDelegate);
            Logger.Instance.SetThreshold(LoggerSeverity.kINFO);

            comboBox1.DropDownStyle = ComboBoxStyle.DropDownList;
            comboBox1.Items.Clear();
            foreach (YoloModelProfile profile in YoloModelProfile.All)
            {
                comboBox1.Items.Add(profile);
            }

            comboBox1.SelectedItem = YoloModelProfile.Default;
            TrySelectModelProfile(textBox1.Text);
            TrySelectModelProfile(textBox2.Text);

            Logger.Instance.INFO("WinFormsAppDemo initialized.");

            int deviceCount = CudaDevice.GetDeviceCount();
            if (deviceCount <= 0)
            {
                Logger.Instance.ERROR("Cannot find any available devices (GPUs)!");
            }
            else
            {
                Logger.Instance.INFO($"Available Devices: {deviceCount}");
                for (int deviceIdx = 0; deviceIdx < deviceCount; deviceIdx++)
                {
                    CudaDeviceProp properties = CudaDevice.GetDeviceProperties(deviceIdx);
                    Logger.Instance.INFO($"  Device {deviceIdx}: \"{properties.Name}\"");
                }
            }

            Logger.Instance.INFO($"The latest version of CUDA supported by the driver: {CudaRuntime.DriverGetVersion()}");
            Logger.Instance.INFO($"The CUDA Runtime version: {CudaRuntime.RuntimeGetVersion()}");
        }

        private void button1_Click(object sender, EventArgs e)
        {
            SelectFilePath(textBox1, "Select ONNX Model", "ONNX Files (*.onnx)|*.onnx|All Files (*.*)|*.*");
        }

        private async void button2_Click(object sender, EventArgs e)
        {
            string modelPath = textBox1.Text.Trim();
            if (!File.Exists(modelPath))
            {
                Logger.Instance.ERROR("Please select a valid ONNX model first.");
                return;
            }

            Enabled = false;
            try
            {
                string enginePath = await Task.Run(() =>
                {
                    using Builder builder = new();
                    Logger.Instance.INFO($"platformHasFastFp16: {builder.platformHasFastFp16()}");
                    Logger.Instance.INFO($"platformHasFastInt8: {builder.platformHasFastInt8()}");
                    Logger.Instance.INFO($"maxDLABatchSize: {builder.maxDLABatchSize()}");

                    using NetworkDefinition networkDefinition = builder.createNetworkV2(TrtNetworkDefinitionCreationFlag.kEXPLICIT_BATCH);
                    using BuilderConfig builderConfig = builder.createBuilderConfig();
                    using OnnxParser onnxParser = new(networkDefinition);

                    if (!onnxParser.parseFromFile(modelPath, 2))
                    {
                        throw new InvalidOperationException("Parse ONNX model failed.");
                    }

                    builderConfig.setFlag(TrtBuilderFlag.kFP16);
                    using CudaStream buildStream = new();
                    builderConfig.setProfileStream(buildStream);

                    using HostMemory hostMemory = builder.buildSerializedNetwork(networkDefinition, builderConfig);

                    string directoryPath = Path.GetDirectoryName(modelPath) ?? string.Empty;
                    string fileNameWithoutExtension = Path.GetFileNameWithoutExtension(modelPath);
                    string enginePath = Path.Combine(directoryPath, $"{fileNameWithoutExtension}.engine");
                    File.WriteAllBytes(enginePath, hostMemory.getByteData());
                    return enginePath;
                });

                textBox2.Text = enginePath;
                TrySelectModelProfile(enginePath);
                Logger.Instance.INFO($"Engine saved to: {enginePath}");
            }
            catch (Exception ex)
            {
                Logger.Instance.ERROR($"ONNX -> Engine failed: {ex.Message}");
            }
            finally
            {
                Enabled = true;
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            SelectFilePath(textBox2, "Select Engine Model", "TensorRT Engine (*.engine)|*.engine|All Files (*.*)|*.*");
        }

        private void button4_Click(object sender, EventArgs e)
        {
            string filePath = textBox2.Text.Trim();
            if (!File.Exists(filePath))
            {
                Logger.Instance.ERROR("Please select a valid Engine model first.");
                return;
            }

            TrySelectModelProfile(filePath);
            DisposeParallelSessions();
            DisposeLoadedModel();

            try
            {
                loadedModel = LoadedYoloModel.Load(runtime, filePath, GetSelectedProfile());

                Logger.Instance.INFO($"Input tensor: {loadedModel.InputName} Dims:{loadedModel.InputDims}");
                foreach (TensorBindingMetadata output in loadedModel.Outputs)
                {
                    Logger.Instance.INFO($"Output tensor: {output.Name} Dims:{output.Dims}");
                }

                Logger.Instance.INFO($"Selected profile: {GetSelectedProfile().DisplayName}");
                Logger.Instance.INFO(YoloTaskRenderer.DescribeModelOutputs(GetSelectedProfile(), loadedModel.Outputs));
            }
            catch (Exception ex)
            {
                Logger.Instance.ERROR($"Load engine failed: {ex.Message}");
                DisposeLoadedModel();
            }
        }

        private void button5_Click(object sender, EventArgs e)
        {
            if (!EnsureModelLoaded() || !TryLoadImage(out Mat image))
            {
                return;
            }

            using (image)
            {
                Stopwatch totalWatch = Stopwatch.StartNew();
                YoloImagePreprocessResult preprocessResult = YoloTaskRenderer.Preprocess(image, GetInputWidth(), GetInputHeight());
                InferenceTimings timings = loadedModel!.PrimarySession.Run(preprocessResult.Tensor);

                Logger.Instance.INFO($"copyFromHostAsync time: {timings.InputCopyMs:0.00} ms");
                Logger.Instance.INFO($"inference time: {timings.ComputeMs:0.00} ms");
                Logger.Instance.INFO($"copyToHostAsync time: {timings.OutputCopyMs:0.00} ms");

                using Mat rendered = RenderResult(image, preprocessResult, out int resultCount);
                totalWatch.Stop();

                Logger.Instance.INFO($"Result count: {resultCount}");
                Logger.Instance.INFO($"The sum inference time: {totalWatch.ElapsedMilliseconds} ms");
                DisplayImage(rendered);
            }
        }

        private void button6_Click(object sender, EventArgs e)
        {
            if (!EnsureModelLoaded() || !TryLoadImage(out Mat image))
            {
                return;
            }

            using (image)
            {
                YoloImagePreprocessResult preprocessResult = YoloTaskRenderer.Preprocess(image, GetInputWidth(), GetInputHeight());
                int count = GetLoopCount();
                double averageMs = loadedModel!.PrimarySession.RunBenchmark(preprocessResult.Tensor, count);

                Logger.Instance.INFO($"The task 0 average time for reasoning {count} times is: {averageMs:0.00} ms");

                using Mat rendered = RenderResult(image, preprocessResult, out int resultCount);
                Logger.Instance.INFO($"Result count: {resultCount}");
                DisplayImage(rendered);
            }
        }

        private void button7_Click(object sender, EventArgs e)
        {
            SelectFilePath(textBox3, "Select Image", "Image Files|*.jpg;*.jpeg;*.png;*.bmp|All Files (*.*)|*.*");
        }

        private void button8_Click(object sender, EventArgs e)
        {
            if (!EnsureModelLoaded() || !TryLoadImage(out Mat image))
            {
                return;
            }

            using (image)
            {
                EnsureParallelSessions();

                YoloImagePreprocessResult preprocessResult = YoloTaskRenderer.Preprocess(image, GetInputWidth(), GetInputHeight());
                int count = GetLoopCount();

                List<(string Name, InferenceSession Session)> sessions =
                [
                    ("The task 0", loadedModel!.PrimarySession),
                    ("The task 1", parallelSessions[0]),
                    ("The task 2", parallelSessions[1]),
                    ("The task 3", parallelSessions[2])
                ];

                Logger.Instance.INFO($"Start infer:{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}");
                Task[] tasks = sessions
                    .Select(item => Task.Run(() =>
                    {
                        double averageMs = item.Session.RunBenchmark(preprocessResult.Tensor, count);
                        Logger.Instance.INFO($"{item.Name} average time for reasoning {count} times is: {averageMs:0.00} ms");
                    }))
                    .ToArray();

                Task.WaitAll(tasks);
                Logger.Instance.INFO($"Finsh infer:{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}");

                using Mat rendered = RenderResult(image, preprocessResult, out int resultCount);
                Logger.Instance.INFO($"Result count: {resultCount}");
                DisplayImage(rendered);
            }
        }

        private static void AppendTextBox(TextBoxBase box, string text)
        {
            if (box.InvokeRequired)
            {
                box.BeginInvoke(new Action<TextBoxBase, string>(AppendTextBox), box, text);
                return;
            }

            box.AppendText(text + Environment.NewLine);
            box.ScrollToCaret();
        }

        private void SelectFilePath(TextBox targetBox, string title, string filter)
        {
            using OpenFileDialog dialog = new();
            dialog.InitialDirectory = GetInitialDirectory(targetBox.Text);
            dialog.Title = title;
            dialog.Filter = filter;
            dialog.RestoreDirectory = true;

            if (dialog.ShowDialog() != DialogResult.OK)
            {
                return;
            }

            targetBox.Text = dialog.FileName;
            TrySelectModelProfile(dialog.FileName);
        }

        private string GetInitialDirectory(string currentPath)
        {
            if (!string.IsNullOrWhiteSpace(currentPath))
            {
                string? currentDirectory = Path.GetDirectoryName(currentPath);
                if (!string.IsNullOrWhiteSpace(currentDirectory) && Directory.Exists(currentDirectory))
                {
                    return currentDirectory;
                }
            }

            return Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
        }

        private void TrySelectModelProfile(string? modelPath)
        {
            YoloModelProfile? profile = YoloModelProfile.TryMatchByPath(modelPath);
            if (profile is not null)
            {
                comboBox1.SelectedItem = profile;
            }
        }

        private YoloModelProfile GetSelectedProfile()
        {
            return comboBox1.SelectedItem as YoloModelProfile ?? YoloModelProfile.Default;
        }

        private bool EnsureModelLoaded()
        {
            if (loadedModel is null)
            {
                Logger.Instance.ERROR("Please load an Engine model first.");
                return false;
            }

            return true;
        }

        private bool TryLoadImage(out Mat image)
        {
            image = Cv2.ImRead(textBox3.Text.Trim());
            if (!image.Empty())
            {
                return true;
            }

            Logger.Instance.ERROR("Please select a valid image first.");
            image.Dispose();
            image = new Mat();
            return false;
        }

        private int GetLoopCount()
        {
            return int.TryParse(textBox4.Text.Trim(), out int count) && count > 0 ? count : 1;
        }

        private int GetInputHeight()
        {
            return loadedModel!.InputDims.nbDims >= 3
                ? (int)loadedModel.InputDims.GetDimension(loadedModel.InputDims.nbDims - 2)
                : (int)loadedModel.InputDims.GetDimension(0);
        }

        private int GetInputWidth()
        {
            return loadedModel!.InputDims.nbDims >= 2
                ? (int)loadedModel.InputDims.GetDimension(loadedModel.InputDims.nbDims - 1)
                : (int)loadedModel.InputDims.GetDimension(0);
        }

        private void EnsureParallelSessions()
        {
            if (loadedModel is null)
            {
                throw new InvalidOperationException("Model is not loaded.");
            }

            if (parallelSessions.Count == 3)
            {
                return;
            }

            DisposeParallelSessions();
            for (int i = 0; i < 3; i++)
            {
                parallelSessions.Add(loadedModel.CreateParallelSession(runtime));
            }
        }

        private Mat RenderResult(Mat image, YoloImagePreprocessResult preprocessResult, out int resultCount)
        {
            return YoloTaskRenderer.DrawResults(
                image,
                loadedModel!.PrimarySession.Outputs,
                GetSelectedProfile(),
                preprocessResult,
                out resultCount);
        }

        private void DisplayImage(Mat image)
        {
            using Mat canvas = new(pictureBox1.Height, pictureBox1.Width, image.Type(), Scalar.Black);

            double scale = Math.Min(
                pictureBox1.Width / (double)image.Width,
                pictureBox1.Height / (double)image.Height);

            OpenCvSharp.Size scaledSize = new(
                Math.Max(1, (int)(image.Width * scale)),
                Math.Max(1, (int)(image.Height * scale)));

            using Mat resized = new();
            Cv2.Resize(image, resized, scaledSize, 0, 0);

            int x = (pictureBox1.Width - resized.Width) / 2;
            int y = (pictureBox1.Height - resized.Height) / 2;
            using (Mat roi = new(canvas, new Rect(x, y, resized.Width, resized.Height)))
            {
                resized.CopyTo(roi);
            }

            using MemoryStream stream = new();
            using Bitmap bitmap = BitmapConverter.ToBitmap(canvas);
            bitmap.Save(stream, System.Drawing.Imaging.ImageFormat.Bmp);
            stream.Position = 0;

            pictureBox1.Image?.Dispose();
            pictureBox1.Image = new Bitmap(stream);
        }

        private void DisposeLoadedModel()
        {
            loadedModel?.Dispose();
            loadedModel = null;
        }

        private void DisposeParallelSessions()
        {
            foreach (InferenceSession session in parallelSessions)
            {
                session.Dispose();
            }

            parallelSessions.Clear();
        }

        private void Form1_FormClosed(object? sender, FormClosedEventArgs e)
        {
            pictureBox1.Image?.Dispose();
            DisposeParallelSessions();
            DisposeLoadedModel();
            runtime.Dispose();
        }
    }
}
