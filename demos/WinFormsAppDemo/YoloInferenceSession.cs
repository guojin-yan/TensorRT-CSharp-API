using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Nvinfer;
using System.Diagnostics;

namespace WinFormsAppDemo;

internal sealed class TensorBindingMetadata
{
    public required string Name { get; init; }

    public required Dims Dims { get; init; }

    public ulong ElementCount => (ulong)Dims.GetElementProduct();
}

internal sealed class OutputBufferBinding : IDisposable
{
    public OutputBufferBinding(TensorBindingMetadata metadata)
    {
        Metadata = metadata;
        DeviceMemory = new Cuda1DMemory<float>(metadata.ElementCount);
        HostBuffer = new float[(int)metadata.ElementCount];
    }

    public TensorBindingMetadata Metadata { get; }

    public Cuda1DMemory<float> DeviceMemory { get; }

    public float[] HostBuffer { get; }

    public void Dispose()
    {
        DeviceMemory.Dispose();
    }
}

internal readonly record struct InferenceTimings(double InputCopyMs, double ComputeMs, double OutputCopyMs)
{
    public double TotalMs => InputCopyMs + ComputeMs + OutputCopyMs;
}

internal sealed class InferenceSession : IDisposable
{
    private readonly string _inputName;
    private readonly CudaEngine _engine;
    private readonly JYPPX.TensorRtSharp.Nvinfer.ExecutionContext _context;
    private readonly Cuda1DMemory<float> _inputBuffer;
    private readonly List<OutputBufferBinding> _outputs;
    private readonly CudaStream _inputStream;
    private readonly CudaStream _computeStream;
    private readonly CudaStream _outputStream;

    private InferenceSession(
        string inputName,
        CudaEngine engine,
        JYPPX.TensorRtSharp.Nvinfer.ExecutionContext context,
        Cuda1DMemory<float> inputBuffer,
        List<OutputBufferBinding> outputs,
        CudaStream inputStream,
        CudaStream computeStream,
        CudaStream outputStream)
    {
        _inputName = inputName;
        _engine = engine;
        _context = context;
        _inputBuffer = inputBuffer;
        _outputs = outputs;
        _inputStream = inputStream;
        _computeStream = computeStream;
        _outputStream = outputStream;
    }

    public IReadOnlyList<OutputBufferBinding> Outputs => _outputs;

    public static InferenceSession Create(
        CudaEngine engine,
        string inputName,
        Dims inputDims,
        IReadOnlyList<TensorBindingMetadata> outputMetadatas)
    {
        JYPPX.TensorRtSharp.Nvinfer.ExecutionContext context =
            engine.createExecutionContext(TrtExecutionContextAllocationStrategy.kSTATIC);

        Cuda1DMemory<float> inputBuffer = new((ulong)inputDims.GetElementProduct());
        CudaStream inputStream = new();
        CudaStream computeStream = new();
        CudaStream outputStream = new();

        context.setInputTensorAddress(inputName, inputBuffer.get());

        List<OutputBufferBinding> outputs = [];
        foreach (TensorBindingMetadata metadata in outputMetadatas)
        {
            OutputBufferBinding binding = new(metadata);
            context.setOutputTensorAddress(metadata.Name, binding.DeviceMemory.get());
            outputs.Add(binding);
        }

        return new InferenceSession(
            inputName,
            engine,
            context,
            inputBuffer,
            outputs,
            inputStream,
            computeStream,
            outputStream);
    }

    public InferenceTimings Run(float[] inputTensor)
    {
        Stopwatch sw = Stopwatch.StartNew();
        _inputBuffer.copyFromHostAsync(inputTensor, _inputStream);
        _inputStream.Synchronize();
        sw.Stop();
        double inputCopyMs = sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        _context.executeV3(_computeStream);
        _computeStream.Synchronize();
        sw.Stop();
        double computeMs = sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        foreach (OutputBufferBinding output in _outputs)
        {
            output.DeviceMemory.copyToHostAsync(output.HostBuffer, _outputStream);
        }

        _outputStream.Synchronize();
        sw.Stop();

        return new InferenceTimings(inputCopyMs, computeMs, sw.Elapsed.TotalMilliseconds);
    }

    public double RunBenchmark(float[] inputTensor, int count)
    {
        Stopwatch sw = Stopwatch.StartNew();
        for (int i = 0; i < count; i++)
        {
            Run(inputTensor);
        }

        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / count;
    }

    public void Dispose()
    {
        foreach (OutputBufferBinding output in _outputs)
        {
            output.Dispose();
        }

        _inputBuffer.Dispose();
        _context.Dispose();
        _engine.Dispose();
        _inputStream.Dispose();
        _computeStream.Dispose();
        _outputStream.Dispose();
    }
}

internal sealed class LoadedYoloModel : IDisposable
{
    private LoadedYoloModel(
        string enginePath,
        byte[] engineBytes,
        YoloModelProfile profile,
        string inputName,
        Dims inputDims,
        List<TensorBindingMetadata> outputs,
        InferenceSession primarySession)
    {
        EnginePath = enginePath;
        EngineBytes = engineBytes;
        Profile = profile;
        InputName = inputName;
        InputDims = inputDims;
        Outputs = outputs;
        PrimarySession = primarySession;
    }

    public string EnginePath { get; }

    public byte[] EngineBytes { get; }

    public YoloModelProfile Profile { get; }

    public string InputName { get; }

    public Dims InputDims { get; }

    public IReadOnlyList<TensorBindingMetadata> Outputs { get; }

    public InferenceSession PrimarySession { get; }

    public static LoadedYoloModel Load(Runtime runtime, string enginePath, YoloModelProfile profile)
    {
        byte[] engineBytes = File.ReadAllBytes(enginePath);
        CudaEngine engine = runtime.deserializeCudaEngineByBlob(engineBytes, (ulong)engineBytes.Length);

        string inputName = string.Empty;
        Dims inputDims = new();
        List<TensorBindingMetadata> outputs = [];

        int tensorCount = engine.getNbIOTensors();
        for (int i = 0; i < tensorCount; i++)
        {
            string tensorName = engine.getIOTensorName(i);
            TrtTensorIOMode tensorMode = engine.getTensorIOMode(tensorName);
            Dims tensorDims = engine.getTensorShape(tensorName);

            if (tensorMode == TrtTensorIOMode.kINPUT && string.IsNullOrWhiteSpace(inputName))
            {
                inputName = tensorName;
                inputDims = tensorDims;
            }
            else if (tensorMode == TrtTensorIOMode.kOUTPUT)
            {
                outputs.Add(new TensorBindingMetadata
                {
                    Name = tensorName,
                    Dims = tensorDims
                });
            }
        }

        if (string.IsNullOrWhiteSpace(inputName) || outputs.Count == 0)
        {
            engine.Dispose();
            throw new InvalidOperationException("Failed to resolve model input/output tensors.");
        }

        InferenceSession primarySession = InferenceSession.Create(engine, inputName, inputDims, outputs);
        return new LoadedYoloModel(enginePath, engineBytes, profile, inputName, inputDims, outputs, primarySession);
    }

    public InferenceSession CreateParallelSession(Runtime runtime)
    {
        CudaEngine engine = runtime.deserializeCudaEngineByBlob(EngineBytes, (ulong)EngineBytes.Length);
        return InferenceSession.Create(engine, InputName, InputDims, Outputs);
    }

    public void Dispose()
    {
        PrimarySession.Dispose();
    }
}
