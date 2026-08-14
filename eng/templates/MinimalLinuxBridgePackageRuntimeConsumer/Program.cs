using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp;
using JYPPX.TensorRtSharp.Shared.Interop;

static TensorRtApiLine ParseTensorRtLine(string[] arguments)
{
    int index = Array.FindIndex(arguments, static argument => argument == "--tensor-rt-line");
    string value = index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : "";
    return value switch
    {
        "8" => TensorRtApiLine.TensorRt8,
        "10" => TensorRtApiLine.TensorRt10,
        "11" => TensorRtApiLine.TensorRt11,
        _ => throw new ArgumentException("--tensor-rt-line must be 8, 10, or 11.")
    };
}

try
{
    TensorRtApiLine line = ParseTensorRtLine(args);
    TensorRtEnvironmentSnapshot environment = TensorRtEnvironmentProbe.GetCurrent();
    Console.WriteLine(
        "RuntimeEnvironment TRT=" + environment.BuildInfo.TensorRtVersion +
        " CUDA=" + environment.BuildInfo.CudaToolkitVersion +
        " TensorRtAvailable=" + environment.RuntimeInfo.TensorRtAvailable +
        " CudaAvailable=" + environment.RuntimeInfo.CudaToolkitAvailable);
    Console.WriteLine("CudaDeviceCount=" + CudaDevice.Count);
    Console.WriteLine("CudaDeviceName=" + CudaDevice.GetInfo(CudaDevice.Current).Name);

    using TensorRtLogger logger = new(line);
    using TensorRtRuntime runtime = new(logger);
    using TensorRtBuilder builder = new(logger);
    using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
    using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
    using CudaStream stream = new();

    config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 32UL * 1024UL * 1024UL);
    TensorRtDims shape = new(new[] { 1, 4 });
    using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, shape);
    using TensorRtLayer identity = network.AddIdentity(inputTensor);
    using TensorRtTensor outputTensor = identity.GetOutput(0);
    outputTensor.Name = "output";
    network.MarkOutput(outputTensor);

    using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
    byte[] serializedEngine = hostMemory.ToArray();
    Console.WriteLine("EngineSerializedBytes=" + serializedEngine.Length);

    using TensorRtEngine engine = runtime.Deserialize(serializedEngine);
    using TensorRtExecutionContext context = engine.CreateExecutionContext();
    float[] inputValues = { 1.25f, -2.5f, 3.75f, 9.5f };
    using TensorRtInferenceBindings bindings = new(engine, context);
    bindings.CopyInputFromHost("input", inputValues, shape);
    bindings.AllocateDeviceBuffer("output", shape);
    bindings.BindAll();

    TensorRtExecutionContextReadiness readiness = bindings.GetReadiness(runShapeInference: true);
    Console.WriteLine("ReadyForEnqueue=" + readiness.IsReadyForEnqueue);
    TensorRtInferenceExecutionSummary execution = bindings.EnqueueAsync(
        stream,
        synchronize: true,
        runShapeInference: false);
    float[] outputValues = bindings.ReadOutputSingles("output", inputValues.Length);
    bool outputMatch = inputValues
        .Zip(outputValues, static (expected, actual) => Math.Abs(expected - actual) <= 0.0001f)
        .All(static match => match);

    Console.WriteLine("ExecutionSummary=" + execution);
    Console.WriteLine("EnqueueCompleted=True");
    Console.WriteLine("StreamSynchronized=" + execution.Synchronized);
    Console.WriteLine("IdentityOutputMatch=" + outputMatch);
    Console.WriteLine("MinimalPackageRuntimeSmoke=" + (outputMatch ? "Passed" : "Failed"));
    return outputMatch ? 0 : 1;
}
catch (Exception exception)
{
    Console.Error.WriteLine("MinimalPackageRuntimeSmoke=Failed");
    Console.Error.WriteLine(exception);
    return 1;
}
