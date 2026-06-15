using System;
using System.IO;
using System.Linq;
using JYPPX.CudaSharp;
using JYPPX.SampleSupport;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace OnnxToEngineSample;

internal static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            return Run(args);
        }
        catch (Exception exception) when (TensorRtSampleSupport.IsDeploymentException(exception))
        {
            Console.WriteLine($"OnnxToEngine=Skipped Reason={exception.Message}");
            return 0;
        }
        catch (ArgumentException exception)
        {
            Console.WriteLine($"OnnxToEngine=InvalidArguments Reason={exception.Message}");
            PrintUsage();
            return 2;
        }
    }

    private static int Run(string[] args)
    {
        TensorRtApiLine line = TensorRtSampleSupport.ResolveLine(SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "10"));
        int batch = SampleCommandLine.GetIntArgument(args, "--batch", 2);
        if (batch < 1 || batch > 4)
        {
            throw new ArgumentOutOfRangeException(nameof(batch), "Batch must be in the optimization profile range [1, 4].");
        }

        byte[] model = OnnxIdentityModel.CreateDynamicBatchModel();
        Console.WriteLine($"OnnxToEngine TensorRtLine={(int)line} ModelBytes={model.Length} Batch={batch}");

        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = TensorRtSampleSupport.SelectAdapter(snapshot, line);
        Console.WriteLine($"Preflight TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion} Runtime={adapter.RuntimeCreationSupported} Builder={adapter.BuilderCreationSupported}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            Console.WriteLine($"OnnxToEngine=Skipped Reason={adapter.StatusMessage}");
            return 0;
        }

        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        using CudaStream stream = new CudaStream();

        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        config.SetProfileStream(stream);
        config.SetOptimizationLevel(3);
        config.SetEngineCapability(TensorRtEngineCapability.Standard);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtOnnxParser parser = new TensorRtOnnxParser(logger, network);
        if (!parser.Parse(model, "sample-dynamic-identity.onnx"))
        {
            throw new InvalidOperationException(parser.GetErrorSummary());
        }

        using TensorRtOptimizationProfile profile = builder.CreateOptimizationProfile();
        profile.SetShape(
            "input",
            new TensorRtDims(new[] { 1, 4 }),
            new TensorRtDims(new[] { 2, 4 }),
            new TensorRtDims(new[] { 4, 4 }));
        int profileIndex = config.AddOptimizationProfile(profile);

        string enginePath = Path.Combine(Path.GetTempPath(), $"jyppx-onnx-to-engine-{Guid.NewGuid():N}.plan");
        try
        {
            using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
            hostMemory.SaveToFile(enginePath);
            using TensorRtEngine engine = runtime.DeserializeFromFile(enginePath);
            using TensorRtExecutionContext context = engine.CreateExecutionContext();

            TensorRtDims runtimeShape = new TensorRtDims(new[] { batch, 4 });
            float[] inputValues = Enumerable.Range(0, batch * 4).Select(index => index + 0.5f).ToArray();
            using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context, profileIndex);
            bindings.SetInputShape("input", runtimeShape)
                    .CopyInputFromHost("input", inputValues, runtimeShape);
            bindings.AllocateDeviceBuffer("output", runtimeShape, checked(inputValues.Length * sizeof(float)));
            bindings.BindAll();

            TensorRtInferenceExecutionSummary executionSummary = null!;
            float elapsedMilliseconds = stream.MeasureElapsedTime(cudaStream =>
            {
                executionSummary = bindings.EnqueueAsync(cudaStream, synchronize: false, runShapeInference: false);
            });

            float[] outputValues = bindings.ReadOutputSingles("output", inputValues.Length);
            if (!inputValues.SequenceEqual(outputValues))
            {
                throw new InvalidOperationException($"Output mismatch. Input=[{string.Join(", ", inputValues)}] Output=[{string.Join(", ", outputValues)}]");
            }

            Console.WriteLine($"Parsed=True ProfileIndex={profileIndex} EngineFileRoundTrip=True");
            Console.WriteLine($"BindingReport Ready={bindings.Report.IsReadyForEnqueue} Inputs={bindings.Report.GetInputs().Count} Outputs={bindings.Report.GetOutputs().Count}");
            Console.WriteLine($"Execution {executionSummary} ElapsedMs={elapsedMilliseconds:0.###} OutputMatch=True");
            Console.WriteLine("OnnxToEngine Passed=True");
            return 0;
        }
        finally
        {
            if (File.Exists(enginePath))
            {
                File.Delete(enginePath);
            }
        }
    }

    private static void PrintUsage()
    {
        Console.WriteLine("OnnxToEngine sample");
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project samples/OnnxToEngine -- --tensor-rt-line 10 --batch 2");
        Console.WriteLine("Options:");
        Console.WriteLine("  --tensor-rt-line <8|10|11>  TensorRT adapter line. Default: 10.");
        Console.WriteLine("  --batch <1..4>              Runtime batch inside the optimization profile. Default: 2.");
    }
}

internal static class OnnxIdentityModel
{
    public static byte[] CreateDynamicBatchModel()
    {
        ProtoWriter model = new ProtoWriter();
        model.Int64(1, 8);
        model.String(2, "JYPPX.TensorRtSharp");
        model.Message(7, CreateGraph());
        model.Message(8, CreateOpsetImport(13));
        return model.ToArray();
    }

    private static byte[] CreateGraph()
    {
        ProtoWriter graph = new ProtoWriter();
        graph.Message(1, CreateIdentityNode());
        graph.String(2, "jyppx_dynamic_identity_graph");
        graph.Message(11, CreateValueInfo("input"));
        graph.Message(12, CreateValueInfo("output"));
        return graph.ToArray();
    }

    private static byte[] CreateIdentityNode()
    {
        ProtoWriter node = new ProtoWriter();
        node.String(1, "input");
        node.String(2, "output");
        node.String(3, "identity");
        node.String(4, "Identity");
        return node.ToArray();
    }

    private static byte[] CreateValueInfo(string name)
    {
        ProtoWriter valueInfo = new ProtoWriter();
        valueInfo.String(1, name);
        valueInfo.Message(2, CreateTensorFloatType());
        return valueInfo.ToArray();
    }

    private static byte[] CreateTensorFloatType()
    {
        ProtoWriter tensorType = new ProtoWriter();
        tensorType.Message(1, CreateTensorType());
        return tensorType.ToArray();
    }

    private static byte[] CreateTensorType()
    {
        ProtoWriter type = new ProtoWriter();
        type.UInt64(1, 1);
        type.Message(2, CreateShape());
        return type.ToArray();
    }

    private static byte[] CreateShape()
    {
        ProtoWriter shape = new ProtoWriter();
        shape.Message(1, CreateDimension("batch"));
        shape.Message(1, CreateDimension(4));
        return shape.ToArray();
    }

    private static byte[] CreateDimension(string parameterName)
    {
        ProtoWriter dimension = new ProtoWriter();
        dimension.String(2, parameterName);
        return dimension.ToArray();
    }

    private static byte[] CreateDimension(long value)
    {
        ProtoWriter dimension = new ProtoWriter();
        dimension.Int64(1, value);
        return dimension.ToArray();
    }

    private static byte[] CreateOpsetImport(long version)
    {
        ProtoWriter opset = new ProtoWriter();
        opset.Int64(2, version);
        return opset.ToArray();
    }
}

internal sealed class ProtoWriter
{
    private readonly MemoryStream _stream = new MemoryStream();

    public void Int64(int fieldNumber, long value)
    {
        WriteTag(fieldNumber, 0);
        WriteVarint(unchecked((ulong)value));
    }

    public void UInt64(int fieldNumber, ulong value)
    {
        WriteTag(fieldNumber, 0);
        WriteVarint(value);
    }

    public void String(int fieldNumber, string value)
    {
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(value);
        Bytes(fieldNumber, bytes);
    }

    public void Message(int fieldNumber, byte[] value)
    {
        Bytes(fieldNumber, value);
    }

    public byte[] ToArray()
    {
        return _stream.ToArray();
    }

    private void Bytes(int fieldNumber, byte[] value)
    {
        WriteTag(fieldNumber, 2);
        WriteVarint((ulong)value.Length);
        _stream.Write(value, 0, value.Length);
    }

    private void WriteTag(int fieldNumber, int wireType)
    {
        WriteVarint((ulong)((fieldNumber << 3) | wireType));
    }

    private void WriteVarint(ulong value)
    {
        while (value >= 0x80)
        {
            _stream.WriteByte((byte)(value | 0x80));
            value >>= 7;
        }

        _stream.WriteByte((byte)value);
    }
}
