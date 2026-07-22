using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            if (args.Length != 4)
            {
                Console.Error.WriteLine(
                    "Usage: RefittedPlan.PackageConsumer <plan> <float-input> <raw-output> <expected-output-sha256>");
                return 2;
            }

            string planPath = Path.GetFullPath(args[0]);
            string inputPath = Path.GetFullPath(args[1]);
            string outputPath = Path.GetFullPath(args[2]);
            string expectedOutputSha256 = args[3].Trim().ToLowerInvariant();
            if (!File.Exists(planPath))
            {
                throw new FileNotFoundException("Persisted refitted plan was not found.", planPath);
            }

            if (!File.Exists(inputPath))
            {
                throw new FileNotFoundException("Float input artifact was not found.", inputPath);
            }

            byte[] inputBytes = File.ReadAllBytes(inputPath);
            if (inputBytes.Length == 0 || inputBytes.Length % sizeof(float) != 0)
            {
                throw new InvalidDataException("Float input byte length must be non-zero and divisible by four.");
            }

            float[] inputValues = new float[inputBytes.Length / sizeof(float)];
            Buffer.BlockCopy(inputBytes, 0, inputValues, 0, inputBytes.Length);

            Console.WriteLine("PackageReferenceOnly=True");
            Console.WriteLine("ProjectReference=False");
            Console.WriteLine("ManualManagedAssemblyLoad=False");
            Console.WriteLine("CoreAssemblyLocation=" + typeof(TensorRtRuntime).Assembly.Location);
            Console.WriteLine("BridgeFileName=" + NativeBridgePathResolver.GetBridgeFileName());
            Console.WriteLine("PlanSha256=" + ComputeFileSha256(planPath));
            Console.WriteLine("PlanLengthBytes=" + new FileInfo(planPath).Length);
            Console.WriteLine("InputSha256=" + ComputeSha256(inputBytes));
            Console.WriteLine("InputLengthBytes=" + inputBytes.LongLength);

            TensorRtEnvironmentSnapshot environment = TensorRtEnvironmentProbe.GetCurrent();
            Console.WriteLine(
                "RuntimeEnvironment TRT=" + environment.BuildInfo.TensorRtVersion +
                " CUDA=" + environment.BuildInfo.CudaToolkitVersion +
                " TensorRtAvailable=" + environment.RuntimeInfo.TensorRtAvailable +
                " CudaAvailable=" + environment.RuntimeInfo.CudaToolkitAvailable);

            RunPersistedPlan(planPath, outputPath, inputValues, expectedOutputSha256);

            Console.WriteLine("OwnerScopeExited=True");
            Console.WriteLine("PackageConsumerRuntime=Passed");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine("PackageConsumerRuntime=Failed");
            Console.Error.WriteLine(exception.ToString());
            return 1;
        }
    }

    private static void RunPersistedPlan(
        string planPath,
        string outputPath,
        float[] inputValues,
        string expectedOutputSha256)
    {
        using TensorRtLogger logger = new TensorRtLogger(TensorRtApiLine.TensorRt10);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtEngine engine = runtime.DeserializeFromFile(planPath);
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context, profileIndex: 0);
        using CudaStream stream = new CudaStream(CudaStreamCreationFlags.NonBlocking);

        TensorRtEngineTensorBinding input = Single(bindings.Report.GetInputs(), "input");
        TensorRtEngineTensorBinding output = Single(bindings.Report.GetOutputs(), "output");
        EnsureFloatDeviceTensor(input, "input");
        EnsureFloatDeviceTensor(output, "output");

        TensorRtDims inputShape = ResolveConcreteShape(context, input);
        TensorRtDims outputShape = ResolveConcreteShape(context, output);
        int inputElementCount = CountElements(inputShape);
        int outputElementCount = CountElements(outputShape);
        if (inputElementCount != inputValues.Length)
        {
            throw new InvalidDataException(
                $"Input element count mismatch. Tensor={inputElementCount} File={inputValues.Length}.");
        }

        if (outputElementCount <= 0)
        {
            throw new InvalidDataException("Output element count must be positive.");
        }

        bindings.CopyInputFromHost(input.Name, inputValues, inputShape);
        bindings.AllocateDeviceBuffer(output.Name, outputShape, output.EstimateByteSize(outputShape));
        bindings.BindAll();
        TensorRtExecutionContextReadiness readiness = bindings.GetReadiness(runShapeInference: true);
        if (!readiness.IsReadyForEnqueue)
        {
            throw new InvalidOperationException("Bindings are not ready for enqueue: " + readiness);
        }

        TensorRtInferenceExecutionSummary execution = bindings.EnqueueAsync(
            stream,
            synchronize: true,
            runShapeInference: false);
        float[] outputValues = bindings.ReadOutputSingles(output.Name, outputElementCount);
        byte[] outputBytes = new byte[checked(outputValues.Length * sizeof(float))];
        Buffer.BlockCopy(outputValues, 0, outputBytes, 0, outputBytes.Length);

        string? outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        File.WriteAllBytes(outputPath, outputBytes);
        string outputSha256 = ComputeSha256(outputBytes);
        int predictedIndex = ArgMax(outputValues);
        bool exactMatch = string.Equals(outputSha256, expectedOutputSha256, StringComparison.OrdinalIgnoreCase);

        Console.WriteLine("EngineRefittable=" + engine.IsRefittable);
        Console.WriteLine("EngineIOTensorCount=" + engine.IOTensorCount);
        Console.WriteLine("EngineLayerCount=" + engine.LayerCount);
        Console.WriteLine("EngineOptimizationProfileCount=" + engine.OptimizationProfileCount);
        Console.WriteLine("InputTensor=" + input.Name);
        Console.WriteLine("InputShape=" + FormatShape(inputShape));
        Console.WriteLine("InputElementCount=" + inputElementCount);
        Console.WriteLine("OutputTensor=" + output.Name);
        Console.WriteLine("OutputShape=" + FormatShape(outputShape));
        Console.WriteLine("OutputElementCount=" + outputElementCount);
        Console.WriteLine("BindingsReadyForEnqueue=" + readiness.IsReadyForEnqueue);
        Console.WriteLine("EnqueueCompleted=" + execution.Synchronized);
        Console.WriteLine("ExecutionSummary=" + execution);
        Console.WriteLine("OutputLengthBytes=" + outputBytes.LongLength);
        Console.WriteLine("OutputSha256=" + outputSha256);
        Console.WriteLine("PredictedIndex=" + predictedIndex);
        Console.WriteLine("OutputExactMatch=" + exactMatch);

        if (!exactMatch)
        {
            throw new InvalidDataException(
                $"Output SHA256 mismatch. Expected={expectedOutputSha256} Actual={outputSha256}.");
        }
    }

    private static TensorRtEngineTensorBinding Single(
        System.Collections.Generic.IReadOnlyList<TensorRtEngineTensorBinding> tensors,
        string role)
    {
        if (tensors.Count != 1)
        {
            throw new NotSupportedException($"The package consumer requires exactly one {role} tensor. Count={tensors.Count}.");
        }

        return tensors[0];
    }

    private static void EnsureFloatDeviceTensor(TensorRtEngineTensorBinding tensor, string role)
    {
        if (tensor.DataType != TensorRtDataType.Float || tensor.Location != TensorRtTensorLocation.Device)
        {
            throw new NotSupportedException(
                $"The {role} tensor must be a device float tensor. Name={tensor.Name} Type={tensor.DataType} Location={tensor.Location}.");
        }
    }

    private static TensorRtDims ResolveConcreteShape(
        TensorRtExecutionContext context,
        TensorRtEngineTensorBinding tensor)
    {
        TensorRtDims shape = tensor.EngineShape;
        if (shape.Values.Length == 0 || shape.Values.Any(static value => value <= 0))
        {
            shape = context.GetTensorShape(tensor.Name);
        }

        if (shape.Values.Length == 0 || shape.Values.Any(static value => value <= 0))
        {
            throw new InvalidDataException($"Tensor '{tensor.Name}' does not have a concrete runtime shape: {shape}.");
        }

        return shape;
    }

    private static int CountElements(TensorRtDims shape)
    {
        int count = 1;
        foreach (int value in shape.Values)
        {
            count = checked(count * value);
        }

        return count;
    }

    private static int ArgMax(float[] values)
    {
        if (values.Length == 0)
        {
            throw new ArgumentException("Output values must not be empty.", nameof(values));
        }

        int index = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[index])
            {
                index = i;
            }
        }

        return index;
    }

    private static string FormatShape(TensorRtDims shape)
    {
        return "[" + string.Join(",", shape.Values) + "]";
    }

    private static string ComputeFileSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static string ComputeSha256(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }
}
