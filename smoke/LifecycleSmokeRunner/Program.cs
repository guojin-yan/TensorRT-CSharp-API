using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        int iterations = JYPPX.SampleSupport.SampleCommandLine.GetPositiveIntArgument(args, "--iterations", 5);
        string tensorRtLine = JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "auto");
        bool skipCuda = HasFlag(args, "--skip-cuda");
        bool skipTensorRt = HasFlag(args, "--skip-tensorrt");

        Console.WriteLine($"LifecycleSmokeRunner Iterations={iterations} TensorRtLine={tensorRtLine} SkipCuda={skipCuda} SkipTensorRt={skipTensorRt}");

        if (!skipCuda)
        {
            RunCudaLifecycle(iterations);
        }

        if (!skipTensorRt)
        {
            RunTensorRtLifecycle(iterations, tensorRtLine);
        }

        Console.WriteLine("LifecycleSmokeRunner Passed=True");

    }

    static void RunCudaLifecycle(int iterations)
    {
        CudaEnvironmentSnapshot snapshot = CudaEnvironmentProbe.GetCurrent();
        Console.WriteLine($"CUDA Vendor={snapshot.CudaRuntimeInfo.VendorDependencyAvailable} DeviceCount={snapshot.CudaRuntimeInfo.DeviceCount}");
        if (!snapshot.CudaRuntimeInfo.VendorDependencyAvailable)
        {
            Console.WriteLine(snapshot.CudaRuntimeInfo.StatusMessage);
            return;
        }

        for (int i = 0; i < iterations; i++)
        {
            using CudaStream stream = new CudaStream();
            using CudaEvent cudaEvent = new CudaEvent();
            using CudaMemory memory = new CudaMemory(128);

            byte[] source = Enumerable.Range(0, 128).Select(value => (byte)((value + i) % 251)).ToArray();
            memory.CopyFrom(source);
            byte[] roundTrip = new byte[source.Length];
            memory.CopyTo(roundTrip);
            if (!source.SequenceEqual(roundTrip))
            {
                throw new InvalidOperationException($"CUDA round-trip mismatch at iteration {i}.");
            }

            memory.Fill((byte)(0x40 + (i % 32)), 128);
            byte[] filled = memory.ToArray(128);
            if (!filled.All(value => value == (byte)(0x40 + (i % 32))))
            {
                throw new InvalidOperationException($"CUDA fill mismatch at iteration {i}.");
            }

            cudaEvent.Record(stream);
            stream.Synchronize();
            cudaEvent.Synchronize();
        }

        Console.WriteLine($"CUDA LifecycleIterations={iterations} Passed=True");
    }

    static void RunTensorRtLifecycle(int iterations, string tensorRtLine)
    {
        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        Console.WriteLine($"TensorRT Bridge={snapshot.BuildInfo.BridgeName} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");

        TensorRtApiLine? line = ResolveTensorRtLine(snapshot, tensorRtLine);
        if (line == null)
        {
            Console.WriteLine("TensorRT lifecycle skipped because no requested adapter line is available.");
            return;
        }

        for (int i = 0; i < iterations; i++)
        {
            bool loggerOk = TensorRtEnvironmentProbe.TryCreateLogger(line.Value, out string loggerMessage);
            bool runtimeOk = TensorRtEnvironmentProbe.TryCreateRuntime(line.Value, out string runtimeMessage);
            bool builderOk = TensorRtEnvironmentProbe.TryCreateBuilder(line.Value, out string builderMessage);
            string chainMessage;
            bool chainOk = line.Value switch
            {
                TensorRtApiLine.TensorRt8 => TensorRtEnvironmentProbe.TryRunTensorRt8MinimalBuildChain(out chainMessage),
                TensorRtApiLine.TensorRt10 => TensorRtEnvironmentProbe.TryRunTensorRt10MinimalBuildChain(out chainMessage),
                TensorRtApiLine.TensorRt11 => TensorRtEnvironmentProbe.TryRunTensorRt11MinimalBuildChain(out chainMessage),
                _ => throw new InvalidOperationException($"Unsupported TensorRT line: {line.Value}."),
            };

            if (!loggerOk || !runtimeOk || !builderOk || !chainOk)
            {
                throw new InvalidOperationException(
                    $"TensorRT lifecycle failed at iteration {i}. " +
                    $"Logger={loggerOk}:{loggerMessage}; Runtime={runtimeOk}:{runtimeMessage}; " +
                    $"Builder={builderOk}:{builderMessage}; Chain={chainOk}:{chainMessage}");
            }

            string advancedMessage = RunTensorRtAdvancedApiSmoke(line.Value);
            Console.WriteLine($"TensorRT AdvancedApi Iteration={i + 1} {advancedMessage}");
        }

        Console.WriteLine($"TensorRT Line={line} LifecycleIterations={iterations} Passed=True");
    }

    static string RunTensorRtAdvancedApiSmoke(TensorRtApiLine line)
    {
        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetEngineCapability(TensorRtEngineCapability.Standard);
        config.SetHardwareCompatibilityLevel(TensorRtHardwareCompatibilityLevel.None);
        string builderConfigDeploymentState = ProbeBuilderConfigDeploymentState(config, line);
        using TensorRtNetworkDefinition network = builder.CreateNetwork();

        string parserState = ProbeOnnxParser(logger, network);
        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using TensorRtEngineInspector inspector = engine.CreateInspector();
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        using CudaStream stream = new CudaStream();

        IReadOnlyList<TensorRtTensorInfo> tensors = engine.GetIOTensors();
        List<CudaMemory> buffers = new List<CudaMemory>();
        string readinessState = "NotCollected";
        try
        {
            foreach (TensorRtTensorInfo tensor in tensors)
            {
                if (tensor.IOMode == TensorRtIOMode.Input)
                {
                    context.SetInputShape(tensor.Name, tensor.Shape);
                }

                int byteCount = EstimateTensorBytes(tensor);
                CudaMemory buffer = new CudaMemory(byteCount);
                buffers.Add(buffer);
                if (tensor.IOMode == TensorRtIOMode.Input)
                {
                    buffer.Fill(0, byteCount);
                }

                context.SetTensorAddress(tensor.Name, buffer);
            }

            TensorRtExecutionContextReadiness readiness = context.GetReadiness(engine, runShapeInference: true);
            TensorRtEngineBindingReport bindingReport = engine.GetBindingReport(context, 0, runShapeInference: false);
            readinessState = $"Ready:{readiness.IsReadyForEnqueue}/Bound:{readiness.AllTensorAddressesBound}/Missing:{readiness.ShapeInferenceMissingTensorCount?.ToString() ?? "n/a"}/BindingReport:{bindingReport.Tensors.Count}:{bindingReport.IsReadyForEnqueue}";
            context.EnqueueAsync(stream);
            stream.Synchronize();
        }
        finally
        {
            foreach (CudaMemory buffer in buffers)
            {
                buffer.Dispose();
            }
        }

        string inspectorText = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
        return $"Parser={parserState} BuilderConfig=[{builderConfigDeploymentState}] InspectorBytes={inspectorText.Length} Readiness={readinessState} Enqueue=True IOTensors={tensors.Count}";
    }

    static string ProbeOnnxParser(TensorRtLogger logger, TensorRtNetworkDefinition network)
    {
        try
        {
            using TensorRtOnnxParser parser = new TensorRtOnnxParser(logger, network);
            return $"Available ErrorCount={parser.ErrorCount}";
        }
        catch (BridgeProbeException exception) when (exception.StatusCode == BridgeStatusCode.DependencyMissing || exception.StatusCode == BridgeStatusCode.NotSupported)
        {
            return $"Unavailable {exception.StatusCode}";
        }
    }

    static string ProbeBuilderConfigDeploymentState(TensorRtBuilderConfig config, TensorRtApiLine line)
    {
        TensorRtEngineCapability capability = config.GetEngineCapability();
        TensorRtHardwareCompatibilityLevel hardwareCompatibility = config.GetHardwareCompatibilityLevel();
        TensorRtPreviewFeature previewFeature = line == TensorRtApiLine.TensorRt10
            ? TensorRtPreviewFeature.ProfileSharing0806Trt10
            : TensorRtPreviewFeature.FasterDynamicShapes0805;
        bool previewEnabled = config.GetPreviewFeature(previewFeature);
        string runtimePlatform = "Unsupported";
        if (line == TensorRtApiLine.TensorRt10)
        {
            config.SetRuntimePlatform(TensorRtRuntimePlatform.SameAsBuild);
            runtimePlatform = config.GetRuntimePlatform().ToString();
        }

        return $"Capability={capability} HardwareCompatibility={hardwareCompatibility} PreviewFeature={previewFeature}:{previewEnabled} RuntimePlatform={runtimePlatform}";
    }

    static int EstimateTensorBytes(TensorRtTensorInfo tensor)
    {
        long elementCount = 1;
        foreach (int dimension in tensor.Shape.Values)
        {
            int normalizedDimension = dimension <= 0 ? 1 : dimension;
            checked
            {
                elementCount *= normalizedDimension;
            }
        }

        long byteCount = checked(elementCount * GetDataTypeSize(tensor.DataType));
        if (byteCount <= 0 || byteCount > int.MaxValue)
        {
            throw new InvalidOperationException($"Tensor {tensor.Name} has unsupported byte size {byteCount}.");
        }

        return (int)byteCount;
    }

    static int GetDataTypeSize(TensorRtDataType dataType)
    {
        switch (dataType)
        {
            case TensorRtDataType.Half:
            case TensorRtDataType.BFloat16:
                return 2;
            case TensorRtDataType.Float:
            case TensorRtDataType.Int32:
                return 4;
            case TensorRtDataType.Int64:
                return 8;
            case TensorRtDataType.Int8:
            case TensorRtDataType.Bool:
            case TensorRtDataType.UInt8:
            case TensorRtDataType.Float8:
                return 1;
            default:
                return 4;
        }
    }

    static TensorRtApiLine? ResolveTensorRtLine(TensorRtEnvironmentSnapshot snapshot, string requestedLine)
    {
        if (string.Equals(requestedLine, "8", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(requestedLine, "trt8", StringComparison.OrdinalIgnoreCase))
        {
            return snapshot.TensorRt8.RuntimeCreationSupported ? TensorRtApiLine.TensorRt8 : null;
        }

        if (string.Equals(requestedLine, "10", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(requestedLine, "trt10", StringComparison.OrdinalIgnoreCase))
        {
            return snapshot.TensorRt10.RuntimeCreationSupported && snapshot.TensorRt10.BuilderCreationSupported ? TensorRtApiLine.TensorRt10 : null;
        }

        if (string.Equals(requestedLine, "11", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(requestedLine, "trt11", StringComparison.OrdinalIgnoreCase))
        {
            return snapshot.TensorRt11.RuntimeCreationSupported && snapshot.TensorRt11.BuilderCreationSupported ? TensorRtApiLine.TensorRt11 : null;
        }

        if (snapshot.TensorRt10.RuntimeCreationSupported && snapshot.TensorRt10.BuilderCreationSupported)
        {
            return TensorRtApiLine.TensorRt10;
        }

        if (snapshot.TensorRt11.RuntimeCreationSupported && snapshot.TensorRt11.BuilderCreationSupported)
        {
            return TensorRtApiLine.TensorRt11;
        }

        if (snapshot.TensorRt8.RuntimeCreationSupported)
        {
            return TensorRtApiLine.TensorRt8;
        }

        return null;
    }


    static bool HasFlag(string[] args, string name)
    {
        return args.Any(argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
    }
}
