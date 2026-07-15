using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        TensorRtApiLine line = ResolveLine(JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "10"));
        int batch = JYPPX.SampleSupport.SampleCommandLine.GetIntArgument(args, "--batch", 2);
        const int width = 4;
        if (batch < 1 || batch > 4)
        {
            throw new ArgumentOutOfRangeException(nameof(batch), "Batch must be in the static smoke-test range [1, 4].");
        }

        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = line switch
        {
            TensorRtApiLine.TensorRt8 => snapshot.TensorRt8,
            TensorRtApiLine.TensorRt10 => snapshot.TensorRt10,
            TensorRtApiLine.TensorRt11 => snapshot.TensorRt11,
            _ => snapshot.TensorRt10
        };
        Console.WriteLine($"NetworkLayersSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion} Batch={batch}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
            return;
        }

        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetMaxAuxStreams(0);
        config.SetProfilingVerbosity(TensorRtProfilingVerbosity.Detailed);

        int elementCount = batch * width;
        float[] inputValues = Enumerable.Range(0, elementCount).Select(static index => index + 1.0f).ToArray();
        float[] biasValues = Enumerable.Range(0, elementCount).Select(static index => 0.25f + index * 0.5f).ToArray();
        float[] expectedValues = inputValues.Zip(biasValues, static (left, right) => left + right).ToArray();

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { batch, width }));
        inputTensor.SetDimensionName(0, "batch");
        inputTensor.SetDimensionName(1, "width");
        inputTensor.BroadcastAcrossBatch = false;
        string dynamicRangeState;
        if (line == TensorRtApiLine.TensorRt11)
        {
            dynamicRangeState = "Skipped:TensorRT11RemovedDynamicRangeIsSet";
        }
        else
        {
            inputTensor.SetDynamicRange(-8.0f, 8.0f);
            dynamicRangeState = $"Set={inputTensor.IsDynamicRangeSet} Range=[{inputTensor.DynamicRangeMinimum},{inputTensor.DynamicRangeMaximum}]";
            inputTensor.ResetDynamicRange();
        }

        using TensorRtLayer constantLayer = network.AddConstant(new TensorRtDims(new[] { batch, width }), TensorRtWeights.FromSingleArray(biasValues));
        constantLayer.Name = "bias_constant";
        constantLayer.SetMetadata("constant-bias");
        using TensorRtTensor biasTensor = constantLayer.GetOutput(0);
        biasTensor.Name = "bias";

        using TensorRtLayer sumLayer = network.AddElementWise(inputTensor, biasTensor, TensorRtElementWiseOperation.Sum);
        sumLayer.Name = "sum";
        sumLayer.SetMetadata("elementwise-sum");
        using TensorRtTensor outputTensor = sumLayer.GetOutput(0);
        outputTensor.Name = "sum_output";
        outputTensor.SetDimensionName(0, "batch");
        network.MarkOutput(outputTensor);

        using TensorRtLayer ownerBoundNonPluginLayer = network.GetLayer(network.LayerCount - 1);
        if (ownerBoundNonPluginLayer.TryGetPluginV2Metadata(out TensorRtPluginV2LayerMetadata? unexpectedPluginMetadata, out string pluginMetadataDiagnostic))
        {
            throw new InvalidOperationException($"A non-plugin layer unexpectedly returned PluginV2 metadata: {unexpectedPluginMetadata}");
        }
        Console.WriteLine($"PluginV2LayerMetadataRejected=True Layer={ownerBoundNonPluginLayer.Name}:{ownerBoundNonPluginLayer.Type} Diagnostic={pluginMetadataDiagnostic}");

        if (ownerBoundNonPluginLayer.TryGetPluginV3Metadata(out TensorRtPluginV3LayerMetadata? unexpectedPluginV3Metadata, out string pluginV3MetadataDiagnostic))
        {
            throw new InvalidOperationException($"A non-plugin layer unexpectedly returned PluginV3 metadata: {unexpectedPluginV3Metadata}");
        }
        Console.WriteLine($"PluginV3LayerMetadataRejected=True Layer={ownerBoundNonPluginLayer.Name}:{ownerBoundNonPluginLayer.Type} Diagnostic={pluginV3MetadataDiagnostic}");

        Console.WriteLine($"LayerMetadata Constant={constantLayer.Name}:{constantLayer.Type}:I{constantLayer.InputCount}:O{constantLayer.OutputCount}:{constantLayer.GetMetadata()} Sum={sumLayer.Name}:{sumLayer.Type}:I{sumLayer.InputCount}:O{sumLayer.OutputCount}:{sumLayer.GetMetadata()}");
        Console.WriteLine($"TensorDeploymentMetadata InputRoles={inputTensor.IsNetworkInput}/{inputTensor.IsNetworkOutput} OutputRoles={outputTensor.IsNetworkInput}/{outputTensor.IsNetworkOutput} Dims={inputTensor.GetDimensionName(0)}/{inputTensor.GetDimensionName(1)}->{outputTensor.GetDimensionName(0)} Broadcast={inputTensor.BroadcastAcrossBatch} DynamicRange={dynamicRangeState}");
        Console.WriteLine($"Network Inputs={network.InputCount} Outputs={network.OutputCount} Input={inputTensor.Name}:{inputTensor.Shape} Output={outputTensor.Name}:{outputTensor.Shape}");

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using TensorRtEngineInspector inspector = engine.CreateInspector();
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        using CudaStream stream = new CudaStream();
        using CudaMemory inputMemory = new CudaMemory(inputValues.Length * sizeof(float));
        using CudaMemory outputMemory = new CudaMemory(inputValues.Length * sizeof(float));

        inputMemory.CopyFrom(inputValues);
        outputMemory.Fill(0, outputMemory.SizeInBytes);

        context.SetTensorAddress("input", inputMemory);
        context.SetTensorAddress("sum_output", outputMemory);
        context.EnqueueAsync(stream);
        stream.Synchronize();

        float[] outputValues = outputMemory.ToSingleArray(inputValues.Length);
        bool outputMatch = expectedValues.Zip(outputValues, static (expected, actual) => Math.Abs(expected - actual) < 0.0001f).All(static value => value);
        if (!outputMatch)
        {
            throw new InvalidOperationException($"ElementWise network output mismatch. Expected=[{string.Join(", ", expectedValues)}] Actual=[{string.Join(", ", outputValues)}]");
        }

        IReadOnlyList<TensorRtTensorInfo> tensors = engine.GetIOTensors();
        string tensorSummary = string.Join("; ", tensors.Select(tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.DataType}:{tensor.Shape}"));
        string inspectorText = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
        Console.WriteLine($"HostMemory={hostMemory.SizeInBytes} EngineIOTensors={engine.IOTensorCount} IOTensors=[{tensorSummary}] InspectorBytes={inspectorText.Length}");
        Console.WriteLine("ElementWiseOutputMatch=True");

    }

    static TensorRtApiLine ResolveLine(string value)
    {
        if (string.Equals(value, "8", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "trt8", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt8;
        }

        if (string.Equals(value, "10", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "trt10", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt10;
        }

        if (string.Equals(value, "11", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "trt11", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt11;
        }

        throw new ArgumentException("TensorRT line must be 8, 10, or 11.", nameof(value));
    }
}
