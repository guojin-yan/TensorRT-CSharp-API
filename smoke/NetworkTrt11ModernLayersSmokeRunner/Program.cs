using System;
using System.Linq;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        TensorRtApiLine line = ResolveLine(JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "11"));
        if (line != TensorRtApiLine.TensorRt11)
        {
            Console.WriteLine($"Skipped=True Message=NetworkTrt11ModernLayersSmokeRunner is a TensorRT 11 focused smoke. RequestedLine={(int)line}");
            return;
        }

        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = snapshot.TensorRt11;
        Console.WriteLine($"NetworkTrt11ModernLayersSmokeRunner TensorRtLine=11 TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
            return;
        }

        const int elementCount = 4;
        float[] inputValues = { 1.0f, 2.0f, 3.0f, 4.0f };

        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetMaxAuxStreams(0);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtTensor input = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, elementCount }));
        input.SetDimensionName(0, "batch");
        input.SetDimensionName(1, "features");
        string dimensionBeforeClear = $"{input.GetDimensionName(0)}/{input.GetDimensionName(1)}";
        input.ClearDimensionName(1);
        input.SetDimensionName(1, "features");
        input.AllowedFormats = TensorRtTensorFormats.Linear;

        using TensorRtLayer squeezeAxesLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromInt32Array(new[] { 0 }));
        squeezeAxesLayer.Name = "squeeze_axes";
        using TensorRtTensor squeezeAxes = squeezeAxesLayer.GetOutput(0);
        squeezeAxes.Name = "squeeze_axes_tensor";

        using TensorRtLayer squeeze = network.AddSqueeze(input, squeezeAxes);
        squeeze.Name = "squeeze_batch_axis";
        using TensorRtTensor squeezed = squeeze.GetOutput(0);
        squeezed.Name = "squeezed";

        using TensorRtLayer unsqueezeAxesLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromInt32Array(new[] { 0 }));
        unsqueezeAxesLayer.Name = "unsqueeze_axes";
        using TensorRtTensor unsqueezeAxes = unsqueezeAxesLayer.GetOutput(0);
        unsqueezeAxes.Name = "unsqueeze_axes_tensor";

        using TensorRtLayer unsqueeze = network.AddUnsqueeze(squeezed, unsqueezeAxes);
        unsqueeze.Name = "unsqueeze_batch_axis";
        using TensorRtTensor output = unsqueeze.GetOutput(0);
        output.Name = "output";
        network.MarkOutput(output);

        Console.WriteLine($"TensorMetadata InputDims={input.Shape} DimNames={dimensionBeforeClear}->{input.GetDimensionName(0)}/{input.GetDimensionName(1)} AllowedFormats={input.AllowedFormats} InputShapeTensor={input.IsShapeTensor} InputExecutionTensor={input.IsExecutionTensor} AxesShapeTensor={squeezeAxes.IsShapeTensor} AxesExecutionTensor={squeezeAxes.IsExecutionTensor}");
        Console.WriteLine($"LayerMetadata Squeeze={squeeze.Name}:{squeeze.Type} Unsqueeze={unsqueeze.Name}:{unsqueeze.Type} Squeezed={squeezed.Shape} Output={output.Shape}");

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        using CudaStream stream = new CudaStream();
        using CudaMemory inputMemory = new CudaMemory(inputValues.Length * sizeof(float));
        using CudaMemory outputMemory = new CudaMemory(inputValues.Length * sizeof(float));
        inputMemory.CopyFrom(inputValues);
        outputMemory.Fill(0, outputMemory.SizeInBytes);

        context.SetTensorAddress("input", inputMemory);
        context.SetTensorAddress("output", outputMemory);
        TensorRtExecutionContextReadiness readiness = context.GetReadiness(engine, runShapeInference: true);
        context.EnqueueAsync(stream);
        stream.Synchronize();

        float[] outputValues = outputMemory.ToSingleArray(inputValues.Length);
        bool outputMatch = inputValues.SequenceEqual(outputValues);
        if (!outputMatch)
        {
            throw new InvalidOperationException($"Squeeze/unsqueeze output mismatch. Input=[{string.Join(", ", inputValues)}] Output=[{string.Join(", ", outputValues)}]");
        }

        TensorRtEngineBindingReport bindingReport = engine.GetBindingReport(context, 0, runShapeInference: false);
        Console.WriteLine($"HostMemory={hostMemory.SizeInBytes} IOTensors={engine.IOTensorCount} Readiness={readiness.IsReadyForEnqueue} BindingReady={bindingReport.IsReadyForEnqueue} OutputMatch=True");

    }

    static TensorRtApiLine ResolveLine(string value)
    {
        if (string.Equals(value, "11", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "trt11", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt11;
        }

        if (string.Equals(value, "10", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "trt10", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt10;
        }

        if (string.Equals(value, "8", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "trt8", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt8;
        }

        throw new ArgumentException("TensorRT line must be 8, 10, or 11.", nameof(value));
    }
}
