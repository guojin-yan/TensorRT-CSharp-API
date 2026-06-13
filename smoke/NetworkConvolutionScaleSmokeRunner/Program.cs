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

        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = line switch
        {
            TensorRtApiLine.TensorRt8 => snapshot.TensorRt8,
            TensorRtApiLine.TensorRt10 => snapshot.TensorRt10,
            TensorRtApiLine.TensorRt11 => snapshot.TensorRt11,
            _ => snapshot.TensorRt10
        };
        Console.WriteLine($"NetworkConvolutionScaleSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
            return;
        }

        float[] inputValues = { 1.0f, 2.0f, -3.0f, 4.0f };
        float[] expectedValues = inputValues.Select(static value => value + 0.5f).ToArray();

        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetMaxAuxStreams(0);
        config.SetAverageTimingIterations(1);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        network.Name = "convolution_scale_padding_network";
        using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 2, 2 }));

        TensorRtWeights convKernel = TensorRtWeights.FromSingleArray(new[] { 2.0f });
        TensorRtWeights convBias = TensorRtWeights.FromSingleArray(new[] { 1.0f });
        using TensorRtLayer convolutionLayer = network.AddConvolution(inputTensor, 1, new TensorRtDims(new[] { 1, 1 }), convKernel, convBias);
        convolutionLayer.Name = "conv_1x1";
        convolutionLayer.SetConvolutionGroups(1);
        convolutionLayer.SetConvolutionStride(new TensorRtDims(new[] { 1, 1 }));
        convolutionLayer.SetConvolutionPadding(new TensorRtDims(new[] { 0, 0 }));
        convolutionLayer.SetConvolutionPrePadding(new TensorRtDims(new[] { 0, 0 }));
        convolutionLayer.SetConvolutionPostPadding(new TensorRtDims(new[] { 0, 0 }));
        convolutionLayer.SetConvolutionDilation(new TensorRtDims(new[] { 1, 1 }));
        convolutionLayer.SetConvolutionPaddingMode(TensorRtPaddingMode.ExplicitRoundDown);
        AssertDims("Convolution padding", new[] { 0, 0 }, convolutionLayer.GetConvolutionPadding());
        using TensorRtTensor convolutionTensor = convolutionLayer.GetOutput(0);
        convolutionTensor.Name = "conv_output";

        TensorRtWeights scaleShift = TensorRtWeights.FromSingleArray(new[] { 0.0f });
        TensorRtWeights scaleScale = TensorRtWeights.FromSingleArray(new[] { 0.5f });
        TensorRtWeights scalePower = TensorRtWeights.FromSingleArray(new[] { 1.0f });
        using TensorRtLayer scaleLayer = network.AddScale(convolutionTensor, TensorRtScaleMode.Uniform, scaleShift, scaleScale, scalePower, channelAxis: 1);
        scaleLayer.Name = "scale_uniform";
        using TensorRtTensor scaleTensor = scaleLayer.GetOutput(0);
        scaleTensor.Name = "scale_output";

        using TensorRtLayer paddingLayer = network.AddPadding(scaleTensor, new TensorRtDims(new[] { 0, 0 }), new TensorRtDims(new[] { 0, 0 }));
        paddingLayer.Name = "padding_identity";
        using TensorRtTensor outputTensor = paddingLayer.GetOutput(0);
        outputTensor.Name = "output";
        network.MarkOutput(outputTensor);

        Console.WriteLine(
            $"LayerMetadata ConvMaps={convolutionLayer.GetConvolutionOutputMaps()} Groups={convolutionLayer.GetConvolutionGroups()} " +
            $"Stride={convolutionLayer.GetConvolutionStride()} Padding={convolutionLayer.GetConvolutionPadding()} Pre={convolutionLayer.GetConvolutionPrePadding()} Post={convolutionLayer.GetConvolutionPostPadding()} Dilation={convolutionLayer.GetConvolutionDilation()} PaddingMode={convolutionLayer.GetConvolutionPaddingMode()} " +
            $"ScaleMode={scaleLayer.GetScaleMode()} ScaleAxis={scaleLayer.GetScaleChannelAxis()} PaddingPre={paddingLayer.GetPaddingPrePadding()} PaddingPost={paddingLayer.GetPaddingPostPadding()}");

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using TensorRtEngineInspector inspector = engine.CreateInspector();
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        using CudaStream stream = new CudaStream();
        using CudaEvent inputConsumedEvent = new CudaEvent();
        using CudaMemory inputMemory = new CudaMemory(inputValues.Length * sizeof(float));
        using CudaMemory outputMemory = new CudaMemory(expectedValues.Length * sizeof(float));
        using CudaMemory? contextDeviceMemory = TryCreateAndSetContextDeploymentMemory(context, engine);

        context.PersistentCacheLimitInBytes = 0;
        context.SetInputConsumedEvent(inputConsumedEvent);

        inputMemory.CopyFrom(inputValues);
        outputMemory.Fill(0, outputMemory.SizeInBytes);
        context.SetTensorAddress("input", inputMemory);
        context.SetTensorAddress("output", outputMemory);

        Console.WriteLine($"ContextDeployment DeviceMemorySize={TryGetContextDeviceMemorySize(context)} PersistentCache={context.PersistentCacheLimitInBytes} InputBound={context.IsTensorAddressBound("input")} OutputBound={context.IsTensorAddressBound("output")}");
        context.EnqueueAsync(stream);
        stream.Synchronize();

        float[] outputValues = outputMemory.ToSingleArray(expectedValues.Length);
        AssertClose("ConvolutionScalePadding", expectedValues, outputValues, 0.0001f);

        IReadOnlyList<TensorRtTensorInfo> tensors = engine.GetIOTensors();
        string tensorSummary = string.Join("; ", tensors.Select(tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.DataType}:{tensor.Shape}"));
        string inspectorText = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
        Console.WriteLine($"Network Name={network.Name} Layers={network.LayerCount} HostMemory={hostMemory.SizeInBytes} EngineMemory={engine.DeviceMemorySizeInBytes} IOTensors=[{tensorSummary}] InspectorBytes={inspectorText.Length}");
        Console.WriteLine("ConvolutionScalePaddingOutputMatch=True");

    }

    static CudaMemory? TryCreateAndSetContextDeploymentMemory(TensorRtExecutionContext context, TensorRtEngine engine)
    {
        if (engine.DeviceMemorySizeInBytes == 0 || engine.DeviceMemorySizeInBytes > int.MaxValue)
        {
            Console.WriteLine($"ContextDeviceMemory Skipped=True Size={engine.DeviceMemorySizeInBytes}");
            return null;
        }

        CudaMemory deviceMemory = new CudaMemory(checked((int)engine.DeviceMemorySizeInBytes));
        context.SetDeviceMemory(deviceMemory);
        Console.WriteLine($"ContextDeviceMemory Set=True Size={deviceMemory.SizeInBytes}");
        return deviceMemory;
    }

    static string TryGetContextDeviceMemorySize(TensorRtExecutionContext context)
    {
        try
        {
            return context.DeviceMemorySizeInBytes.ToString();
        }
        catch (BridgeProbeException exception) when (exception.StatusCode == BridgeStatusCode.NotSupported)
        {
            return $"NotSupported:{exception.Message}";
        }
    }

    static void AssertClose(string name, float[] expected, float[] actual, float tolerance)
    {
        bool match = expected.Length == actual.Length &&
            expected.Zip(actual, (expectedValue, actualValue) => Math.Abs(expectedValue - actualValue) <= tolerance).All(static value => value);
        if (!match)
        {
            throw new InvalidOperationException($"{name} output mismatch. Expected=[{string.Join(", ", expected)}] Actual=[{string.Join(", ", actual)}]");
        }
    }

    static void AssertDims(string name, int[] expected, TensorRtDims actual)
    {
        bool match = expected.SequenceEqual(actual.Values);
        if (!match)
        {
            throw new InvalidOperationException($"{name} mismatch. Expected=[{string.Join(", ", expected)}] Actual={actual}");
        }
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
