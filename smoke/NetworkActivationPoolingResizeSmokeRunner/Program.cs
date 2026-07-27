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
        Console.WriteLine($"NetworkActivationPoolingResizeSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
            return;
        }

        float[] inputValues =
        {
            -1.0f, 2.0f, -3.0f, 4.0f,
            5.0f, -6.0f, 7.0f, -8.0f,
            -9.0f, 10.0f, -11.0f, 12.0f,
            13.0f, -14.0f, 15.0f, -16.0f
        };
        float[] expectedValues = { 5.0f, 7.0f, 13.0f, 15.0f };

        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetMaxAuxStreams(0);
        config.SetAverageTimingIterations(1);
        config.SetProfilingVerbosity(TensorRtProfilingVerbosity.Detailed);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        network.Name = "activation_pooling_resize_network";
        using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 4, 4 }));
        inputTensor.AllowedFormats = TensorRtTensorFormats.Linear;
        inputTensor.Location = TensorRtTensorLocation.Device;
        string dynamicRangeText;
        if (line == TensorRtApiLine.TensorRt11)
        {
            dynamicRangeText = "Skipped:TensorRT11RemovedTensorDynamicRange";
        }
        else
        {
            inputTensor.SetDynamicRange(-16.0f, 16.0f);
            dynamicRangeText = $"[{inputTensor.DynamicRangeMinimum},{inputTensor.DynamicRangeMaximum}]";
            inputTensor.ResetDynamicRange();
        }

        Console.WriteLine($"InputTensorMetadata Name={inputTensor.Name} DataType={inputTensor.DataType} Location={inputTensor.Location} Formats={inputTensor.AllowedFormats} DynamicRange={dynamicRangeText}");

        using TensorRtLayer activationLayer = network.AddActivation(inputTensor, TensorRtActivationType.Relu);
        activationLayer.Name = "relu";
        activationLayer.SetActivationType(TensorRtActivationType.LeakyRelu);
        activationLayer.SetActivationAlpha(0.0);
        activationLayer.SetActivationBeta(0.0);
        string layerPrecisionText;
        if (line == TensorRtApiLine.TensorRt11)
        {
            layerPrecisionText = $"Skipped:TensorRT11PrecisionSetterPending OutputType={activationLayer.GetOutputType(0)}";
        }
        else
        {
            activationLayer.Precision = TensorRtDataType.Float;
            activationLayer.SetOutputType(0, TensorRtDataType.Float);
            layerPrecisionText = $"ActivationIsSet={activationLayer.IsPrecisionSet} Precision={activationLayer.Precision} OutputTypeSet={activationLayer.IsOutputTypeSet(0)} OutputType={activationLayer.GetOutputType(0)}";
            activationLayer.ResetOutputType(0);
            activationLayer.ResetPrecision();
        }

        Console.WriteLine($"LayerPrecision {layerPrecisionText}");
        using TensorRtTensor activationTensor = activationLayer.GetOutput(0);
        activationTensor.Name = "relu_output";

        using TensorRtLayer poolingLayer = network.AddPooling(activationTensor, TensorRtPoolingType.Max, new TensorRtDims(new[] { 2, 2 }));
        poolingLayer.Name = "max_pool_2x2";
        poolingLayer.SetPoolingType(TensorRtPoolingType.Max);
        poolingLayer.SetPoolingStride(new TensorRtDims(new[] { 2, 2 }));
        poolingLayer.SetPoolingPadding(new TensorRtDims(new[] { 0, 0 }));
        poolingLayer.SetPoolingPrePadding(new TensorRtDims(new[] { 0, 0 }));
        poolingLayer.SetPoolingPostPadding(new TensorRtDims(new[] { 0, 0 }));
        poolingLayer.SetPoolingPaddingMode(TensorRtPaddingMode.ExplicitRoundDown);
        poolingLayer.SetPoolingAverageCountExcludesPadding(true);
        poolingLayer.SetPoolingBlendFactor(0.0);
        using TensorRtTensor poolingTensor = poolingLayer.GetOutput(0);
        poolingTensor.Name = "pool_output";

        using TensorRtLayer resizeLayer = network.AddResize(poolingTensor, new TensorRtDims(new[] { 1, 1, 2, 2 }), TensorRtResizeMode.Nearest);
        resizeLayer.Name = "resize_identity_shape";
        resizeLayer.SetResizeCoordinateTransformation(TensorRtResizeCoordinateTransformation.Asymmetric);
        resizeLayer.SetResizeSelectorForSinglePixel(TensorRtResizeSelector.Formula);
        resizeLayer.SetResizeNearestRounding(TensorRtResizeRoundMode.Floor);
        resizeLayer.SetResizeCubicCoefficient(-0.75);
        resizeLayer.SetResizeExcludeOutside(false);
        string resizeAlignCornersText = SetResizeAlignCornersIfSupported(line, resizeLayer, false);
        string resizeAdvancedText = $"Coord={resizeLayer.GetResizeCoordinateTransformation()}:Selector={resizeLayer.GetResizeSelectorForSinglePixel()}:Round={resizeLayer.GetResizeNearestRounding()}:Cubic={resizeLayer.GetResizeCubicCoefficient()}:ExcludeOutside={resizeLayer.GetResizeExcludeOutside()}:{resizeAlignCornersText}";
        using TensorRtTensor outputTensor = resizeLayer.GetOutput(0);
        outputTensor.Name = "resize_output";
        network.MarkOutput(outputTensor);

        config.SetLayerDeviceType(activationLayer, TensorRtDeviceType.Gpu);
        Console.WriteLine($"LayerDeviceType IsSet={config.IsLayerDeviceTypeSet(activationLayer)} Device={config.GetLayerDeviceType(activationLayer)}");
        config.ResetLayerDeviceType(activationLayer);

        string networkFlagsText;
        try
        {
            networkFlagsText = $"{network.Flags}:ExplicitBatch={network.GetFlag(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch)}";
        }
        catch (BridgeProbeException exception)
        {
            networkFlagsText = $"Unavailable:{exception.StatusCode}";
        }

        using TensorRtLayer firstLayer = network.GetLayer(0);
        Console.WriteLine($"LayerMetadata Activation={activationLayer.Name}:{activationLayer.Type}:Type={activationLayer.GetActivationType()}:Alpha={activationLayer.GetActivationAlpha()}:Beta={activationLayer.GetActivationBeta()} Pooling={poolingLayer.Name}:{poolingLayer.Type}:Type={poolingLayer.GetPoolingType()}:Window={poolingLayer.GetPoolingWindowSize()}:Stride={poolingLayer.GetPoolingStride()}:Padding={poolingLayer.GetPoolingPadding()}:Pre={poolingLayer.GetPoolingPrePadding()}:Post={poolingLayer.GetPoolingPostPadding()}:Mode={poolingLayer.GetPoolingPaddingMode()}:AvgExcludes={poolingLayer.GetPoolingAverageCountExcludesPadding()}:Blend={poolingLayer.GetPoolingBlendFactor()} Resize={resizeLayer.Name}:{resizeLayer.Type}:Mode={resizeLayer.GetResizeMode()}:OutputDims={resizeLayer.GetResizeOutputDimensions()}:{resizeAdvancedText}");
        Console.WriteLine($"Network Name={network.Name} Flags={networkFlagsText} Inputs={network.InputCount} Outputs={network.OutputCount} Layers={network.LayerCount} FirstLayer={firstLayer.Name} Input={inputTensor.Name}:{inputTensor.Shape} Activation={activationTensor.Name}:{activationTensor.Shape} Pooling={poolingTensor.Name}:{poolingTensor.Shape} Output={outputTensor.Name}:{outputTensor.Shape}");

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using TensorRtEngineInspector inspector = engine.CreateInspector();
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        context.Name = "activation_pooling_resize_context";
        context.DebugSync = false;
        context.EnqueueEmitsProfile = true;
        using CudaStream stream = new CudaStream();
        using CudaMemory inputMemory = new CudaMemory(inputValues.Length * sizeof(float));
        using CudaMemory outputMemory = new CudaMemory(expectedValues.Length * sizeof(float));

        inputMemory.CopyFrom(inputValues);
        outputMemory.Fill(0, outputMemory.SizeInBytes);

        context.SetTensorAddress("input", inputMemory);
        context.SetTensorAddress("resize_output", outputMemory);
        Console.WriteLine($"EngineMetadata Name={engine.Name} Layers={engine.LayerCount} Refittable={engine.IsRefittable} Capability={engine.Capability} Tactics={engine.TacticSources} Profiling={engine.ProfilingVerbosity} InputFormat={engine.GetTensorFormat("input")} OutputFormat={engine.GetTensorFormat("resize_output")} InputBytes={engine.GetTensorBytesPerComponent("input")} InputComponents={engine.GetTensorComponentsPerElement("input")} InputVectorDim={engine.GetTensorVectorizedDimension("input")} InputLocation={engine.GetTensorLocation("input")} OutputLocation={engine.GetTensorLocation("resize_output")} ShapeIO={engine.IsShapeInferenceIO("input")}");
        Console.WriteLine($"ContextMetadata Name={context.Name} Profile={context.OptimizationProfileIndex} DebugSync={context.DebugSync} EnqueueEmitsProfile={context.EnqueueEmitsProfile} InputBound={context.IsTensorAddressBound("input")} OutputBound={context.IsTensorAddressBound("resize_output")} InputShape={context.GetTensorShape("input")} OutputShape={context.GetTensorShape("resize_output")} InputStrides={context.GetTensorStrides("input")} OutputStrides={context.GetTensorStrides("resize_output")}");
        bool errorBufferAvailable = context.TryGetErrorBuffer(out string errorBuffer, out string errorBufferDiagnostic);
        Console.WriteLine($"ExecutionContextErrorBufferCopy Available={errorBufferAvailable} Length={errorBuffer.Length} Diagnostic={errorBufferDiagnostic}");
        context.EnqueueAsync(stream);
        stream.Synchronize();

        float[] outputValues = outputMemory.ToSingleArray(expectedValues.Length);
        AssertClose("ActivationPoolingResize", expectedValues, outputValues, 0.0001f);

        IReadOnlyList<TensorRtTensorInfo> tensors = engine.GetIOTensors();
        string tensorSummary = string.Join("; ", tensors.Select(tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.DataType}:{tensor.Shape}"));
        string inspectorText = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
        Console.WriteLine($"HostMemory={hostMemory.SizeInBytes} EngineIOTensors={engine.IOTensorCount} IOTensors=[{tensorSummary}] InspectorBytes={inspectorText.Length}");
        Console.WriteLine("ActivationPoolingResizeOutputMatch=True");

    }

    static string SetResizeAlignCornersIfSupported(TensorRtApiLine line, TensorRtLayer resizeLayer, bool alignCorners)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            return "AlignCorners=Skipped:TRT8Only";
        }

        resizeLayer.SetResizeAlignCorners(alignCorners);
        return $"AlignCorners={resizeLayer.GetResizeAlignCorners()}";
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
