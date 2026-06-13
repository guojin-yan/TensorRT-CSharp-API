using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        TensorRtApiLine line = ResolveLine(JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "11"));
        if (line != TensorRtApiLine.TensorRt11)
        {
            Console.WriteLine($"Skipped=True Message=NetworkTrt11AdvancedLayersSmokeRunner is a TensorRT 11 focused smoke. RequestedLine={(int)line}");
            return;
        }

        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = snapshot.TensorRt11;
        Console.WriteLine($"NetworkTrt11AdvancedLayersSmokeRunner TensorRtLine=11 TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
            return;
        }

        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);

        List<string> results = new List<string>
        {
            Probe("cast", builder, ProbeCast),
            Probe("non-zero", builder, ProbeNonZero),
            Probe("ragged-softmax", builder, ProbeRaggedSoftMax),
            Probe("nms", builder, ProbeNms),
            Probe("reverse-sequence", builder, ProbeReverseSequence),
            Probe("einsum", builder, ProbeEinsum),
            Probe("loop-control-flow", builder, ProbeLoopControlFlow),
            Probe("if-conditional", builder, ProbeIfConditional),
            Probe("fill-int64", builder, ProbeFillInt64)
        };

        foreach (string result in results)
        {
            Console.WriteLine(result);
        }

        int created = results.FindAll(static result => result.Contains("Status=created", StringComparison.Ordinal)).Count;
        Console.WriteLine($"AdvancedLayers Created={created}/{results.Count}");
        if (created < results.Count)
        {
            throw new InvalidOperationException($"Expected all TensorRT 11 advanced layer probes to create successfully, but only {created}/{results.Count} succeeded.");
        }

    }

    static string Probe(string name, TensorRtBuilder builder, Func<TensorRtNetworkDefinition, string> probe)
    {
        try
        {
            using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
            return $"{name}:Status=created {probe(network)}";
        }
        catch (Exception exception)
        {
            return $"{name}:Status=failed Error={exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeCast(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor input = network.AddInput("cast_input", TensorRtDataType.Float, new TensorRtDims(new[] { 4 }));
        using TensorRtLayer cast = network.AddCast(input, TensorRtDataType.Int32);
        cast.Name = "cast_to_int32";
        cast.SetMetadata("advanced-cast");
        cast.SetInput(0, input);
        cast.SetCastToType(TensorRtDataType.Int32);
        return $"Layer={cast.Type} Outputs={cast.OutputCount} ToType={cast.GetCastToType()} Metadata={cast.GetMetadata()} Ranks={cast.GetRankCount()}";
    }

    static string ProbeNonZero(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor input = network.AddInput("nonzero_input", TensorRtDataType.Float, new TensorRtDims(new[] { 4 }));
        using TensorRtLayer nonZero = network.AddNonZero(input, TensorRtDataType.Int64);
        nonZero.Name = "non_zero_int64";
        bool setOk = nonZero.SetNonZeroIndicesType(TensorRtDataType.Int64);
        return $"Layer={nonZero.Type} Outputs={nonZero.OutputCount} IndicesType={nonZero.GetNonZeroIndicesType()} SetOk={setOk}";
    }

    static string ProbeRaggedSoftMax(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor input = network.AddInput("ragged_input", TensorRtDataType.Float, new TensorRtDims(new[] { 2, 3 }));
        using TensorRtLayer boundsLayer = network.AddConstant(new TensorRtDims(new[] { 2, 1 }), TensorRtWeights.FromInt32Array(new[] { 3, 2 }));
        using TensorRtTensor bounds = boundsLayer.GetOutput(0);
        using TensorRtLayer ragged = network.AddRaggedSoftMax(input, bounds);
        ragged.Name = "ragged_softmax";
        ragged.SetMetadata("advanced-ragged-softmax");
        return $"Layer={ragged.Type} Outputs={ragged.OutputCount} Metadata={ragged.GetMetadata()}";
    }

    static string ProbeNms(TensorRtNetworkDefinition network)
    {
        using TensorRtLayer boxesLayer = network.AddConstant(
            new TensorRtDims(new[] { 1, 2, 4 }),
            TensorRtWeights.FromSingleArray(new[] { 0.0f, 0.0f, 1.0f, 1.0f, 0.1f, 0.1f, 1.1f, 1.1f }));
        using TensorRtLayer scoresLayer = network.AddConstant(new TensorRtDims(new[] { 1, 2, 1 }), TensorRtWeights.FromSingleArray(new[] { 0.9f, 0.8f }));
        using TensorRtLayer maxOutputLayer = network.AddConstant(new TensorRtDims(Array.Empty<int>()), TensorRtWeights.FromInt32Array(new[] { 1 }));
        using TensorRtTensor boxes = boxesLayer.GetOutput(0);
        using TensorRtTensor scores = scoresLayer.GetOutput(0);
        using TensorRtTensor maxOutput = maxOutputLayer.GetOutput(0);
        using TensorRtLayer nms = network.AddNms(boxes, scores, maxOutput, TensorRtDataType.Int64);
        using TensorRtLayer iouLayer = network.AddConstant(new TensorRtDims(Array.Empty<int>()), TensorRtWeights.FromSingleArray(new[] { 0.5f }));
        using TensorRtLayer scoreLayer = network.AddConstant(new TensorRtDims(Array.Empty<int>()), TensorRtWeights.FromSingleArray(new[] { 0.25f }));
        using TensorRtTensor iouThreshold = iouLayer.GetOutput(0);
        using TensorRtTensor scoreThreshold = scoreLayer.GetOutput(0);
        nms.Name = "nms_postprocess";
        nms.SetNmsBoundingBoxFormat(TensorRtBoundingBoxFormat.CornerPairs);
        nms.SetNmsTopKBoxLimit(100);
        nms.SetNmsIouThresholdTensor(iouThreshold);
        nms.SetNmsScoreThresholdTensor(scoreThreshold);
        bool indicesOk = nms.SetNmsIndicesType(TensorRtDataType.Int64);
        return $"Layer={nms.Type} Outputs={nms.OutputCount} Format={nms.GetNmsBoundingBoxFormat()} TopK={nms.GetNmsTopKBoxLimit()} IndicesType={nms.GetNmsIndicesType()} IndicesOk={indicesOk}";
    }

    static string ProbeReverseSequence(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor input = network.AddInput("reverse_input", TensorRtDataType.Float, new TensorRtDims(new[] { 2, 3 }));
        using TensorRtLayer lengthsLayer = network.AddConstant(new TensorRtDims(new[] { 2 }), TensorRtWeights.FromInt32Array(new[] { 2, 3 }));
        using TensorRtTensor lengths = lengthsLayer.GetOutput(0);
        using TensorRtLayer reverse = network.AddReverseSequence(input, lengths);
        reverse.Name = "reverse_sequence";
        reverse.SetReverseSequenceBatchAxis(0);
        reverse.SetReverseSequenceSequenceAxis(1);
        return $"Layer={reverse.Type} Outputs={reverse.OutputCount} BatchAxis={reverse.GetReverseSequenceBatchAxis()} SequenceAxis={reverse.GetReverseSequenceSequenceAxis()}";
    }

    static string ProbeEinsum(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor left = network.AddInput("einsum_left", TensorRtDataType.Float, new TensorRtDims(new[] { 4 }));
        using TensorRtTensor right = network.AddInput("einsum_right", TensorRtDataType.Float, new TensorRtDims(new[] { 4 }));
        using TensorRtLayer einsum = network.AddEinsum("i,i->i", left, right);
        einsum.Name = "einsum_vector_product";
        einsum.SetMetadata("advanced-einsum");
        return $"Layer={einsum.Type} Outputs={einsum.OutputCount} Metadata={einsum.GetMetadata()}";
    }

    static string ProbeLoopControlFlow(TensorRtNetworkDefinition network)
    {
        using TensorRtLoop loop = network.AddLoop();
        loop.Name = "loop_probe";
        using TensorRtLayer countLayer = network.AddConstant(new TensorRtDims(Array.Empty<int>()), TensorRtWeights.FromInt32Array(new[] { 2 }));
        using TensorRtTensor count = countLayer.GetOutput(0);
        using TensorRtLayer tripLimit = loop.AddTripLimit(count, TensorRtTripLimitKind.Count);
        tripLimit.Name = "loop_trip_count";
        return $"Loop={loop.Name} TripLayer={tripLimit.Type} TripKind={tripLimit.GetTripLimitKind()} Boundary={tripLimit.GetLoopBoundaryLoopName()}";
    }

    static string ProbeIfConditional(TensorRtNetworkDefinition network)
    {
        using TensorRtIfConditional conditional = network.AddIfConditional();
        conditional.Name = "if_probe";
        using TensorRtLayer conditionLayer = network.AddConstant(new TensorRtDims(Array.Empty<int>()), TensorRtWeights.FromBooleanArray(new[] { true }));
        using TensorRtTensor condition = conditionLayer.GetOutput(0);
        using TensorRtLayer setCondition = conditional.SetCondition(condition);
        setCondition.Name = "if_condition";
        return $"Conditional={conditional.Name} ConditionLayer={setCondition.Type} Boundary={setCondition.GetIfConditionalBoundaryName()}";
    }

    static string ProbeFillInt64(TensorRtNetworkDefinition network)
    {
        using TensorRtLayer fill = network.AddFill(new TensorRtDims(new[] { 2 }), TensorRtFillOperation.Linspace);
        fill.Name = "fill_int64_probe";
        fill.SetFillAlphaInt64(1);
        fill.SetFillBetaInt64(4);
        return $"Layer={fill.Type} Outputs={fill.OutputCount} Alpha={fill.GetFillAlphaInt64()} Beta={fill.GetFillBetaInt64()} IsInt64={fill.IsFillAlphaBetaInt64()}";
    }

    static TensorRtApiLine ResolveLine(string value)
    {
        if (string.Equals(value, "11", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "trt11", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt11;
        }

        throw new ArgumentException("TensorRT line must be 11 for this sample.", nameof(value));
    }
}
