using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

TensorRtApiLine line = ResolveLine(JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "10"));
TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
TensorRtAdapterInfo adapter = line switch
{
    TensorRtApiLine.TensorRt8 => snapshot.TensorRt8,
    TensorRtApiLine.TensorRt10 => snapshot.TensorRt10,
    TensorRtApiLine.TensorRt11 => snapshot.TensorRt11,
    _ => snapshot.TensorRt10
};

Console.WriteLine($"NetworkCompatLayerMetadataSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
{
    Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
    return;
}

using TensorRtLogger logger = new TensorRtLogger(line);
using TensorRtBuilder builder = new TensorRtBuilder(logger);

List<string> results = new List<string>
{
    Probe("constant-dimensions", builder, ProbeConstantDimensions),
    Probe("cast", builder, ProbeCast),
    Probe("einsum", builder, ProbeEinsum),
    Probe("assertion", builder, ProbeAssertion),
    Probe("one-hot", builder, ProbeOneHot),
    Probe("grid-sample", builder, ProbeGridSample),
    Probe("normalization", builder, network => ProbeNormalization(network, line)),
    Probe("nms", builder, ProbeNms),
    Probe("reverse-sequence", builder, ProbeReverseSequence)
};

foreach (string result in results)
{
    Console.WriteLine(result);
}

int passed = results.FindAll(static result => result.Contains("Status=created", StringComparison.Ordinal)).Count;
Console.WriteLine($"CompatLayerMetadata Created={passed}/{results.Count}");
if (passed < 7)
{
    throw new InvalidOperationException($"Expected at least 7 TensorRT compatibility layer metadata probes to create successfully, but only {passed}/{results.Count} succeeded.");
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

static string ProbeConstantDimensions(TensorRtNetworkDefinition network)
{
    TensorRtDims dims = new TensorRtDims(new[] { 2 });
    using TensorRtLayer constant = network.AddConstant(dims, TensorRtWeights.FromSingleArray(new[] { 1.0f, 2.0f }));
    constant.Name = "compat_constant";
    constant.SetConstantLayerDimensions(dims);
    return $"Layer={constant.Type} Outputs={constant.OutputCount} Dims={constant.GetConstantLayerDimensions()}";
}

static string ProbeCast(TensorRtNetworkDefinition network)
{
    using TensorRtTensor input = network.AddInput("compat_cast_input", TensorRtDataType.Float, new TensorRtDims(new[] { 2 }));
    using TensorRtLayer cast = network.AddCast(input, TensorRtDataType.Int32);
    cast.Name = "compat_cast";
    cast.SetCastToType(TensorRtDataType.Float);
    cast.SetCastToType(TensorRtDataType.Int32);
    return $"Layer={cast.Type} Outputs={cast.OutputCount} ToType={cast.GetCastToType()}";
}

static string ProbeEinsum(TensorRtNetworkDefinition network)
{
    using TensorRtTensor left = network.AddInput("compat_einsum_left", TensorRtDataType.Float, new TensorRtDims(new[] { 2 }));
    using TensorRtTensor right = network.AddInput("compat_einsum_right", TensorRtDataType.Float, new TensorRtDims(new[] { 2 }));
    using TensorRtLayer einsum = network.AddEinsum("i,i->i", left, right);
    einsum.Name = "compat_einsum";
    bool changed = einsum.SetEinsumEquation("i,i->i");
    return $"Layer={einsum.Type} Outputs={einsum.OutputCount} Equation={einsum.GetEinsumEquation()} Changed={changed}";
}

static string ProbeAssertion(TensorRtNetworkDefinition network)
{
    using TensorRtLayer conditionLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromBooleanArray(new[] { true }));
    using TensorRtTensor condition = conditionLayer.GetOutput(0);
    using TensorRtLayer assertion = network.AddAssertion(condition, "compat assertion");
    assertion.Name = "compat_assertion";
    assertion.SetAssertionMessage("compat assertion updated");
    return $"Layer={assertion.Type} Outputs={assertion.OutputCount} Message={assertion.GetAssertionMessage()}";
}

static string ProbeOneHot(TensorRtNetworkDefinition network)
{
    using TensorRtLayer indicesLayer = network.AddConstant(new TensorRtDims(new[] { 2 }), TensorRtWeights.FromInt32Array(new[] { 0, 1 }));
    using TensorRtLayer valuesLayer = network.AddConstant(new TensorRtDims(new[] { 2 }), TensorRtWeights.FromSingleArray(new[] { 0.0f, 1.0f }));
    using TensorRtLayer depthLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromInt32Array(new[] { 2 }));
    using TensorRtTensor indices = indicesLayer.GetOutput(0);
    using TensorRtTensor values = valuesLayer.GetOutput(0);
    using TensorRtTensor depth = depthLayer.GetOutput(0);
    using TensorRtLayer oneHot = network.AddOneHot(indices, values, depth, 1);
    oneHot.Name = "compat_one_hot";
    oneHot.SetOneHotAxis(1);
    return $"Layer={oneHot.Type} Outputs={oneHot.OutputCount} Axis={oneHot.GetOneHotAxis()}";
}

static string ProbeGridSample(TensorRtNetworkDefinition network)
{
    using TensorRtTensor input = network.AddInput("compat_grid_input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 2, 2 }));
    using TensorRtLayer gridLayer = network.AddConstant(
        new TensorRtDims(new[] { 1, 2, 2, 2 }),
        TensorRtWeights.FromSingleArray(new[] { -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f }));
    using TensorRtTensor grid = gridLayer.GetOutput(0);
    using TensorRtLayer gridSample = network.AddGridSample(input, grid);
    gridSample.Name = "compat_grid_sample";
    gridSample.SetGridSampleInterpolationMode(TensorRtInterpolationMode.Nearest);
    gridSample.SetGridSampleAlignCorners(true);
    gridSample.SetGridSampleMode(TensorRtSampleMode.Fill);
    return $"Layer={gridSample.Type} Outputs={gridSample.OutputCount} Interpolation={gridSample.GetGridSampleInterpolationMode()} Align={gridSample.GetGridSampleAlignCorners()} Mode={gridSample.GetGridSampleMode()}";
}

static string ProbeNormalization(TensorRtNetworkDefinition network, TensorRtApiLine line)
{
    using TensorRtTensor input = network.AddInput("compat_norm_input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 2, 2 }));
    using TensorRtLayer scaleLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromSingleArray(new[] { 1.0f }));
    using TensorRtLayer biasLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromSingleArray(new[] { 0.0f }));
    using TensorRtTensor scale = scaleLayer.GetOutput(0);
    using TensorRtTensor bias = biasLayer.GetOutput(0);
    using TensorRtLayer normalization = network.AddNormalizationV2(input, scale, bias, 1u << 1);
    normalization.Name = "compat_normalization";
    normalization.SetNormalizationEpsilon(0.001);
    normalization.SetNormalizationAxes(1u << 1);
    normalization.SetNormalizationGroupCount(1);
    string precision = SetNormalizationComputePrecisionIfSupported(line, normalization, TensorRtDataType.Float);
    return $"Layer={normalization.Type} Outputs={normalization.OutputCount} Epsilon={normalization.GetNormalizationEpsilon()} Axes={normalization.GetNormalizationAxes()} Groups={normalization.GetNormalizationGroupCount()} {precision}";
}

static string SetNormalizationComputePrecisionIfSupported(TensorRtApiLine line, TensorRtLayer normalization, TensorRtDataType dataType)
{
    if (line == TensorRtApiLine.TensorRt11)
    {
        return "ComputePrecision=Skipped:TRT8Or10Only";
    }

    normalization.SetNormalizationComputePrecision(dataType);
    return $"ComputePrecision={normalization.GetNormalizationComputePrecision()}";
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
    using TensorRtLayer nms = network.AddNms(boxes, scores, maxOutput, TensorRtDataType.Int32);
    nms.Name = "compat_nms";
    nms.SetNmsBoundingBoxFormat(TensorRtBoundingBoxFormat.CornerPairs);
    nms.SetNmsTopKBoxLimit(100);
    return $"Layer={nms.Type} Outputs={nms.OutputCount} Format={nms.GetNmsBoundingBoxFormat()} TopK={nms.GetNmsTopKBoxLimit()}";
}

static string ProbeReverseSequence(TensorRtNetworkDefinition network)
{
    using TensorRtTensor input = network.AddInput("compat_reverse_input", TensorRtDataType.Float, new TensorRtDims(new[] { 2, 4 }));
    using TensorRtLayer lengthsLayer = network.AddConstant(new TensorRtDims(new[] { 2 }), TensorRtWeights.FromInt32Array(new[] { 3, 4 }));
    using TensorRtTensor lengths = lengthsLayer.GetOutput(0);
    using TensorRtLayer reverseSequence = network.AddReverseSequence(input, lengths);
    reverseSequence.Name = "compat_reverse_sequence";
    reverseSequence.SetReverseSequenceBatchAxis(0);
    reverseSequence.SetReverseSequenceSequenceAxis(1);
    return $"Layer={reverseSequence.Type} Outputs={reverseSequence.OutputCount} BatchAxis={reverseSequence.GetReverseSequenceBatchAxis()} SequenceAxis={reverseSequence.GetReverseSequenceSequenceAxis()} ImplicitBatch={network.HasImplicitBatchDimension}";
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
