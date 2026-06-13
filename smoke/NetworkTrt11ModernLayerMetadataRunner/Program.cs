using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = snapshot.TensorRt11;
        Console.WriteLine($"NetworkTrt11ModernLayerMetadataRunner TensorRtLine=11 TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
        bool builderSafeRegistryAvailable = TensorRtEnvironmentProbe.IsBuilderSafePluginRegistryAvailable(TensorRtApiLine.TensorRt11, TensorRtEngineCapability.Safety);
        Console.WriteLine($"BuilderSafePluginRegistry Exists={builderSafeRegistryAvailable}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
            return;
        }

        using TensorRtLogger logger = new TensorRtLogger(TensorRtApiLine.TensorRt11);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);

        List<string> results = new List<string>
        {
            Probe("scatter", builder, ProbeScatter),
            Probe("one-hot", builder, ProbeOneHot),
            Probe("cumulative", builder, ProbeCumulative),
            Probe("assertion", builder, ProbeAssertion),
            Probe("grid-sample", builder, ProbeGridSample),
            Probe("normalization-v2", builder, ProbeNormalizationV2),
            Probe("dynamic-quantize-v2", builder, ProbeDynamicQuantizeV2),
            Probe("dequantize-fill-types", builder, ProbeDequantizeFillTypes),
            Probe("rotary-embedding", builder, ProbeRotaryEmbedding),
            Probe("kv-cache-update", builder, ProbeKvCacheUpdate),
            Probe("safe-network-v2", builder, ProbeSafeNetworkV2),
            Probe("attention-v2", builder, ProbeAttentionV2)
        };

        List<string> optionalResults = new List<string>
        {
            ProbeOptional("moe", builder, ProbeMoE, "TensorRT 11 MoE layer creation is gated by the current TensorRT/GPU capability set."),
            ProbeOptional("dist-collective", builder, ProbeDistCollective, "TensorRT 11 DistCollective layer creation is gated by the current TensorRT multi-device capability set.")
        };

        string timingCacheResult = ProbeTimingCache(builder);

        foreach (string result in results)
        {
            Console.WriteLine(result);
        }

        foreach (string result in optionalResults)
        {
            Console.WriteLine(result);
        }

        Console.WriteLine(timingCacheResult);

        int passed = results.FindAll(static result => result.Contains("Status=created", StringComparison.Ordinal)).Count;
        int dims64EvidenceCount = results.FindAll(static result => result.Contains("Shape64=", StringComparison.Ordinal) && result.Contains("Ext64=", StringComparison.Ordinal)).Count;
        Console.WriteLine($"ModernLayerMetadata Created={passed}/{results.Count}");
        Console.WriteLine($"ModernLayerMetadata Dims64Evidence={dims64EvidenceCount}/{results.Count}");
        if (passed < results.Count)
        {
            throw new InvalidOperationException($"Expected all {results.Count} TensorRT 11 modern layer metadata probes to create successfully, but only {passed} succeeded.");
        }

        if (dims64EvidenceCount < results.Count)
        {
            throw new InvalidOperationException($"Expected all {results.Count} TensorRT 11 probes to produce Dims64 evidence, but only {dims64EvidenceCount} did.");
        }

        if (!timingCacheResult.Contains("Status=created", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"Expected TensorRT 11 timing cache probe to complete successfully: {timingCacheResult}");
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

    static string ProbeOptional(string name, TensorRtBuilder builder, Func<TensorRtNetworkDefinition, string> probe, string skipReason)
    {
        try
        {
            using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
            return $"{name}:Status=created Optional=True {probe(network)}";
        }
        catch (Exception exception) when (exception.GetType().Name == "BridgeProbeException" && exception.Message.Contains("null TensorRT object", StringComparison.OrdinalIgnoreCase))
        {
            return $"{name}:Status=skipped Optional=True Reason={skipReason} Error={exception.GetType().Name}:{exception.Message}";
        }
        catch (Exception exception)
        {
            return $"{name}:Status=failed Optional=True Error={exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTimingCache(TensorRtBuilder builder)
    {
        try
        {
            using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
            using TensorRtTimingCache cache = config.CreateTimingCache();
            using TensorRtTimingCache inputCache = config.CreateTimingCache();
            long initialCount = cache.QueryKeyCount();
            byte[] flatKeys = cache.QueryKeyBytes();
            int keyListCount = cache.QueryKeys().Count;
            bool combined = cache.Combine(inputCache, ignoreMismatch: true);
            string queryState = "skipped";
            if (flatKeys.Length >= TensorRtTimingCache.KeySizeInBytes)
            {
                byte[] firstKey = new byte[TensorRtTimingCache.KeySizeInBytes];
                Array.Copy(flatKeys, firstKey, firstKey.Length);
                bool queried = cache.TryQuery(firstKey, out TensorRtTimingCacheValue queryValue);
                queryState = $"{queried}:{queryValue.IsValid}";
            }

            bool reset = cache.Reset();
            long resetCount = cache.QueryKeyCount();
            return $"timing-cache:Status=created Count={initialCount}->{resetCount} KeysBytes={flatKeys.Length} KeyList={keyListCount} Combine={combined} Reset={reset} Query={queryState}";
        }
        catch (Exception exception)
        {
            return $"timing-cache:Status=failed Error={exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeScatter(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor data = network.AddInput("scatter_data", TensorRtDataType.Float, new TensorRtDims(new[] { 4 }));
        using TensorRtLayer indicesLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromInt32Array(new[] { 2 }));
        using TensorRtTensor indices = indicesLayer.GetOutput(0);
        using TensorRtLayer updatesLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromSingleArray(new[] { 9.0f }));
        using TensorRtTensor updates = updatesLayer.GetOutput(0);
        using TensorRtLayer scatter = network.AddScatter(data, indices, updates, TensorRtScatterMode.Element);
        scatter.Name = "scatter_element";
        scatter.SetScatterAxis(0);
        scatter.SetScatterMode(TensorRtScatterMode.Element);
        return $"Layer={scatter.Type} Outputs={scatter.OutputCount} Mode={scatter.GetScatterMode()} Axis={scatter.GetScatterAxis()} TensorSlots=[{DescribeLayerTensors(scatter)}]";
    }

    static string ProbeOneHot(TensorRtNetworkDefinition network)
    {
        using TensorRtLayer indicesLayer = network.AddConstant(new TensorRtDims(new[] { 2 }), TensorRtWeights.FromInt32Array(new[] { 0, 2 }));
        using TensorRtTensor indices = indicesLayer.GetOutput(0);
        using TensorRtLayer valuesLayer = network.AddConstant(new TensorRtDims(new[] { 2 }), TensorRtWeights.FromSingleArray(new[] { 0.0f, 1.0f }));
        using TensorRtTensor values = valuesLayer.GetOutput(0);
        using TensorRtLayer depthLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromInt32Array(new[] { 3 }));
        using TensorRtTensor depth = depthLayer.GetOutput(0);
        using TensorRtLayer oneHot = network.AddOneHot(indices, values, depth, 1);
        oneHot.Name = "one_hot_axis_1";
        oneHot.SetOneHotAxis(1);
        return $"Layer={oneHot.Type} Outputs={oneHot.OutputCount} Axis={oneHot.GetOneHotAxis()} TensorSlots=[{DescribeLayerTensors(oneHot)}]";
    }

    static string ProbeCumulative(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor input = network.AddInput("cumulative_input", TensorRtDataType.Float, new TensorRtDims(new[] { 4 }));
        using TensorRtLayer axisLayer = network.AddConstant(new TensorRtDims(Array.Empty<int>()), TensorRtWeights.FromInt32Array(new[] { 0 }));
        using TensorRtTensor axis = axisLayer.GetOutput(0);
        using TensorRtLayer cumulative = network.AddCumulative(input, axis, TensorRtCumulativeOperation.Sum, exclusive: true, reverse: true);
        cumulative.Name = "cumulative_sum";
        cumulative.SetCumulativeOperation(TensorRtCumulativeOperation.Sum);
        cumulative.SetCumulativeExclusive(true);
        cumulative.SetCumulativeReverse(true);
        bool shapeOutputMarked = network.MarkOutputForShapes(axis);
        bool shapeOutputUnmarked = network.UnmarkOutputForShapes(axis);
        return $"Layer={cumulative.Type} Outputs={cumulative.OutputCount} Operation={cumulative.GetCumulativeOperation()} Exclusive={cumulative.GetCumulativeExclusive()} Reverse={cumulative.GetCumulativeReverse()} ShapeOutput={shapeOutputMarked}->{shapeOutputUnmarked} TensorSlots=[{DescribeLayerTensors(cumulative)}]";
    }

    static string ProbeAssertion(TensorRtNetworkDefinition network)
    {
        using TensorRtLayer conditionLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromBooleanArray(new[] { true }));
        using TensorRtTensor condition = conditionLayer.GetOutput(0);
        using TensorRtLayer assertion = network.AddAssertion(condition, "metadata probe assertion");
        assertion.Name = "assertion_probe";
        assertion.SetAssertionMessage("metadata probe assertion updated");
        return $"Layer={assertion.Type} Outputs={assertion.OutputCount} Message={assertion.GetAssertionMessage()} TensorSlots=[{DescribeLayerTensors(assertion)}]";
    }

    static string ProbeGridSample(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor input = network.AddInput("grid_input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 2, 2 }));
        float[] gridValues =
        {
            -1.0f, -1.0f,
             1.0f, -1.0f,
            -1.0f,  1.0f,
             1.0f,  1.0f
        };
        using TensorRtLayer gridLayer = network.AddConstant(new TensorRtDims(new[] { 1, 2, 2, 2 }), TensorRtWeights.FromSingleArray(gridValues));
        using TensorRtTensor grid = gridLayer.GetOutput(0);
        using TensorRtLayer gridSample = network.AddGridSample(input, grid);
        gridSample.Name = "grid_sample_probe";
        gridSample.SetGridSampleInterpolationMode(TensorRtInterpolationMode.Nearest);
        gridSample.SetGridSampleAlignCorners(true);
        gridSample.SetGridSampleMode(TensorRtSampleMode.Fill);
        return $"Layer={gridSample.Type} Outputs={gridSample.OutputCount} Interpolation={gridSample.GetGridSampleInterpolationMode()} AlignCorners={gridSample.GetGridSampleAlignCorners()} SampleMode={gridSample.GetGridSampleMode()} TensorSlots=[{DescribeLayerTensors(gridSample)}]";
    }

    static string ProbeNormalizationV2(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor input = network.AddInput("norm_input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 2, 2 }));
        using TensorRtLayer scaleLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromSingleArray(new[] { 1.0f }));
        using TensorRtTensor scale = scaleLayer.GetOutput(0);
        using TensorRtLayer biasLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromSingleArray(new[] { 0.0f }));
        using TensorRtTensor bias = biasLayer.GetOutput(0);
        using TensorRtLayer normalization = network.AddNormalizationV2(input, scale, bias, 1u << 1);
        normalization.Name = "normalization_v2_probe";
        normalization.SetNormalizationEpsilon(0.001);
        normalization.SetNormalizationAxes(1u << 1);
        normalization.SetNormalizationGroupCount(1);
        return $"Layer={normalization.Type} Outputs={normalization.OutputCount} Epsilon={normalization.GetNormalizationEpsilon()} Axes={normalization.GetNormalizationAxes()} Groups={normalization.GetNormalizationGroupCount()} IsV2={normalization.IsNormalizationV2()} TensorSlots=[{DescribeLayerTensors(normalization)}]";
    }

    static string ProbeDynamicQuantizeV2(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor input = network.AddInput("dq_input", TensorRtDataType.Float, new TensorRtDims(new[] { 32 }));
        using TensorRtLayer quantize = network.AddDynamicQuantizeV2(input, new TensorRtDims(new[] { 32 }), TensorRtDataType.Float8, TensorRtDataType.Float);
        quantize.Name = "dynamic_quantize_v2_probe";
        quantize.SetDynamicQuantizeToType(TensorRtDataType.Float8);
        quantize.SetDynamicQuantizeScaleType(TensorRtDataType.Float);
        quantize.SetDynamicQuantizeBlockShape(new TensorRtDims(new[] { 32 }));
        quantize.SetDynamicQuantizeAxis(0);
        quantize.SetDynamicQuantizeBlockSize(32);
        return $"Layer={quantize.Type} Outputs={quantize.OutputCount} ToType={quantize.GetDynamicQuantizeToType()} ScaleType={quantize.GetDynamicQuantizeScaleType()} BlockShape={quantize.GetDynamicQuantizeBlockShape()} Axis={quantize.GetDynamicQuantizeAxis()} BlockSize={quantize.GetDynamicQuantizeBlockSize()} TensorSlots=[{DescribeLayerTensors(quantize)}]";
    }

    static string ProbeDequantizeFillTypes(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor quantized = network.AddInput("dequantize_input", TensorRtDataType.Int8, new TensorRtDims(new[] { 4 }));
        using TensorRtLayer scaleLayer = network.AddConstant(new TensorRtDims(Array.Empty<int>()), TensorRtWeights.FromSingleArray(new[] { 0.5f }));
        using TensorRtTensor scale = scaleLayer.GetOutput(0);
        using TensorRtLayer dequantize = network.AddDequantize(quantized, scale);
        dequantize.Name = "dequantize_to_type_probe";
        dequantize.SetDequantizeToType(TensorRtDataType.Half);

        using TensorRtLayer fill = network.AddFill(new TensorRtDims(new[] { 2, 2 }), TensorRtFillOperation.Linspace);
        fill.Name = "fill_to_type_probe";
        fill.SetFillToType(TensorRtDataType.Int32);
        return $"DequantizeLayer={dequantize.Type} ToType={dequantize.GetDequantizeToType()} FillLayer={fill.Type} FillToType={fill.GetFillToType()} DequantizeSlots=[{DescribeLayerTensors(dequantize)}] FillSlots=[{DescribeLayerTensors(fill)}]";
    }

    static string ProbeRotaryEmbedding(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor input = network.AddInput("rope_input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 2, 4 }));
        using TensorRtLayer cosLayer = network.AddConstant(new TensorRtDims(new[] { 1, 2, 2 }), TensorRtWeights.FromSingleArray(new[]
        {
            1.0f, 0.0f,
            0.5f, 0.5f
        }));
        using TensorRtTensor cosCache = cosLayer.GetOutput(0);
        using TensorRtLayer sinLayer = network.AddConstant(new TensorRtDims(new[] { 1, 2, 2 }), TensorRtWeights.FromSingleArray(new[]
        {
            0.0f, 1.0f,
            0.5f, 0.5f
        }));
        using TensorRtTensor sinCache = sinLayer.GetOutput(0);
        using TensorRtLayer rotary = network.AddRotaryEmbedding(input, cosCache, sinCache, interleaved: true, rotaryEmbeddingDimension: 4);
        rotary.Name = "rotary_embedding_probe";
        rotary.SetRotaryEmbeddingInterleaved(false);
        bool dimensionAccepted = rotary.SetRotaryEmbeddingDimension(4);
        return $"Layer={rotary.Type} Outputs={rotary.OutputCount} Interleaved={rotary.GetRotaryEmbeddingInterleaved()} RotaryDim={rotary.GetRotaryEmbeddingDimension()} RotaryDimAccepted={dimensionAccepted} TensorSlots=[{DescribeLayerTensors(rotary)}]";
    }

    static string ProbeKvCacheUpdate(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor cache = network.AddInput("kv_cache", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 4, 8 }));
        using TensorRtTensor update = network.AddInput("kv_update", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 1, 8 }));
        using TensorRtLayer indicesLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromInt32Array(new[] { 0 }));
        using TensorRtTensor writeIndices = indicesLayer.GetOutput(0);
        using TensorRtLayer kvCacheUpdate = network.AddKvCacheUpdate(cache, update, writeIndices);
        kvCacheUpdate.Name = "kv_cache_update_probe";
        bool modeAccepted = kvCacheUpdate.SetKvCacheUpdateMode(TensorRtKvCacheMode.Linear);
        bool formAccepted = kvCacheUpdate.SetKvCacheUpdateForm(TensorRtAttentionIoForm.PackedNhd);
        using TensorRtLayer lengthsLayer = network.AddConstant(new TensorRtDims(new[] { 2 }), TensorRtWeights.FromInt32Array(new[] { 0, 1 }));
        using TensorRtTensor lengths = lengthsLayer.GetOutput(0);
        bool lengthsAccepted = kvCacheUpdate.SetKvCacheUpdateLengths(lengths);
        bool hasLengths = kvCacheUpdate.TryGetKvCacheUpdateLengths(out TensorRtTensor? updateLengths);
        updateLengths?.Dispose();
        return $"Layer={kvCacheUpdate.Type} Outputs={kvCacheUpdate.OutputCount} Mode={kvCacheUpdate.GetKvCacheUpdateMode()} ModeAccepted={modeAccepted} Form={kvCacheUpdate.GetKvCacheUpdateForm()} FormAccepted={formAccepted} LengthsAccepted={lengthsAccepted} HasLengths={hasLengths} TensorSlots=[{DescribeLayerTensors(kvCacheUpdate)}]";
    }

    static string ProbeAttentionV2(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor query = network.AddInput("attention_query", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 2, 4 }));
        using TensorRtTensor key = network.AddInput("attention_key", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 2, 4 }));
        using TensorRtTensor value = network.AddInput("attention_value", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 2, 4 }));
        using TensorRtAttention attention = network.AddAttentionV2(query, key, value, TensorRtAttentionNormalizationOperation.Softmax, TensorRtCausalMaskKind.None);
        attention.Name = "attention_v2_probe";
        attention.Metadata = "{\"probe\":\"attention-v2\"}";
        bool normalizationAccepted = attention.SetNormalizationOperation(TensorRtAttentionNormalizationOperation.Softmax);
        bool causalAccepted = attention.SetCausalKind(TensorRtCausalMaskKind.None);
        bool decomposableAccepted = attention.SetDecomposable(true);
        bool rankAccepted = attention.SetRankCount(1);
        bool queryFormAccepted = attention.SetQueryForm(TensorRtAttentionIoForm.PaddedBhnd);
        bool keyValueFormAccepted = attention.SetKeyValueForm(TensorRtAttentionIoForm.PaddedBhnd);

        using TensorRtLayer maskLayer = network.AddConstant(new TensorRtDims(new[] { 1, 1, 2, 2 }), TensorRtWeights.FromBooleanArray(new[] { true, true, true, true }));
        using TensorRtTensor mask = maskLayer.GetOutput(0);
        bool maskAccepted = attention.SetMask(mask);
        bool hasMask = attention.TryGetMask(out TensorRtTensor? maskFromAttention);
        maskFromAttention?.Dispose();

        using TensorRtLayer kvLengthsLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromInt32Array(new[] { 2 }));
        using TensorRtTensor kvLengths = kvLengthsLayer.GetOutput(0);
        bool keyValueLengthsAccepted = attention.SetKeyValueLengths(kvLengths);
        bool hasKeyValueLengths = attention.TryGetKeyValueLengths(out TensorRtTensor? keyValueLengthsFromAttention);
        keyValueLengthsFromAttention?.Dispose();

        using TensorRtLayer quantizeScaleLayer = network.AddConstant(new TensorRtDims(Array.Empty<int>()), TensorRtWeights.FromSingleArray(new[] { 0.5f }));
        using TensorRtTensor quantizeScale = quantizeScaleLayer.GetOutput(0);
        bool quantizeScaleAccepted = attention.SetNormalizationQuantizeScale(quantizeScale);
        bool quantizeTypeAccepted = attention.SetNormalizationQuantizeToType(TensorRtDataType.Float8);
        bool hasQuantizeScale = attention.TryGetNormalizationQuantizeScale(out TensorRtTensor? quantizeScaleFromAttention);
        quantizeScaleFromAttention?.Dispose();

        using TensorRtTensor input0 = attention.GetInput(0);
        using TensorRtTensor output0 = attention.GetOutput(0);
        bool queryLengthsInitiallySet = attention.TryGetQueryLengths(out TensorRtTensor? queryLengths);
        queryLengths?.Dispose();

        TensorRtDims64 outputShape64 = output0.Shape64;
        return $"Attention={attention.Name} Metadata={attention.Metadata} Inputs={attention.InputCount} Outputs={attention.OutputCount} Input0={input0.Name} Output0={output0.Name}:Shape64={outputShape64}:Ext64={FormatDims64Extents(outputShape64)} Norm={attention.GetNormalizationOperation()} NormAccepted={normalizationAccepted} Causal={attention.GetCausalKind()} CausalAccepted={causalAccepted} Decomposable={attention.GetDecomposable()} DecomposableAccepted={decomposableAccepted} Rank={attention.GetRankCount()} RankAccepted={rankAccepted} QueryForm={attention.GetQueryForm()} QueryFormAccepted={queryFormAccepted} KeyValueForm={attention.GetKeyValueForm()} KeyValueFormAccepted={keyValueFormAccepted} MaskAccepted={maskAccepted} HasMask={hasMask} KeyValueLengthsAccepted={keyValueLengthsAccepted} HasKeyValueLengths={hasKeyValueLengths} QuantizeScaleAccepted={quantizeScaleAccepted} QuantizeTypeAccepted={quantizeTypeAccepted} QuantizeToType={attention.GetNormalizationQuantizeToType()} HasQuantizeScale={hasQuantizeScale} QueryLengthsInitiallySet={queryLengthsInitiallySet}";
    }

    static string ProbeSafeNetworkV2(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor quantized = network.AddInput("v2_dequantize_input", TensorRtDataType.Int8, new TensorRtDims(new[] { 4 }));
        using TensorRtLayer scaleLayer = network.AddConstant(new TensorRtDims(Array.Empty<int>()), TensorRtWeights.FromSingleArray(new[] { 0.5f }));
        bool weightsNameSet = network.SetWeightsName(scaleLayer, "safe_network_v2_scale_weights");
        using TensorRtTensor scale = scaleLayer.GetOutput(0);
        using TensorRtLayer dequantize = network.AddDequantizeV2(quantized, scale, TensorRtDataType.Half);
        dequantize.Name = "dequantize_v2_probe";

        using TensorRtLayer fill = network.AddFillV2(new TensorRtDims(new[] { 2, 2 }), TensorRtFillOperation.Linspace, TensorRtDataType.Int32);
        fill.Name = "fill_v2_probe";

        using TensorRtTensor gatherData = network.AddInput("gather_v2_data", TensorRtDataType.Float, new TensorRtDims(new[] { 3 }));
        using TensorRtLayer gatherIndicesLayer = network.AddConstant(new TensorRtDims(new[] { 2 }), TensorRtWeights.FromInt32Array(new[] { 0, 2 }));
        using TensorRtTensor gatherIndices = gatherIndicesLayer.GetOutput(0);
        using TensorRtLayer gather = network.AddGatherV2(gatherData, gatherIndices, TensorRtGatherMode.Default);
        gather.Name = "gather_v2_probe";

        using TensorRtTensor preluInput = network.AddInput("prelu_input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 2, 2 }));
        using TensorRtLayer slopeLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromSingleArray(new[] { 0.25f }));
        using TensorRtTensor slope = slopeLayer.GetOutput(0);
        using TensorRtLayer prelu = network.AddParametricReLU(preluInput, slope);
        prelu.Name = "parametric_relu_probe";
        using TensorRtTensor dequantizeOutput = dequantize.GetOutput(0);
        TensorRtDims64 shape64 = dequantizeOutput.Shape64;

        using TensorRtTensor query = network.AddInput("boundary_query", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 2, 4 }));
        using TensorRtTensor key = network.AddInput("boundary_key", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 2, 4 }));
        using TensorRtTensor value = network.AddInput("boundary_value", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 2, 4 }));
        using TensorRtAttention attention = network.AddAttentionV2(query, key, value, TensorRtAttentionNormalizationOperation.Softmax, TensorRtCausalMaskKind.None);
        attention.Name = "boundary_attention_probe";
        string boundaryAttentionName = "<not-found>";
        for (int layerIndex = 0; layerIndex < network.LayerCount; ++layerIndex)
        {
            using TensorRtLayer layer = network.GetLayer(layerIndex);
            if (layer.Type == TensorRtLayerType.AttentionInput || layer.Type == TensorRtLayerType.AttentionOutput)
            {
                using TensorRtAttention boundaryAttention = layer.GetAttentionFromBoundary();
                boundaryAttentionName = boundaryAttention.Name;
                break;
            }
        }

        return $"DequantizeV2={dequantize.Type}:{dequantize.GetDequantizeToType()}:Shape64={shape64}:Ext64={FormatDims64Extents(shape64)} WeightsNameSet={weightsNameSet} FillV2={fill.Type}:{fill.GetFillToType()} GatherV2={gather.Type}:{gather.GetGatherMode()} ParametricReLU={prelu.Type} BoundaryAttention={boundaryAttentionName} TensorSlots=[{DescribeLayerTensors(prelu)}]";
    }

    static string ProbeDistCollective(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor input = network.AddInput("dist_collective_input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 4 }));
        using TensorRtLayer collective = network.AddDistCollective(
            input,
            TensorRtCollectiveOperation.AllReduce,
            TensorRtDistributedReduceOperation.Sum,
            root: -1,
            groups: null);
        collective.Name = "dist_collective_probe";
        using TensorRtTensor output = collective.GetOutput(0);
        TensorRtDims64 shape64 = output.Shape64;
        return $"Layer={collective.Type} Outputs={collective.OutputCount} Shape64={shape64}:Ext64={FormatDims64Extents(shape64)} TensorSlots=[{DescribeLayerTensors(collective)}]";
    }

    static string ProbeMoE(TensorRtNetworkDefinition network)
    {
        using TensorRtTensor hiddenStates = network.AddInput("moe_hidden_states", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 64 }));
        using TensorRtTensor selectedExperts = network.AddInput("moe_selected_experts", TensorRtDataType.Int32, new TensorRtDims(new[] { 1, 1, 1 }));
        using TensorRtTensor scores = network.AddInput("moe_scores", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 1 }));
        using TensorRtLayer moe = network.AddMoE(hiddenStates, selectedExperts, scores);
        moe.Name = "moe_probe";
        moe.SetMoEActivationType(TensorRtMoEActivationType.SiLU);
        moe.SetMoEQuantizationToType(TensorRtDataType.Float8);
        moe.SetMoEQuantizationBlockShape(new TensorRtDims(new[] { 1, 1, 1, 16 }));
        moe.SetMoEDynamicQuantizationOutputScaleType(TensorRtDataType.Float);
        moe.SetMoESwigluParameters(7.5f, 1.25f, 0.5f);
        moe.SetMoESwigluLimit(8.0f);
        moe.SetMoESwigluAlpha(1.5f);
        moe.SetMoESwigluBeta(0.25f);
        return $"Layer={moe.Type} Outputs={moe.OutputCount} Activation={moe.GetMoEActivationType()} QuantType={moe.GetMoEQuantizationToType()} BlockShape={moe.GetMoEQuantizationBlockShape()} DynQScaleType={moe.GetMoEDynamicQuantizationOutputScaleType()} Swiglu={moe.GetMoESwigluLimit()}/{moe.GetMoESwigluAlpha()}/{moe.GetMoESwigluBeta()} TensorSlots=[{DescribeLayerTensors(moe)}]";
    }

    static string DescribeLayerTensors(TensorRtLayer layer)
    {
        List<string> parts = new List<string>();
        for (int index = 0; index < layer.InputCount; ++index)
        {
            TensorRtLayerTensorMetadata metadata = layer.GetInputTensorMetadata(index);
            parts.Add($"I{index}:{metadata.Summary}:Shape64={metadata.Shape64}:Ext64={FormatExtents64(metadata)}");
        }

        for (int index = 0; index < layer.OutputCount; ++index)
        {
            TensorRtLayerTensorMetadata metadata = layer.GetOutputTensorMetadata(index);
            parts.Add($"O{index}:{metadata.Summary}:Shape64={metadata.Shape64}:Ext64={FormatExtents64(metadata)}");
        }

        return string.Join(";", parts);
    }

    static string FormatExtents64(TensorRtLayerTensorMetadata metadata)
    {
        return metadata.DimensionExtents64.Count == 0
            ? "<none>"
            : string.Join("/", metadata.DimensionExtents64);
    }

    static string FormatDims64Extents(TensorRtDims64 dims)
    {
        return dims.IsUnknownRank || dims.Values.Length == 0
            ? "<none>"
            : string.Join("/", dims.Values);
    }
}
