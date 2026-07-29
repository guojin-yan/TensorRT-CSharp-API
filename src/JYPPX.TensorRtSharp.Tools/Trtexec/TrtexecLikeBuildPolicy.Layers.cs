using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

internal static partial class TrtexecLikeBuildPolicy
{
    private static void ApplyLayerPolicies(
        TensorRtNetworkDefinition network,
        string layerPrecisions,
        string layerOutputTypes,
        List<string> log)
    {
        bool hasPrecisionRules = !string.IsNullOrWhiteSpace(layerPrecisions);
        bool hasOutputRules = !string.IsNullOrWhiteSpace(layerOutputTypes);
        if (!hasPrecisionRules && !hasOutputRules)
        {
            return;
        }

        if (network.Line == TensorRtApiLine.TensorRt11)
        {
            if (hasPrecisionRules)
            {
                log.Add($"TrtexecBuildPolicy Name=LayerPrecisions Applied=False Requested={layerPrecisions} VersionGuard=TRT11 Reason=layer-set-precision-removed ReadbackMatch=False");
            }

            if (hasOutputRules)
            {
                log.Add($"TrtexecBuildPolicy Name=LayerOutputTypes Applied=False Requested={layerOutputTypes} VersionGuard=TRT11 Reason=layer-set-output-type-removed ReadbackMatch=False");
            }

            return;
        }

        IReadOnlyList<TrtexecLikeLayerTypeRule> precisionRules = hasPrecisionRules
            ? ParseLayerRules(layerPrecisions, "--layerPrecisions", allowMultipleTypes: false)
            : Array.Empty<TrtexecLikeLayerTypeRule>();
        IReadOnlyList<TrtexecLikeLayerTypeRule> outputRules = hasOutputRules
            ? ParseLayerRules(layerOutputTypes, "--layerOutputTypes", allowMultipleTypes: true)
            : Array.Empty<TrtexecLikeLayerTypeRule>();
        HashSet<int> matchedPrecisionRules = new HashSet<int>();
        HashSet<int> matchedOutputRules = new HashSet<int>();
        int matchedPrecisionLayers = 0;
        int matchedOutputLayers = 0;
        bool precisionReadbackMatch = true;
        bool outputReadbackMatch = true;

        if (hasPrecisionRules && precisionRules.SelectMany(static rule => rule.DataTypes).Any(type => !IsDataTypeSupported(network.Line, type)))
        {
            log.Add($"TrtexecBuildPolicy Name=LayerPrecisions Applied=False Requested={layerPrecisions} VersionGuard={network.Line} Reason=data-type-not-supported-on-api-line ReadbackMatch=False");
            hasPrecisionRules = false;
            precisionRules = Array.Empty<TrtexecLikeLayerTypeRule>();
        }

        if (hasOutputRules && outputRules.SelectMany(static rule => rule.DataTypes).Any(type => !IsDataTypeSupported(network.Line, type)))
        {
            log.Add($"TrtexecBuildPolicy Name=LayerOutputTypes Applied=False Requested={layerOutputTypes} VersionGuard={network.Line} Reason=data-type-not-supported-on-api-line ReadbackMatch=False");
            hasOutputRules = false;
            outputRules = Array.Empty<TrtexecLikeLayerTypeRule>();
        }

        // Resolve and validate the complete rule set before mutating any layer.
        for (int layerIndex = 0; layerIndex < network.LayerCount; layerIndex++)
        {
            using TensorRtLayer layer = network.GetLayer(layerIndex);
            string layerName = layer.Name;
            TrackMatchingRules(layerName, precisionRules, matchedPrecisionRules);
            TrackMatchingRules(layerName, outputRules, matchedOutputRules);

            TrtexecLikeLayerTypeRule? outputRule = ResolveLayerRule(layerName, outputRules);
            if (outputRule != null)
            {
                IReadOnlyList<TensorRtDataType> requestedTypes = outputRule.DataTypes;
                if (requestedTypes.Count != 1 && requestedTypes.Count != layer.OutputCount)
                {
                    throw new InvalidOperationException($"--layerOutputTypes rule '{outputRule}' requires one broadcast type or exactly {layer.OutputCount} output types for layer '{layerName}'.");
                }
            }
        }

        EnsureEveryRuleMatched("--layerPrecisions", precisionRules, matchedPrecisionRules);
        EnsureEveryRuleMatched("--layerOutputTypes", outputRules, matchedOutputRules);

        for (int layerIndex = 0; layerIndex < network.LayerCount; layerIndex++)
        {
            using TensorRtLayer layer = network.GetLayer(layerIndex);
            string layerName = layer.Name;
            TrtexecLikeLayerTypeRule? precisionRule = ResolveLayerRule(layerName, precisionRules);
            if (precisionRule != null)
            {
                TensorRtDataType requested = precisionRule.DataTypes[0];
                layer.Precision = requested;
                precisionReadbackMatch &= layer.IsPrecisionSet && layer.Precision == requested;
                matchedPrecisionLayers++;
            }

            TrtexecLikeLayerTypeRule? outputRule = ResolveLayerRule(layerName, outputRules);
            if (outputRule != null)
            {
                IReadOnlyList<TensorRtDataType> requestedTypes = outputRule.DataTypes;
                for (int outputIndex = 0; outputIndex < layer.OutputCount; outputIndex++)
                {
                    TensorRtDataType requested = requestedTypes.Count == 1 ? requestedTypes[0] : requestedTypes[outputIndex];
                    layer.SetOutputType(outputIndex, requested);
                    outputReadbackMatch &= layer.IsOutputTypeSet(outputIndex) && layer.GetOutputType(outputIndex) == requested;
                }

                matchedOutputLayers++;
            }
        }
        if (hasPrecisionRules)
        {
            log.Add($"TrtexecBuildPolicy Name=LayerPrecisions Applied={precisionReadbackMatch} Requested={layerPrecisions} Matched={matchedPrecisionLayers} Rules={precisionRules.Count} ReadbackMatch={precisionReadbackMatch}");
        }

        if (hasOutputRules)
        {
            log.Add($"TrtexecBuildPolicy Name=LayerOutputTypes Applied={outputReadbackMatch} Requested={layerOutputTypes} Matched={matchedOutputLayers} Rules={outputRules.Count} ReadbackMatch={outputReadbackMatch}");
        }

        if (!precisionReadbackMatch || !outputReadbackMatch)
        {
            throw new InvalidOperationException("Layer precision policy did not match TensorRT layer readback.");
        }
    }
}
