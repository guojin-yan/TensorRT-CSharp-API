using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

internal sealed class TrtexecLikeIoFormatSpec
{
    public TrtexecLikeIoFormatSpec(string dataTypeToken, TensorRtDataType dataType, string formatText, TensorRtTensorFormats formats)
    {
        DataTypeToken = dataTypeToken;
        DataType = dataType;
        FormatText = formatText;
        Formats = formats;
    }

    public string DataTypeToken { get; }

    public TensorRtDataType DataType { get; }

    public string FormatText { get; }

    public TensorRtTensorFormats Formats { get; }

    public override string ToString() => DataTypeToken + ":" + FormatText;
}

internal sealed class TrtexecLikeLayerTypeRule
{
    public TrtexecLikeLayerTypeRule(string pattern, IReadOnlyList<string> dataTypeTokens, IReadOnlyList<TensorRtDataType> dataTypes)
    {
        Pattern = pattern;
        DataTypeTokens = dataTypeTokens;
        DataTypes = dataTypes;
    }

    public string Pattern { get; }

    public IReadOnlyList<string> DataTypeTokens { get; }

    public IReadOnlyList<TensorRtDataType> DataTypes { get; }

    public bool HasWildcard => Pattern.IndexOf('*') >= 0;

    public bool Matches(string layerName)
    {
        int wildcard = Pattern.IndexOf('*');
        if (wildcard < 0)
        {
            return string.Equals(Pattern, layerName, StringComparison.Ordinal);
        }

        string prefix = Pattern.Substring(0, wildcard);
        string suffix = Pattern.Substring(wildcard + 1);
        return layerName.StartsWith(prefix, StringComparison.Ordinal) &&
            layerName.EndsWith(suffix, StringComparison.Ordinal) &&
            layerName.Length >= prefix.Length + suffix.Length;
    }

    public override string ToString() => Pattern + ":" + string.Join("+", DataTypeTokens);
}

internal static class TrtexecLikeBuildPolicy
{
    public static string NormalizeIoFormats(string value, string optionName)
    {
        return string.IsNullOrWhiteSpace(value)
            ? string.Empty
            : string.Join(",", ParseIoFormats(value, optionName));
    }

    public static string NormalizePrecisionConstraints(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        return value.Trim().ToLowerInvariant() switch
        {
            "none" => "none",
            "prefer" => "prefer",
            "obey" => "obey",
            _ => throw new ArgumentException("--precisionConstraints must be none, prefer, or obey.")
        };
    }

    public static string NormalizeLayerPrecisions(string value)
    {
        return string.IsNullOrWhiteSpace(value)
            ? string.Empty
            : string.Join(",", ParseLayerRules(value, "--layerPrecisions", allowMultipleTypes: false));
    }

    public static string NormalizeLayerOutputTypes(string value)
    {
        return string.IsNullOrWhiteSpace(value)
            ? string.Empty
            : string.Join(",", ParseLayerRules(value, "--layerOutputTypes", allowMultipleTypes: true));
    }

    public static void ValidatePolicyCombination(
        string precisionConstraints,
        string layerPrecisions,
        string layerOutputTypes)
    {
        bool hasLayerPolicy = !string.IsNullOrWhiteSpace(layerPrecisions) || !string.IsNullOrWhiteSpace(layerOutputTypes);
        if (hasLayerPolicy &&
            !string.Equals(precisionConstraints, "prefer", StringComparison.Ordinal) &&
            !string.Equals(precisionConstraints, "obey", StringComparison.Ordinal))
        {
            throw new ArgumentException("--layerPrecisions and --layerOutputTypes require --precisionConstraints=prefer or obey.");
        }
    }

    public static void Apply(
        TensorRtBuilderConfig config,
        TensorRtNetworkDefinition network,
        TrtexecLikeDeploymentOptions options,
        List<string> log)
    {
        if (config == null)
        {
            throw new ArgumentNullException(nameof(config));
        }

        if (network == null)
        {
            throw new ArgumentNullException(nameof(network));
        }

        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (log == null)
        {
            throw new ArgumentNullException(nameof(log));
        }

        ApplyIoFormats(network, options.InputIOFormats, isInput: true, log);
        ApplyIoFormats(network, options.OutputIOFormats, isInput: false, log);
        ApplyPrecisionConstraints(config, options.PrecisionConstraints, log);
        ApplyLayerPolicies(network, options.LayerPrecisions, options.LayerOutputTypes, log);
    }

    internal static IReadOnlyList<TrtexecLikeIoFormatSpec> ParseIoFormats(string value, string optionName)
    {
        List<TrtexecLikeIoFormatSpec> specs = new List<TrtexecLikeIoFormatSpec>();
        foreach (string item in value.Split(','))
        {
            string trimmed = item.Trim();
            int separator = trimmed.IndexOf(':');
            if (separator <= 0 || separator == trimmed.Length - 1 || trimmed.IndexOf(':', separator + 1) >= 0)
            {
                throw new ArgumentException(optionName + " entries must use type:format[+format] syntax.");
            }

            (string dataTypeToken, TensorRtDataType dataType) = ParseDataType(trimmed.Substring(0, separator), optionName);
            string[] rawFormats = trimmed.Substring(separator + 1).Split('+');
            if (rawFormats.Length == 0 || rawFormats.Any(string.IsNullOrWhiteSpace))
            {
                throw new ArgumentException(optionName + " requires one or more non-empty tensor formats.");
            }

            List<string> formatTokens = new List<string>();
            TensorRtTensorFormats formats = TensorRtTensorFormats.None;
            foreach (string rawFormat in rawFormats)
            {
                (string formatToken, TensorRtTensorFormats format) = ParseTensorFormat(rawFormat, optionName);
                if (!formatTokens.Contains(formatToken, StringComparer.Ordinal))
                {
                    formatTokens.Add(formatToken);
                    formats |= format;
                }
            }

            specs.Add(new TrtexecLikeIoFormatSpec(dataTypeToken, dataType, string.Join("+", formatTokens), formats));
        }

        if (specs.Count == 0)
        {
            throw new ArgumentException(optionName + " requires at least one type:format specification.");
        }

        return specs;
    }

    internal static IReadOnlyList<TrtexecLikeLayerTypeRule> ParseLayerRules(
        string value,
        string optionName,
        bool allowMultipleTypes)
    {
        List<TrtexecLikeLayerTypeRule> rules = new List<TrtexecLikeLayerTypeRule>();
        foreach (string item in value.Split(','))
        {
            string trimmed = item.Trim();
            int separator = trimmed.LastIndexOf(':');
            if (separator <= 0 || separator == trimmed.Length - 1)
            {
                throw new ArgumentException(optionName + " entries must use layerPattern:type[+type] syntax.");
            }

            string pattern = trimmed.Substring(0, separator).Trim();
            if (pattern.Length == 0 || pattern.Count(static character => character == '*') > 1)
            {
                throw new ArgumentException(optionName + " layer patterns must be non-empty and contain at most one '*'.");
            }

            string[] rawTypes = trimmed.Substring(separator + 1).Split('+');
            if (rawTypes.Length == 0 || rawTypes.Any(string.IsNullOrWhiteSpace))
            {
                throw new ArgumentException(optionName + " requires one or more non-empty data types.");
            }

            if (!allowMultipleTypes && rawTypes.Length != 1)
            {
                throw new ArgumentException(optionName + " accepts exactly one data type per layer pattern.");
            }

            List<string> dataTypeTokens = new List<string>();
            List<TensorRtDataType> dataTypes = new List<TensorRtDataType>();
            foreach (string rawType in rawTypes)
            {
                (string token, TensorRtDataType dataType) = ParseDataType(rawType, optionName);
                dataTypeTokens.Add(token);
                dataTypes.Add(dataType);
            }

            rules.Add(new TrtexecLikeLayerTypeRule(pattern, dataTypeTokens, dataTypes));
        }

        if (rules.Count == 0)
        {
            throw new ArgumentException(optionName + " requires at least one layer rule.");
        }

        return rules;
    }

    internal static TrtexecLikeLayerTypeRule? ResolveLayerRule(
        string layerName,
        IReadOnlyList<TrtexecLikeLayerTypeRule> rules)
    {
        TrtexecLikeLayerTypeRule? exact = null;
        TrtexecLikeLayerTypeRule? wildcard = null;
        foreach (TrtexecLikeLayerTypeRule rule in rules)
        {
            if (!rule.Matches(layerName))
            {
                continue;
            }

            if (rule.HasWildcard)
            {
                wildcard = rule;
            }
            else
            {
                exact = rule;
            }
        }

        return exact ?? wildcard;
    }

    private static void ApplyIoFormats(
        TensorRtNetworkDefinition network,
        string specification,
        bool isInput,
        List<string> log)
    {
        if (string.IsNullOrWhiteSpace(specification))
        {
            return;
        }

        string policyName = isInput ? "InputIOFormats" : "OutputIOFormats";
        string optionName = isInput ? "--inputIOFormats" : "--outputIOFormats";
        IReadOnlyList<TrtexecLikeIoFormatSpec> specs = ParseIoFormats(specification, optionName);
        int tensorCount = isInput ? network.InputCount : network.OutputCount;
        if (tensorCount == 0)
        {
            throw new InvalidOperationException(optionName + " cannot be applied because the parsed network has no matching I/O tensors.");
        }

        if (specs.Count != 1 && specs.Count != tensorCount)
        {
            throw new InvalidOperationException($"{optionName} requires one broadcast specification or exactly {tensorCount} specifications.");
        }

        List<TensorRtTensor> tensors = new List<TensorRtTensor>(tensorCount);
        try
        {
            for (int index = 0; index < tensorCount; index++)
            {
                tensors.Add(isInput ? network.GetInput(index) : network.GetOutput(index));
            }

            for (int index = 0; index < tensors.Count; index++)
            {
                TensorRtDataType requestedType = specs.Count == 1 ? specs[0].DataType : specs[index].DataType;
                if (!IsDataTypeSupported(network.Line, requestedType))
                {
                    log.Add($"TrtexecBuildPolicy Name={policyName} Applied=False Requested={specification} TensorIndex={index} VersionGuard={network.Line} Reason=data-type-not-supported-on-api-line ReadbackMatch=False");
                    return;
                }

                if (network.Line == TensorRtApiLine.TensorRt11 && tensors[index].DataType != requestedType)
                {
                    log.Add($"TrtexecBuildPolicy Name={policyName} Applied=False Requested={specification} TensorIndex={index} RequestedType={requestedType} ReadbackType={tensors[index].DataType} VersionGuard=TRT11 Reason=tensor-set-type-removed ReadbackMatch=False");
                    return;
                }
            }

            bool readbackMatch = true;
            for (int index = 0; index < tensors.Count; index++)
            {
                TrtexecLikeIoFormatSpec spec = specs.Count == 1 ? specs[0] : specs[index];
                TensorRtTensor tensor = tensors[index];
                if (network.Line != TensorRtApiLine.TensorRt11)
                {
                    tensor.DataType = spec.DataType;
                }

                tensor.AllowedFormats = spec.Formats;
                readbackMatch &= tensor.DataType == spec.DataType && tensor.AllowedFormats == spec.Formats;
            }

            log.Add($"TrtexecBuildPolicy Name={policyName} Applied={readbackMatch} Requested={specification} TensorCount={tensorCount} Broadcast={specs.Count == 1} TypeMode={(network.Line == TensorRtApiLine.TensorRt11 ? "validated-existing" : "set-and-readback")} ReadbackMatch={readbackMatch}");
            if (!readbackMatch)
            {
                throw new InvalidOperationException(policyName + " did not match TensorRT tensor readback.");
            }
        }
        finally
        {
            foreach (TensorRtTensor tensor in tensors)
            {
                tensor.Dispose();
            }
        }
    }

    private static void ApplyPrecisionConstraints(
        TensorRtBuilderConfig config,
        string precisionConstraints,
        List<string> log)
    {
        if (string.IsNullOrWhiteSpace(precisionConstraints))
        {
            return;
        }

        if (config.Line == TensorRtApiLine.TensorRt11)
        {
            log.Add($"TrtexecBuildPolicy Name=PrecisionConstraints Applied=False Requested={precisionConstraints} VersionGuard=TRT11 Reason=precision-constraint-builder-flags-removed ReadbackMatch=False");
            return;
        }

        bool expectPrefer = string.Equals(precisionConstraints, "prefer", StringComparison.Ordinal);
        bool expectObey = string.Equals(precisionConstraints, "obey", StringComparison.Ordinal);
        config.SetFlag(TensorRtBuilderFlag.PreferPrecisionConstraints, expectPrefer);
        config.SetFlag(TensorRtBuilderFlag.ObeyPrecisionConstraints, expectObey);
        bool readbackPrefer = config.GetFlag(TensorRtBuilderFlag.PreferPrecisionConstraints);
        bool readbackObey = config.GetFlag(TensorRtBuilderFlag.ObeyPrecisionConstraints);
        bool readbackMatch = readbackPrefer == expectPrefer && readbackObey == expectObey;
        log.Add($"TrtexecBuildPolicy Name=PrecisionConstraints Applied={readbackMatch} Requested={precisionConstraints} ReadbackPrefer={readbackPrefer} ReadbackObey={readbackObey} ReadbackMatch={readbackMatch}");
        if (!readbackMatch)
        {
            throw new InvalidOperationException("Precision constraint flags did not match TensorRT builder-config readback.");
        }
    }

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

    private static void TrackMatchingRules(
        string layerName,
        IReadOnlyList<TrtexecLikeLayerTypeRule> rules,
        HashSet<int> matchedRules)
    {
        for (int index = 0; index < rules.Count; index++)
        {
            if (rules[index].Matches(layerName))
            {
                matchedRules.Add(index);
            }
        }
    }

    private static void EnsureEveryRuleMatched(
        string optionName,
        IReadOnlyList<TrtexecLikeLayerTypeRule> rules,
        HashSet<int> matchedRules)
    {
        for (int index = 0; index < rules.Count; index++)
        {
            if (!matchedRules.Contains(index))
            {
                throw new InvalidOperationException($"{optionName} rule '{rules[index]}' did not match any network layer.");
            }
        }
    }

    private static bool IsDataTypeSupported(TensorRtApiLine line, TensorRtDataType dataType)
    {
        return line != TensorRtApiLine.TensorRt8 ||
            (dataType != TensorRtDataType.BFloat16 && dataType != TensorRtDataType.Int64);
    }

    private static (string Token, TensorRtDataType DataType) ParseDataType(string value, string optionName)
    {
        return value.Trim().ToLowerInvariant() switch
        {
            "fp32" => ("fp32", TensorRtDataType.Float),
            "fp16" => ("fp16", TensorRtDataType.Half),
            "bf16" => ("bf16", TensorRtDataType.BFloat16),
            "int32" => ("int32", TensorRtDataType.Int32),
            "int64" => ("int64", TensorRtDataType.Int64),
            "int8" => ("int8", TensorRtDataType.Int8),
            "uint8" => ("uint8", TensorRtDataType.UInt8),
            "bool" => ("bool", TensorRtDataType.Bool),
            _ => throw new ArgumentException(optionName + " contains an unsupported data type. Expected fp32, fp16, bf16, int32, int64, int8, uint8, or bool.")
        };
    }

    private static (string Token, TensorRtTensorFormats Format) ParseTensorFormat(string value, string optionName)
    {
        return value.Trim().ToLowerInvariant() switch
        {
            "chw" => ("chw", TensorRtTensorFormats.Linear),
            "chw2" => ("chw2", TensorRtTensorFormats.Chw2),
            "chw4" => ("chw4", TensorRtTensorFormats.Chw4),
            "hwc8" => ("hwc8", TensorRtTensorFormats.Hwc8),
            "chw16" => ("chw16", TensorRtTensorFormats.Chw16),
            "chw32" => ("chw32", TensorRtTensorFormats.Chw32),
            "dhwc8" => ("dhwc8", TensorRtTensorFormats.Dhwc8),
            "cdhw32" => ("cdhw32", TensorRtTensorFormats.Cdhw32),
            "hwc" => ("hwc", TensorRtTensorFormats.Hwc),
            "dhwc" => ("dhwc", TensorRtTensorFormats.Dhwc),
            "dla_linear" => ("dla_linear", TensorRtTensorFormats.DlaLinear),
            "hwc16" => ("hwc16", TensorRtTensorFormats.Hwc16),
            "dla_hwc4" => ("dla_hwc4", TensorRtTensorFormats.DlaHwc4),
            _ => throw new ArgumentException(optionName + " contains an unsupported tensor format.")
        };
    }
}
