using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

internal static partial class TrtexecLikeBuildPolicy
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
}
