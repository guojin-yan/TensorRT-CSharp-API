using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public static partial class TrtexecLikeParser
{
    private static ulong ParseWorkspaceBytes(string value)
    {
        return ParseMemorySizeBytes(value, "--workspace");
    }

    private static ulong? ParseOptionalMemorySizeBytes(string value, string argumentName)
    {
        return string.IsNullOrWhiteSpace(value) ? null : ParseMemorySizeBytes(value, argumentName);
    }

    private static long? ParseOptionalLongMemorySizeBytes(string value, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        ulong bytes = ParseMemorySizeBytes(value, argumentName);
        if (bytes > long.MaxValue)
        {
            throw new ArgumentException($"{argumentName} must not exceed {long.MaxValue} bytes.");
        }

        return (long)bytes;
    }

    private static TensorRtTilingOptimizationLevel? ParseOptionalTilingOptimizationLevel(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        string normalized = value.Trim().Replace("-", string.Empty).Replace("_", string.Empty).ToLowerInvariant();
        return normalized switch
        {
            "0" or "none" or "off" => TensorRtTilingOptimizationLevel.None,
            "1" or "fast" => TensorRtTilingOptimizationLevel.Fast,
            "2" or "moderate" => TensorRtTilingOptimizationLevel.Moderate,
            "3" or "full" => TensorRtTilingOptimizationLevel.Full,
            _ => throw new ArgumentException("--tilingOptimizationLevel must be none, fast, moderate, full, or 0..3.")
        };
    }

    private static TensorRtQuantizationFlags? ParseOptionalQuantizationFlags(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        TensorRtQuantizationFlags flags = TensorRtQuantizationFlags.None;
        foreach (string token in value.Split(new[] { ',', '|', ';' }, StringSplitOptions.RemoveEmptyEntries))
        {
            string normalized = token.Trim().Replace("-", string.Empty).Replace("_", string.Empty).ToLowerInvariant();
            flags |= normalized switch
            {
                "none" or "0" => TensorRtQuantizationFlags.None,
                "calibratebeforefusion" => TensorRtQuantizationFlags.CalibrateBeforeFusion,
                _ => throw new ArgumentException("--quantizationFlags supports none or calibrateBeforeFusion.")
            };
        }

        return flags;
    }

    private static IReadOnlyList<TrtexecLikeMemoryPoolSize> ParseMemoryPoolSizes(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return Array.Empty<TrtexecLikeMemoryPoolSize>();
        }

        List<TrtexecLikeMemoryPoolSize> sizes = new List<TrtexecLikeMemoryPoolSize>();
        foreach (string item in value.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
        {
            string trimmed = item.Trim();
            int separator = trimmed.IndexOf(':');
            if (separator <= 0 || separator == trimmed.Length - 1)
            {
                throw new ArgumentException("--memPoolSize entries must use poolName:sizeMiB syntax.");
            }

            string name = trimmed.Substring(0, separator);
            string sizeText = trimmed.Substring(separator + 1);
            ulong sizeMiB = ParseMemorySizeMiB(sizeText, "--memPoolSize");
            sizes.Add(new TrtexecLikeMemoryPoolSize(name, sizeMiB));
        }

        return sizes;
    }

    private static string NormalizeProfilingVerbosity(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return "layer_names_only";
        }

        string normalized = value.Trim().Replace("-", "_").ToLowerInvariant();
        return normalized switch
        {
            "none" => "none",
            "layer_names_only" or "layernamesonly" or "layer_names" or "names" => "layer_names_only",
            "detailed" or "detail" or "verbose" => "detailed",
            _ => throw new ArgumentException("--profilingVerbosity must be none, layer_names_only, or detailed.")
        };
    }

    private static string NormalizeTacticSources(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        List<string> normalizedSources = new List<string>();
        foreach (string item in value.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
        {
            string trimmed = item.Trim();
            if (trimmed.Length < 2 || (trimmed[0] != '+' && trimmed[0] != '-'))
            {
                throw new ArgumentException("--tacticSources entries must begin with + or -.");
            }

            string source = trimmed.Substring(1)
                .Replace("-", string.Empty, StringComparison.Ordinal)
                .Replace("_", string.Empty, StringComparison.Ordinal)
                .ToLowerInvariant();
            string canonicalSource = source switch
            {
                "cublas" => "CUBLAS",
                "cublaslt" => "CUBLAS_LT",
                "cudnn" => "CUDNN",
                "edgemaskconvolutions" or "edgemask" => "EDGE_MASK_CONVOLUTIONS",
                "jitconvolutions" or "jit" => "JIT_CONVOLUTIONS",
                _ => throw new ArgumentException($"Unsupported --tacticSources entry '{trimmed}'.")
            };
            normalizedSources.Add(trimmed[0] + canonicalSource);
        }

        return string.Join(",", normalizedSources);
    }

    private static string NormalizeSparsity(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        return value.Trim().ToLowerInvariant() switch
        {
            "disable" or "disabled" => "disable",
            "enable" or "enabled" => "enable",
            "force" => "force",
            _ => throw new ArgumentException("--sparsity must be disable, enable, or force.")
        };
    }
}
