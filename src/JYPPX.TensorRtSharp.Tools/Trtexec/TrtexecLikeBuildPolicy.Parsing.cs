using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

internal static partial class TrtexecLikeBuildPolicy
{
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
}
