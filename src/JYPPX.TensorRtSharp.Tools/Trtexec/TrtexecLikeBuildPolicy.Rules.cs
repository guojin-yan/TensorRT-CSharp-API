using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

internal static partial class TrtexecLikeBuildPolicy
{
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
}
