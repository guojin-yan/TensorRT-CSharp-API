using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

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
