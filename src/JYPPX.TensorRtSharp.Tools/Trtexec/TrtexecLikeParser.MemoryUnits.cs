using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public static partial class TrtexecLikeParser
{
    private static ulong ParseMemorySizeMiB(string value, string argumentName)
    {
        ulong bytes = ParseMemorySizeBytes(value, argumentName);
        const ulong mib = 1024UL * 1024UL;
        if (bytes % mib != 0)
        {
            throw new ArgumentException($"{argumentName} size must resolve to a whole number of MiB.");
        }

        return bytes / mib;
    }

    private static ulong ParseMemorySizeBytes(string value, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException($"{argumentName} requires a memory size.");
        }

        string trimmed = value.Trim();
        string numberText = trimmed;
        decimal multiplier = 1024m * 1024m;
        if (TryTrimSuffix(trimmed, "gib", out numberText) || TryTrimSuffix(trimmed, "gb", out numberText))
        {
            multiplier = 1024m * 1024m * 1024m;
        }
        else if (TryTrimSuffix(trimmed, "mib", out numberText) || TryTrimSuffix(trimmed, "mb", out numberText))
        {
            multiplier = 1024m * 1024m;
        }
        else if (TryTrimSuffix(trimmed, "kib", out numberText) || TryTrimSuffix(trimmed, "kb", out numberText))
        {
            multiplier = 1024m;
        }
        else if (TryTrimSuffix(trimmed, "b", out numberText))
        {
            multiplier = 1m;
        }

        if (!decimal.TryParse(numberText.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out decimal parsed) || parsed < 0m)
        {
            throw new ArgumentException($"{argumentName} must be a non-negative memory size, defaulting to MiB when no suffix is supplied.");
        }

        decimal bytes = parsed * multiplier;
        if (bytes != decimal.Truncate(bytes) || bytes > ulong.MaxValue)
        {
            throw new ArgumentException($"{argumentName} size must resolve to a whole number of bytes.");
        }

        return checked((ulong)bytes);
    }

    private static bool TryTrimSuffix(string value, string suffix, out string withoutSuffix)
    {
        if (value.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
        {
            withoutSuffix = value.Substring(0, value.Length - suffix.Length);
            return true;
        }

        withoutSuffix = value;
        return false;
    }
}
