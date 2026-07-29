using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class TrtexecLikeDeploymentOptions
{
    private static void AddDiagnostic(List<string> diagnostics, string name, string value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            diagnostics.Add(name + "=" + value);
        }
    }

    private static void AddDiagnostic(List<string> diagnostics, string name, bool enabled)
    {
        if (enabled)
        {
            diagnostics.Add(name + "=True");
        }
    }

    private static void AddDiagnostic(List<string> diagnostics, string name, int? value)
    {
        if (value.HasValue)
        {
            diagnostics.Add(name + "=" + value.Value.ToString(CultureInfo.InvariantCulture));
        }
    }

    private static void AddDiagnostic(List<string> diagnostics, string name, long? value)
    {
        if (value.HasValue)
        {
            diagnostics.Add(name + "=" + value.Value.ToString(CultureInfo.InvariantCulture));
        }
    }

    private static void Add(List<string> args, string name, string value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            args.Add(name);
            args.Add(QuoteIfNeeded(value));
        }
    }

    private static void AddSwitch(List<string> args, string name, bool enabled)
    {
        if (enabled)
        {
            args.Add(name);
        }
    }

    private static string QuoteIfNeeded(string value)
    {
        return value.IndexOf(' ') >= 0 ? "\"" + value + "\"" : value;
    }

    private static string FormatNullable(int? value)
    {
        return value.HasValue ? value.Value.ToString(CultureInfo.InvariantCulture) : string.Empty;
    }

    private static string FormatBytesMiB(ulong? value)
    {
        if (!value.HasValue)
        {
            return string.Empty;
        }

        const ulong mib = 1024UL * 1024UL;
        return value.Value % mib == 0
            ? (value.Value / mib).ToString(CultureInfo.InvariantCulture)
            : value.Value.ToString(CultureInfo.InvariantCulture) + "B";
    }

    private static string FormatBytes(long? value)
    {
        return value.HasValue
            ? value.Value.ToString(CultureInfo.InvariantCulture) + "B"
            : string.Empty;
    }
}
