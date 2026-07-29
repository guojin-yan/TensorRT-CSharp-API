using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public static partial class TrtexecLikeParser
{
    private static string GetValue(string[] args, string name, string defaultValue)
    {
        for (int index = 0; index < args.Length - 1; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }

            string prefix = name + "=";
            if (args[index].StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return args[index].Substring(prefix.Length);
            }
        }

        if (args.Length > 0)
        {
            string prefix = name + "=";
            string last = args[args.Length - 1];
            if (last.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return last.Substring(prefix.Length);
            }
        }

        return defaultValue;
    }

    private static bool HasSwitch(string[] args, string name)
    {
        return args.Any(argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase) ||
            argument.StartsWith(name + "=", StringComparison.OrdinalIgnoreCase));
    }

    private static int? ParseThreadMode(string[] args)
    {
        for (int index = 0; index < args.Length; index++)
        {
            string argument = args[index];
            if (string.Equals(argument, "--threads", StringComparison.OrdinalIgnoreCase))
            {
                if (index + 1 < args.Length && !IsOptionName(args[index + 1]))
                {
                    return ParsePositiveInt(args[index + 1], "--threads");
                }

                return 1;
            }

            const string prefix = "--threads=";
            if (argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                string value = argument.Substring(prefix.Length);
                return string.IsNullOrWhiteSpace(value) ? 1 : ParsePositiveInt(value, "--threads");
            }
        }

        return null;
    }

    private static string FullPathOrEmpty(string path)
    {
        return string.IsNullOrWhiteSpace(path) ? string.Empty : Path.GetFullPath(path);
    }

    private static bool PathsEqual(string left, string right)
    {
        return !string.IsNullOrWhiteSpace(left) &&
            !string.IsNullOrWhiteSpace(right) &&
            string.Equals(Path.GetFullPath(left), Path.GetFullPath(right), StringComparison.OrdinalIgnoreCase);
    }

    private static IReadOnlyList<string> ParseList(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return Array.Empty<string>();
        }

        return value.Split(new[] { ';', ',' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(static item => item.Trim())
            .Where(static item => item.Length > 0)
            .ToArray();
    }

    private static IReadOnlyList<string> ParsePluginLibraries(string[] args)
    {
        List<string> values = new List<string>();
        AddValues(args, values, "--plugins");
        AddValues(args, values, "--plugin");
        AddValues(args, values, "--dynamicPlugins");
        AddValues(args, values, "--setPluginsToSerialize");

        if (values.Count == 0)
        {
            return Array.Empty<string>();
        }

        return values
            .SelectMany(static value => ParseList(value))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static void AddValues(string[] args, List<string> values, string name)
    {
        for (int index = 0; index < args.Length; index++)
        {
            string argument = args[index];
            if (string.Equals(argument, name, StringComparison.OrdinalIgnoreCase))
            {
                if (index == args.Length - 1 || IsOptionName(args[index + 1]))
                {
                    throw new ArgumentException($"{name} requires a plugin library path or path list.");
                }

                values.Add(args[index + 1]);
                index++;
                continue;
            }

            string prefix = name + "=";
            if (argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                string value = argument.Substring(prefix.Length);
                if (string.IsNullOrWhiteSpace(value))
                {
                    throw new ArgumentException($"{name} requires a plugin library path or path list.");
                }

                values.Add(value);
            }
        }
    }

    private static bool IsOptionName(string value)
    {
        return !string.IsNullOrWhiteSpace(value) && value.StartsWith("--", StringComparison.Ordinal);
    }

    private static bool ShouldTreatEngineAliasAsLoad(string[] args, string engineAliasPath)
    {
        return !string.IsNullOrWhiteSpace(engineAliasPath) &&
            string.IsNullOrWhiteSpace(GetValue(args, "--onnx", string.Empty)) &&
            string.IsNullOrWhiteSpace(GetValue(args, "--model", string.Empty)) &&
            string.IsNullOrWhiteSpace(GetValue(args, "--onnxFile", string.Empty)) &&
            !HasSwitch(args, "--buildOnly");
    }

    private static string FirstNonEmpty(string first, string second)
    {
        return string.IsNullOrWhiteSpace(first) ? second : first;
    }
}
