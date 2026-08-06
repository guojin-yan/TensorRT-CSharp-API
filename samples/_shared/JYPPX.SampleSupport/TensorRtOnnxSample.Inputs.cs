using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.SampleSupport;

internal sealed class OnnxSampleInputOptions
{
    public OnnxSampleInputOptions(
        string name,
        TensorRtDims shape,
        TensorRtDims minShape,
        TensorRtDims optShape,
        TensorRtDims maxShape,
        bool hasProfileOverride,
        string inputPattern,
        string inputPath,
        string inputDataPath)
    {
        Name = name ?? string.Empty;
        Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        MinShape = minShape ?? throw new ArgumentNullException(nameof(minShape));
        OptShape = optShape ?? throw new ArgumentNullException(nameof(optShape));
        MaxShape = maxShape ?? throw new ArgumentNullException(nameof(maxShape));
        HasProfileOverride = hasProfileOverride;
        InputPattern = string.IsNullOrWhiteSpace(inputPattern) ? "ramp" : inputPattern;
        InputPath = inputPath ?? string.Empty;
        InputDataPath = inputDataPath ?? string.Empty;
    }

    public string Name { get; }

    public TensorRtDims Shape { get; }

    public TensorRtDims MinShape { get; }

    public TensorRtDims OptShape { get; }

    public TensorRtDims MaxShape { get; }

    public bool HasProfileOverride { get; }

    public string InputPattern { get; }

    public string InputPath { get; }

    public string InputDataPath { get; }

    public bool UsesExternalInput => !string.IsNullOrWhiteSpace(InputPath) || !string.IsNullOrWhiteSpace(InputDataPath);

    public string SourceClassification => !string.IsNullOrWhiteSpace(InputDataPath)
        ? "float-data-file"
        : !string.IsNullOrWhiteSpace(InputPath)
            ? "raw-byte-file"
            : "deterministic-pattern";

    public string SourcePath => !string.IsNullOrWhiteSpace(InputDataPath) ? InputDataPath : InputPath;

    public static IReadOnlyList<OnnxSampleInputOptions> FromArgs(
        string[] args,
        string legacyName,
        TensorRtDims legacyShape,
        TensorRtDims legacyMinShape,
        TensorRtDims legacyOptShape,
        TensorRtDims legacyMaxShape,
        bool legacyHasProfileOverride,
        string legacyPattern,
        string legacyInputPath,
        string legacyInputDataPath)
    {
        string inputShapesText = SampleCommandLine.GetStringArgument(args, "--input-shapes", string.Empty);
        string minShapesText = SampleCommandLine.GetStringArgument(args, "--min-shapes", string.Empty);
        string optShapesText = SampleCommandLine.GetStringArgument(args, "--opt-shapes", string.Empty);
        string maxShapesText = SampleCommandLine.GetStringArgument(args, "--max-shapes", string.Empty);
        string patternsText = SampleCommandLine.GetStringArgument(args, "--input-patterns", string.Empty);
        string byteInputsText = SampleCommandLine.GetStringArgument(args, "--load-byte-inputs", string.Empty);
        string floatInputsText = SampleCommandLine.GetStringArgument(args, "--load-inputs", string.Empty);
        bool usesNamedContract = !string.IsNullOrWhiteSpace(inputShapesText) ||
            !string.IsNullOrWhiteSpace(minShapesText) ||
            !string.IsNullOrWhiteSpace(optShapesText) ||
            !string.IsNullOrWhiteSpace(maxShapesText) ||
            !string.IsNullOrWhiteSpace(patternsText) ||
            !string.IsNullOrWhiteSpace(byteInputsText) ||
            !string.IsNullOrWhiteSpace(floatInputsText);

        if (!usesNamedContract)
        {
            return new[]
            {
                new OnnxSampleInputOptions(
                    legacyName,
                    legacyShape,
                    legacyMinShape,
                    legacyOptShape,
                    legacyMaxShape,
                    legacyHasProfileOverride,
                    legacyPattern,
                    legacyInputPath,
                    legacyInputDataPath)
            };
        }

        if (string.IsNullOrWhiteSpace(inputShapesText))
        {
            throw new ArgumentException("Named multi-input mode requires --input-shapes tensor:dims mappings for every model input.");
        }

        RejectLegacyInputArguments(args);
        Dictionary<string, TensorRtDims> shapes = ParseShapeMap(inputShapesText, "--input-shapes");
        Dictionary<string, TensorRtDims> minShapes = ParseOptionalShapeMap(minShapesText, "--min-shapes");
        Dictionary<string, TensorRtDims> optShapes = ParseOptionalShapeMap(optShapesText, "--opt-shapes");
        Dictionary<string, TensorRtDims> maxShapes = ParseOptionalShapeMap(maxShapesText, "--max-shapes");
        Dictionary<string, string> patterns = ParseStringMap(patternsText, "--input-patterns", resolveFile: false);
        Dictionary<string, string> byteInputs = ParseStringMap(byteInputsText, "--load-byte-inputs", resolveFile: true);
        Dictionary<string, string> floatInputs = ParseStringMap(floatInputsText, "--load-inputs", resolveFile: true);

        ValidateExactKeySet(shapes, patterns.Keys.Concat(byteInputs.Keys).Concat(floatInputs.Keys), "input source mappings");
        bool hasProfileOverride = minShapes.Count != 0 || optShapes.Count != 0 || maxShapes.Count != 0;
        if (hasProfileOverride)
        {
            ValidateExactKeySet(shapes, minShapes.Keys, "--min-shapes");
            ValidateExactKeySet(shapes, optShapes.Keys, "--opt-shapes");
            ValidateExactKeySet(shapes, maxShapes.Keys, "--max-shapes");
        }

        List<OnnxSampleInputOptions> result = new List<OnnxSampleInputOptions>(shapes.Count);
        foreach (KeyValuePair<string, TensorRtDims> pair in shapes)
        {
            int sourceCount = (patterns.ContainsKey(pair.Key) ? 1 : 0) +
                (byteInputs.ContainsKey(pair.Key) ? 1 : 0) +
                (floatInputs.ContainsKey(pair.Key) ? 1 : 0);
            if (sourceCount != 1)
            {
                throw new ArgumentException($"Input tensor '{pair.Key}' must have exactly one source in --input-patterns, --load-byte-inputs, or --load-inputs.");
            }

            string pattern = patterns.TryGetValue(pair.Key, out string? patternValue) ? patternValue : string.Empty;
            string bytePath = byteInputs.TryGetValue(pair.Key, out string? bytePathValue) ? bytePathValue : string.Empty;
            string floatPath = floatInputs.TryGetValue(pair.Key, out string? floatPathValue) ? floatPathValue : string.Empty;
            TensorRtDims minShape = hasProfileOverride ? minShapes[pair.Key] : pair.Value;
            TensorRtDims optShape = hasProfileOverride ? optShapes[pair.Key] : pair.Value;
            TensorRtDims maxShape = hasProfileOverride ? maxShapes[pair.Key] : pair.Value;
            result.Add(new OnnxSampleInputOptions(
                pair.Key,
                pair.Value,
                minShape,
                optShape,
                maxShape,
                hasProfileOverride,
                pattern,
                bytePath,
                floatPath));
        }

        return result;
    }

    private static void RejectLegacyInputArguments(string[] args)
    {
        string[] incompatible =
        {
            "--input-name", "--input-shape", "--min-shape", "--opt-shape", "--max-shape",
            "--input-pattern", "--input", "--input-data"
        };
        foreach (string option in incompatible)
        {
            if (args.Any(argument => string.Equals(argument, option, StringComparison.OrdinalIgnoreCase)))
            {
                throw new ArgumentException($"{option} cannot be combined with named multi-input options.");
            }
        }
    }

    private static Dictionary<string, TensorRtDims> ParseOptionalShapeMap(string value, string optionName)
    {
        return string.IsNullOrWhiteSpace(value)
            ? new Dictionary<string, TensorRtDims>(StringComparer.Ordinal)
            : ParseShapeMap(value, optionName);
    }

    private static Dictionary<string, TensorRtDims> ParseShapeMap(string value, string optionName)
    {
        Dictionary<string, TensorRtDims> result = new Dictionary<string, TensorRtDims>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, string> pair in ParseSegments(value, optionName))
        {
            if (!result.TryAdd(pair.Key, TensorRtOnnxSample.ParseShape(pair.Value, optionName)))
            {
                throw new ArgumentException($"{optionName} contains a duplicate mapping for tensor '{pair.Key}'.");
            }
        }

        return result;
    }

    private static Dictionary<string, string> ParseStringMap(string value, string optionName, bool resolveFile)
    {
        Dictionary<string, string> result = new Dictionary<string, string>(StringComparer.Ordinal);
        if (string.IsNullOrWhiteSpace(value))
        {
            return result;
        }

        foreach (KeyValuePair<string, string> pair in ParseSegments(value, optionName))
        {
            string mappedValue = pair.Value;
            if (resolveFile)
            {
                mappedValue = Path.GetFullPath(mappedValue.Trim('"'));
                if (!File.Exists(mappedValue))
                {
                    throw new FileNotFoundException($"Input file for {optionName} was not found.", mappedValue);
                }
            }
            else if (!string.Equals(mappedValue, "zeros", StringComparison.OrdinalIgnoreCase) &&
                     !string.Equals(mappedValue, "ones", StringComparison.OrdinalIgnoreCase) &&
                     !string.Equals(mappedValue, "ramp", StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException($"{optionName} pattern for tensor '{pair.Key}' must be zeros, ones, or ramp.");
            }

            if (!result.TryAdd(pair.Key, mappedValue))
            {
                throw new ArgumentException($"{optionName} contains a duplicate mapping for tensor '{pair.Key}'.");
            }
        }

        return result;
    }

    internal static IReadOnlyList<KeyValuePair<string, string>> ParseSegments(string value, string optionName)
    {
        List<KeyValuePair<string, string>> result = new List<KeyValuePair<string, string>>();
        foreach (string rawSegment in value.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
        {
            string segment = rawSegment.Trim();
            int separator = segment.IndexOf(':');
            if (separator <= 0 || separator == segment.Length - 1)
            {
                throw new ArgumentException($"{optionName} entries must use tensor:value format.");
            }

            string name = segment.Substring(0, separator).Trim();
            string mappedValue = segment.Substring(separator + 1).Trim();
            if (name.Length == 0 || mappedValue.Length == 0)
            {
                throw new ArgumentException($"{optionName} entries must use non-empty tensor:value pairs.");
            }

            result.Add(new KeyValuePair<string, string>(name, mappedValue));
        }

        if (result.Count == 0)
        {
            throw new ArgumentException($"{optionName} must contain at least one tensor:value mapping.");
        }

        return result;
    }

    private static void ValidateExactKeySet(
        IReadOnlyDictionary<string, TensorRtDims> shapes,
        IEnumerable<string> mappedNames,
        string description)
    {
        string[] names = mappedNames.ToArray();
        string[] duplicateNames = names.GroupBy(static name => name, StringComparer.Ordinal)
            .Where(static group => group.Count() > 1)
            .Select(static group => group.Key)
            .OrderBy(static name => name, StringComparer.Ordinal)
            .ToArray();
        if (duplicateNames.Length > 0)
        {
            throw new ArgumentException($"{description} assign more than one value to: {string.Join(", ", duplicateNames)}.");
        }

        HashSet<string> expected = new HashSet<string>(shapes.Keys, StringComparer.Ordinal);
        HashSet<string> actual = new HashSet<string>(names, StringComparer.Ordinal);
        string[] missing = expected.Except(actual).OrderBy(static name => name, StringComparer.Ordinal).ToArray();
        string[] extra = actual.Except(expected).OrderBy(static name => name, StringComparer.Ordinal).ToArray();
        if (missing.Length != 0 || extra.Length != 0)
        {
            throw new ArgumentException($"{description} must match --input-shapes exactly. Missing=[{string.Join(",", missing)}] Extra=[{string.Join(",", extra)}].");
        }
    }
}

internal sealed class OnnxSampleInputTensor
{
    public OnnxSampleInputTensor(
        string name,
        TensorRtDims shape,
        float[] values,
        string sourceClassification,
        string sourcePath)
    {
        Name = name ?? string.Empty;
        Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        ElementCount = values.Length;
        ByteLength = checked(values.Length * sizeof(float));
        Preview = values.Take(8).ToArray();
        byte[] bytes = new byte[ByteLength];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        Sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        SourceClassification = sourceClassification ?? string.Empty;
        SourcePath = sourcePath ?? string.Empty;
    }

    public string Name { get; }

    public TensorRtDims Shape { get; }

    public int ElementCount { get; }

    public int ByteLength { get; }

    public IReadOnlyList<float> Preview { get; }

    public string Sha256 { get; }

    public string SourceClassification { get; }

    public string SourcePath { get; }
}

internal sealed class OnnxSampleNetworkInput
{
    public OnnxSampleNetworkInput(string name, TensorRtDataType dataType, TensorRtDims networkShape, OnnxSampleInputOptions options)
    {
        Name = name ?? string.Empty;
        DataType = dataType;
        NetworkShape = networkShape ?? throw new ArgumentNullException(nameof(networkShape));
        Options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public string Name { get; }

    public TensorRtDataType DataType { get; }

    public TensorRtDims NetworkShape { get; }

    public OnnxSampleInputOptions Options { get; }
}
