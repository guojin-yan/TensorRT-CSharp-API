using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace JYPPX.SampleSupport;

internal enum OnnxSampleReferenceNaNPolicy
{
    Reject = 0,
    Equal = 1
}

internal enum OnnxSampleReferenceInfinityPolicy
{
    Exact = 0,
    Reject = 1
}

internal sealed class OnnxSampleReferenceOptions
{
    public OnnxSampleReferenceOptions(
        IReadOnlyDictionary<string, string> outputFiles,
        float absoluteTolerance,
        float relativeTolerance,
        OnnxSampleReferenceNaNPolicy nanPolicy,
        OnnxSampleReferenceInfinityPolicy infinityPolicy)
    {
        OutputFiles = new Dictionary<string, string>(outputFiles ?? throw new ArgumentNullException(nameof(outputFiles)), StringComparer.Ordinal);
        if (!float.IsFinite(absoluteTolerance) || absoluteTolerance < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(absoluteTolerance), "Absolute tolerance must be finite and non-negative.");
        }
        if (!float.IsFinite(relativeTolerance) || relativeTolerance < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(relativeTolerance), "Relative tolerance must be finite and non-negative.");
        }

        AbsoluteTolerance = absoluteTolerance;
        RelativeTolerance = relativeTolerance;
        NaNPolicy = nanPolicy;
        InfinityPolicy = infinityPolicy;
    }

    public IReadOnlyDictionary<string, string> OutputFiles { get; }

    public float AbsoluteTolerance { get; }

    public float RelativeTolerance { get; }

    public OnnxSampleReferenceNaNPolicy NaNPolicy { get; }

    public OnnxSampleReferenceInfinityPolicy InfinityPolicy { get; }

    public bool Requested => OutputFiles.Count != 0;

    public static OnnxSampleReferenceOptions FromArgs(string[] args)
    {
        string mappingsText = SampleCommandLine.GetStringArgument(args, "--reference-outputs", string.Empty);
        Dictionary<string, string> mappings = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!string.IsNullOrWhiteSpace(mappingsText))
        {
            foreach (KeyValuePair<string, string> pair in OnnxSampleInputOptions.ParseSegments(mappingsText, "--reference-outputs"))
            {
                string path = Path.GetFullPath(pair.Value.Trim('"'));
                if (!File.Exists(path))
                {
                    throw new FileNotFoundException("Reference output file was not found.", path);
                }
                if (!mappings.TryAdd(pair.Key, path))
                {
                    throw new ArgumentException($"--reference-outputs contains a duplicate mapping for tensor '{pair.Key}'.");
                }
            }
        }

        float absoluteTolerance = ParseNonNegativeFiniteFloat(args, "--reference-abs-tolerance", 0.0f);
        float relativeTolerance = ParseNonNegativeFiniteFloat(args, "--reference-rel-tolerance", 0.0f);
        OnnxSampleReferenceNaNPolicy nanPolicy = ParseNaNPolicy(
            SampleCommandLine.GetStringArgument(args, "--reference-nan-policy", "reject"));
        OnnxSampleReferenceInfinityPolicy infinityPolicy = ParseInfinityPolicy(
            SampleCommandLine.GetStringArgument(args, "--reference-infinity-policy", "exact"));
        return new OnnxSampleReferenceOptions(mappings, absoluteTolerance, relativeTolerance, nanPolicy, infinityPolicy);
    }

    private static float ParseNonNegativeFiniteFloat(string[] args, string optionName, float defaultValue)
    {
        string text = SampleCommandLine.GetStringArgument(args, optionName, defaultValue.ToString("R", CultureInfo.InvariantCulture));
        if (!float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) ||
            !float.IsFinite(value) ||
            value < 0.0f)
        {
            throw new ArgumentException($"{optionName} must be a finite non-negative number.");
        }

        return value;
    }

    private static OnnxSampleReferenceNaNPolicy ParseNaNPolicy(string value)
    {
        return value.Trim().ToLowerInvariant() switch
        {
            "reject" => OnnxSampleReferenceNaNPolicy.Reject,
            "equal" => OnnxSampleReferenceNaNPolicy.Equal,
            _ => throw new ArgumentException("--reference-nan-policy must be reject or equal.")
        };
    }

    private static OnnxSampleReferenceInfinityPolicy ParseInfinityPolicy(string value)
    {
        return value.Trim().ToLowerInvariant() switch
        {
            "exact" => OnnxSampleReferenceInfinityPolicy.Exact,
            "reject" => OnnxSampleReferenceInfinityPolicy.Reject,
            _ => throw new ArgumentException("--reference-infinity-policy must be exact or reject.")
        };
    }
}

internal sealed class OnnxSampleReferenceTensorData
{
    public int SchemaVersion { get; set; }

    public string TensorName { get; set; } = string.Empty;

    public int[] Shape { get; set; } = Array.Empty<int>();

    public float[] Values { get; set; } = Array.Empty<float>();

    public string SourceClassification { get; set; } = string.Empty;
}

internal sealed class OnnxSampleReferenceTensorComparison
{
    public OnnxSampleReferenceTensorComparison(
        string tensorName,
        string referencePath,
        string referenceSha256,
        string sourceClassification,
        IReadOnlyList<int> actualShape,
        IReadOnlyList<int> referenceShape,
        int actualElementCount,
        int referenceElementCount,
        int comparedElementCount,
        int mismatchCount,
        int firstMismatchIndex,
        float maximumAbsoluteError,
        float maximumRelativeError,
        bool completed,
        bool passed,
        string diagnostic)
    {
        TensorName = tensorName ?? string.Empty;
        ReferencePath = referencePath ?? string.Empty;
        ReferenceSha256 = referenceSha256 ?? string.Empty;
        SourceClassification = sourceClassification ?? string.Empty;
        ActualShape = (actualShape ?? Array.Empty<int>()).ToArray();
        ReferenceShape = (referenceShape ?? Array.Empty<int>()).ToArray();
        ActualElementCount = actualElementCount;
        ReferenceElementCount = referenceElementCount;
        ComparedElementCount = comparedElementCount;
        MismatchCount = mismatchCount;
        FirstMismatchIndex = firstMismatchIndex;
        MaximumAbsoluteError = maximumAbsoluteError;
        MaximumRelativeError = maximumRelativeError;
        Completed = completed;
        Passed = passed;
        Diagnostic = diagnostic ?? string.Empty;
    }

    public string TensorName { get; }

    public string ReferencePath { get; }

    public string ReferenceSha256 { get; }

    public string SourceClassification { get; }

    public IReadOnlyList<int> ActualShape { get; }

    public IReadOnlyList<int> ReferenceShape { get; }

    public int ActualElementCount { get; }

    public int ReferenceElementCount { get; }

    public int ComparedElementCount { get; }

    public int MismatchCount { get; }

    public int FirstMismatchIndex { get; }

    public float MaximumAbsoluteError { get; }

    public float MaximumRelativeError { get; }

    public bool Completed { get; }

    public bool Passed { get; }

    public string Diagnostic { get; }
}

internal sealed class OnnxSampleReferenceValidationResult
{
    public OnnxSampleReferenceValidationResult(
        bool requested,
        bool completed,
        bool passed,
        OnnxSampleReferenceOptions options,
        IReadOnlyList<OnnxSampleReferenceTensorComparison> tensorComparisons,
        IReadOnlyList<string> diagnostics)
    {
        Requested = requested;
        Completed = completed;
        Passed = passed;
        AbsoluteTolerance = options.AbsoluteTolerance;
        RelativeTolerance = options.RelativeTolerance;
        NaNPolicy = options.NaNPolicy.ToString().ToLowerInvariant();
        InfinityPolicy = options.InfinityPolicy.ToString().ToLowerInvariant();
        TensorComparisons = (tensorComparisons ?? Array.Empty<OnnxSampleReferenceTensorComparison>()).ToArray();
        Diagnostics = (diagnostics ?? Array.Empty<string>()).ToArray();
    }

    public bool Requested { get; }

    public bool Completed { get; }

    public bool Passed { get; }

    public float AbsoluteTolerance { get; }

    public float RelativeTolerance { get; }

    public string NaNPolicy { get; }

    public string InfinityPolicy { get; }

    public IReadOnlyList<OnnxSampleReferenceTensorComparison> TensorComparisons { get; }

    public IReadOnlyList<string> Diagnostics { get; }

    public static OnnxSampleReferenceValidationResult NotRequested(OnnxSampleReferenceOptions options)
    {
        return new OnnxSampleReferenceValidationResult(
            requested: false,
            completed: false,
            passed: false,
            options,
            Array.Empty<OnnxSampleReferenceTensorComparison>(),
            Array.Empty<string>());
    }
}

internal static partial class TensorRtOnnxSample
{
    private static readonly JsonSerializerOptions ReferenceJsonOptions = new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
    };

    public static bool ReferenceValuesMatchForTesting(
        float actual,
        float expected,
        float absoluteTolerance,
        float relativeTolerance,
        string nanPolicy,
        string infinityPolicy,
        out float absoluteError,
        out float relativeError)
    {
        OnnxSampleReferenceOptions options = new OnnxSampleReferenceOptions(
            new Dictionary<string, string>(StringComparer.Ordinal),
            absoluteTolerance,
            relativeTolerance,
            string.Equals(nanPolicy, "equal", StringComparison.OrdinalIgnoreCase)
                ? OnnxSampleReferenceNaNPolicy.Equal
                : OnnxSampleReferenceNaNPolicy.Reject,
            string.Equals(infinityPolicy, "reject", StringComparison.OrdinalIgnoreCase)
                ? OnnxSampleReferenceInfinityPolicy.Reject
                : OnnxSampleReferenceInfinityPolicy.Exact);
        return ReferenceValuesMatch(actual, expected, options, out absoluteError, out relativeError);
    }

    private static OnnxSampleReferenceValidationResult ValidateReferenceOutputs(
        IReadOnlyList<OnnxSampleOutputTensor> outputs,
        OnnxSampleReferenceOptions options)
    {
        if (!options.Requested)
        {
            return OnnxSampleReferenceValidationResult.NotRequested(options);
        }

        HashSet<string> outputNames = new HashSet<string>(outputs.Select(static output => output.Name), StringComparer.Ordinal);
        string[] missing = outputNames.Except(options.OutputFiles.Keys).OrderBy(static name => name, StringComparer.Ordinal).ToArray();
        string[] extra = options.OutputFiles.Keys.Except(outputNames).OrderBy(static name => name, StringComparer.Ordinal).ToArray();
        if (missing.Length != 0 || extra.Length != 0)
        {
            throw new ArgumentException($"--reference-outputs must match captured outputs exactly. Missing=[{string.Join(",", missing)}] Extra=[{string.Join(",", extra)}].");
        }

        List<OnnxSampleReferenceTensorComparison> comparisons = new List<OnnxSampleReferenceTensorComparison>(outputs.Count);
        List<string> diagnostics = new List<string>();
        foreach (OnnxSampleOutputTensor output in outputs)
        {
            string path = options.OutputFiles[output.Name];
            byte[] bytes = File.ReadAllBytes(path);
            string sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
            OnnxSampleReferenceTensorData? reference = JsonSerializer.Deserialize<OnnxSampleReferenceTensorData>(bytes, ReferenceJsonOptions);
            if (reference == null)
            {
                throw new InvalidDataException($"Reference output '{path}' did not contain a JSON object.");
            }
            if (reference.SchemaVersion != 1)
            {
                throw new InvalidDataException($"Reference output '{path}' must use schemaVersion=1.");
            }
            if (string.IsNullOrWhiteSpace(reference.SourceClassification))
            {
                throw new InvalidDataException($"Reference output '{path}' must declare sourceClassification.");
            }
            if (reference.Shape == null || reference.Shape.Length == 0 || reference.Shape.Any(static dimension => dimension <= 0))
            {
                throw new InvalidDataException($"Reference output '{path}' must declare a non-empty positive shape.");
            }
            if (reference.Values == null || reference.Values.Length == 0)
            {
                throw new InvalidDataException($"Reference output '{path}' must contain at least one value.");
            }

            OnnxSampleReferenceTensorComparison comparison = CompareReferenceOutput(output, path, sha256, reference, options);
            comparisons.Add(comparison);
            if (!comparison.Passed)
            {
                diagnostics.Add($"{output.Name}: {comparison.Diagnostic}");
            }
        }

        bool completed = comparisons.All(static comparison => comparison.Completed);
        bool passed = completed && comparisons.All(static comparison => comparison.Passed);
        if (passed)
        {
            diagnostics.Add("All captured output tensors matched their structured references.");
        }

        return new OnnxSampleReferenceValidationResult(
            requested: true,
            completed,
            passed,
            options,
            comparisons,
            diagnostics);
    }

    private static OnnxSampleReferenceTensorComparison CompareReferenceOutput(
        OnnxSampleOutputTensor output,
        string path,
        string sha256,
        OnnxSampleReferenceTensorData reference,
        OnnxSampleReferenceOptions options)
    {
        int[] referenceShape = reference.Shape ?? Array.Empty<int>();
        float[] referenceValues = reference.Values ?? Array.Empty<float>();
        string metadataDiagnostic = string.Empty;
        if (!string.Equals(reference.TensorName, output.Name, StringComparison.Ordinal))
        {
            metadataDiagnostic = "reference tensorName does not match the runtime output name";
        }
        else if (!output.Shape.Values.SequenceEqual(referenceShape))
        {
            metadataDiagnostic = "reference shape does not match the runtime output shape";
        }
        else if (referenceValues.Length != output.Values.Length)
        {
            metadataDiagnostic = "reference value count does not match the runtime output element count";
        }

        if (metadataDiagnostic.Length != 0)
        {
            return new OnnxSampleReferenceTensorComparison(
                output.Name,
                path,
                sha256,
                reference.SourceClassification,
                output.Shape.Values,
                referenceShape,
                output.Values.Length,
                referenceValues.Length,
                0,
                1,
                -1,
                0.0f,
                0.0f,
                completed: false,
                passed: false,
                metadataDiagnostic);
        }

        int mismatchCount = 0;
        int firstMismatchIndex = -1;
        float maximumAbsoluteError = 0.0f;
        float maximumRelativeError = 0.0f;
        for (int index = 0; index < output.Values.Length; index++)
        {
            bool matches = ReferenceValuesMatch(
                output.Values[index],
                referenceValues[index],
                options,
                out float absoluteError,
                out float relativeError);
            maximumAbsoluteError = Math.Max(maximumAbsoluteError, absoluteError);
            maximumRelativeError = Math.Max(maximumRelativeError, relativeError);
            if (!matches)
            {
                mismatchCount++;
                if (firstMismatchIndex < 0)
                {
                    firstMismatchIndex = index;
                }
            }
        }

        bool passed = mismatchCount == 0;
        return new OnnxSampleReferenceTensorComparison(
            output.Name,
            path,
            sha256,
            reference.SourceClassification,
            output.Shape.Values,
            referenceShape,
            output.Values.Length,
            referenceValues.Length,
            output.Values.Length,
            mismatchCount,
            firstMismatchIndex,
            maximumAbsoluteError,
            maximumRelativeError,
            completed: true,
            passed,
            passed ? "all reference values matched" : $"{mismatchCount} reference values did not match");
    }

    private static bool ReferenceValuesMatch(
        float actual,
        float expected,
        OnnxSampleReferenceOptions options,
        out float absoluteError,
        out float relativeError)
    {
        if (float.IsNaN(actual) || float.IsNaN(expected))
        {
            bool bothNaN = float.IsNaN(actual) && float.IsNaN(expected);
            absoluteError = bothNaN ? 0.0f : float.MaxValue;
            relativeError = absoluteError;
            return options.NaNPolicy == OnnxSampleReferenceNaNPolicy.Equal && bothNaN;
        }

        if (float.IsInfinity(actual) || float.IsInfinity(expected))
        {
            bool exact = actual.Equals(expected);
            absoluteError = exact ? 0.0f : float.MaxValue;
            relativeError = absoluteError;
            return options.InfinityPolicy == OnnxSampleReferenceInfinityPolicy.Exact && exact;
        }

        absoluteError = Math.Abs(actual - expected);
        float scale = Math.Max(Math.Abs(actual), Math.Abs(expected));
        relativeError = scale == 0.0f ? absoluteError : absoluteError / scale;
        return absoluteError <= options.AbsoluteTolerance || absoluteError <= options.RelativeTolerance * scale;
    }
}
