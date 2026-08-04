using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class OnnxEngineBuildService
{
    private static readonly JsonSerializerOptions ReferenceJsonOptions = new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
    };

    private static OnnxEngineReferenceValidationArtifact ValidateReferenceOutputs(
        IReadOnlyList<OnnxEngineRuntimeOutputTensor> outputs,
        TrtexecLikeRuntimeOptions options)
    {
        if (!options.RequestsReferenceValidation)
        {
            return OnnxEngineReferenceValidationArtifact.NotRequested;
        }

        List<string> diagnostics = new List<string>();
        Dictionary<string, string> mappings = ParseReferenceOutputMappings(options.ReferenceOutputs, diagnostics);
        HashSet<string> outputNames = new HashSet<string>(outputs.Select(static output => output.Name), StringComparer.Ordinal);
        foreach (string unknownName in mappings.Keys.Where(name => !outputNames.Contains(name)).OrderBy(static name => name, StringComparer.Ordinal))
        {
            diagnostics.Add($"unknown-output-mapping:{unknownName}");
        }

        List<OnnxEngineReferenceTensorComparisonArtifact> comparisons = new List<OnnxEngineReferenceTensorComparisonArtifact>(outputs.Count);
        bool allComparable = diagnostics.Count == 0;
        foreach (OnnxEngineRuntimeOutputTensor output in outputs)
        {
            if (!mappings.TryGetValue(output.Name, out string? referencePath))
            {
                diagnostics.Add($"missing-output-mapping:{output.Name}");
                comparisons.Add(CreateUnavailableReferenceComparison(output, string.Empty, "reference mapping is missing"));
                allComparable = false;
                continue;
            }

            try
            {
                OnnxEngineReferenceTensorData reference = ReadReferenceTensor(referencePath);
                OnnxEngineReferenceTensorComparisonArtifact comparison = CompareReferenceTensor(output, referencePath, reference, options, out bool comparable);
                comparisons.Add(comparison);
                allComparable &= comparable;
                if (!comparison.Passed)
                {
                    diagnostics.Add($"tensor-validation-failed:{output.Name}:{comparison.Diagnostic}");
                }
            }
            catch (Exception exception) when (exception is IOException || exception is UnauthorizedAccessException || exception is JsonException || exception is ArgumentException || exception is InvalidOperationException)
            {
                string diagnostic = SanitizeReferenceDiagnostic(exception.Message);
                diagnostics.Add($"reference-read-failed:{output.Name}:{diagnostic}");
                comparisons.Add(CreateUnavailableReferenceComparison(output, referencePath, diagnostic));
                allComparable = false;
            }
        }

        bool completed = allComparable && comparisons.Count == outputs.Count;
        bool passed = completed && comparisons.All(static comparison => comparison.Passed);
        return new OnnxEngineReferenceValidationArtifact(
            requested: true,
            completed,
            passed,
            options.ReferenceAbsoluteTolerance,
            options.ReferenceRelativeTolerance,
            options.ReferenceNaNPolicy.ToString().ToLowerInvariant(),
            options.ReferenceInfinityPolicy.ToString().ToLowerInvariant(),
            diagnostics,
            comparisons);
    }

    private static Dictionary<string, string> ParseReferenceOutputMappings(string value, List<string> diagnostics)
    {
        Dictionary<string, string> mappings = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string segment in (value ?? string.Empty).Split(new[] { ';', ',' }, StringSplitOptions.RemoveEmptyEntries))
        {
            int separator = segment.IndexOf(':');
            if (separator <= 0 || separator == segment.Length - 1)
            {
                diagnostics.Add("invalid-reference-mapping:" + SanitizeReferenceDiagnostic(segment));
                continue;
            }

            string tensorName = segment.Substring(0, separator).Trim();
            string pathText = segment.Substring(separator + 1).Trim().Trim('"');
            if (tensorName.Length == 0 || pathText.Length == 0)
            {
                diagnostics.Add("invalid-reference-mapping:" + SanitizeReferenceDiagnostic(segment));
                continue;
            }

            string fullPath;
            try
            {
                fullPath = Path.GetFullPath(pathText);
            }
            catch (Exception exception) when (exception is ArgumentException || exception is NotSupportedException)
            {
                diagnostics.Add($"invalid-reference-path:{tensorName}:{SanitizeReferenceDiagnostic(exception.Message)}");
                continue;
            }

            if (!mappings.TryAdd(tensorName, fullPath))
            {
                diagnostics.Add("duplicate-reference-mapping:" + tensorName);
            }
        }

        return mappings;
    }

    private static OnnxEngineReferenceTensorData ReadReferenceTensor(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        OnnxEngineReferenceTensorData? document = JsonSerializer.Deserialize<OnnxEngineReferenceTensorData>(bytes, ReferenceJsonOptions);
        if (document == null)
        {
            throw new InvalidOperationException("Reference JSON document is empty.");
        }
        if (document.SchemaVersion != 1)
        {
            throw new ArgumentException("Reference JSON schemaVersion must be 1.");
        }
        if (string.IsNullOrWhiteSpace(document.TensorName))
        {
            throw new ArgumentException("Reference JSON tensorName is required.");
        }
        if (document.Shape.Count == 0 || document.Shape.Any(static value => value <= 0))
        {
            throw new ArgumentException("Reference JSON shape must contain positive dimensions.");
        }
        if (string.IsNullOrWhiteSpace(document.SourceClassification))
        {
            throw new ArgumentException("Reference JSON sourceClassification is required.");
        }

        document.FileSha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        return document;
    }

    private static OnnxEngineReferenceTensorComparisonArtifact CompareReferenceTensor(
        OnnxEngineRuntimeOutputTensor output,
        string referencePath,
        OnnxEngineReferenceTensorData reference,
        TrtexecLikeRuntimeOptions options,
        out bool comparable)
    {
        string fullPath = Path.GetFullPath(referencePath);
        if (!OnnxEngineReferenceTensorComparer.MetadataMatches(
            output.Name,
            output.Shape,
            output.Values.Length,
            reference,
            out string metadataDiagnostic))
        {
            comparable = false;
            return CreateReferenceMetadataMismatch(output, fullPath, reference, metadataDiagnostic);
        }

        comparable = true;
        int mismatchCount = 0;
        int firstMismatchIndex = -1;
        float maximumAbsoluteError = 0.0f;
        float maximumRelativeError = 0.0f;
        for (int index = 0; index < output.Values.Length; index++)
        {
            float actual = output.Values[index];
            float expected = reference.Values[index];
            bool matches = ReferenceValuesMatch(actual, expected, options, out float absoluteError, out float relativeError);
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
        string diagnostic = passed
            ? "all reference values matched"
            : $"{mismatchCount} value(s) exceeded tolerance or special-value policy; first mismatch index {firstMismatchIndex}";
        return new OnnxEngineReferenceTensorComparisonArtifact(
            output.Name,
            fullPath,
            reference.FileSha256,
            reference.SourceClassification,
            output.Shape,
            reference.Shape,
            output.Values.Length,
            reference.Values.Count,
            output.Values.Length,
            mismatchCount,
            firstMismatchIndex,
            maximumAbsoluteError,
            maximumRelativeError,
            passed,
            diagnostic);
    }

    private static bool ReferenceValuesMatch(
        float actual,
        float expected,
        TrtexecLikeRuntimeOptions options,
        out float absoluteError,
        out float relativeError)
    {
        return OnnxEngineReferenceValueComparer.Matches(
            actual,
            expected,
            options.ReferenceAbsoluteTolerance,
            options.ReferenceRelativeTolerance,
            options.ReferenceNaNPolicy,
            options.ReferenceInfinityPolicy,
            out absoluteError,
            out relativeError);
    }

    private static OnnxEngineReferenceTensorComparisonArtifact CreateReferenceMetadataMismatch(
        OnnxEngineRuntimeOutputTensor output,
        string referencePath,
        OnnxEngineReferenceTensorData reference,
        string diagnostic)
    {
        return new OnnxEngineReferenceTensorComparisonArtifact(
            output.Name,
            referencePath,
            reference.FileSha256,
            reference.SourceClassification,
            output.Shape,
            reference.Shape,
            output.Values.Length,
            reference.Values.Count,
            0,
            0,
            -1,
            0.0f,
            0.0f,
            passed: false,
            diagnostic);
    }

    private static OnnxEngineReferenceTensorComparisonArtifact CreateUnavailableReferenceComparison(
        OnnxEngineRuntimeOutputTensor output,
        string referencePath,
        string diagnostic)
    {
        return new OnnxEngineReferenceTensorComparisonArtifact(
            output.Name,
            referencePath,
            string.Empty,
            string.Empty,
            output.Shape,
            Array.Empty<int>(),
            output.Values.Length,
            0,
            0,
            0,
            -1,
            0.0f,
            0.0f,
            passed: false,
            diagnostic);
    }

    private static string SanitizeReferenceDiagnostic(string value)
    {
        return (value ?? string.Empty).Replace('\r', ' ').Replace('\n', ' ').Trim();
    }

    private static int CountElements(TensorRtDims shape)
    {
        if (shape == null)
        {
            throw new ArgumentNullException(nameof(shape));
        }

        int result = 1;
        foreach (int value in shape.Values)
        {
            if (value <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(shape), "Shape must be concrete and positive.");
            }

            result = checked(result * value);
        }

        return result;
    }

    private static bool CanEstimate(TensorRtDims? shape)
    {
        if (shape == null || shape.Values.Length == 0)
        {
            return false;
        }

        foreach (int value in shape.Values)
        {
            if (value <= 0)
            {
                return false;
            }
        }

        return true;
    }

    private static bool ShapesEqual(TensorRtDims? left, TensorRtDims right)
    {
        return left != null && left.Values.SequenceEqual(right.Values);
    }

    private static bool ValuesEqual(IReadOnlyList<float> left, IReadOnlyList<float> right)
    {
        if (left.Count != right.Count)
        {
            return false;
        }

        for (int index = 0; index < left.Count; index++)
        {
            if (!float.IsFinite(left[index]) ||
                !float.IsFinite(right[index]) ||
                Math.Abs(left[index] - right[index]) > 1e-5f)
            {
                return false;
            }
        }

        return true;
    }

    private static string FormatShape(IReadOnlyList<int> values)
    {
        return string.Join("x", values);
    }

}
