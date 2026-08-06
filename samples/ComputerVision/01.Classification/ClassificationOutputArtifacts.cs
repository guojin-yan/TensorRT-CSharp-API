using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using JYPPX.SampleSupport;
using JYPPX.TensorRtSharp;

namespace ClassificationSample;

public enum ClassificationScoreTransform
{
    Raw = 0,
    Softmax = 1
}

public sealed class ClassificationPrediction
{
    public ClassificationPrediction(int index, string label, float score)
    {
        Index = index;
        Label = label ?? string.Empty;
        Score = score;
    }

    public int Index { get; }

    public string Label { get; }

    public float Score { get; }
}

public sealed class ClassificationReferenceContext
{
    public ClassificationReferenceContext(
        string tensorName,
        int[] shape,
        string valueKind,
        string modelSha256,
        string inputTensorSha256,
        string preprocessContractSha256,
        string outputTensorContractSha256,
        string labelsSha256,
        string taskSemanticsSha256)
    {
        TensorName = tensorName ?? string.Empty;
        Shape = shape == null ? Array.Empty<int>() : (int[])shape.Clone();
        ValueKind = valueKind ?? string.Empty;
        ModelSha256 = modelSha256 ?? string.Empty;
        InputTensorSha256 = inputTensorSha256 ?? string.Empty;
        PreprocessContractSha256 = preprocessContractSha256 ?? string.Empty;
        OutputTensorContractSha256 = outputTensorContractSha256 ?? string.Empty;
        LabelsSha256 = labelsSha256 ?? string.Empty;
        TaskSemanticsSha256 = taskSemanticsSha256 ?? string.Empty;
    }

    public string TensorName { get; }

    public int[] Shape { get; }

    public string ValueKind { get; }

    public string ModelSha256 { get; }

    public string InputTensorSha256 { get; }

    public string PreprocessContractSha256 { get; }

    public string OutputTensorContractSha256 { get; }

    public string LabelsSha256 { get; }

    public string TaskSemanticsSha256 { get; }
}

public sealed class ClassificationReferenceValidationResult
{
    public ClassificationReferenceValidationResult(
        bool requested,
        bool completed,
        bool passed,
        string referencePath,
        string referenceSha256,
        string sourceClassification,
        int comparedElementCount,
        int mismatchCount,
        int firstMismatchIndex,
        float maximumAbsoluteError,
        float maximumRelativeError,
        float absoluteTolerance,
        float relativeTolerance,
        string nanPolicy,
        string infinityPolicy,
        string diagnostic)
    {
        Requested = requested;
        Completed = completed;
        Passed = passed;
        ReferencePath = referencePath ?? string.Empty;
        ReferenceSha256 = referenceSha256 ?? string.Empty;
        SourceClassification = sourceClassification ?? string.Empty;
        ComparedElementCount = comparedElementCount;
        MismatchCount = mismatchCount;
        FirstMismatchIndex = firstMismatchIndex;
        MaximumAbsoluteError = maximumAbsoluteError;
        MaximumRelativeError = maximumRelativeError;
        AbsoluteTolerance = absoluteTolerance;
        RelativeTolerance = relativeTolerance;
        NaNPolicy = nanPolicy ?? string.Empty;
        InfinityPolicy = infinityPolicy ?? string.Empty;
        Diagnostic = diagnostic ?? string.Empty;
    }

    public static ClassificationReferenceValidationResult NotRequested { get; } = new ClassificationReferenceValidationResult(
        false, false, false, string.Empty, string.Empty, string.Empty, 0, 0, -1, 0.0f, 0.0f, 0.0f, 0.0f, "reject", "exact", "reference validation was not requested");

    public bool Requested { get; }

    public bool Completed { get; }

    public bool Passed { get; }

    public string ReferencePath { get; }

    public string ReferenceSha256 { get; }

    public string SourceClassification { get; }

    public int ComparedElementCount { get; }

    public int MismatchCount { get; }

    public int FirstMismatchIndex { get; }

    public float MaximumAbsoluteError { get; }

    public float MaximumRelativeError { get; }

    public float AbsoluteTolerance { get; }

    public float RelativeTolerance { get; }

    public string NaNPolicy { get; }

    public string InfinityPolicy { get; }

    public string Diagnostic { get; }
}

public static class ClassificationOutputProcessor
{
    private static readonly JsonSerializerOptions ReferenceJsonOptions = new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
    };

    public static ClassificationScoreTransform ParseScoreTransform(string value)
    {
        string normalized = (value ?? string.Empty).Trim().ToLowerInvariant();
        return normalized switch
        {
            "" or "raw" or "logits" => ClassificationScoreTransform.Raw,
            "softmax" or "probabilities" => ClassificationScoreTransform.Softmax,
            _ => throw new ArgumentException("Score transform must be raw or softmax.", nameof(value))
        };
    }

    public static string ValueKind(ClassificationScoreTransform transform)
    {
        return transform == ClassificationScoreTransform.Softmax ? "probabilities" : "logits";
    }

    public static float[] Transform(float[] values, ClassificationScoreTransform transform)
    {
        if (values == null || values.Length == 0)
        {
            throw new ArgumentException("Classification output must not be empty.", nameof(values));
        }

        float[] copy = (float[])values.Clone();
        if (transform == ClassificationScoreTransform.Raw)
        {
            return copy;
        }

        if (copy.Any(static value => !float.IsFinite(value)))
        {
            throw new ArgumentException("Softmax requires finite output values.", nameof(values));
        }

        float maximum = copy.Max();
        double sum = 0.0;
        for (int index = 0; index < copy.Length; index++)
        {
            copy[index] = MathF.Exp(copy[index] - maximum);
            sum += copy[index];
        }
        if (!double.IsFinite(sum) || sum <= 0.0)
        {
            throw new InvalidOperationException("Softmax normalization failed.");
        }
        for (int index = 0; index < copy.Length; index++)
        {
            copy[index] = (float)(copy[index] / sum);
        }

        return copy;
    }

    public static IReadOnlyList<ClassificationPrediction> GetTopK(float[] values, IReadOnlyList<string> labels, int topK)
    {
        if (values == null || values.Length == 0)
        {
            throw new ArgumentException("Classification values must not be empty.", nameof(values));
        }
        if (topK <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(topK));
        }

        IReadOnlyList<string> safeLabels = labels ?? Array.Empty<string>();
        return values
            .Select(static (score, index) => (Index: index, Score: score))
            .OrderByDescending(static item => item.Score)
            .ThenBy(static item => item.Index)
            .Take(Math.Min(topK, values.Length))
            .Select(item => new ClassificationPrediction(
                item.Index,
                item.Index < safeLabels.Count && !string.IsNullOrWhiteSpace(safeLabels[item.Index])
                    ? safeLabels[item.Index]
                    : item.Index.ToString(CultureInfo.InvariantCulture),
                item.Score))
            .ToArray();
    }

    public static ClassificationReferenceValidationResult ValidateReference(
        string referencePath,
        ClassificationReferenceContext context,
        float[] actualValues,
        float absoluteTolerance,
        float relativeTolerance,
        string nanPolicy,
        string infinityPolicy)
    {
        if (string.IsNullOrWhiteSpace(referencePath))
        {
            return ClassificationReferenceValidationResult.NotRequested;
        }
        if (context == null)
        {
            throw new ArgumentNullException(nameof(context));
        }
        if (actualValues == null || actualValues.Length == 0)
        {
            throw new ArgumentException("Actual classification values must not be empty.", nameof(actualValues));
        }
        ValidateTolerance(absoluteTolerance, nameof(absoluteTolerance));
        ValidateTolerance(relativeTolerance, nameof(relativeTolerance));
        string normalizedNaNPolicy = NormalizeNaNPolicy(nanPolicy);
        string normalizedInfinityPolicy = NormalizeInfinityPolicy(infinityPolicy);
        string fullPath = Path.GetFullPath(referencePath);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Classification reference JSON was not found.", fullPath);
        }

        byte[] bytes = File.ReadAllBytes(fullPath);
        ReadOnlySpan<byte> jsonBytes = bytes;
        if (jsonBytes.StartsWith(new byte[] { 0xef, 0xbb, 0xbf }))
        {
            jsonBytes = jsonBytes[3..];
        }
        ClassificationReferenceDocument? reference = JsonSerializer.Deserialize<ClassificationReferenceDocument>(jsonBytes, ReferenceJsonOptions);
        if (reference == null || reference.SchemaVersion != 1 || string.IsNullOrWhiteSpace(reference.SourceClassification))
        {
            throw new InvalidDataException("Classification reference must contain schemaVersion=1 and sourceClassification.");
        }

        string referenceSha256 = ComputeSha256(bytes);
        string metadataDiagnostic = MetadataDiagnostic(reference, context, actualValues.Length);
        if (!string.IsNullOrEmpty(metadataDiagnostic))
        {
            return new ClassificationReferenceValidationResult(
                true, false, false, fullPath, referenceSha256, reference.SourceClassification, 0, 0, -1, 0.0f, 0.0f,
                absoluteTolerance, relativeTolerance, normalizedNaNPolicy, normalizedInfinityPolicy, metadataDiagnostic);
        }

        int mismatchCount = 0;
        int firstMismatchIndex = -1;
        float maximumAbsoluteError = 0.0f;
        float maximumRelativeError = 0.0f;
        for (int index = 0; index < actualValues.Length; index++)
        {
            bool matches = ValuesMatch(
                actualValues[index],
                reference.Values[index],
                absoluteTolerance,
                relativeTolerance,
                normalizedNaNPolicy,
                normalizedInfinityPolicy,
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
        return new ClassificationReferenceValidationResult(
            true, true, passed, fullPath, referenceSha256, reference.SourceClassification, actualValues.Length, mismatchCount,
            firstMismatchIndex, maximumAbsoluteError, maximumRelativeError, absoluteTolerance, relativeTolerance,
            normalizedNaNPolicy, normalizedInfinityPolicy, passed
                ? "all classification reference values matched"
                : $"{mismatchCount} value(s) exceeded tolerance or special-value policy; first mismatch index {firstMismatchIndex}");
    }

    public static string ComputeOutputTensorContractSha256(string tensorName, int[] shape, string valueKind)
    {
        string canonical = $"tensorName={tensorName};shape={string.Join(",", shape ?? Array.Empty<int>())};dataType=float32;valueKind={valueKind}";
        return ComputeSha256(Encoding.UTF8.GetBytes(canonical));
    }

    public static string ComputeTaskSemanticsSha256(ClassificationScoreTransform transform, int topK, string labelsSha256)
    {
        string canonical = $"task=classification;scoreTransform={transform.ToString().ToLowerInvariant()};topK={topK.ToString(CultureInfo.InvariantCulture)};argmaxRule=max-score-then-lowest-index;labelsSha256={labelsSha256}";
        return ComputeSha256(Encoding.UTF8.GetBytes(canonical));
    }

    public static string ComputeFileSha256OrEmpty(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            return string.Empty;
        }

        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    public static string ComputeFloatSha256(float[] values)
    {
        byte[] bytes = new byte[checked(values.Length * sizeof(float))];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return ComputeSha256(bytes);
    }

    private static string MetadataDiagnostic(ClassificationReferenceDocument reference, ClassificationReferenceContext context, int actualCount)
    {
        if (!string.Equals(reference.TensorName, context.TensorName, StringComparison.Ordinal)) return "reference tensorName does not match runtime output";
        if (reference.Shape == null) return "reference shape must not be null";
        if (!reference.Shape.SequenceEqual(context.Shape)) return "reference shape does not match runtime output shape";
        if (reference.Values == null) return "reference values must not be null";
        if (reference.Values.Length != actualCount) return "reference value count does not match runtime output element count";
        if (!string.Equals(reference.ValueKind, context.ValueKind, StringComparison.Ordinal)) return "reference valueKind does not match score transform";
        if (!string.Equals(reference.ModelSha256, context.ModelSha256, StringComparison.Ordinal)) return "reference modelSha256 does not match runtime model";
        if (!string.Equals(reference.InputTensorSha256, context.InputTensorSha256, StringComparison.Ordinal)) return "reference inputTensorSha256 does not match runtime input";
        if (!string.Equals(reference.PreprocessContractSha256, context.PreprocessContractSha256, StringComparison.Ordinal)) return "reference preprocessContractSha256 does not match runtime preprocessing";
        if (!string.Equals(reference.OutputTensorContractSha256, context.OutputTensorContractSha256, StringComparison.Ordinal)) return "reference outputTensorContractSha256 does not match runtime output contract";
        if (!string.Equals(reference.LabelsSha256, context.LabelsSha256, StringComparison.Ordinal)) return "reference labelsSha256 does not match runtime labels";
        if (!string.Equals(reference.TaskSemanticsSha256, context.TaskSemanticsSha256, StringComparison.Ordinal)) return "reference taskSemanticsSha256 does not match runtime task semantics";
        if (!AllSha256(reference)) return "reference provenance fingerprints must be 64-character SHA256 values";
        return string.Empty;
    }

    private static bool AllSha256(ClassificationReferenceDocument reference)
    {
        return IsSha256(reference.ModelSha256) && IsSha256(reference.InputTensorSha256) &&
            IsSha256(reference.PreprocessContractSha256) && IsSha256(reference.OutputTensorContractSha256) &&
            IsSha256(reference.LabelsSha256) && IsSha256(reference.TaskSemanticsSha256);
    }

    private static bool IsSha256(string value) => value != null && value.Length == 64 && value.All(static character =>
        (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'));

    private static bool ValuesMatch(
        float actual,
        float expected,
        float absoluteTolerance,
        float relativeTolerance,
        string nanPolicy,
        string infinityPolicy,
        out float absoluteError,
        out float relativeError)
    {
        if (float.IsNaN(actual) || float.IsNaN(expected))
        {
            bool both = float.IsNaN(actual) && float.IsNaN(expected);
            absoluteError = both ? 0.0f : float.MaxValue;
            relativeError = absoluteError;
            return both && string.Equals(nanPolicy, "equal", StringComparison.Ordinal);
        }
        if (float.IsInfinity(actual) || float.IsInfinity(expected))
        {
            bool exact = actual.Equals(expected);
            absoluteError = exact ? 0.0f : float.MaxValue;
            relativeError = absoluteError;
            return exact && string.Equals(infinityPolicy, "exact", StringComparison.Ordinal);
        }

        absoluteError = Math.Abs(actual - expected);
        float scale = Math.Max(Math.Abs(actual), Math.Abs(expected));
        relativeError = scale == 0.0f ? absoluteError : absoluteError / scale;
        return absoluteError <= absoluteTolerance || absoluteError <= relativeTolerance * scale;
    }

    private static void ValidateTolerance(float value, string name)
    {
        if (!float.IsFinite(value) || value < 0.0f)
        {
            throw new ArgumentOutOfRangeException(name, "Tolerance must be finite and non-negative.");
        }
    }

    private static string NormalizeNaNPolicy(string value)
    {
        string normalized = (value ?? string.Empty).Trim().ToLowerInvariant();
        return normalized switch
        {
            "" or "reject" => "reject",
            "equal" => "equal",
            _ => throw new ArgumentException("NaN policy must be reject or equal.", nameof(value))
        };
    }

    private static string NormalizeInfinityPolicy(string value)
    {
        string normalized = (value ?? string.Empty).Trim().ToLowerInvariant();
        return normalized switch
        {
            "" or "exact" => "exact",
            "reject" => "reject",
            _ => throw new ArgumentException("Infinity policy must be exact or reject.", nameof(value))
        };
    }

    private static string ComputeSha256(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    private sealed class ClassificationReferenceDocument
    {
        public int SchemaVersion { get; set; }
        public string TensorName { get; set; } = string.Empty;
        public int[] Shape { get; set; } = Array.Empty<int>();
        public float[] Values { get; set; } = Array.Empty<float>();
        public string ValueKind { get; set; } = string.Empty;
        public string ModelSha256 { get; set; } = string.Empty;
        public string InputTensorSha256 { get; set; } = string.Empty;
        public string PreprocessContractSha256 { get; set; } = string.Empty;
        public string OutputTensorContractSha256 { get; set; } = string.Empty;
        public string LabelsSha256 { get; set; } = string.Empty;
        public string TaskSemanticsSha256 { get; set; } = string.Empty;
        public string SourceClassification { get; set; } = string.Empty;
    }
}

internal static class ClassificationOutputReportWriter
{
    public const string SchemaVersion = "classification-output.v1";

    public static void Write(
        string path,
        OnnxSampleOptions options,
        OnnxSampleResult result,
        ClassificationImagePreprocessResult? preprocess,
        string labelsPath,
        string labelsSha256,
        int labelCount,
        ClassificationScoreTransform transform,
        float[] transformedValues,
        IReadOnlyList<ClassificationPrediction> topK,
        ClassificationReferenceContext referenceContext,
        ClassificationReferenceValidationResult validation)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return;
        }

        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        string inputSourceKind = preprocess != null
            ? "preprocessed-image-tensor"
            : options.UsesExternalInput ? "external-tensor" : "synthetic-pattern";
        string proofClassification = preprocess != null
            ? "real-input-reference-candidate-runtime"
            : options.UsesExternalInput ? "external-tensor-runtime" : "synthetic-input-runtime";
        bool validationRequested = validation.Requested || result.ReferenceValidation.Requested;
        bool validationPassed = (!validation.Requested || validation.Passed) &&
            (!result.ReferenceValidation.Requested || (result.ReferenceValidation.Completed && result.ReferenceValidation.Passed));
        var document = new
        {
            schemaVersion = SchemaVersion,
            task = "classification",
            success = validationPassed,
            inferenceRan = true,
            outputCaptured = true,
            outputValidated = validationRequested && validationPassed,
            proofClassification,
            model = new
            {
                path = options.ModelPath,
                sha256 = referenceContext.ModelSha256
            },
            labels = new
            {
                path = labelsPath ?? string.Empty,
                sha256 = labelsSha256,
                count = labelCount
            },
            input = new
            {
                sourceKind = inputSourceKind,
                tensorName = result.InputName,
                shape = result.InputShape.Values,
                tensorSha256 = referenceContext.InputTensorSha256,
                preprocessContractSha256 = referenceContext.PreprocessContractSha256,
                image = preprocess == null ? null : new
                {
                    sourcePath = preprocess.SourcePath,
                    sourceSha256 = preprocess.SourceSha256,
                    sourceWidth = preprocess.SourceWidth,
                    sourceHeight = preprocess.SourceHeight,
                    tensorPath = preprocess.TensorPath,
                    targetWidth = preprocess.TargetWidth,
                    targetHeight = preprocess.TargetHeight,
                    resizeMode = preprocess.Options.ResizeMode,
                    resizeShorterSide = preprocess.Options.ResizeShorterSide,
                    resizedWidth = preprocess.ResizedWidth,
                    resizedHeight = preprocess.ResizedHeight,
                    cropX = preprocess.CropX,
                    cropY = preprocess.CropY,
                    tensorLayout = preprocess.Options.TensorLayout,
                    colorOrder = preprocess.Options.ColorOrder,
                    scale = preprocess.Options.Scale,
                    mean = preprocess.Options.Mean,
                    standardDeviation = preprocess.Options.StandardDeviation
                }
            },
            inputTensors = result.Inputs.Select(static input => new
            {
                tensorName = input.Name,
                shape = input.Shape.Values,
                input.ElementCount,
                input.ByteLength,
                input.Preview,
                input.Sha256,
                input.SourceClassification,
                input.SourcePath
            }),
            output = new
            {
                tensorName = result.OutputName,
                shape = result.OutputShape.Values,
                elementCount = result.OutputValues.Length,
                valueKind = ClassificationOutputProcessor.ValueKind(transform),
                rawValues = result.OutputValues,
                rawSha256 = ClassificationOutputProcessor.ComputeFloatSha256(result.OutputValues),
                values = transformedValues,
                valueSha256 = ClassificationOutputProcessor.ComputeFloatSha256(transformedValues),
                outputTensorContractSha256 = referenceContext.OutputTensorContractSha256,
                taskSemanticsSha256 = referenceContext.TaskSemanticsSha256,
                topK
            },
            runtime = new
            {
                tensorRtLine = (int)result.Line,
                profileIndex = result.ProfileIndex,
                engineDeviceMemoryBytes = result.EngineDeviceMemoryBytes,
                elapsedMilliseconds = result.ElapsedMilliseconds,
                executionSummary = result.ExecutionSummary.ToString()
            },
            referenceValidation = validation,
            runtimeReferenceValidation = result.ReferenceValidation,
            boundary = new
            {
                isIndependentFrameworkGolden = false,
                ownerReviewedGolden = false,
                repositoryRedistributionApproved = false,
                isPackageConsumerRuntimeProof = false,
                isPublicPackageProof = false,
                isPostPublishProof = false,
                canPublishPublicly = false,
                canCloseReleaseIssue = false,
                statement = "This report captures Classification preprocessing, runtime output, and optional task-specific reference comparison. It is not an Owner-accepted golden, public-package, post-publish, or release proof."
            }
        };
        JsonSerializerOptions jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
        };
        File.WriteAllText(fullPath, JsonSerializer.Serialize(document, jsonOptions) + Environment.NewLine, new UTF8Encoding(false));
    }
}
