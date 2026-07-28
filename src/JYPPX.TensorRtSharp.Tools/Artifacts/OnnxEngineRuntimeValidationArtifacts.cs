using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json.Serialization;

namespace JYPPX.TensorRtSharp.Tools;

public static class OnnxEngineReferenceValueComparer
{
    public static bool Matches(
        float actual,
        float expected,
        float absoluteTolerance,
        float relativeTolerance,
        TrtexecLikeReferenceNaNPolicy nanPolicy,
        TrtexecLikeReferenceInfinityPolicy infinityPolicy,
        out float absoluteError,
        out float relativeError)
    {
        if (!float.IsFinite(absoluteTolerance) || absoluteTolerance < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(absoluteTolerance), "Absolute tolerance must be finite and non-negative.");
        }
        if (!float.IsFinite(relativeTolerance) || relativeTolerance < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(relativeTolerance), "Relative tolerance must be finite and non-negative.");
        }

        if (float.IsNaN(actual) || float.IsNaN(expected))
        {
            bool bothNaN = float.IsNaN(actual) && float.IsNaN(expected);
            absoluteError = bothNaN ? 0.0f : float.MaxValue;
            relativeError = absoluteError;
            return nanPolicy == TrtexecLikeReferenceNaNPolicy.Equal && bothNaN;
        }

        if (float.IsInfinity(actual) || float.IsInfinity(expected))
        {
            bool exact = actual.Equals(expected);
            absoluteError = exact ? 0.0f : float.MaxValue;
            relativeError = absoluteError;
            return infinityPolicy == TrtexecLikeReferenceInfinityPolicy.Exact && exact;
        }

        absoluteError = Math.Abs(actual - expected);
        float scale = Math.Max(Math.Abs(actual), Math.Abs(expected));
        relativeError = scale == 0.0f ? absoluteError : absoluteError / scale;
        return absoluteError <= absoluteTolerance || absoluteError <= relativeTolerance * scale;
    }
}

public static class OnnxEngineReferenceTensorComparer
{
    public static bool MetadataMatches(
        string actualTensorName,
        IReadOnlyList<int> actualShape,
        int actualElementCount,
        OnnxEngineReferenceTensorData reference,
        out string diagnostic)
    {
        if (reference == null)
        {
            throw new ArgumentNullException(nameof(reference));
        }
        if (!string.Equals(reference.TensorName, actualTensorName, StringComparison.Ordinal))
        {
            diagnostic = "reference tensorName does not match engine output name";
            return false;
        }
        if (!(actualShape ?? Array.Empty<int>()).SequenceEqual(reference.Shape))
        {
            diagnostic = "reference shape does not match runtime output shape";
            return false;
        }
        if (reference.Values.Count != actualElementCount)
        {
            diagnostic = "reference value count does not match runtime output element count";
            return false;
        }

        diagnostic = string.Empty;
        return true;
    }
}

public sealed class OnnxEngineReferenceTensorData
{
    [JsonConstructor]
    public OnnxEngineReferenceTensorData(
        int schemaVersion,
        string tensorName,
        IReadOnlyList<int> shape,
        IReadOnlyList<float> values,
        string sourceClassification)
    {
        SchemaVersion = schemaVersion;
        TensorName = tensorName ?? string.Empty;
        Shape = shape?.ToArray() ?? Array.Empty<int>();
        Values = values?.ToArray() ?? Array.Empty<float>();
        SourceClassification = sourceClassification ?? string.Empty;
    }

    public int SchemaVersion { get; }

    public string TensorName { get; }

    public IReadOnlyList<int> Shape { get; }

    public IReadOnlyList<float> Values { get; }

    public string SourceClassification { get; }

    [JsonIgnore]
    internal string FileSha256 { get; set; } = string.Empty;
}

public sealed class OnnxEngineRuntimeInputArtifact
{
    public OnnxEngineRuntimeInputArtifact(
        string tensorName,
        IReadOnlyList<int> shape,
        IReadOnlyList<float> values,
        string sourceClassification,
        string sourcePath)
        : this(
            tensorName,
            shape,
            values?.Count ?? 0,
            values == null ? Array.Empty<float>() : values.Take(8).ToArray(),
            ToBytes(values),
            sourceClassification,
            sourcePath)
    {
    }

    internal OnnxEngineRuntimeInputArtifact(
        string tensorName,
        IReadOnlyList<int> shape,
        int elementCount,
        IReadOnlyList<float> preview,
        byte[] bytes,
        string sourceClassification,
        string sourcePath)
    {
        TensorName = tensorName ?? string.Empty;
        Shape = shape?.ToArray() ?? Array.Empty<int>();
        ElementCount = Math.Max(0, elementCount);
        Preview = preview?.Take(8).ToArray() ?? Array.Empty<float>();
        ByteLength = bytes?.LongLength ?? 0;
        Sha256 = bytes == null || bytes.Length == 0
            ? string.Empty
            : Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        SourceClassification = sourceClassification ?? string.Empty;
        SourcePath = sourcePath ?? string.Empty;
    }

    public string TensorName { get; }

    public IReadOnlyList<int> Shape { get; }

    public int ElementCount { get; }

    public long ByteLength { get; }

    public IReadOnlyList<float> Preview { get; }

    public string Sha256 { get; }

    public string SourceClassification { get; }

    public string SourcePath { get; }

    private static byte[] ToBytes(IReadOnlyList<float>? values)
    {
        if (values == null || values.Count == 0)
        {
            return Array.Empty<byte>();
        }

        float[] copy = values.ToArray();
        byte[] bytes = new byte[checked(copy.Length * sizeof(float))];
        Buffer.BlockCopy(copy, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}

public sealed class OnnxEngineReferenceTensorComparisonArtifact
{
    public OnnxEngineReferenceTensorComparisonArtifact(
        string tensorName,
        string referencePath,
        string referenceSha256,
        string referenceSourceClassification,
        IReadOnlyList<int> actualShape,
        IReadOnlyList<int> referenceShape,
        int actualElementCount,
        int referenceElementCount,
        int comparedElementCount,
        int mismatchCount,
        int firstMismatchIndex,
        float maximumAbsoluteError,
        float maximumRelativeError,
        bool passed,
        string diagnostic)
    {
        TensorName = tensorName ?? string.Empty;
        ReferencePath = referencePath ?? string.Empty;
        ReferenceSha256 = referenceSha256 ?? string.Empty;
        ReferenceSourceClassification = referenceSourceClassification ?? string.Empty;
        ActualShape = actualShape?.ToArray() ?? Array.Empty<int>();
        ReferenceShape = referenceShape?.ToArray() ?? Array.Empty<int>();
        ActualElementCount = Math.Max(0, actualElementCount);
        ReferenceElementCount = Math.Max(0, referenceElementCount);
        ComparedElementCount = Math.Max(0, comparedElementCount);
        MismatchCount = Math.Max(0, mismatchCount);
        FirstMismatchIndex = firstMismatchIndex;
        MaximumAbsoluteError = maximumAbsoluteError;
        MaximumRelativeError = maximumRelativeError;
        Passed = passed;
        Diagnostic = diagnostic ?? string.Empty;
    }

    public string TensorName { get; }

    public string ReferencePath { get; }

    public string ReferenceSha256 { get; }

    public string ReferenceSourceClassification { get; }

    public IReadOnlyList<int> ActualShape { get; }

    public IReadOnlyList<int> ReferenceShape { get; }

    public int ActualElementCount { get; }

    public int ReferenceElementCount { get; }

    public int ComparedElementCount { get; }

    public int MismatchCount { get; }

    public int FirstMismatchIndex { get; }

    public float MaximumAbsoluteError { get; }

    public float MaximumRelativeError { get; }

    public bool Passed { get; }

    public string Diagnostic { get; }
}

public sealed class OnnxEngineReferenceValidationArtifact
{
    public OnnxEngineReferenceValidationArtifact(
        bool requested,
        bool completed,
        bool passed,
        float absoluteTolerance,
        float relativeTolerance,
        string nanPolicy,
        string infinityPolicy,
        IReadOnlyList<string> diagnostics,
        IReadOnlyList<OnnxEngineReferenceTensorComparisonArtifact> tensorComparisons)
    {
        Requested = requested;
        Completed = completed;
        Passed = passed;
        AbsoluteTolerance = absoluteTolerance;
        RelativeTolerance = relativeTolerance;
        NaNPolicy = nanPolicy ?? string.Empty;
        InfinityPolicy = infinityPolicy ?? string.Empty;
        Diagnostics = diagnostics?.ToArray() ?? Array.Empty<string>();
        TensorComparisons = tensorComparisons?.ToArray() ?? Array.Empty<OnnxEngineReferenceTensorComparisonArtifact>();
    }

    public static OnnxEngineReferenceValidationArtifact NotRequested { get; } = new OnnxEngineReferenceValidationArtifact(
        requested: false,
        completed: false,
        passed: false,
        absoluteTolerance: 0.0f,
        relativeTolerance: 0.0f,
        nanPolicy: "reject",
        infinityPolicy: "exact",
        diagnostics: Array.Empty<string>(),
        tensorComparisons: Array.Empty<OnnxEngineReferenceTensorComparisonArtifact>());

    public bool Requested { get; }

    public bool Completed { get; }

    public bool Passed { get; }

    public float AbsoluteTolerance { get; }

    public float RelativeTolerance { get; }

    public string NaNPolicy { get; }

    public string InfinityPolicy { get; }

    public IReadOnlyList<string> Diagnostics { get; }

    public IReadOnlyList<OnnxEngineReferenceTensorComparisonArtifact> TensorComparisons { get; }
}
