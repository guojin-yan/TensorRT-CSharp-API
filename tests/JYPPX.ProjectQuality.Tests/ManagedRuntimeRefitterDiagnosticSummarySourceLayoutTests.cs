using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedRuntimeRefitterDiagnosticSummarySourceLayoutTests
{
    private const string RuntimeOriginalNormalizedSha256 =
        "436e702ad3182c377999a01358d18e5eb6a6b2b57b3497e3d332344fa3eb65dc";
    private const string RefitterOriginalNormalizedSha256 =
        "615ad27b6336d49c6313ff3bb9e60ade2f9445efc2362cfce7145cb32ca429bd";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "Runtime",
            "TensorRtRuntimeDiagnosticSnapshot.cs",
            new[] { "TensorRtRuntimeDiagnosticSnapshot", "ToSummary", "ToString" }
        },
        {
            "Runtime",
            "TensorRtRuntimeDiagnosticSummary.cs",
            new[] { "TensorRtRuntimeDiagnosticSummary", "ToString" }
        },
        {
            "Refit",
            "TensorRtRefitterDiagnosticSnapshot.cs",
            new[] { "TensorRtRefitterDiagnosticSnapshot", "ToSummary", "ToString" }
        },
        {
            "Refit",
            "TensorRtRefitterDiagnosticSummary.cs",
            new[] { "TensorRtRefitterDiagnosticSummary", "ToString" }
        }
    };

    public static TheoryData<string, string, string[]> SummaryProperties => new()
    {
        {
            "Runtime",
            "TensorRtRuntimeDiagnosticSummary.cs",
            new[]
            {
                "Line", "DlaCore", "DlaCoreCount", "MaxThreads", "EngineHostCodeAllowed",
                "TempfileControlFlags", "HasTemporaryDirectory", "HasLogger", "HasErrorRecorder", "ErrorCount",
                "CopiedErrorRecordCount", "HasErrorOverflowed", "DiagnosticCount", "RuntimeEvidenceKind",
                "IsRuntimeExecutionEvidence", "IsRuntimeExecutionProof", "PointerFreeCopiedSummary",
                "CanPromoteRuntimeProof", "CanPromoteReleaseProof", "CanDeleteDeferredRecord"
            }
        },
        {
            "Refit",
            "TensorRtRefitterDiagnosticSummary.cs",
            new[]
            {
                "Line", "MaxThreads", "WeightsValidation", "HasLogger", "HasErrorRecorder", "ErrorCount",
                "CopiedErrorRecordCount", "HasErrorOverflowed", "DynamicRangeTensorCount",
                "CopiedDynamicRangeTensorNameCount", "MissingNamedWeightCount", "CopiedMissingNamedWeightCount",
                "AllNamedWeightCount", "CopiedAllNamedWeightCount", "DiagnosticCount"
            }
        }
    };

    [Theory]
    [MemberData(nameof(FileOwnersAndMethods))]
    public void FilesOwnExactTopLevelTypesAndMethods(string module, string fileName, string[] expected)
    {
        string source = ReadSource(module, fileName);
        Assert.Equal(new[] { expected[0] }, EnumerateTopLevelTypeNames(source));
        Assert.Equal(expected.Skip(1), EnumerateMethodNames(source));
    }

    [Theory]
    [MemberData(nameof(SummaryProperties))]
    public void SummariesKeepExactPublicProperties(string module, string fileName, string[] expectedProperties)
    {
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(ReadSource(module, fileName)));
    }

    [Fact]
    public void SummariesKeepOneInternalConstructorEach()
    {
        foreach ((string module, string fileName, string typeName) in new[]
                 {
                     ("Runtime", "TensorRtRuntimeDiagnosticSummary.cs", "TensorRtRuntimeDiagnosticSummary"),
                     ("Refit", "TensorRtRefitterDiagnosticSummary.cs", "TensorRtRefitterDiagnosticSummary")
                 })
        {
            Assert.Single(Regex.Matches(
                ReadSource(module, fileName),
                $@"^    internal {typeName}\(",
                RegexOptions.Multiline));
        }
    }

    [Fact]
    public void SnapshotsDoNotOwnSummaryDeclarations()
    {
        Assert.DoesNotContain(
            "public sealed class TensorRtRuntimeDiagnosticSummary",
            ReadSource("Runtime", "TensorRtRuntimeDiagnosticSnapshot.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public sealed class TensorRtRefitterDiagnosticSummary",
            ReadSource("Refit", "TensorRtRefitterDiagnosticSnapshot.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedSummariesRemainPointerFree()
    {
        foreach ((string module, string fileName) in new[]
                 {
                     ("Runtime", "TensorRtRuntimeDiagnosticSummary.cs"),
                     ("Refit", "TensorRtRefitterDiagnosticSummary.cs")
                 })
        {
            string source = ReadSource(module, fileName);
            Assert.DoesNotContain("public IntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public UIntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public nint", source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ReaderCandidatesAndConsumersUseBothCompleteSourceSets()
    {
        string reader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));
        foreach (string sourceFile in new[]
                 {
                     "TensorRtRuntimeDiagnosticSnapshot.cs",
                     "TensorRtRuntimeDiagnosticSummary.cs",
                     "TensorRtRefitterDiagnosticSnapshot.cs",
                     "TensorRtRefitterDiagnosticSummary.cs"
                 })
        {
            Assert.Contains(sourceFile, reader, StringComparison.Ordinal);
        }

        string candidates = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "deferred-readonly-candidate-list.json"));
        Assert.Contains("TensorRtRuntimeDiagnosticSummary.cs", candidates, StringComparison.Ordinal);

        foreach (string consumerName in new[]
                 {
                     "CallbackAllocatorBoundaryTests.cs",
                     "DeferredReadonlyUpliftBatchTests.cs",
                     "ReadonlyDiagnosticsCandidateImplementationEvidenceTests.cs",
                     "RefitterEngineInspectorDiagnosticsTests.cs"
                 })
        {
            string consumer = File.ReadAllText(Path.Combine(
                RepositoryPaths.Root,
                "tests",
                "JYPPX.ProjectQuality.Tests",
                consumerName));
            Assert.Contains("RepositorySourceReader.Read", consumer, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void SourceOrganizationReferencesDedicatedSummaryOwners()
    {
        foreach (string relativePath in new[]
                 {
                     Path.Combine("docs", "articles", "en", "source-organization.md"),
                     Path.Combine("docs", "articles", "zh-cn", "source-organization.md"),
                     Path.Combine("docs", "articles", "zh-cn", "windows-api-completion.md")
                 })
        {
            string docs = File.ReadAllText(Path.Combine(RepositoryPaths.Root, relativePath));
            Assert.Contains("TensorRtRuntimeDiagnosticSummary", docs, StringComparison.Ordinal);
            Assert.Contains("TensorRtRefitterDiagnosticSummary", docs, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void RuntimeFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "Runtime",
            "TensorRtRuntimeDiagnosticSnapshot.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("Runtime", "TensorRtRuntimeDiagnosticSummary.cs"));
        Assert.Equal(RuntimeOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void RefitterFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "Refit",
            "TensorRtRefitterDiagnosticSnapshot.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("Refit", "TensorRtRefitterDiagnosticSummary.cs"));
        Assert.Equal(RefitterOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string[] EnumerateTopLevelTypeNames(string source)
    {
        return Regex.Matches(
                Normalize(source),
                @"^public sealed class (?<name>[A-Za-z_][A-Za-z0-9_]*)$",
                RegexOptions.Multiline)
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumerateMethodNames(string source)
    {
        return Regex.Matches(
                source,
                @"^\s*(?:public|private|internal)\s+(?!(?:delegate)\b)(?:static\s+)?[^\r\n=]+?\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
                RegexOptions.Multiline)
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumeratePublicPropertyNames(string source)
    {
        return Regex.Matches(
                source,
                @"^\s*public\s+(?:static\s+)?[^\s(]+\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:=>|\{)",
                RegexOptions.Multiline)
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string ReadTopLevelSegment(string module, string fileName)
    {
        string source = Normalize(ReadSource(module, fileName));
        int start = source.IndexOf("/// <summary>", StringComparison.Ordinal);
        Assert.True(start >= 0);
        return source[start..];
    }

    private static string ComputeSha256(string value)
    {
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(value)))
            .ToLowerInvariant();
    }

    private static string Normalize(string value)
    {
        return value.Replace("\r\n", "\n", StringComparison.Ordinal)
            .Replace('\r', '\n');
    }

    private static string ReadSource(string module, string fileName)
    {
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            module,
            fileName));
    }
}
