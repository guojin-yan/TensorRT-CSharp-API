using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedOnnxParserDiagnosticSummarySourceLayoutTests
{
    private const string ParserOriginalNormalizedSha256 =
        "9239ea696671216091e72abcc5f3606c5be7e4021033d312cfb4ddb39cc8a2c9";
    private const string RefitterOriginalNormalizedSha256 =
        "0231f71dfc55d28c26d09ff152010165bf2b9d51214b9b1b1533d87e25f58e1a";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtOnnxParserDiagnosticSnapshot.cs",
            "TensorRtOnnxParserDiagnosticSnapshot",
            new[] { "ToSummary", "ToString" }
        },
        {
            "TensorRtOnnxParserDiagnosticSummary.cs",
            "TensorRtOnnxParserDiagnosticSummary",
            new[] { "ToString" }
        },
        {
            "TensorRtOnnxParserRefitterDiagnosticSnapshot.cs",
            "TensorRtOnnxParserRefitterDiagnosticSnapshot",
            new[] { "ToSummary", "ToString" }
        },
        {
            "TensorRtOnnxParserRefitterDiagnosticSummary.cs",
            "TensorRtOnnxParserRefitterDiagnosticSummary",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> SummaryProperties => new()
    {
        {
            "TensorRtOnnxParserDiagnosticSummary.cs",
            new[]
            {
                "Line", "ErrorCount", "CopiedDiagnosticCount", "DiagnosticSummaryLength", "RuntimeEvidenceKind",
                "IsRuntimeExecutionEvidence", "IsRuntimeExecutionProof", "PointerFreeCopiedSummary",
                "CanPromoteRuntimeProof", "CanPromoteReleaseProof", "CanDeleteDeferredRecord",
                "UsedVCPluginLibraryCount", "IdentityOperatorSupported"
            }
        },
        {
            "TensorRtOnnxParserRefitterDiagnosticSummary.cs",
            new[]
            {
                "Line", "ErrorCount", "CopiedDiagnosticCount", "DiagnosticSummaryLength", "RuntimeEvidenceKind",
                "IsRuntimeExecutionEvidence", "IsRuntimeExecutionProof", "PointerFreeCopiedSummary",
                "CanPromoteRuntimeProof", "CanPromoteReleaseProof", "CanDeleteDeferredRecord"
            }
        }
    };

    [Theory]
    [MemberData(nameof(FileOwnersAndMethods))]
    public void FilesOwnExactTopLevelTypesAndMethods(string fileName, string expectedType, string[] expectedMethods)
    {
        string source = ReadSource(fileName);
        Assert.Equal(new[] { expectedType }, EnumerateTopLevelTypeNames(source));
        Assert.Equal(expectedMethods, EnumerateMethodNames(source));
    }

    [Theory]
    [MemberData(nameof(SummaryProperties))]
    public void SummariesKeepExactPublicProperties(string fileName, string[] expectedProperties)
    {
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(ReadSource(fileName)));
    }

    [Fact]
    public void SummariesKeepOnePublicConstructorEach()
    {
        foreach ((string fileName, string typeName) in new[]
                 {
                     ("TensorRtOnnxParserDiagnosticSummary.cs", "TensorRtOnnxParserDiagnosticSummary"),
                     ("TensorRtOnnxParserRefitterDiagnosticSummary.cs", "TensorRtOnnxParserRefitterDiagnosticSummary")
                 })
        {
            Assert.Single(Regex.Matches(
                ReadSource(fileName),
                $@"^    public {typeName}\(",
                RegexOptions.Multiline));
        }
    }

    [Fact]
    public void SnapshotsDoNotOwnSummaryDeclarations()
    {
        Assert.DoesNotContain(
            "public sealed class TensorRtOnnxParserDiagnosticSummary",
            ReadSource("TensorRtOnnxParserDiagnosticSnapshot.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public sealed class TensorRtOnnxParserRefitterDiagnosticSummary",
            ReadSource("TensorRtOnnxParserRefitterDiagnosticSnapshot.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedSummariesRemainPointerFreeAndNonProof()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtOnnxParserDiagnosticSummary.cs",
                     "TensorRtOnnxParserRefitterDiagnosticSummary.cs"
                 })
        {
            string source = ReadSource(fileName);
            Assert.DoesNotContain("public IntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public UIntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public nint", source, StringComparison.Ordinal);
            Assert.Contains("RuntimeEvidenceKind => \"copied-readonly-summary\"", source, StringComparison.Ordinal);
            Assert.Contains("IsRuntimeExecutionProof => false", source, StringComparison.Ordinal);
            Assert.Contains("CanPromoteRuntimeProof => false", source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ReaderAndConsumersUseBothCompleteSourceSets()
    {
        string reader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));
        foreach (string sourceFile in new[]
                 {
                     "TensorRtOnnxParserDiagnosticSnapshot.cs",
                     "TensorRtOnnxParserDiagnosticSummary.cs",
                     "TensorRtOnnxParserRefitterDiagnosticSnapshot.cs",
                     "TensorRtOnnxParserRefitterDiagnosticSummary.cs"
                 })
        {
            Assert.Contains(sourceFile, reader, StringComparison.Ordinal);
        }

        foreach (string consumerName in new[]
                 {
                     "DeferredBTier41To45ProofClosureTests.cs",
                     "DeferredReadonlyUpliftBatchTests.cs",
                     "ParserRefitterBoundaryTests.cs",
                     "RuntimeSerializationOnnxSupportTests.cs"
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
    public void OnnxArticleAndSourceOrganizationReferenceDedicatedSummaries()
    {
        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "blog-onnx-parser-engine-roundtrip.md"));
        string organization = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "source-organization.md"));

        foreach (string summaryFile in new[]
                 {
                     "TensorRtOnnxParserDiagnosticSummary.cs",
                     "TensorRtOnnxParserRefitterDiagnosticSummary.cs"
                 })
        {
            Assert.Contains(summaryFile, article, StringComparison.Ordinal);
        }
        Assert.Contains("ONNX parser", organization, StringComparison.Ordinal);
        Assert.Contains("ParserRefitter diagnostic summary", organization, StringComparison.Ordinal);
    }

    [Fact]
    public void ParserFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtOnnxParserDiagnosticSnapshot.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtOnnxParserDiagnosticSummary.cs"));
        Assert.Equal(ParserOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void RefitterFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtOnnxParserRefitterDiagnosticSnapshot.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtOnnxParserRefitterDiagnosticSummary.cs"));
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

    private static string ReadTopLevelSegment(string fileName)
    {
        string source = Normalize(ReadSource(fileName));
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

    private static string ReadSource(string fileName)
    {
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Parsing",
            fileName));
    }
}
