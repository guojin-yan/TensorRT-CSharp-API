using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedOnnxConfigModelSupportSourceLayoutTests
{
    private const string ConfigOriginalNormalizedSha256 =
        "137d359e84a49296caaf3df0860d3c424f7d58fe56b60e2024d2b2ea4440a33a";
    private const string ModelSupportOriginalNormalizedSha256 =
        "520cec9d6e6bc4ac58bf37eea8b93abb981b89e075b7b6945541a1dbfa14996d";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtOnnxConfig.cs",
            "TensorRtOnnxConfig",
            new[] { "IncreaseVerbosity", "DecreaseVerbosity", "ToSnapshot", "Dispose" }
        },
        {
            "TensorRtOnnxConfigSnapshot.cs",
            "TensorRtOnnxConfigSnapshot",
            new[] { "ToSummary", "ToString" }
        },
        {
            "TensorRtOnnxConfigSummary.cs",
            "TensorRtOnnxConfigSummary",
            new[] { "ToString" }
        },
        {
            "TensorRtOnnxModelSupportReport.cs",
            "TensorRtOnnxModelSupportReport",
            new[] { "ToSummary", "ToString" }
        },
        {
            "TensorRtOnnxModelSupportSummary.cs",
            "TensorRtOnnxModelSupportSummary",
            new[] { "ToString" }
        },
        {
            "TensorRtOnnxSubgraphSupportInfo.cs",
            "TensorRtOnnxSubgraphSupportInfo",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> PublicProperties => new()
    {
        {
            "TensorRtOnnxConfig.cs",
            new[]
            {
                "Line", "ModelDataType", "VerbosityLevel", "ModelFileName", "TextFileName",
                "FullTextFileName", "PrintLayerInfo"
            }
        },
        {
            "TensorRtOnnxConfigSnapshot.cs",
            new[]
            {
                "Line", "ModelDataType", "VerbosityLevel", "ModelFileName", "TextFileName",
                "FullTextFileName", "PrintLayerInfo"
            }
        },
        {
            "TensorRtOnnxConfigSummary.cs",
            new[]
            {
                "Line", "ModelDataType", "VerbosityLevel", "PrintLayerInfo", "HasModelFileName",
                "HasTextFileName", "HasFullTextFileName", "ModelFileNameLength", "TextFileNameLength",
                "FullTextFileNameLength"
            }
        },
        {
            "TensorRtOnnxModelSupportReport.cs",
            new[] { "IsSupported", "SupportedSubgraphCount", "UnsupportedSubgraphCount", "Subgraphs" }
        },
        {
            "TensorRtOnnxModelSupportSummary.cs",
            new[]
            {
                "IsSupported", "ReportedSupportedSubgraphCount", "ReportedUnsupportedSubgraphCount",
                "CopiedSubgraphCount", "CopiedSupportedSubgraphCount", "CopiedUnsupportedSubgraphCount",
                "CopiedNodeCount", "CopiedSubgraphCountsMatchReportedCounts", "RuntimeEvidenceKind",
                "IsRuntimeExecutionEvidence", "IsRuntimeExecutionProof", "PointerFreeCopiedSummary",
                "CanPromoteRuntimeProof", "CanPromoteReleaseProof", "CanDeleteDeferredRecord"
            }
        },
        {
            "TensorRtOnnxSubgraphSupportInfo.cs",
            new[] { "Index", "IsSupported", "Nodes" }
        }
    };

    public static TheoryData<string, string, string> ConstructorOwners => new()
    {
        { "TensorRtOnnxConfig.cs", "public", "TensorRtOnnxConfig" },
        { "TensorRtOnnxConfig.cs", "internal", "TensorRtOnnxConfig" },
        { "TensorRtOnnxConfigSnapshot.cs", "public", "TensorRtOnnxConfigSnapshot" },
        { "TensorRtOnnxConfigSummary.cs", "public", "TensorRtOnnxConfigSummary" },
        { "TensorRtOnnxModelSupportReport.cs", "public", "TensorRtOnnxModelSupportReport" },
        { "TensorRtOnnxModelSupportSummary.cs", "internal", "TensorRtOnnxModelSupportSummary" },
        { "TensorRtOnnxSubgraphSupportInfo.cs", "public", "TensorRtOnnxSubgraphSupportInfo" }
    };

    [Theory]
    [MemberData(nameof(FileOwnersAndMethods))]
    public void FilesOwnExactTopLevelTypesAndMethods(
        string fileName,
        string expectedType,
        string[] expectedMethods)
    {
        string source = ReadSource(fileName);
        Assert.Equal(new[] { expectedType }, EnumerateTopLevelTypeNames(source));
        Assert.Equal(expectedMethods, EnumerateMethodNames(source));
    }

    [Theory]
    [MemberData(nameof(PublicProperties))]
    public void FilesKeepExactPublicPropertyOrder(string fileName, string[] expectedProperties)
    {
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(ReadSource(fileName)));
    }

    [Theory]
    [MemberData(nameof(ConstructorOwners))]
    public void ConstructorsStayWithTheirNamesakeTypes(string fileName, string access, string typeName)
    {
        Assert.Single(Regex.Matches(
            ReadSource(fileName),
            $@"^    {access} {typeName}\(",
            RegexOptions.Multiline));
    }

    [Fact]
    public void PrimaryFilesDoNotRetainSecondaryTypeDeclarations()
    {
        string config = ReadSource("TensorRtOnnxConfig.cs");
        Assert.DoesNotContain("public sealed class TensorRtOnnxConfigSnapshot", config, StringComparison.Ordinal);
        Assert.DoesNotContain("public sealed class TensorRtOnnxConfigSummary", config, StringComparison.Ordinal);

        string report = ReadSource("TensorRtOnnxModelSupportReport.cs");
        Assert.DoesNotContain("public sealed class TensorRtOnnxModelSupportSummary", report, StringComparison.Ordinal);
        Assert.DoesNotContain("public sealed class TensorRtOnnxSubgraphSupportInfo", report, StringComparison.Ordinal);
    }

    [Fact]
    public void SplitTypesRemainPointerFreeAndKeepNonProofClassification()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtOnnxConfig.cs",
                     "TensorRtOnnxConfigSnapshot.cs",
                     "TensorRtOnnxConfigSummary.cs",
                     "TensorRtOnnxModelSupportReport.cs",
                     "TensorRtOnnxModelSupportSummary.cs",
                     "TensorRtOnnxSubgraphSupportInfo.cs"
                 })
        {
            string source = ReadSource(fileName);
            Assert.DoesNotContain("public IntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public UIntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public nint", source, StringComparison.Ordinal);
        }

        string report = ReadSource("TensorRtOnnxModelSupportReport.cs");
        Assert.Contains("does not promote", report, StringComparison.Ordinal);
        Assert.Contains("runtime execution or package-consumer proof", report, StringComparison.Ordinal);

        string summary = ReadSource("TensorRtOnnxModelSupportSummary.cs");
        Assert.Contains("RuntimeEvidenceKind => \"copied-readonly-summary\"", summary, StringComparison.Ordinal);
        Assert.Contains("IsRuntimeExecutionProof => false", summary, StringComparison.Ordinal);
        Assert.Contains("CanPromoteRuntimeProof => false", summary, StringComparison.Ordinal);
        Assert.Contains("CanPromoteReleaseProof => false", summary, StringComparison.Ordinal);
        Assert.Contains("CanDeleteDeferredRecord => false", summary, StringComparison.Ordinal);
    }

    [Fact]
    public void ReaderKeepsBothThreeFileSourceSetsInOriginalOrder()
    {
        string reader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));

        AssertInOrder(
            reader,
            "TensorRtOnnxConfig.cs",
            "TensorRtOnnxConfigSnapshot.cs",
            "TensorRtOnnxConfigSummary.cs");
        AssertInOrder(
            reader,
            "TensorRtOnnxModelSupportReport.cs",
            "TensorRtOnnxModelSupportSummary.cs",
            "TensorRtOnnxSubgraphSupportInfo.cs");
    }

    [Fact]
    public void DirectConsumersUseTheUnifiedSourceReader()
    {
        foreach (string consumerName in new[]
                 {
                     "OnnxConfigLifecycleCoverageConvergenceTests.cs",
                     "OnnxConfigScalarControlsTests.cs",
                     "OnnxConfigSnapshotSummaryTests.cs",
                     "DeferredReadonlyUpliftBatchTests.cs",
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
    public void DocumentationReferencesDedicatedOnnxConfigAndSupportTypes()
    {
        foreach (string organizationPath in new[]
                 {
                     Path.Combine("docs", "articles", "en", "source-organization.md"),
                     Path.Combine("docs", "articles", "zh-cn", "source-organization.md")
                 })
        {
            string organization = File.ReadAllText(Path.Combine(RepositoryPaths.Root, organizationPath));
            Assert.Contains("TensorRtOnnxConfigSnapshot.cs", organization, StringComparison.Ordinal);
            Assert.Contains("TensorRtOnnxConfigSummary.cs", organization, StringComparison.Ordinal);
            Assert.Contains("TensorRtOnnxModelSupportSummary.cs", organization, StringComparison.Ordinal);
            Assert.Contains("TensorRtOnnxSubgraphSupportInfo.cs", organization, StringComparison.Ordinal);
        }

        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "blog-onnx-parser-engine-roundtrip.md"));
        Assert.Contains("TensorRtOnnxModelSupportSummary.cs", article, StringComparison.Ordinal);
        Assert.Contains("TensorRtOnnxSubgraphSupportInfo.cs", article, StringComparison.Ordinal);
        Assert.Contains("不构成 runtime 或 release proof", article, StringComparison.Ordinal);
    }

    [Fact]
    public void ConfigFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtOnnxConfig.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtOnnxConfigSnapshot.cs"));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtOnnxConfigSummary.cs"));
        Assert.Equal(ConfigOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void ModelSupportFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtOnnxModelSupportReport.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtOnnxModelSupportSummary.cs"));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtOnnxSubgraphSupportInfo.cs"));
        Assert.Equal(ModelSupportOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string[] EnumerateTopLevelTypeNames(string source)
    {
        return Regex.Matches(
                Normalize(source),
                @"^public sealed class (?<name>[A-Za-z_][A-Za-z0-9_]*)(?: : IDisposable)?$",
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

    private static void AssertInOrder(string source, params string[] markers)
    {
        int previous = -1;
        foreach (string marker in markers)
        {
            int current = source.IndexOf(marker, previous + 1, StringComparison.Ordinal);
            Assert.True(current > previous, $"Expected marker in source-set order: {marker}");
            previous = current;
        }
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
