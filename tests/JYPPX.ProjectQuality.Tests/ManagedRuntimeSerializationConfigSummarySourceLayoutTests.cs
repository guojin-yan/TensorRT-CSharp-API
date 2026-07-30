using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedRuntimeSerializationConfigSummarySourceLayoutTests
{
    private const string RuntimeOriginalNormalizedSha256 =
        "da5c8921ce02c2fcd66c1aa3d2214c941b15e31c1edc7cb4c24c167a4932658d";
    private const string SerializationOriginalNormalizedSha256 =
        "91a5979d9a8f23239b24da43094572d978ec2aea1aa16ec9f3c8bb411351766a";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "Runtime",
            "TensorRtRuntimeConfig.cs",
            new[] { "TensorRtRuntimeConfig", "ToSummary", "Dispose" }
        },
        {
            "Runtime",
            "TensorRtRuntimeConfigSummary.cs",
            new[] { "TensorRtRuntimeConfigSummary", "ToString" }
        },
        {
            "Serialization",
            "TensorRtSerializationConfig.cs",
            new[] { "TensorRtSerializationConfig", "SetFlag", "ClearFlag", "GetFlag", "ToSummary", "Dispose" }
        },
        {
            "Serialization",
            "TensorRtSerializationConfigSummary.cs",
            new[] { "TensorRtSerializationConfigSummary", "ToString" }
        }
    };

    public static TheoryData<string, string, string[]> SummaryProperties => new()
    {
        {
            "Runtime",
            "TensorRtRuntimeConfigSummary.cs",
            new[]
            {
                "Line", "AllocationStrategy", "PointerFreeCopiedSummary", "CanPromoteRuntimeProof",
                "CanDeleteDeferredRecord"
            }
        },
        {
            "Serialization",
            "TensorRtSerializationConfigSummary.cs",
            new[]
            {
                "Line", "Flags", "PointerFreeCopiedSummary", "CanPromoteRuntimeProof", "CanDeleteDeferredRecord"
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
                     ("Runtime", "TensorRtRuntimeConfigSummary.cs", "TensorRtRuntimeConfigSummary"),
                     ("Serialization", "TensorRtSerializationConfigSummary.cs", "TensorRtSerializationConfigSummary")
                 })
        {
            Assert.Single(Regex.Matches(
                ReadSource(module, fileName),
                $@"^    internal {typeName}\(",
                RegexOptions.Multiline));
        }
    }

    [Fact]
    public void OwnersDoNotOwnSummaryDeclarations()
    {
        Assert.DoesNotContain(
            "public sealed class TensorRtRuntimeConfigSummary",
            ReadSource("Runtime", "TensorRtRuntimeConfig.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public sealed class TensorRtSerializationConfigSummary",
            ReadSource("Serialization", "TensorRtSerializationConfig.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedSummariesRemainPointerFreeAndNonProof()
    {
        foreach ((string module, string fileName) in new[]
                 {
                     ("Runtime", "TensorRtRuntimeConfigSummary.cs"),
                     ("Serialization", "TensorRtSerializationConfigSummary.cs")
                 })
        {
            string source = ReadSource(module, fileName);
            Assert.DoesNotContain("public IntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public UIntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public nint", source, StringComparison.Ordinal);
            Assert.Contains("PointerFreeCopiedSummary => true", source, StringComparison.Ordinal);
            Assert.Contains("CanPromoteRuntimeProof => false", source, StringComparison.Ordinal);
            Assert.Contains("CanDeleteDeferredRecord => false", source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ReaderAndConsumerUseBothCompleteSourceSets()
    {
        string reader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));
        foreach (string sourceFile in new[]
                 {
                     "TensorRtRuntimeConfig.cs",
                     "TensorRtRuntimeConfigSummary.cs",
                     "TensorRtSerializationConfig.cs",
                     "TensorRtSerializationConfigSummary.cs"
                 })
        {
            Assert.Contains(sourceFile, reader, StringComparison.Ordinal);
        }

        string consumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RuntimeSerializationOnnxSupportTests.cs"));
        Assert.Contains("RepositorySourceReader.Read", consumer, StringComparison.Ordinal);
    }

    [Fact]
    public void EvidenceMatrixAndSourceOrganizationReferenceDedicatedSummaries()
    {
        string matrix = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "readonly-summary-evidence-matrix.md"));
        string organization = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "source-organization.md"));

        Assert.Contains("TensorRtRuntimeConfigSummary.cs", matrix, StringComparison.Ordinal);
        Assert.Contains("TensorRtSerializationConfigSummary.cs", matrix, StringComparison.Ordinal);
        Assert.Contains("RuntimeConfig", organization, StringComparison.Ordinal);
        Assert.Contains("SerializationConfig summary", organization, StringComparison.Ordinal);
    }

    [Fact]
    public void RuntimeFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("Runtime", "TensorRtRuntimeConfig.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("Runtime", "TensorRtRuntimeConfigSummary.cs"));
        Assert.Equal(RuntimeOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void SerializationFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("Serialization", "TensorRtSerializationConfig.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("Serialization", "TensorRtSerializationConfigSummary.cs"));
        Assert.Equal(SerializationOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
