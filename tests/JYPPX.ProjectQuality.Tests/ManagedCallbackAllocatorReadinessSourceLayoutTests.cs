using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedCallbackAllocatorReadinessSourceLayoutTests
{
    private const string OriginalNormalizedSha256 =
        "50be00404eaf752c7c78c39e79d231a43f7334895024e755a8d0bc543b99d064";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtCallbackAllocatorReadiness.cs",
            "TensorRtCallbackAllocatorReadiness",
            new[] { "Evaluate", "AddRange" }
        },
        {
            "TensorRtCallbackAllocatorReadinessSnapshot.cs",
            "TensorRtCallbackAllocatorReadinessSnapshot",
            new[] { "ToString" }
        }
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

    [Fact]
    public void SnapshotKeepsExactPublicProperties()
    {
        string[] expectedProperties =
        {
            "EvidenceKind", "RuntimeEvidenceKind", "RealCallbackRuntime", "IsRealCallbackRuntimeProof",
            "LoggerCallbackReady", "ProfilerCallbackReady", "ProgressMonitorCallbackReady",
            "AllocatorOwnerDryRunReady", "AllocatorLedgerSafetyGateReady", "OutputAllocatorOwnerDesignReady",
            "OutputAllocatorRuntimeGateReady", "DebugListenerOwnerDesignReady",
            "DebugListenerNoThrowVTableGateReady", "DebugListenerRuntimeProofPrecheckReady",
            "RealCallbackInvocationProofReady", "IsPublishSafeForManagedCallbacks",
            "IsRuntimeInvocationProofComplete", "RuntimeProofBlocked", "BlockedPrerequisites",
            "BlockedReasonCount", "Status", "Summary"
        };

        Assert.Equal(
            expectedProperties,
            EnumeratePublicPropertyNames(ReadSource("TensorRtCallbackAllocatorReadinessSnapshot.cs")));
    }

    [Fact]
    public void SnapshotKeepsOneInternalConstructor()
    {
        Assert.Single(Regex.Matches(
            ReadSource("TensorRtCallbackAllocatorReadinessSnapshot.cs"),
            @"^    internal TensorRtCallbackAllocatorReadinessSnapshot\(",
            RegexOptions.Multiline));
    }

    [Fact]
    public void EvaluatorDoesNotOwnSnapshotDeclaration()
    {
        Assert.DoesNotContain(
            "public sealed class TensorRtCallbackAllocatorReadinessSnapshot",
            ReadSource("TensorRtCallbackAllocatorReadiness.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedSnapshotRemainsPointerFree()
    {
        string source = ReadSource("TensorRtCallbackAllocatorReadinessSnapshot.cs");
        Assert.DoesNotContain("public IntPtr", source, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", source, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", source, StringComparison.Ordinal);
    }

    [Fact]
    public void SourceReaderAndDirectConsumerUseTheCompleteSourceSet()
    {
        string testReader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));
        string consumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "CallbackAllocatorReadinessSnapshotTests.cs"));

        Assert.Contains("TensorRtCallbackAllocatorReadiness.cs", testReader, StringComparison.Ordinal);
        Assert.Contains("TensorRtCallbackAllocatorReadinessSnapshot.cs", testReader, StringComparison.Ordinal);
        Assert.Contains("RepositorySourceReader.Read", consumer, StringComparison.Ordinal);
    }

    [Fact]
    public void DocsReferenceDedicatedSnapshotOwner()
    {
        string guide = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "callback-allocator-boundary-guide.md"));

        Assert.Contains("TensorRtCallbackAllocatorReadiness.cs", guide, StringComparison.Ordinal);
        Assert.Contains("TensorRtCallbackAllocatorReadinessSnapshot.cs", guide, StringComparison.Ordinal);
    }

    [Fact]
    public void SplitFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtCallbackAllocatorReadiness.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtCallbackAllocatorReadinessSnapshot.cs"));
        Assert.Equal(OriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string[] EnumerateTopLevelTypeNames(string source)
    {
        return Regex.Matches(
                Normalize(source),
                @"^public (?:static class|sealed class) (?<name>[A-Za-z_][A-Za-z0-9_]*)$",
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
            "Callbacks",
            "MemoryAllocation",
            fileName));
    }
}
