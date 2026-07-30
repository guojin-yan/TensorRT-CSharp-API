using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedCallbackMetadataDesignGateSourceLayoutTests
{
    private const string AlgorithmGateOriginalNormalizedSha256 =
        "e622a6d3c2a77e78d59cb25fb36d50ac9bacf5001bb71f2051086a57ba9e4667";
    private const string AllocatorInterfaceGateOriginalNormalizedSha256 =
        "5e6fbe72e3bdb90ba6ce14904f2d2e4274f294c203d7f93b853d74931245274a";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtAlgorithmSnapshotDesignGate.cs",
            "TensorRtAlgorithmSnapshotDesignGate",
            new[] { "EvaluateKnownSurface", "Evaluate" }
        },
        {
            "TensorRtAlgorithmSnapshotDesignGateResult.cs",
            "TensorRtAlgorithmSnapshotDesignGateResult",
            new[] { "ToString" }
        },
        {
            "TensorRtAllocatorInterfaceInfoDesignGate.cs",
            "TensorRtAllocatorInterfaceInfoDesignGate",
            new[] { "EvaluateKnownSurface", "Evaluate" }
        },
        {
            "TensorRtAllocatorInterfaceInfoDesignGateResult.cs",
            "TensorRtAllocatorInterfaceInfoDesignGateResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtAlgorithmSnapshotDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "DiagnosticsKind", "RuntimeEvidenceKind", "IsRuntimeExecutionEvidence",
                "IsRuntimeExecutionProof", "Line", "LineSupportsAlgorithmSelector",
                "SelectorCallbackOwnerModeled", "AlgorithmResultLifetimeModeled",
                "CopiedTimingWorkspaceShapeReady", "CopiedContextShapeReady", "CopiedIoInfoShapeReady",
                "CopiedVariantShapeReady", "AlgorithmPointerExposed", "AlgorithmContextPointerExposed",
                "AlgorithmIoInfoPointerExposed", "AlgorithmVariantPointerExposed",
                "AlgorithmSelectorCallbackTrampolineEnabled", "DirectAlgorithmRowsDeferred",
                "DirectSelectorCallbackRowsDeferred", "CopiedMetadataShapeReady", "PointerFreeSurfaceReady",
                "DesignGateReady", "CanPromoteWithoutRuntimeProof", "FullPackageConsumerRuntimeEvidenceReady",
                "CanPromoteRuntimeProof", "RuntimeProofBlocked", "DeferredRowsStillRequired",
                "CandidateInterfaces", "CandidateMethods", "RequiredOutputMode", "NextSafeImplementationStep",
                "CandidateMethodCount", "BlockedPrerequisites", "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtAllocatorInterfaceInfoDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "DiagnosticsKind", "RuntimeEvidenceKind", "IsRuntimeExecutionEvidence",
                "IsRuntimeExecutionProof", "Line", "LineSupportsAllocatorInterfaceInfo",
                "CopiedInterfaceInfoMetadataReady", "TemporaryStorageAllocatorSnapshotAvailable",
                "OutputAllocatorSnapshotAvailable", "AllocatorOwnerLifetimeModeled",
                "DeviceMemoryOwnershipModeled", "AsyncStreamLifetimeModeled", "AllocatorPointerExposed",
                "DeviceMemoryPointerExposed", "AllocationCallbackInvocationEnabled",
                "DirectAllocatorCallbackRowsDeferred", "CopiedMetadataShapeReady", "PointerFreeSurfaceReady",
                "DesignGateReady", "CanPromoteWithoutRuntimeProof", "CanPromoteRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "CandidateInterfaces", "CandidateMethods",
                "RequiredOutputMode", "NextSafeImplementationStep", "CandidateMethodCount",
                "BlockedPrerequisites", "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
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

    [Theory]
    [MemberData(nameof(ResultProperties))]
    public void ResultsKeepExactPublicProperties(string fileName, string[] expectedProperties)
    {
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(ReadSource(fileName)));
    }

    [Fact]
    public void ResultsKeepOneInternalConstructorEach()
    {
        foreach ((string fileName, string typeName) in new[]
                 {
                     ("TensorRtAlgorithmSnapshotDesignGateResult.cs",
                         "TensorRtAlgorithmSnapshotDesignGateResult"),
                     ("TensorRtAllocatorInterfaceInfoDesignGateResult.cs",
                         "TensorRtAllocatorInterfaceInfoDesignGateResult")
                 })
        {
            Assert.Single(Regex.Matches(
                ReadSource(fileName),
                $@"^    internal {typeName}\(",
                RegexOptions.Multiline));
        }
    }

    [Fact]
    public void EvaluatorsDoNotOwnResultDeclarations()
    {
        Assert.DoesNotContain(
            "public readonly struct TensorRtAlgorithmSnapshotDesignGateResult",
            ReadSource("TensorRtAlgorithmSnapshotDesignGate.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtAllocatorInterfaceInfoDesignGateResult",
            ReadSource("TensorRtAllocatorInterfaceInfoDesignGate.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtAlgorithmSnapshotDesignGateResult.cs",
                     "TensorRtAllocatorInterfaceInfoDesignGateResult.cs"
                 })
        {
            string source = ReadSource(fileName);
            Assert.DoesNotContain("public IntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public UIntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public nint", source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void EvidenceReadersEnumerateBothCompleteSourceSets()
    {
        string candidates = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "deferred-readonly-candidate-list.json"));
        string testReader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));
        string[] sourceFiles =
        {
            "TensorRtAlgorithmSnapshotDesignGate.cs",
            "TensorRtAlgorithmSnapshotDesignGateResult.cs",
            "TensorRtAllocatorInterfaceInfoDesignGate.cs",
            "TensorRtAllocatorInterfaceInfoDesignGateResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, candidates, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        string algorithmConsumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "AlgorithmSnapshotDesignGateTests.cs"));
        string allocatorConsumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "AllocatorInterfaceInfoDesignGateTests.cs"));
        Assert.Contains("TensorRtAlgorithmSnapshotDesignGate", algorithmConsumer, StringComparison.Ordinal);
        Assert.Contains("TensorRtAllocatorInterfaceInfoDesignGate", allocatorConsumer, StringComparison.Ordinal);
    }

    [Fact]
    public void DocsReferenceDedicatedResultOwners()
    {
        string algorithmDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "algorithm-snapshot-design-gate.md"));
        string allocatorDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "allocator-interface-info-design-gate.md"));

        Assert.Contains("TensorRtAlgorithmSnapshotDesignGate.cs", algorithmDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtAlgorithmSnapshotDesignGateResult.cs", algorithmDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtAllocatorInterfaceInfoDesignGate.cs", allocatorDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtAllocatorInterfaceInfoDesignGateResult.cs", allocatorDoc, StringComparison.Ordinal);
    }

    [Fact]
    public void AlgorithmGateFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtAlgorithmSnapshotDesignGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtAlgorithmSnapshotDesignGateResult.cs"));
        Assert.Equal(AlgorithmGateOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void AllocatorInterfaceGateFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtAllocatorInterfaceInfoDesignGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtAllocatorInterfaceInfoDesignGateResult.cs"));
        Assert.Equal(AllocatorInterfaceGateOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string[] EnumerateTopLevelTypeNames(string source)
    {
        return Regex.Matches(
                Normalize(source),
                @"^public (?:static class|readonly struct) (?<name>[A-Za-z_][A-Za-z0-9_]*)$",
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
        string module = fileName.StartsWith("TensorRtAlgorithm", StringComparison.Ordinal)
            ? "Core"
            : "MemoryAllocation";
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Callbacks",
            module,
            fileName));
    }
}
