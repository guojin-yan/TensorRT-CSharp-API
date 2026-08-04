using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedOutputAllocatorOwnershipAttachDetachSourceLayoutTests
{
    private const string OwnershipGateOriginalNormalizedSha256 =
        "651c3cf8339d9d5909f43bee454a6cd077bbc5184590fcd6534cc24b817e7ff5";
    private const string AttachDetachGateOriginalNormalizedSha256 =
        "15bcf6677919fdfbbc1eed97e52b6dd4c9a6c84280e7ab8bb46779b2b8d0dfd5";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtOutputBufferOwnershipSafetyGate.cs",
            "TensorRtOutputBufferOwnershipSafetyGate",
            new[] { "Evaluate", "Evaluate" }
        },
        {
            "TensorRtOutputBufferOwnershipSafetyGateResult.cs",
            "TensorRtOutputBufferOwnershipSafetyGateResult",
            new[] { "ToString" }
        },
        {
            "TensorRtOutputAllocatorAttachDetachDesignGate.cs",
            "TensorRtOutputAllocatorAttachDetachDesignGate",
            new[] { "Evaluate" }
        },
        {
            "TensorRtOutputAllocatorAttachDetachDesignGateResult.cs",
            "TensorRtOutputAllocatorAttachDetachDesignGateResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtOutputBufferOwnershipSafetyGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "TensorName", "RequestedSize", "Alignment",
                "ShapeRank", "HasCurrentMemory", "NotifyShapeCount", "ReallocateOutputCount",
                "AttachDetachDesignGateReady", "OwnerDesignReady", "PointerFreeSurfaceReady",
                "CopiedCurrentMemoryMetadataReady", "CopiedShapeMetadataReady", "CopiedRequestMetadataReady",
                "OutputBufferOwnershipRuntimeReady", "CurrentMemoryReusePolicyReady",
                "BorrowedPointerEscapeBlocked", "OwnedDevicePointerReleasePolicyReady",
                "ShapeNotificationOrderingReady", "ReallocateOutputRuntimeReady",
                "FullPackageConsumerRuntimeEvidenceReady", "SafetyGateReady", "CanAttemptRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtOutputAllocatorAttachDetachDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "LineSupportsOutputAllocator", "OwnerDesignReady",
                "ManagedOwnerStateMachineReady", "PointerFreeSurfaceReady", "DetachClearControlAvailable",
                "AttachControlAvailable", "LineSpecificAttachDetachReady", "StableNativeOwnerAddressReady",
                "NoThrowNativeVTableReady", "NativeVTableReady", "OutputBufferOwnershipRuntimeReady",
                "DesignGateReady", "CanAttemptRuntimeProof", "RuntimeProofBlocked", "DeferredRowsStillRequired",
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
                     ("TensorRtOutputBufferOwnershipSafetyGateResult.cs",
                         "TensorRtOutputBufferOwnershipSafetyGateResult"),
                     ("TensorRtOutputAllocatorAttachDetachDesignGateResult.cs",
                         "TensorRtOutputAllocatorAttachDetachDesignGateResult")
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
            "public readonly struct TensorRtOutputBufferOwnershipSafetyGateResult",
            ReadSource("TensorRtOutputBufferOwnershipSafetyGate.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtOutputAllocatorAttachDetachDesignGateResult",
            ReadSource("TensorRtOutputAllocatorAttachDetachDesignGate.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtOutputBufferOwnershipSafetyGateResult.cs",
                     "TensorRtOutputAllocatorAttachDetachDesignGateResult.cs"
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
        string readiness = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Test-RuntimePackageReadiness.ps1"));
        string testReader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));
        string[] sourceFiles =
        {
            "TensorRtOutputBufferOwnershipSafetyGate.cs",
            "TensorRtOutputBufferOwnershipSafetyGateResult.cs",
            "TensorRtOutputAllocatorAttachDetachDesignGate.cs",
            "TensorRtOutputAllocatorAttachDetachDesignGateResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        string ownershipConsumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "OutputBufferOwnershipSafetyGateTests.cs"));
        string attachDetachConsumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "OutputAllocatorAttachDetachDesignGateTests.cs"));
        Assert.Contains("RepositorySourceReader.Read", ownershipConsumer, StringComparison.Ordinal);
        Assert.Contains("RepositorySourceReader.Read", attachDetachConsumer, StringComparison.Ordinal);
    }

    [Fact]
    public void DocsReferenceDedicatedResultOwners()
    {
        string ownershipDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "output-buffer-ownership-safety-gate.md"));
        string attachDetachDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "output-allocator-attach-detach-design-gate.md"));

        Assert.Contains("TensorRtOutputBufferOwnershipSafetyGate.cs", ownershipDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtOutputBufferOwnershipSafetyGateResult.cs", ownershipDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtOutputAllocatorAttachDetachDesignGate.cs", attachDetachDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtOutputAllocatorAttachDetachDesignGateResult.cs", attachDetachDoc, StringComparison.Ordinal);
    }

    [Fact]
    public void OwnershipGateFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtOutputBufferOwnershipSafetyGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtOutputBufferOwnershipSafetyGateResult.cs"));
        Assert.Equal(OwnershipGateOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void AttachDetachGateFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtOutputAllocatorAttachDetachDesignGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtOutputAllocatorAttachDetachDesignGateResult.cs"));
        Assert.Equal(AttachDetachGateOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Callbacks",
            "MemoryAllocation",
            fileName));
    }
}
