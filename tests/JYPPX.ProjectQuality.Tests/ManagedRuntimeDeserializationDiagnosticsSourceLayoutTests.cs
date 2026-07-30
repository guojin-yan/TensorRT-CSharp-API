using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedRuntimeDeserializationDiagnosticsSourceLayoutTests
{
    private const string BoundaryOriginalNormalizedSha256 =
        "6591ff58bbfb5cb6e1f01cb9a52caa7f310c27029548854f3632ed2f0275f0ec";
    private const string DependencyOriginalNormalizedSha256 =
        "2fd282093b86550e509b45658dd8738c837efd8b15993c27877e36409426f290";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtRuntimeDeserializationBoundaryPrecheck.cs",
            "TensorRtRuntimeDeserializationBoundaryPrecheck",
            new[] { "EvaluateKnownSurface", "Evaluate" }
        },
        {
            "TensorRtRuntimeDeserializationBoundaryPrecheckResult.cs",
            "TensorRtRuntimeDeserializationBoundaryPrecheckResult",
            new[] { "ToString" }
        },
        {
            "TensorRtRuntimeDeserializationDependencyDiagnostics.cs",
            "TensorRtRuntimeDeserializationDependencyDiagnostics",
            new[] { "EvaluateKnownSurface", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtRuntimeDeserializationDependencyDiagnosticsResult.cs",
            "TensorRtRuntimeDeserializationDependencyDiagnosticsResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtRuntimeDeserializationBoundaryPrecheckResult.cs",
            new[]
            {
                "EvidenceKind", "DiagnosticsKind", "RuntimeEvidenceKind", "IsRuntimeExecutionEvidence",
                "IsRuntimeExecutionProof", "Line", "LineSupportsRuntimeDeserialization",
                "LineSupportsDeserializeCudaEngineV2", "ManagedByteArrayDeserializeReady",
                "ManagedArraySegmentDeserializeReady", "ManagedReadOnlySpanDeserializeReady",
                "ManagedStreamDeserializeReady", "ManagedFileDeserializeReady", "HostMemoryDeserializeReady",
                "SerializedBufferCopiedBeforeInterop", "PinnedBufferScopedToInteropCall",
                "BorrowedSerializedBufferEscaped", "HostMemoryHandleOwnedByWrapper", "EngineHandleOwnedByWrapper",
                "EnginePointerExposed", "EnginePointerProduced", "DirectDeserializeCudaEngineRowsDeferred",
                "DirectDeserializeCudaEngineRowsImplemented", "DirectDeserializeCudaEngineV2RowsDeferred",
                "LoadRuntimeDeferred", "PluginLibraryDependencyDiagnosticsReady", "PointerFreeSurfaceReady",
                "ManagedDeserializeSurfaceReady", "SafeDeserializeBridgeReady", "PrecheckReady",
                "CanAttemptRuntimeProof", "CanPromoteWithoutRuntimeProof",
                "FullPackageConsumerRuntimeEvidenceReady", "CanPromoteRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "BlockedPrerequisites", "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtRuntimeDeserializationDependencyDiagnosticsResult.cs",
            new[]
            {
                "EvidenceKind", "DiagnosticsKind", "RuntimeEvidenceKind", "IsRuntimeExecutionEvidence",
                "IsRuntimeExecutionProof", "Line", "PrecheckReady", "ManagedDeserializeSurfaceReady",
                "SafeDeserializeBridgeReady", "LineSupportsDeserializeCudaEngineV2",
                "DirectDeserializeCudaEngineRowsDeferred", "DirectDeserializeCudaEngineV2RowsDeferred",
                "LoadRuntimeDeferred", "FullPackageConsumerReportPresent", "FullPackageConsumerSmokeRequested",
                "FullPackageConsumerSmokeResult", "DependencyProbeOnly", "BlockedByCudaDriver",
                "DriverRuntimeMismatchClassified", "PackageConsumerRuntimeProofPresent", "ExternalRuntimeProofRequired",
                "RuntimeProofOwnerActionRequired", "RuntimeProofBlockerCategory",
                "PackageConsumerEvidenceClassification", "PluginLibraryDependencyDiagnosticsComplete",
                "LoadRuntimeOwnershipModeled", "CanAttemptRuntimeProof", "CanPromoteRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "PointerFreeSurfaceReady", "WhyNotRuntimeProof",
                "NextOwnerAction", "BlockedPrerequisites", "BlockedPrerequisiteCount", "Status", "Diagnostic"
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
                     ("TensorRtRuntimeDeserializationBoundaryPrecheckResult.cs",
                         "TensorRtRuntimeDeserializationBoundaryPrecheckResult"),
                     ("TensorRtRuntimeDeserializationDependencyDiagnosticsResult.cs",
                         "TensorRtRuntimeDeserializationDependencyDiagnosticsResult")
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
            "public readonly struct TensorRtRuntimeDeserializationBoundaryPrecheckResult",
            ReadSource("TensorRtRuntimeDeserializationBoundaryPrecheck.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtRuntimeDeserializationDependencyDiagnosticsResult",
            ReadSource("TensorRtRuntimeDeserializationDependencyDiagnostics.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtRuntimeDeserializationBoundaryPrecheckResult.cs",
                     "TensorRtRuntimeDeserializationDependencyDiagnosticsResult.cs"
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
            "TensorRtRuntimeDeserializationBoundaryPrecheck.cs",
            "TensorRtRuntimeDeserializationBoundaryPrecheckResult.cs",
            "TensorRtRuntimeDeserializationDependencyDiagnostics.cs",
            "TensorRtRuntimeDeserializationDependencyDiagnosticsResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        string consumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RuntimeDeserializationBoundaryPrecheckTests.cs"));
        Assert.Contains("RepositorySourceReader.Read", consumer, StringComparison.Ordinal);
    }

    [Fact]
    public void DocsReferenceDedicatedResultOwners()
    {
        string boundaryDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "runtime-deserialization-boundary-precheck.md"));
        string dependencyDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "runtime-deserialization-dependency-diagnostics.md"));

        Assert.Contains("TensorRtRuntimeDeserializationBoundaryPrecheck.cs", boundaryDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtRuntimeDeserializationBoundaryPrecheckResult.cs", boundaryDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtRuntimeDeserializationDependencyDiagnostics.cs", dependencyDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtRuntimeDeserializationDependencyDiagnosticsResult.cs", dependencyDoc, StringComparison.Ordinal);
    }

    [Fact]
    public void BoundaryFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtRuntimeDeserializationBoundaryPrecheck.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtRuntimeDeserializationBoundaryPrecheckResult.cs"));
        Assert.Equal(BoundaryOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void DependencyFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtRuntimeDeserializationDependencyDiagnostics.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtRuntimeDeserializationDependencyDiagnosticsResult.cs"));
        Assert.Equal(DependencyOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
            "Runtime",
            fileName));
    }
}
