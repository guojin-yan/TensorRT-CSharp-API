using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerProofGapAttachEntrySourceLayoutTests
{
    private const string ProofGapOriginalNormalizedSha256 =
        "77c574d69087f51d8d7ac6bea6bffe24a2c4f64dc926501d05b31facd560f055";
    private const string AttachEntryOriginalNormalizedSha256 =
        "58872e6d29ad8d7ee9a885a35f20b935d31ca3db98a1f8161429300f57c958d9";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerCallbackProofGapReport.cs",
            "TensorRtDebugListenerCallbackProofGapReport",
            new[] { "Evaluate", "AddGapIfFalse", "AddGaps", "AddGap" }
        },
        {
            "TensorRtDebugListenerCallbackProofGapReportResult.cs",
            "TensorRtDebugListenerCallbackProofGapReportResult",
            new[] { "GetRuntimeProofBlockerCategory", "GetNextOwnerAction" }
        },
        {
            "TensorRtDebugListenerNativeAttachEntryDesignGate.cs",
            "TensorRtDebugListenerNativeAttachEntryDesignGate",
            new[] { "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeAttachEntryDesignGateResult.cs",
            "TensorRtDebugListenerNativeAttachEntryDesignGateResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtDebugListenerCallbackProofGapReportResult.cs",
            new[]
            {
                "EvidenceKind", "RuntimeEvidenceKind", "RealCallbackRuntime", "IsRealCallbackRuntimeProof",
                "CallbackKind", "Line", "TensorRtLine", "RuntimePackageKey", "NonNullAttachStillDisabled",
                "NativeAttachEntryReady", "NativeVTableInstallBlocked", "NoThrowCallbackEntryReady",
                "ExceptionStatusMappingReady", "InFlightAccountingReady", "BorrowedDebugTensorMetadataCopied",
                "DetachRollbackReady", "ProcessDebugTensorRuntimeInvoked", "FullPackageConsumerRuntimeProofReady",
                "PointerFreeSurfaceReady", "AttemptedNoInvocation", "InvocationCount", "FailureCount",
                "InFlightCallbackCount", "CanPromoteRealCallbackRuntime", "CanAttemptRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "GapReasons", "GapReasonCount",
                "PrimaryGapReason", "RuntimeProofBlockerCategory", "PackageConsumerRuntimeProofRequired",
                "RuntimeInvocationRequired", "EvidenceSource", "NextOwnerAction", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtDebugListenerNativeAttachEntryDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "NativeNoThrowVTableDesignGateReady",
                "NativeOwnerAddressDesignGateReady", "NativeAttachNoThrowPreflightReady",
                "NativeDetachEntryLocated", "NativeAttachEntryLocated", "LineSpecificAttachEntryDesignReady",
                "AttachEntryNoThrowReady", "AttachEntryVersionGuardReady", "AttachEntryOwnershipReady",
                "DetachBeforeReleaseReady", "NativeOwnerLifecycleReady", "NativeVTableDesignReady",
                "ManagedCallbackKeepAliveDesignReady", "BorrowedDebugTensorMetadataCopyDesignReady",
                "BorrowedDebugTensorPointerEscapeBlocked", "DesignGateReady", "CanImplementNativeAttach",
                "BorrowedDebugTensorLifetimeRuntimeReady", "BorrowedDebugTensorDataLifetimeRuntimeReady",
                "ProcessDebugTensorRuntimeReady", "FullPackageConsumerRuntimeEvidenceReady", "CanAttemptRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
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
                     ("TensorRtDebugListenerCallbackProofGapReportResult.cs",
                         "TensorRtDebugListenerCallbackProofGapReportResult"),
                     ("TensorRtDebugListenerNativeAttachEntryDesignGateResult.cs",
                         "TensorRtDebugListenerNativeAttachEntryDesignGateResult")
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
            "public sealed class TensorRtDebugListenerCallbackProofGapReportResult",
            ReadSource("TensorRtDebugListenerCallbackProofGapReport.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerNativeAttachEntryDesignGateResult",
            ReadSource("TensorRtDebugListenerNativeAttachEntryDesignGate.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerCallbackProofGapReportResult.cs",
                     "TensorRtDebugListenerNativeAttachEntryDesignGateResult.cs"
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
            "TensorRtDebugListenerCallbackProofGapReport.cs",
            "TensorRtDebugListenerCallbackProofGapReportResult.cs",
            "TensorRtDebugListenerNativeAttachEntryDesignGate.cs",
            "TensorRtDebugListenerNativeAttachEntryDesignGateResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "DebugListenerCallbackProofGapReportTests.cs",
                     "DebugListenerNativeAttachEntryDesignGateTests.cs"
                 })
        {
            string consumer = File.ReadAllText(Path.Combine(
                RepositoryPaths.Root,
                "tests",
                "JYPPX.ProjectQuality.Tests",
                consumerFile));
            Assert.Contains("RepositorySourceReader.Read", consumer, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void DocsReferenceDedicatedResultOwners()
    {
        string gapDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-callback-proof-gap-report.md"));
        string attachDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-attach-entry-design-gate.md"));

        Assert.Contains("TensorRtDebugListenerCallbackProofGapReport.cs", gapDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerCallbackProofGapReportResult.cs", gapDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryDesignGate.cs", attachDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryDesignGateResult.cs", attachDoc, StringComparison.Ordinal);
    }

    [Fact]
    public void ProofGapFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerCallbackProofGapReport.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerCallbackProofGapReportResult.cs"));
        Assert.Equal(ProofGapOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void AttachEntryFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerNativeAttachEntryDesignGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerNativeAttachEntryDesignGateResult.cs"));
        Assert.Equal(AttachEntryOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string[] EnumerateTopLevelTypeNames(string source)
    {
        return Regex.Matches(
                Normalize(source),
                @"^public (?:static class|sealed class|readonly struct) (?<name>[A-Za-z_][A-Za-z0-9_]*)$",
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
            "Debugging",
            fileName));
    }
}
