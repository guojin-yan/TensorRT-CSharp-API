using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerAttachBridgeInFlightSourceLayoutTests
{
    private const string AttachBridgeOriginalNormalizedSha256 =
        "fc93627c2f93908e91ed05df15b2e39e166175826c4e3a71b7aad4c9be23265f";
    private const string InFlightAccountingOriginalNormalizedSha256 =
        "4391e6ffe21b711338c4a110ea25fcfcfdaf140cc3e562adde545e63918275ae";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerNativeAttachBridgeShapeGate.cs",
            "TensorRtDebugListenerNativeAttachBridgeShapeGate",
            new[] { "Evaluate", "Evaluate", "BecausePointerFree", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeAttachBridgeShapeGateResult.cs",
            "TensorRtDebugListenerNativeAttachBridgeShapeGateResult",
            new[] { "ToString" }
        },
        {
            "TensorRtDebugListenerInFlightAccountingGate.cs",
            "TensorRtDebugListenerInFlightAccountingGate",
            new[] { "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerInFlightAccountingGateResult.cs",
            "TensorRtDebugListenerInFlightAccountingGateResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtDebugListenerNativeAttachBridgeShapeGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus",
                "NativeOwnerLifecycleGateReady", "AttachBridgeShapeReady", "AttachBridgeNoThrowBoundaryReady",
                "AttachBridgeVersionGuardReady", "AttachBridgeOwnershipDiagnosticsReady", "AttachBridgePointerFree",
                "SetDebugListenerNonNullEnabled", "NonNullAttachStillDisabled", "NativeAttachEntryLocated",
                "NativeDetachEntryLocated", "NativeOwnerLifecycleReady", "NativeVTableDesignReady",
                "AttachBridgeShapeGateReady", "ProcessDebugTensorRuntimeReady",
                "FullPackageConsumerRuntimeEvidenceReady", "CanImplementNativeAttach", "CanAttemptRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtDebugListenerInFlightAccountingGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus", "ProcessDebugTensorCount",
                "InFlightCallbackCount", "MaxInFlightCallbackCount", "ReleaseHookCount", "CallbackStatePinned",
                "DelegatePinned", "DisposeRequested", "ExceptionStatusMappingGateReady",
                "CallbackEnterAccountingGateReady", "CallbackLeaveAccountingGateReady",
                "CallbackInFlightNeverNegativeReady", "ReleaseAfterDrainGateReady",
                "CallbackStateUnpinAfterDrainGateReady", "AccountingAddressExposed", "AccountingPointerProduced",
                "NativeAttachEntryLocated", "InFlightAccountingGateReady", "ProcessDebugTensorRuntimeReady",
                "FullPackageConsumerRuntimeEvidenceReady", "CanAttemptRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "BlockedPrerequisites", "BlockedPrerequisiteCount", "Status", "Diagnostic"
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
                     ("TensorRtDebugListenerNativeAttachBridgeShapeGateResult.cs",
                         "TensorRtDebugListenerNativeAttachBridgeShapeGateResult"),
                     ("TensorRtDebugListenerInFlightAccountingGateResult.cs",
                         "TensorRtDebugListenerInFlightAccountingGateResult")
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
            "public readonly struct TensorRtDebugListenerNativeAttachBridgeShapeGateResult",
            ReadSource("TensorRtDebugListenerNativeAttachBridgeShapeGate.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerInFlightAccountingGateResult",
            ReadSource("TensorRtDebugListenerInFlightAccountingGate.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerNativeAttachBridgeShapeGateResult.cs",
                     "TensorRtDebugListenerInFlightAccountingGateResult.cs"
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
            "TensorRtDebugListenerNativeAttachBridgeShapeGate.cs",
            "TensorRtDebugListenerNativeAttachBridgeShapeGateResult.cs",
            "TensorRtDebugListenerInFlightAccountingGate.cs",
            "TensorRtDebugListenerInFlightAccountingGateResult.cs"
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
            "DebugListenerAttachBridgeVTableBatchTests.cs"));
        Assert.Contains("RepositorySourceReader.Read", consumer, StringComparison.Ordinal);
    }

    [Fact]
    public void DocsReferenceDedicatedResultOwners()
    {
        string attachDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-attach-bridge-shape-gate.md"));
        string accountingDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-inflight-accounting-gate.md"));

        Assert.Contains("TensorRtDebugListenerNativeAttachBridgeShapeGate.cs", attachDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerNativeAttachBridgeShapeGateResult.cs",
            attachDoc,
            StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerInFlightAccountingGate.cs", accountingDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerInFlightAccountingGateResult.cs",
            accountingDoc,
            StringComparison.Ordinal);
    }

    [Fact]
    public void AttachBridgeFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerNativeAttachBridgeShapeGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerNativeAttachBridgeShapeGateResult.cs"));
        Assert.Equal(AttachBridgeOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void InFlightAccountingFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerInFlightAccountingGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerInFlightAccountingGateResult.cs"));
        Assert.Equal(InFlightAccountingOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
            .Replace('\r', '\n')
            .Replace(
                "using JYPPX.TensorRtSharp.Shared.Interop;",
                "using JYPPX.Shared.Interop;",
                StringComparison.Ordinal);
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
