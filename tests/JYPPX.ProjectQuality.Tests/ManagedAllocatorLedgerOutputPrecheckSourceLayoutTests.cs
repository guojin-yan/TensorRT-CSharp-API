using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedAllocatorLedgerOutputPrecheckSourceLayoutTests
{
    private const string AllocatorLedgerOriginalNormalizedSha256 =
        "31ba18ac5466a804cf8e3b013192e8082e1384b5b19c96c43d84f32331e0e962";
    private const string OutputPrecheckOriginalNormalizedSha256 =
        "214bb9a84e7caff5a7142f0afc2212bd9149a0fa403f2b29f751721486ae388a";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtAllocatorLedgerSafetyGate.cs",
            "TensorRtAllocatorLedgerSafetyGate",
            new[] { "Evaluate", "GetSnapshot" }
        },
        {
            "TensorRtAllocatorLedgerSafetyGateResult.cs",
            "TensorRtAllocatorLedgerSafetyGateResult",
            new[] { "ToString", "BuildBlockedPrerequisites" }
        },
        {
            "TensorRtOutputAllocatorRuntimeProofPrecheck.cs",
            "TensorRtOutputAllocatorRuntimeProofPrecheck",
            new[] { "Evaluate", "Evaluate", "Evaluate" }
        },
        {
            "TensorRtOutputAllocatorRuntimeProofPrecheckResult.cs",
            "TensorRtOutputAllocatorRuntimeProofPrecheckResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtAllocatorLedgerSafetyGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "Operation", "LastStatus",
                "InternalPrototypeStatus", "NativeLedgerStatus", "NativeLedgerAvailable",
                "InvocationCount", "FailureCount", "InFlightCallbackCount", "MaxInFlightCallbackCount",
                "ActivePrototypeCallCount", "ReleaseHookCount", "CallbackStatePinned", "DelegatePinned",
                "DisposeRequested", "IsAttached", "NativeOwnerId", "StateTransitionCount",
                "LedgerAllocationCount", "LedgerReleaseCount", "LedgerFailureCount", "LastAllocationId",
                "LastReleaseAllocationId", "LastStreamValue", "NativeAttachState", "HasLiveAllocation",
                "NativeLastOperation", "LastDiagnostic", "NativeLedgerDiagnostic", "ReleaseDiagnostic",
                "ManagedKeepAliveReady", "DisposeReleaseReady", "NativeLedgerDesignReady",
                "PointerFreeSurfaceReady", "LineSpecificAttachDetachReady", "DevicePointerLedgerRuntimeReady",
                "StreamLifetimeReady", "FullPackageConsumerRuntimeEvidenceReady", "DeferredRowsStillRequired",
                "CanAttemptRuntimeProof", "RuntimeProofBlocked", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtOutputAllocatorRuntimeProofPrecheckResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "LineSupportsOutputAllocator", "OwnerDesignReady",
                "NativeLedgerDesignReady", "DisposeReleaseReady", "PointerFreeSurfaceReady",
                "AttachDetachDesignGateReady", "AttachControlAvailable", "DetachClearControlAvailable",
                "ManagedOwnerStateMachineReady", "LineSpecificAttachDetachReady", "StableNativeOwnerAddressReady",
                "NoThrowNativeVTableReady", "NativeVTableReady", "DevicePointerLedgerRuntimeReady",
                "StreamLifetimeReady", "OutputBufferOwnershipSafetyGateReady",
                "OutputBufferOwnershipRuntimeReady", "CurrentMemoryReusePolicyReady",
                "BorrowedPointerEscapeBlocked", "OwnedDevicePointerReleasePolicyReady",
                "ShapeNotificationOrderingReady", "ReallocateOutputRuntimeReady",
                "FullPackageConsumerRuntimeEvidenceReady", "DeferredRowsStillRequired",
                "CanAttemptRuntimeProof", "RuntimeProofBlocked", "BlockedPrerequisites",
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
                     ("TensorRtAllocatorLedgerSafetyGateResult.cs",
                         "TensorRtAllocatorLedgerSafetyGateResult"),
                     ("TensorRtOutputAllocatorRuntimeProofPrecheckResult.cs",
                         "TensorRtOutputAllocatorRuntimeProofPrecheckResult")
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
            "public readonly struct TensorRtAllocatorLedgerSafetyGateResult",
            ReadSource("TensorRtAllocatorLedgerSafetyGate.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtOutputAllocatorRuntimeProofPrecheckResult",
            ReadSource("TensorRtOutputAllocatorRuntimeProofPrecheck.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtAllocatorLedgerSafetyGateResult.cs",
                     "TensorRtOutputAllocatorRuntimeProofPrecheckResult.cs"
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
            "TensorRtAllocatorLedgerSafetyGate.cs",
            "TensorRtAllocatorLedgerSafetyGateResult.cs",
            "TensorRtOutputAllocatorRuntimeProofPrecheck.cs",
            "TensorRtOutputAllocatorRuntimeProofPrecheckResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        string ledgerConsumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "AllocatorOwnerLedgerSafetyGateTests.cs"));
        string precheckConsumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "OutputAllocatorRuntimeProofPrecheckTests.cs"));
        string ownershipConsumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "OutputBufferOwnershipSafetyGateTests.cs"));
        Assert.Contains("RepositorySourceReader.Read", ledgerConsumer, StringComparison.Ordinal);
        Assert.Contains("RepositorySourceReader.Read", precheckConsumer, StringComparison.Ordinal);
        Assert.Contains("RepositorySourceReader.Read", ownershipConsumer, StringComparison.Ordinal);
    }

    [Fact]
    public void DocsReferenceDedicatedResultOwners()
    {
        string ledgerDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "allocator-owner-ledger-safety-gate.md"));
        string precheckDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "output-allocator-runtime-proof-precheck.md"));

        Assert.Contains("TensorRtAllocatorLedgerSafetyGate.cs", ledgerDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtAllocatorLedgerSafetyGateResult.cs", ledgerDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtOutputAllocatorRuntimeProofPrecheck.cs", precheckDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtOutputAllocatorRuntimeProofPrecheckResult.cs", precheckDoc, StringComparison.Ordinal);
    }

    [Fact]
    public void AllocatorLedgerFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtAllocatorLedgerSafetyGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtAllocatorLedgerSafetyGateResult.cs"));
        Assert.Equal(AllocatorLedgerOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void OutputPrecheckFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtOutputAllocatorRuntimeProofPrecheck.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtOutputAllocatorRuntimeProofPrecheckResult.cs"));
        Assert.Equal(OutputPrecheckOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
