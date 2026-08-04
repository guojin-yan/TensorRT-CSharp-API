using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerLifecycleGateCallbackStubSourceLayoutTests
{
    private const string LifecycleGateOriginalNormalizedSha256 =
        "365214677f3decd5f41413aa8e4bc5fac32714500b9ad17393087d581a85983f";
    private const string CallbackStubOriginalNormalizedSha256 =
        "11d2de6ce8cde314aa0dc52fb2b023e37e8a15ddf083bec784b82a42f9c88821";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerNativeOwnerLifecycleGate.cs",
            "TensorRtDebugListenerNativeOwnerLifecycleGate",
            new[] { "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeOwnerLifecycleGateResult.cs",
            "TensorRtDebugListenerNativeOwnerLifecycleGateResult",
            new[] { "ToString" }
        },
        {
            "TensorRtDebugListenerNoThrowVTableCallbackStub.cs",
            "TensorRtDebugListenerNoThrowVTableCallbackStub",
            new[] { "Evaluate", "Evaluate", "BuildCallbackRuntimeBlockedReason", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNoThrowVTableCallbackStubResult.cs",
            "TensorRtDebugListenerNoThrowVTableCallbackStubResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtDebugListenerNativeOwnerLifecycleGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus", "ReleaseHookCount",
                "InFlightCallbackCount", "CallbackStatePinned", "DelegatePinned", "DisposeRequested",
                "LastDiagnostic", "ReleaseDiagnostic", "NativeNoThrowDestructorGateReady",
                "NativeOwnerNonCopyableReady", "NativeOwnerCopyBlocked", "NativeOwnerMoveBlocked",
                "NativeOwnerAddressExposed", "NativeOwnerPointerProduced", "DestructorNoThrowScaffoldReady",
                "DestructorExceptionEscapeBlocked", "DestructorAddressExposed", "DestructorPointerProduced",
                "ManagedDisposeSnapshotReady", "LifecycleScaffoldReady", "ReleaseHookOrderingGateReady",
                "DisposeIdempotencyGateReady", "InFlightDrainGateReady", "CallbackStateUnpinAfterDetachGateReady",
                "DelegateUnpinAfterDetachGateReady", "LifecycleAddressExposed", "LifecyclePointerProduced",
                "NativeAttachEntryLocated", "NativeDetachEntryLocated", "NoThrowNativeDestructorReady",
                "ReleaseHookOrderingReady", "DisposeIdempotencyReady", "InFlightDrainBeforeReleaseReady",
                "CallbackStateUnpinAfterDetachReady", "DelegateUnpinAfterDetachReady", "NativeOwnerLifecycleReady",
                "NativeVTableDesignReady", "ProcessDebugTensorRuntimeReady",
                "FullPackageConsumerRuntimeEvidenceReady", "LifecycleGateReady", "CanImplementNativeAttach",
                "CanAttemptRuntimeProof", "RuntimeProofBlocked", "DeferredRowsStillRequired",
                "BlockedPrerequisites", "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtDebugListenerNoThrowVTableCallbackStubResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus", "TensorName", "DataType",
                "Location", "ShapeRank", "ShapeSummary", "IsInput", "IsExecutionTensor", "CallbackEntryCount",
                "CallbackLeaveCount", "FailureCount", "MinimalSafetyReady", "NoThrowVTableScaffoldGateReady",
                "NoThrowVTableScaffoldReady", "CallbackStubShapeReady", "CallbackStubNoThrowReady",
                "CallbackMetadataCopyReady", "CallbackExceptionCaptureReady", "CallbackStatusMappingReady",
                "CallbackInFlightEnterReady", "CallbackInFlightLeaveReady", "CallbackInFlightPairingReady",
                "CallbackInFlightNeverNegativeReady", "BorrowedDebugTensorMetadataCopyReady",
                "BorrowedDebugTensorPointerEscapeBlocked", "DebugTensorPointerExposed",
                "DebugTensorDataPointerExposed", "SetDebugListenerNonNullEnabled", "NativeAttachWouldBeBlocked",
                "NativeVTableInstalled", "CallbackStubGateReady", "ProcessDebugTensorRuntimeReady",
                "CanInstallNativeVTable", "CanCallProcessDebugTensorRuntime", "CanAttemptRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "ReasonCallbackRuntimeStillBlocked",
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
                     ("TensorRtDebugListenerNativeOwnerLifecycleGateResult.cs",
                         "TensorRtDebugListenerNativeOwnerLifecycleGateResult"),
                     ("TensorRtDebugListenerNoThrowVTableCallbackStubResult.cs",
                         "TensorRtDebugListenerNoThrowVTableCallbackStubResult")
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
            "public readonly struct TensorRtDebugListenerNativeOwnerLifecycleGateResult",
            ReadSource("TensorRtDebugListenerNativeOwnerLifecycleGate.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerNoThrowVTableCallbackStubResult",
            ReadSource("TensorRtDebugListenerNoThrowVTableCallbackStub.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerNativeOwnerLifecycleGateResult.cs",
                     "TensorRtDebugListenerNoThrowVTableCallbackStubResult.cs"
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
            "TensorRtDebugListenerNativeOwnerLifecycleGate.cs",
            "TensorRtDebugListenerNativeOwnerLifecycleGateResult.cs",
            "TensorRtDebugListenerNoThrowVTableCallbackStub.cs",
            "TensorRtDebugListenerNoThrowVTableCallbackStubResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "DebugListenerNativeOwnerLifecycleGateTests.cs",
                     "DebugListenerNoThrowVTableCallbackStubTests.cs"
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
        string lifecycleDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-owner-lifecycle-gate.md"));
        string callbackStubDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-nothrow-vtable-callback-stub.md"));

        Assert.Contains("TensorRtDebugListenerNativeOwnerLifecycleGate.cs", lifecycleDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerNativeOwnerLifecycleGateResult.cs",
            lifecycleDoc,
            StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerNoThrowVTableCallbackStub.cs", callbackStubDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerNoThrowVTableCallbackStubResult.cs",
            callbackStubDoc,
            StringComparison.Ordinal);
    }

    [Fact]
    public void LifecycleGateFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerNativeOwnerLifecycleGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerNativeOwnerLifecycleGateResult.cs"));
        Assert.Equal(LifecycleGateOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void CallbackStubFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerNoThrowVTableCallbackStub.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerNoThrowVTableCallbackStubResult.cs"));
        Assert.Equal(CallbackStubOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
