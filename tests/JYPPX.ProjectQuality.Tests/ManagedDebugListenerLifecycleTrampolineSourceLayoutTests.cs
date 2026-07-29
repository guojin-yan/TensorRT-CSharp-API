using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerLifecycleTrampolineSourceLayoutTests
{
    private const string LifecycleDryRunOriginalNormalizedSha256 =
        "182eda1a92bd1f3d7bcfcbd65104e1913213201dda6be998aca9aa2a6c626033";
    private const string CallbackTrampolineOriginalNormalizedSha256 =
        "df0efe7be7752c0c665a517e0360dae57f71d29c140d89181ebcf32f451bc192";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerNativeOwnerLifecycleDryRun.cs",
            "TensorRtDebugListenerNativeOwnerLifecycleDryRun",
            new[] { "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.cs",
            "TensorRtDebugListenerNativeOwnerLifecycleDryRunResult",
            new[] { "ToString" }
        },
        {
            "TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.cs",
            "TensorRtDebugListenerProcessDebugTensorCallbackTrampoline",
            new[] { "Evaluate", "Evaluate", "BuildDiagnostic", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugTensorMetadataSnapshot.cs",
            "TensorRtDebugTensorMetadataSnapshot",
            Array.Empty<string>()
        },
        {
            "TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.cs",
            "TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ModelProperties => new()
    {
        {
            "TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus", "ReleaseHookCount",
                "InFlightCallbackCount", "CallbackStatePinned", "DelegatePinned", "DisposeRequested",
                "LastDiagnostic", "ReleaseDiagnostic", "NativeDetachBeforeReleaseDesignGateReady",
                "NativeAttachEntryDesignGateReady", "NativeNoThrowVTableDesignGateReady",
                "NativeOwnerAddressDesignGateReady", "NativeDetachEntryLocated", "NativeAttachEntryLocated",
                "StableNativeOwnerIdentityReady", "NativeOwnerNonCopyableReady", "NativeOwnerDisposeOrderReady",
                "NativeOwnerReleaseHookReady", "NativeOwnerInFlightDrainReady", "DetachBeforeReleaseReady",
                "ReleaseHookOrderingReady", "DisposeIdempotencyReady", "InFlightDrainBeforeReleaseReady",
                "CallbackStateUnpinAfterDetachReady", "DelegateUnpinAfterDetachReady",
                "NoThrowNativeDestructorReady", "NativeOwnerLifecycleReady", "NativeVTableDesignReady",
                "ManagedCallbackKeepAliveDesignReady", "BorrowedDebugTensorMetadataCopyDesignReady",
                "BorrowedDebugTensorPointerEscapeBlocked", "DryRunReady", "BorrowedDebugTensorLifetimeRuntimeReady",
                "BorrowedDebugTensorDataLifetimeRuntimeReady", "ProcessDebugTensorRuntimeReady",
                "FullPackageConsumerRuntimeEvidenceReady", "CanImplementNativeAttach", "CanAttemptRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtDebugTensorMetadataSnapshot.cs",
            new[]
            {
                "TensorName", "TensorNameLength", "DataType", "Location", "TensorShapeRank", "ShapeSummary",
                "IsInput", "IsOutput", "IsShapeTensor", "IsExecutionTensor", "MetadataCopied"
            }
        },
        {
            "TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "TensorRtLine", "RuntimePackageKey", "Metadata",
                "TrampolineShapeReady", "NativeCallbackEntryLocated", "NoThrowCallbackEntryReady",
                "ExceptionCaptureReady", "CallbackStatusMappingReady", "InFlightAccountingReady",
                "DetachBeforeReleaseReady", "BorrowedDebugTensorMetadataCopyReady",
                "BorrowedDebugTensorPointerExposed", "BorrowedDebugTensorDataPointerExposed",
                "PointerFreeSurfaceReady", "ProcessDebugTensorRuntimeReady", "OptInEnabled",
                "FullPackageConsumerReport", "AttachAttempted", "AttachSucceeded", "NativeVTableInstalled",
                "ProcessDebugTensorInvoked", "InvocationCount", "CallbackStubEntryCount", "CallbackStubLeaveCount",
                "FailureCount", "InFlightCallbackCount", "LastStatus", "LastDiagnostic",
                "CanPromoteRealCallbackRuntime", "CanAttemptRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "ReasonRuntimeProofStillBlocked", "BlockedPrerequisites",
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
    [MemberData(nameof(ModelProperties))]
    public void ModelsKeepExactPublicProperties(string fileName, string[] expectedProperties)
    {
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(ReadSource(fileName)));
    }

    [Fact]
    public void ModelsKeepOneInternalConstructorEach()
    {
        foreach ((string fileName, string typeName) in new[]
                 {
                     ("TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.cs",
                         "TensorRtDebugListenerNativeOwnerLifecycleDryRunResult"),
                     ("TensorRtDebugTensorMetadataSnapshot.cs", "TensorRtDebugTensorMetadataSnapshot"),
                     ("TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.cs",
                         "TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult")
                 })
        {
            Assert.Single(Regex.Matches(
                ReadSource(fileName),
                $@"^    internal {typeName}\(",
                RegexOptions.Multiline));
        }
    }

    [Fact]
    public void EvaluatorsDoNotOwnModelDeclarations()
    {
        string dryRun = ReadSource("TensorRtDebugListenerNativeOwnerLifecycleDryRun.cs");
        string trampoline = ReadSource("TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.cs");

        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerNativeOwnerLifecycleDryRunResult",
            dryRun,
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugTensorMetadataSnapshot",
            trampoline,
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult",
            trampoline,
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedModelsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.cs",
                     "TensorRtDebugTensorMetadataSnapshot.cs",
                     "TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.cs"
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
            "TensorRtDebugListenerNativeOwnerLifecycleDryRun.cs",
            "TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.cs",
            "TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.cs",
            "TensorRtDebugTensorMetadataSnapshot.cs",
            "TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "DebugListenerNativeOwnerLifecycleDryRunTests.cs",
                     "DebugListenerProcessDebugTensorCallbackTrampolineTests.cs"
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
    public void DocsReferenceDedicatedModelOwners()
    {
        string dryRunDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-owner-lifecycle-dry-run.md"));
        string roadmap = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "callback-allocator-safety-bridge-roadmap.md"));

        Assert.Contains("TensorRtDebugListenerNativeOwnerLifecycleDryRun.cs", dryRunDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.cs", dryRunDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.cs", roadmap, StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugTensorMetadataSnapshot.cs", roadmap, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.cs",
            roadmap,
            StringComparison.Ordinal);
    }

    [Fact]
    public void LifecycleDryRunFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerNativeOwnerLifecycleDryRun.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.cs"));
        Assert.Equal(LifecycleDryRunOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void CallbackTrampolineFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugTensorMetadataSnapshot.cs"));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.cs"));
        Assert.Equal(CallbackTrampolineOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
            "Debugging",
            fileName));
    }
}
