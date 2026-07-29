using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerVTableExperimentOwnerAddressSourceLayoutTests
{
    private const string VTableExperimentOriginalNormalizedSha256 =
        "07e05d1b0e898397d08d7f5d36f732377172279d8a8eb679a87fdb7d253627ce";
    private const string OwnerAddressGateOriginalNormalizedSha256 =
        "f796fe581509555374ae79fe0b1cbddcccd6aa0d2d34240afdbacdcb12e75c38";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerNativeOwnerVTableInstallExperiment.cs",
            "TensorRtDebugListenerNativeOwnerVTableInstallExperiment",
            new[] { "Evaluate", "Evaluate", "BuildExperimentBlockedReason", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.cs",
            "TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult",
            new[] { "ToString" }
        },
        {
            "TensorRtDebugListenerNativeOwnerAddressDesignGate.cs",
            "TensorRtDebugListenerNativeOwnerAddressDesignGate",
            new[]
            {
                "Evaluate", "Evaluate", "Evaluate", "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker"
            }
        },
        {
            "TensorRtDebugListenerNativeOwnerAddressDesignGateResult.cs",
            "TensorRtDebugListenerNativeOwnerAddressDesignGateResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus", "NativeOwnerLifecycleGateReady",
                "NativeAttachBridgeShapeGateReady", "NativeNoThrowVTableScaffoldGateReady",
                "BorrowedDebugTensorMetadataGateReady", "NativeVTableInstallPreflightReady", "ExperimentShapeReady",
                "InstallAttemptGuardReady", "NonNullAttachEnabled", "RuntimeProofEnabled",
                "NativeVTableInstallAttempted", "NativeVTableInstalled", "RollbackReady", "DetachBeforeReleaseReady",
                "FailureStatusMappingReady", "PointerFree", "VTableAddressExposed", "VTablePointerProduced",
                "DebugTensorPointerExposed", "DebugTensorDataPointerExposed", "CanEnableSetDebugListenerNonNull",
                "CanInstallNativeVTable", "ProcessDebugTensorRuntimeReady", "CanCallProcessDebugTensorRuntime",
                "CanAttemptRuntimeProof", "RuntimeProofBlocked", "DeferredRowsStillRequired",
                "ReasonNativeOwnerVTableInstallStillBlocked", "BlockedPrerequisites", "BlockedPrerequisiteCount",
                "Status", "Diagnostic"
            }
        },
        {
            "TensorRtDebugListenerNativeOwnerAddressDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "NativeAttachNoThrowPreflightReady",
                "NativeAttachEntryLocated", "NativeDetachEntryLocated", "StableNativeOwnerAddressReady",
                "StableNativeOwnerAddressDesignReady", "ManagedCallbackKeepAliveDesignReady",
                "NativeOwnerNonCopyableReady", "NativeOwnerDisposeOrderReady", "NativeOwnerReleaseHookReady",
                "NativeOwnerInFlightDrainReady", "NoThrowNativeDestructorReady", "NoThrowVTableDesignReady",
                "ExceptionToStatusMappingDesignReady", "BorrowedDebugTensorMetadataCopyDesignReady",
                "BorrowedDebugTensorPointerEscapeBlocked", "NativeOwnerLifecycleReady", "NativeVTableDesignReady",
                "BorrowedDebugTensorLifetimeRuntimeReady", "BorrowedDebugTensorDataLifetimeRuntimeReady",
                "ProcessDebugTensorRuntimeReady", "FullPackageConsumerRuntimeEvidenceReady", "DesignGateReady",
                "CanImplementNativeAttach", "CanAttemptRuntimeProof", "RuntimeProofBlocked",
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
                     ("TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.cs",
                         "TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult"),
                     ("TensorRtDebugListenerNativeOwnerAddressDesignGateResult.cs",
                         "TensorRtDebugListenerNativeOwnerAddressDesignGateResult")
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
            "public readonly struct TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult",
            ReadSource("TensorRtDebugListenerNativeOwnerVTableInstallExperiment.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerNativeOwnerAddressDesignGateResult",
            ReadSource("TensorRtDebugListenerNativeOwnerAddressDesignGate.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.cs",
                     "TensorRtDebugListenerNativeOwnerAddressDesignGateResult.cs"
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
            "TensorRtDebugListenerNativeOwnerVTableInstallExperiment.cs",
            "TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.cs",
            "TensorRtDebugListenerNativeOwnerAddressDesignGate.cs",
            "TensorRtDebugListenerNativeOwnerAddressDesignGateResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "DebugListenerNativeOwnerVTableInstallExperimentTests.cs",
                     "DebugListenerNativeOwnerAddressDesignGateTests.cs"
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
        string experimentDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-owner-vtable-install-experiment.md"));
        string addressDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-owner-address-design-gate.md"));

        Assert.Contains(
            "TensorRtDebugListenerNativeOwnerVTableInstallExperiment.cs",
            experimentDoc,
            StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.cs",
            experimentDoc,
            StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerNativeOwnerAddressDesignGate.cs", addressDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerNativeOwnerAddressDesignGateResult.cs",
            addressDoc,
            StringComparison.Ordinal);
    }

    [Fact]
    public void VTableExperimentFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtDebugListenerNativeOwnerVTableInstallExperiment.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.cs"));
        Assert.Equal(VTableExperimentOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void OwnerAddressFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerNativeOwnerAddressDesignGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerNativeOwnerAddressDesignGateResult.cs"));
        Assert.Equal(OwnerAddressGateOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
