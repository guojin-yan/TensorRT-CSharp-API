using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerVTablePreflightDetachGateSourceLayoutTests
{
    private const string VTablePreflightOriginalNormalizedSha256 =
        "e1ea5ab7ca05ce765c05502b16909793b4d21f449ada90c54a668973653a9462";
    private const string DetachGateOriginalNormalizedSha256 =
        "742672ca8af41077ba376e9ebd8fd1d60d6192c2d8a684ec00fad2af43be444d";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerNativeVTableInstallPreflight.cs",
            "TensorRtDebugListenerNativeVTableInstallPreflight",
            new[] { "Evaluate", "Evaluate", "BuildNativeVTableInstallBlockedReason", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeVTableInstallPreflightResult.cs",
            "TensorRtDebugListenerNativeVTableInstallPreflightResult",
            new[] { "ToString" }
        },
        {
            "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.cs",
            "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate",
            new[] { "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.cs",
            "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtDebugListenerNativeVTableInstallPreflightResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus", "NativeOwnerLifecycleGateReady",
                "NativeAttachBridgeShapeGateReady", "NativeNoThrowVTableScaffoldGateReady",
                "BorrowedDebugTensorMetadataGateReady", "VTableInstallShapeReady", "VTableInstallVersionGuardReady",
                "VTableInstallNoThrowBoundaryReady", "VTableInstallOwnershipDiagnosticsReady",
                "VTableInstallPointerFree", "AttachBridgeSetDebugListenerNonNullEnabled",
                "SetDebugListenerNonNullEnabled", "NonNullAttachStillDisabled", "VTableAddressExposed",
                "VTablePointerProduced", "DebugTensorPointerExposed", "DebugTensorDataPointerExposed",
                "BorrowedDebugTensorPointerEscapeBlocked", "BorrowedDebugTensorDataPointerEscapeBlocked",
                "BorrowedDebugTensorLifetimeReady", "BorrowedDebugTensorDataLifetimeReady",
                "NativeVTableInstallPreflightReady", "NativeVTableInstalled", "NativeVTableInstallRuntimeReady",
                "CanEnableSetDebugListenerNonNull", "CanInstallNativeVTable", "ProcessDebugTensorRuntimeReady",
                "CanCallProcessDebugTensorRuntime", "FullPackageConsumerRuntimeEvidenceReady",
                "CanAttemptRuntimeProof", "RuntimeProofBlocked", "DeferredRowsStillRequired",
                "ReasonNativeVTableInstallStillBlocked", "BlockedPrerequisites", "BlockedPrerequisiteCount",
                "Status", "Diagnostic"
            }
        },
        {
            "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "NativeAttachEntryDesignGateReady",
                "NativeNoThrowVTableDesignGateReady", "NativeOwnerAddressDesignGateReady",
                "NativeDetachEntryLocated", "NativeAttachEntryLocated", "LineSpecificAttachEntryDesignReady",
                "AttachEntryNoThrowReady", "AttachEntryVersionGuardReady", "AttachEntryOwnershipReady",
                "DetachBeforeReleaseReady", "ReleaseHookOrderingReady", "DisposeIdempotencyReady",
                "InFlightDrainBeforeReleaseReady", "CallbackStateUnpinAfterDetachReady",
                "DelegateUnpinAfterDetachReady", "NativeOwnerLifecycleReady", "NativeVTableDesignReady",
                "ManagedCallbackKeepAliveDesignReady", "BorrowedDebugTensorMetadataCopyDesignReady",
                "BorrowedDebugTensorPointerEscapeBlocked", "DesignGateReady", "CanImplementNativeAttach",
                "BorrowedDebugTensorLifetimeRuntimeReady", "BorrowedDebugTensorDataLifetimeRuntimeReady",
                "ProcessDebugTensorRuntimeReady", "FullPackageConsumerRuntimeEvidenceReady",
                "CanAttemptRuntimeProof", "RuntimeProofBlocked", "DeferredRowsStillRequired",
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
                     ("TensorRtDebugListenerNativeVTableInstallPreflightResult.cs",
                         "TensorRtDebugListenerNativeVTableInstallPreflightResult"),
                     ("TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.cs",
                         "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult")
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
            "public readonly struct TensorRtDebugListenerNativeVTableInstallPreflightResult",
            ReadSource("TensorRtDebugListenerNativeVTableInstallPreflight.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult",
            ReadSource("TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerNativeVTableInstallPreflightResult.cs",
                     "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.cs"
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
            "TensorRtDebugListenerNativeVTableInstallPreflight.cs",
            "TensorRtDebugListenerNativeVTableInstallPreflightResult.cs",
            "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.cs",
            "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "DebugListenerNativeVTableInstallPreflightTests.cs",
                     "DebugListenerNativeDetachBeforeReleaseDesignGateTests.cs"
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
        string preflightDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-vtable-install-preflight.md"));
        string detachDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-detach-before-release-design-gate.md"));

        Assert.Contains("TensorRtDebugListenerNativeVTableInstallPreflight.cs", preflightDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerNativeVTableInstallPreflightResult.cs",
            preflightDoc,
            StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.cs",
            detachDoc,
            StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.cs",
            detachDoc,
            StringComparison.Ordinal);
    }

    [Fact]
    public void VTablePreflightFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerNativeVTableInstallPreflight.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerNativeVTableInstallPreflightResult.cs"));
        Assert.Equal(VTablePreflightOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void DetachGateFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.cs"));
        Assert.Equal(DetachGateOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
