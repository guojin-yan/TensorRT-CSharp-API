using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerNoThrowVTableMinimalSafetySourceLayoutTests
{
    private const string VTableGateOriginalNormalizedSha256 =
        "101d1d5077485fd67afdeb75b5ca011ab7f2b2650aa6bcdb5fe906f808633532";
    private const string MinimalSafetyOriginalNormalizedSha256 =
        "54fb0c554b02f6572ec07a792e1172ddc073bd85bdef1b0b567abe266ea0f4d6";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerNativeNoThrowVTableDesignGate.cs",
            "TensorRtDebugListenerNativeNoThrowVTableDesignGate",
            new[] { "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.cs",
            "TensorRtDebugListenerNativeNoThrowVTableDesignGateResult",
            new[] { "ToString" }
        },
        {
            "TensorRtDebugListenerNativeAttachEntryMinimalSafety.cs",
            "TensorRtDebugListenerNativeAttachEntryMinimalSafety",
            new[]
            {
                "Evaluate", "Evaluate", "Evaluate", "BuildNativeAttachBlockedReason", "AddBlockerIfFalse", "AddBlocker"
            }
        },
        {
            "TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.cs",
            "TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "NativeOwnerAddressDesignGateReady",
                "NativeAttachNoThrowPreflightReady", "NativeAttachEntryLocated", "NativeOwnerLifecycleReady",
                "ManagedCallbackKeepAliveDesignReady", "NoThrowNativeDestructorReady", "NoThrowVTableDesignReady",
                "ExceptionToStatusMappingDesignReady", "BorrowedDebugTensorMetadataCopyDesignReady",
                "BorrowedDebugTensorPointerEscapeBlocked", "NativeVTableTrampolineReady",
                "CallbackExceptionCaptureReady", "CallbackStatusMappingReady", "CallbackInFlightAccountingReady",
                "NativeVTableDesignReady", "DesignGateReady", "CanImplementNativeAttach",
                "BorrowedDebugTensorLifetimeRuntimeReady", "BorrowedDebugTensorDataLifetimeRuntimeReady",
                "ProcessDebugTensorRuntimeReady", "FullPackageConsumerRuntimeEvidenceReady", "CanAttemptRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus", "RuntimeScaffoldReady",
                "LifecycleGateReady", "LifecyclePointerFree", "NativeAttachEntryLocated",
                "NativeDetachEntryLocated", "AttachEntryParameterShapeReady", "AttachEntryNoThrowReady",
                "AttachEntryVersionGuardReady", "AttachEntryOwnershipDiagnosticsReady",
                "SetDebugListenerNonNullEnabled", "NonNullAttachStillDisabled", "NativeAttachWouldBeBlocked",
                "MinimalSafetyReady", "ProcessDebugTensorRuntimeReady", "FullPackageConsumerRuntimeEvidenceReady",
                "CanImplementNativeAttach", "CanAttemptRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "ReasonNativeAttachStillBlocked", "BlockedPrerequisites",
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
                     ("TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.cs",
                         "TensorRtDebugListenerNativeNoThrowVTableDesignGateResult"),
                     ("TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.cs",
                         "TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult")
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
            "public readonly struct TensorRtDebugListenerNativeNoThrowVTableDesignGateResult",
            ReadSource("TensorRtDebugListenerNativeNoThrowVTableDesignGate.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult",
            ReadSource("TensorRtDebugListenerNativeAttachEntryMinimalSafety.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.cs",
                     "TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.cs"
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
            "TensorRtDebugListenerNativeNoThrowVTableDesignGate.cs",
            "TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.cs",
            "TensorRtDebugListenerNativeAttachEntryMinimalSafety.cs",
            "TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "DebugListenerNativeNoThrowVTableDesignGateTests.cs",
                     "DebugListenerNativeAttachEntryMinimalSafetyTests.cs"
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
        string vtableDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-nothrow-vtable-design-gate.md"));
        string safetyDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-attach-entry-minimal-safety.md"));

        Assert.Contains("TensorRtDebugListenerNativeNoThrowVTableDesignGate.cs", vtableDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.cs", vtableDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryMinimalSafety.cs", safetyDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.cs", safetyDoc, StringComparison.Ordinal);
    }

    [Fact]
    public void VTableGateFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerNativeNoThrowVTableDesignGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.cs"));
        Assert.Equal(VTableGateOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void MinimalSafetyFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerNativeAttachEntryMinimalSafety.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.cs"));
        Assert.Equal(MinimalSafetyOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
