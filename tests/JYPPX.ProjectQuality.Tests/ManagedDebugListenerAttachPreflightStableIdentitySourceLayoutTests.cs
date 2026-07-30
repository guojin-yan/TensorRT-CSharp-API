using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerAttachPreflightStableIdentitySourceLayoutTests
{
    private const string AttachPreflightOriginalNormalizedSha256 =
        "aa752d93a7ca0a607fe67530c7499e0a912faa6f40523999fcc33f03adc7a549";
    private const string StableIdentityOriginalNormalizedSha256 =
        "a8225511214e996bce81afd31d7dcaa9e44898afd7ba67be809bfa252b2f75a9";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerNativeAttachNoThrowPreflight.cs",
            "TensorRtDebugListenerNativeAttachNoThrowPreflight",
            new[] { "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeAttachNoThrowPreflightResult.cs",
            "TensorRtDebugListenerNativeAttachNoThrowPreflightResult",
            new[] { "ToString" }
        },
        {
            "TensorRtDebugListenerNativeOwnerStableIdentity.cs",
            "TensorRtDebugListenerNativeOwnerStableIdentity",
            new[] { "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeOwnerStableIdentityResult.cs",
            "TensorRtDebugListenerNativeOwnerStableIdentityResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtDebugListenerNativeAttachNoThrowPreflightResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "AttachVTableSafetyGateReady",
                "NativeAttachEntryLocated", "NativeDetachEntryLocated", "StableNativeOwnerAddressDesignReady",
                "ManagedCallbackKeepAliveDesignReady", "NoThrowVTableDesignReady",
                "ExceptionToStatusMappingDesignReady", "BorrowedDebugTensorMetadataCopyDesignReady",
                "BorrowedDebugTensorPointerEscapeBlocked", "NativeVTableDesignReady",
                "BorrowedDebugTensorLifetimeRuntimeReady", "BorrowedDebugTensorDataLifetimeRuntimeReady",
                "ProcessDebugTensorRuntimeReady", "FullPackageConsumerRuntimeEvidenceReady",
                "PreflightReady", "CanImplementNativeAttach", "CanAttemptRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtDebugListenerNativeOwnerStableIdentityResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus",
                "LastDiagnostic", "ReleaseDiagnostic", "NativeAttachEntryRuntimeScaffoldReady",
                "StableNativeOwnerIdentityReady", "NativeOwnerNonCopyableReady",
                "OwnerIdentityDiagnosticsReady", "OwnerIdentityPointerFree", "NativeAttachEntryLocated",
                "NativeDetachEntryLocated", "NoThrowNativeDestructorReady", "NativeOwnerLifecycleReady",
                "ProcessDebugTensorRuntimeReady", "FullPackageConsumerRuntimeEvidenceReady",
                "CanImplementNativeAttach", "CanAttemptRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "BlockedPrerequisites", "BlockedPrerequisiteCount",
                "Status", "Diagnostic"
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
                     ("TensorRtDebugListenerNativeAttachNoThrowPreflightResult.cs",
                         "TensorRtDebugListenerNativeAttachNoThrowPreflightResult"),
                     ("TensorRtDebugListenerNativeOwnerStableIdentityResult.cs",
                         "TensorRtDebugListenerNativeOwnerStableIdentityResult")
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
            "public readonly struct TensorRtDebugListenerNativeAttachNoThrowPreflightResult",
            ReadSource("TensorRtDebugListenerNativeAttachNoThrowPreflight.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerNativeOwnerStableIdentityResult",
            ReadSource("TensorRtDebugListenerNativeOwnerStableIdentity.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerNativeAttachNoThrowPreflightResult.cs",
                     "TensorRtDebugListenerNativeOwnerStableIdentityResult.cs"
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
            "TensorRtDebugListenerNativeAttachNoThrowPreflight.cs",
            "TensorRtDebugListenerNativeAttachNoThrowPreflightResult.cs",
            "TensorRtDebugListenerNativeOwnerStableIdentity.cs",
            "TensorRtDebugListenerNativeOwnerStableIdentityResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        string preflightConsumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "DebugListenerNativeAttachNoThrowPreflightTests.cs"));
        string identityConsumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "DebugListenerNativeOwnerStableIdentityTests.cs"));
        Assert.Contains("RepositorySourceReader.Read", preflightConsumer, StringComparison.Ordinal);
        Assert.Contains("RepositorySourceReader.Read", identityConsumer, StringComparison.Ordinal);
    }

    [Fact]
    public void DocsReferenceDedicatedResultOwners()
    {
        string preflightDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-attach-nothrow-preflight.md"));
        string identityDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-owner-stable-identity.md"));

        Assert.Contains("TensorRtDebugListenerNativeAttachNoThrowPreflight.cs", preflightDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerNativeAttachNoThrowPreflightResult.cs",
            preflightDoc,
            StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerNativeOwnerStableIdentity.cs", identityDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerNativeOwnerStableIdentityResult.cs",
            identityDoc,
            StringComparison.Ordinal);
    }

    [Fact]
    public void AttachPreflightFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtDebugListenerNativeAttachNoThrowPreflight.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtDebugListenerNativeAttachNoThrowPreflightResult.cs"));
        Assert.Equal(AttachPreflightOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void StableIdentityFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtDebugListenerNativeOwnerStableIdentity.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtDebugListenerNativeOwnerStableIdentityResult.cs"));
        Assert.Equal(StableIdentityOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
