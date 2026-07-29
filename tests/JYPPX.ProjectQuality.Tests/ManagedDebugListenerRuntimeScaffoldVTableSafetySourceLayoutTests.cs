using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerRuntimeScaffoldVTableSafetySourceLayoutTests
{
    private const string RuntimeScaffoldOriginalNormalizedSha256 =
        "4ec37c83a74e45d93e3fc3e9150e0c97f979de2e579d4aec9a56fb001c58c6f5";
    private const string VTableSafetyOriginalNormalizedSha256 =
        "1a1dcc857be13403a63a55a73581a144b5c8bf87f422d45c72ba5b4121811dde";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.cs",
            "TensorRtDebugListenerNativeAttachEntryRuntimeScaffold",
            new[] { "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.cs",
            "TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult",
            new[] { "ToString" }
        },
        {
            "TensorRtDebugListenerAttachVTableSafetyGate.cs",
            "TensorRtDebugListenerAttachVTableSafetyGate",
            new[] { "Evaluate", "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerAttachVTableSafetyGateResult.cs",
            "TensorRtDebugListenerAttachVTableSafetyGateResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus",
                "NativeOwnerLifecycleDryRunReady", "NativeAttachEntryLocated", "NativeDetachEntryLocated",
                "AttachEntryParameterShapeReady", "AttachEntryVersionGuardReady", "AttachEntryNoThrowBoundaryReady",
                "AttachEntryOwnershipDiagnosticsReady", "StableNativeOwnerIdentityReady",
                "NativeOwnerNonCopyableReady", "NoThrowNativeDestructorReady", "NativeOwnerLifecycleReady",
                "RuntimeScaffoldReady", "ProcessDebugTensorRuntimeReady", "FullPackageConsumerRuntimeEvidenceReady",
                "CanImplementNativeAttach", "CanAttemptRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "BlockedPrerequisites", "BlockedPrerequisiteCount", "Status",
                "Diagnostic"
            }
        },
        {
            "TensorRtDebugListenerAttachVTableSafetyGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "LineSupportsDebugListener", "OwnerDesignReady",
                "AttachDetachDesignGateReady", "BorrowedTensorSafetyGateReady", "ManagedOwnerStateMachineReady",
                "DebugTensorMetadataCopied", "PointerFreeSurfaceReady",
                "BorrowedDebugTensorPointerEscapeBlocked", "DetachClearControlAvailable", "AttachControlAvailable",
                "LineSpecificAttachDetachReady", "StableNativeOwnerAddressReady", "NoThrowNativeVTableReady",
                "NativeVTableReady", "ExceptionToStatusMappingReady", "BorrowedDebugTensorLifetimeReady",
                "BorrowedDebugTensorDataLifetimeReady", "ProcessDebugTensorRuntimeReady",
                "FullPackageConsumerRuntimeEvidenceReady", "SafetyGateReady", "CanAttemptRuntimeProof",
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
                     ("TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.cs",
                         "TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult"),
                     ("TensorRtDebugListenerAttachVTableSafetyGateResult.cs",
                         "TensorRtDebugListenerAttachVTableSafetyGateResult")
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
            "public readonly struct TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult",
            ReadSource("TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerAttachVTableSafetyGateResult",
            ReadSource("TensorRtDebugListenerAttachVTableSafetyGate.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.cs",
                     "TensorRtDebugListenerAttachVTableSafetyGateResult.cs"
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
            "TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.cs",
            "TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.cs",
            "TensorRtDebugListenerAttachVTableSafetyGate.cs",
            "TensorRtDebugListenerAttachVTableSafetyGateResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "DebugListenerNativeAttachEntryRuntimeScaffoldTests.cs",
                     "DebugListenerAttachVTableSafetyGateTests.cs"
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
        string scaffoldDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-attach-entry-runtime-scaffold.md"));
        string safetyDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-attach-vtable-safety-gate.md"));

        Assert.Contains("TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.cs", scaffoldDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.cs",
            scaffoldDoc,
            StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerAttachVTableSafetyGate.cs", safetyDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerAttachVTableSafetyGateResult.cs", safetyDoc, StringComparison.Ordinal);
    }

    [Fact]
    public void RuntimeScaffoldFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.cs"));
        Assert.Equal(RuntimeScaffoldOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void VTableSafetyFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerAttachVTableSafetyGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerAttachVTableSafetyGateResult.cs"));
        Assert.Equal(VTableSafetyOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
