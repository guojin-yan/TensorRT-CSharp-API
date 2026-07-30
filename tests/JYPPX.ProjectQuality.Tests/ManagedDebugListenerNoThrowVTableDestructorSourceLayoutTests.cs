using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerNoThrowVTableDestructorSourceLayoutTests
{
    private const string VTableScaffoldGateOriginalNormalizedSha256 =
        "01f6ef36f1568ed32fb484297b7068bd469c79f12c26458467a3bd8ccfcc1d98";
    private const string DestructorOriginalNormalizedSha256 =
        "995fdb1b4619cc3893e519c2467e6a261d7e69c64a1f34d028f5a53260de2072";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs",
            "TensorRtDebugListenerNativeNoThrowVTableScaffoldGate",
            new[] { "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.cs",
            "TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult",
            new[] { "ToString" }
        },
        {
            "TensorRtDebugListenerNativeNoThrowDestructor.cs",
            "TensorRtDebugListenerNativeNoThrowDestructor",
            new[] { "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeNoThrowDestructorResult.cs",
            "TensorRtDebugListenerNativeNoThrowDestructorResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus",
                "NativeAttachBridgeShapeGateReady", "ExceptionStatusMappingGateReady",
                "InFlightAccountingGateReady", "NoThrowVTableScaffoldReady", "VTableDestructorNoThrowReady",
                "ProcessDebugTensorCallbackStubNoThrowReady", "ExceptionEscapeBlocked",
                "CallbackExceptionCaptureGateReady", "CallbackStatusMappingGateReady",
                "CallbackInFlightAccountingGateReady", "BorrowedDebugTensorPointerEscapeBlocked",
                "VTableAddressExposed", "VTablePointerProduced", "NativeAttachEntryLocated",
                "NativeVTableDesignReady", "VTableScaffoldGateReady", "ProcessDebugTensorRuntimeReady",
                "FullPackageConsumerRuntimeEvidenceReady", "CanImplementNativeAttach", "CanAttemptRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtDebugListenerNativeNoThrowDestructorResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus", "LastDiagnostic",
                "ReleaseDiagnostic", "NativeOwnerNonCopyableStorageReady", "NativeOwnerNonCopyableReady",
                "NativeOwnerCopyBlocked", "NativeOwnerMoveBlocked", "NativeOwnerAddressExposed",
                "NativeOwnerPointerProduced", "DestructorNoThrowScaffoldReady",
                "DestructorExceptionEscapeBlocked", "DestructorAddressExposed", "DestructorPointerProduced",
                "NativeAttachEntryLocated", "NativeDetachEntryLocated", "NoThrowNativeDestructorReady",
                "NativeOwnerLifecycleReady", "ProcessDebugTensorRuntimeReady",
                "FullPackageConsumerRuntimeEvidenceReady", "CanImplementNativeAttach", "CanAttemptRuntimeProof",
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
                     ("TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.cs",
                         "TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult"),
                     ("TensorRtDebugListenerNativeNoThrowDestructorResult.cs",
                         "TensorRtDebugListenerNativeNoThrowDestructorResult")
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
            "public readonly struct TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult",
            ReadSource("TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerNativeNoThrowDestructorResult",
            ReadSource("TensorRtDebugListenerNativeNoThrowDestructor.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.cs",
                     "TensorRtDebugListenerNativeNoThrowDestructorResult.cs"
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
            "TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs",
            "TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.cs",
            "TensorRtDebugListenerNativeNoThrowDestructor.cs",
            "TensorRtDebugListenerNativeNoThrowDestructorResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "DebugListenerAttachBridgeVTableBatchTests.cs",
                     "DebugListenerNativeNoThrowDestructorTests.cs"
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
            "debug-listener-native-nothrow-vtable-scaffold-gate.md"));
        string destructorDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-nothrow-destructor.md"));

        Assert.Contains("TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs", vtableDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.cs",
            vtableDoc,
            StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerNativeNoThrowDestructor.cs", destructorDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerNativeNoThrowDestructorResult.cs", destructorDoc, StringComparison.Ordinal);
    }

    [Fact]
    public void VTableScaffoldGateFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.cs"));
        Assert.Equal(VTableScaffoldGateOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void DestructorFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerNativeNoThrowDestructor.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerNativeNoThrowDestructorResult.cs"));
        Assert.Equal(DestructorOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
