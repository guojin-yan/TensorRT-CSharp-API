using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerNativeOwnerBorrowedTensorSourceLayoutTests
{
    private const string NativeOwnerStorageOriginalNormalizedSha256 =
        "8f8a50ad374445a5127409ebd089c904c00878e03d5c5bb6151640e67d356d1c";
    private const string BorrowedTensorSafetyOriginalNormalizedSha256 =
        "1b247e203981d698ee2c39fe68a5c1afc7512cfc4c6d3e14f833314d557974c7";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerNativeOwnerNonCopyableStorage.cs",
            "TensorRtDebugListenerNativeOwnerNonCopyableStorage",
            new[] { "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.cs",
            "TensorRtDebugListenerNativeOwnerNonCopyableStorageResult",
            new[] { "ToString" }
        },
        {
            "TensorRtDebugListenerBorrowedTensorSafetyGate.cs",
            "TensorRtDebugListenerBorrowedTensorSafetyGate",
            new[] { "Evaluate", "Evaluate" }
        },
        {
            "TensorRtDebugListenerBorrowedTensorSafetyGateResult.cs",
            "TensorRtDebugListenerBorrowedTensorSafetyGateResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus", "LastDiagnostic",
                "ReleaseDiagnostic", "NativeOwnerStableIdentityReady", "OwnerIdentityDiagnosticsReady",
                "OwnerIdentityPointerFree", "NativeOwnerNonCopyableReady", "NativeOwnerCopyBlocked",
                "NativeOwnerMoveBlocked", "NativeOwnerAddressExposed", "NativeOwnerPointerProduced",
                "NativeAttachEntryLocated", "NativeDetachEntryLocated", "NoThrowNativeDestructorReady",
                "NativeOwnerLifecycleReady", "ProcessDebugTensorRuntimeReady",
                "FullPackageConsumerRuntimeEvidenceReady", "CanImplementNativeAttach", "CanAttemptRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtDebugListenerBorrowedTensorSafetyGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "TensorName", "DataType", "Location", "ShapeRank",
                "ShapeSummary", "IsInput", "IsOutput", "IsShapeTensor", "IsExecutionTensor",
                "ProcessDebugTensorCount", "AttachDetachDesignGateReady", "OwnerDesignReady",
                "DebugTensorMetadataCopied", "PointerFreeSurfaceReady", "BorrowedDebugTensorPointerEscapeBlocked",
                "BorrowedDebugTensorLifetimeReady", "BorrowedDebugTensorDataLifetimeReady",
                "ProcessDebugTensorRuntimeReady", "FullPackageConsumerRuntimeEvidenceReady", "SafetyGateReady",
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
                     ("TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.cs",
                         "TensorRtDebugListenerNativeOwnerNonCopyableStorageResult"),
                     ("TensorRtDebugListenerBorrowedTensorSafetyGateResult.cs",
                         "TensorRtDebugListenerBorrowedTensorSafetyGateResult")
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
            "public readonly struct TensorRtDebugListenerNativeOwnerNonCopyableStorageResult",
            ReadSource("TensorRtDebugListenerNativeOwnerNonCopyableStorage.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerBorrowedTensorSafetyGateResult",
            ReadSource("TensorRtDebugListenerBorrowedTensorSafetyGate.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.cs",
                     "TensorRtDebugListenerBorrowedTensorSafetyGateResult.cs"
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
            "TensorRtDebugListenerNativeOwnerNonCopyableStorage.cs",
            "TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.cs",
            "TensorRtDebugListenerBorrowedTensorSafetyGate.cs",
            "TensorRtDebugListenerBorrowedTensorSafetyGateResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "DebugListenerNativeOwnerNonCopyableStorageTests.cs",
                     "DebugListenerBorrowedTensorSafetyGateTests.cs"
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
        string storageDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-native-owner-noncopyable-storage.md"));
        string borrowedDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-borrowed-tensor-safety-gate.md"));

        Assert.Contains("TensorRtDebugListenerNativeOwnerNonCopyableStorage.cs", storageDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.cs",
            storageDoc,
            StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerBorrowedTensorSafetyGate.cs", borrowedDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerBorrowedTensorSafetyGateResult.cs",
            borrowedDoc,
            StringComparison.Ordinal);
    }

    [Fact]
    public void NativeOwnerStorageFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerNativeOwnerNonCopyableStorage.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.cs"));
        Assert.Equal(NativeOwnerStorageOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void BorrowedTensorSafetyFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerBorrowedTensorSafetyGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerBorrowedTensorSafetyGateResult.cs"));
        Assert.Equal(BorrowedTensorSafetyOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
