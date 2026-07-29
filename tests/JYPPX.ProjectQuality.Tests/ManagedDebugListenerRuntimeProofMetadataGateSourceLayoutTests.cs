using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerRuntimeProofMetadataGateSourceLayoutTests
{
    private const string RuntimeProofOriginalNormalizedSha256 =
        "291d939ad471570afeca435f97f8f566cfa200293314ae5d7840decf6e45db56";
    private const string MetadataGateOriginalNormalizedSha256 =
        "138405aaa834da13b4a206f339fb172dff6ab2e571a90893fbd6376616b8150d";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerRealCallbackRuntimeProof.cs",
            "TensorRtDebugListenerRealCallbackRuntimeProof",
            new[] { "Evaluate", "Evaluate", "BuildDiagnostic", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerRealCallbackRuntimeProofResult.cs",
            "TensorRtDebugListenerRealCallbackRuntimeProofResult",
            new[] { "ToString" }
        },
        {
            "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.cs",
            "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate",
            new[]
            {
                "Evaluate", "Evaluate", "BuildMetadataRuntimeBlockedReason", "AddBlockerIfFalse", "AddBlocker"
            }
        },
        {
            "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.cs",
            "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtDebugListenerRealCallbackRuntimeProofResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "TensorRtLine", "RuntimePackageKey", "OptInEnabled",
                "FullPackageConsumerReport", "RuntimeSmokeReady", "TrampolineShapeReady", "AttachAttempted",
                "AttachSucceeded", "DetachAttempted", "DetachSucceeded", "RollbackAttempted", "RollbackSucceeded",
                "NativeVTableInstalled", "ProcessDebugTensorInvoked", "InvocationCount", "FailureCount",
                "InFlightCallbackCount", "BorrowedDebugTensorMetadataCopied", "PointerFreeSurfaceReady",
                "ProcessDebugTensorRuntimeReady", "AttemptedNoInvocation", "LastStatus", "LastDiagnostic",
                "CanPromoteRealCallbackRuntime", "CanAttemptRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "ReasonRuntimeProofStillBlocked", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus", "TensorName", "TensorNameLength",
                "DataType", "Location", "TensorShapeRank", "ShapeSummary", "IsInput", "IsOutput", "IsShapeTensor",
                "IsExecutionTensor", "BorrowedTensorSafetyGateReady", "CallbackStubGateReady", "MetadataGateReady",
                "TensorNameCopied", "TensorTypeCopied", "TensorLocationCopied", "TensorShapeCopied",
                "TensorFlagsCopied", "BorrowedDebugTensorMetadataCopyReady",
                "BorrowedDebugTensorPointerEscapeBlocked", "BorrowedDebugTensorDataPointerEscapeBlocked",
                "DebugTensorPointerExposed", "DebugTensorDataPointerExposed", "BorrowedDebugTensorLifetimeReady",
                "BorrowedDebugTensorDataLifetimeReady", "SetDebugListenerNonNullEnabled", "NativeVTableInstalled",
                "ProcessDebugTensorRuntimeReady", "CanCallProcessDebugTensorRuntime", "CanAttemptRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "ReasonMetadataRuntimeStillBlocked",
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
                     ("TensorRtDebugListenerRealCallbackRuntimeProofResult.cs",
                         "TensorRtDebugListenerRealCallbackRuntimeProofResult"),
                     ("TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.cs",
                         "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult")
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
            "public readonly struct TensorRtDebugListenerRealCallbackRuntimeProofResult",
            ReadSource("TensorRtDebugListenerRealCallbackRuntimeProof.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult",
            ReadSource("TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerRealCallbackRuntimeProofResult.cs",
                     "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.cs"
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
            "TensorRtDebugListenerRealCallbackRuntimeProof.cs",
            "TensorRtDebugListenerRealCallbackRuntimeProofResult.cs",
            "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.cs",
            "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "DebugListenerRealCallbackRuntimeProofTests.cs",
                     "DebugListenerBorrowedDebugTensorMetadataRuntimeGateTests.cs"
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
        string proofDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-real-callback-runtime-proof.md"));
        string metadataDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-borrowed-debug-tensor-metadata-runtime-gate.md"));

        Assert.Contains("TensorRtDebugListenerRealCallbackRuntimeProof.cs", proofDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerRealCallbackRuntimeProofResult.cs",
            proofDoc,
            StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.cs",
            metadataDoc,
            StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.cs",
            metadataDoc,
            StringComparison.Ordinal);
    }

    [Fact]
    public void RuntimeProofFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerRealCallbackRuntimeProof.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerRealCallbackRuntimeProofResult.cs"));
        Assert.Equal(RuntimeProofOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void MetadataGateFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.cs"));
        Assert.Equal(MetadataGateOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
