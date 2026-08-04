using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerAttachDetachExceptionStatusSourceLayoutTests
{
    private const string AttachDetachOriginalNormalizedSha256 =
        "0f62bbd560d75374ab67a8189e5e1cc76e20aba43b038251dc78f59a19d8e1f8";
    private const string ExceptionStatusOriginalNormalizedSha256 =
        "d5ab2a07e578fc1a781357f6eb66e594398c693c3ac328b3f1e699e692a23632";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerAttachDetachDesignGate.cs",
            "TensorRtDebugListenerAttachDetachDesignGate",
            new[] { "Evaluate" }
        },
        {
            "TensorRtDebugListenerAttachDetachDesignGateResult.cs",
            "TensorRtDebugListenerAttachDetachDesignGateResult",
            new[] { "ToString" }
        },
        {
            "TensorRtDebugListenerExceptionStatusMappingGate.cs",
            "TensorRtDebugListenerExceptionStatusMappingGate",
            new[] { "Evaluate", "Evaluate", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerExceptionStatusMappingGateResult.cs",
            "TensorRtDebugListenerExceptionStatusMappingGateResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtDebugListenerAttachDetachDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "LineSupportsDebugListener",
                "OwnerDesignReady", "ManagedOwnerStateMachineReady", "DebugTensorMetadataCopied",
                "PointerFreeSurfaceReady", "DetachClearControlAvailable", "AttachControlAvailable",
                "LineSpecificAttachDetachReady", "StableNativeOwnerAddressReady",
                "NoThrowNativeVTableReady", "NativeVTableReady", "BorrowedDebugTensorLifetimeReady",
                "DesignGateReady", "CanAttemptRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "BlockedPrerequisites", "BlockedPrerequisiteCount",
                "Status", "Diagnostic"
            }
        },
        {
            "TensorRtDebugListenerExceptionStatusMappingGateResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "LastStatus",
                "AttachBridgeShapeGateReady", "ManagedCallbackExceptionCaptureReady",
                "NativeCallbackExceptionCaptureReady", "CallbackStatusMappingGateReady",
                "ExceptionEscapeBlocked", "DiagnosticCopyReady", "MappingAddressExposed",
                "MappingPointerProduced", "NativeAttachEntryLocated", "ExceptionStatusMappingGateReady",
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
                     ("TensorRtDebugListenerAttachDetachDesignGateResult.cs",
                         "TensorRtDebugListenerAttachDetachDesignGateResult"),
                     ("TensorRtDebugListenerExceptionStatusMappingGateResult.cs",
                         "TensorRtDebugListenerExceptionStatusMappingGateResult")
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
            "public readonly struct TensorRtDebugListenerAttachDetachDesignGateResult",
            ReadSource("TensorRtDebugListenerAttachDetachDesignGate.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerExceptionStatusMappingGateResult",
            ReadSource("TensorRtDebugListenerExceptionStatusMappingGate.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerAttachDetachDesignGateResult.cs",
                     "TensorRtDebugListenerExceptionStatusMappingGateResult.cs"
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
            "TensorRtDebugListenerAttachDetachDesignGate.cs",
            "TensorRtDebugListenerAttachDetachDesignGateResult.cs",
            "TensorRtDebugListenerExceptionStatusMappingGate.cs",
            "TensorRtDebugListenerExceptionStatusMappingGateResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        string attachDetachConsumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "DebugListenerAttachDetachDesignGateTests.cs"));
        string batchConsumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "DebugListenerAttachBridgeVTableBatchTests.cs"));
        Assert.Contains("TensorRtDebugListenerAttachDetachDesignGate", attachDetachConsumer, StringComparison.Ordinal);
        Assert.Contains("RepositorySourceReader.Read", batchConsumer, StringComparison.Ordinal);
    }

    [Fact]
    public void DocsReferenceDedicatedResultOwners()
    {
        string attachDetachDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-attach-detach-design-gate.md"));
        string exceptionDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-exception-status-mapping-gate.md"));

        Assert.Contains("TensorRtDebugListenerAttachDetachDesignGate.cs", attachDetachDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerAttachDetachDesignGateResult.cs",
            attachDetachDoc,
            StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerExceptionStatusMappingGate.cs", exceptionDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerExceptionStatusMappingGateResult.cs",
            exceptionDoc,
            StringComparison.Ordinal);
    }

    [Fact]
    public void AttachDetachFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtDebugListenerAttachDetachDesignGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtDebugListenerAttachDetachDesignGateResult.cs"));
        Assert.Equal(AttachDetachOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void ExceptionStatusFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "TensorRtDebugListenerExceptionStatusMappingGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "TensorRtDebugListenerExceptionStatusMappingGateResult.cs"));
        Assert.Equal(ExceptionStatusOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
