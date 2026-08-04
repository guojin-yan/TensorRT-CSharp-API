using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDebugListenerRuntimeAttemptAttachSmokeSourceLayoutTests
{
    private const string AttemptPreflightOriginalNormalizedSha256 =
        "0a8e1b1412c65e8179385ec8f5a96eb6739ca5f87f8a0b8a17485c7345dff7ef";
    private const string AttachSmokeOriginalNormalizedSha256 =
        "48f7c9536e1dc19df2b9bf75f4fb14d51f71355633e70333fee2e2f6670fd2de";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "TensorRtDebugListenerRuntimeProofAttemptPreflight.cs",
            "TensorRtDebugListenerRuntimeProofAttemptPreflight",
            new[]
            {
                "Evaluate", "Evaluate", "BuildNonNullAttachReason", "BuildNativeVTableReason",
                "BuildRuntimeProofReason", "AddBlockerIfFalse", "AddBlocker"
            }
        },
        {
            "TensorRtDebugListenerRuntimeProofAttemptPreflightResult.cs",
            "TensorRtDebugListenerRuntimeProofAttemptPreflightResult",
            new[] { "ToString" }
        },
        {
            "TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs",
            "TensorRtDebugListenerRealNonNullAttachRuntimeSmoke",
            new[] { "Evaluate", "Evaluate", "BuildDiagnostic", "AddBlockerIfFalse", "AddBlocker" }
        },
        {
            "TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.cs",
            "TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string[]> ResultProperties => new()
    {
        {
            "TensorRtDebugListenerRuntimeProofAttemptPreflightResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "NativeAttachEntryLocated", "NonNullAttachStillDisabled",
                "NativeOwnerLifecycleReady", "CanImplementNativeAttach", "NativeVTableReady",
                "NoThrowVTableDesignReady", "NativeVTableTrampolineReady", "CallbackExceptionCaptureReady",
                "CallbackStatusMappingReady", "CallbackInFlightAccountingReady", "VTableAddressExposed",
                "VTablePointerProduced", "BorrowedDebugTensorPointerEscapeBlocked",
                "BorrowedDebugTensorLifetimeReady", "BorrowedDebugTensorDataLifetimeReady",
                "ProcessDebugTensorRuntimeReady", "FullPackageConsumerRuntimeEvidenceReady",
                "PrecheckCanAttemptRuntimeProof", "CanEnableSetDebugListenerNonNull", "CanInstallNativeVTable",
                "CanCallProcessDebugTensorRuntime", "CanPromoteRealCallbackRuntime",
                "ReasonNonNullAttachStillBlocked", "ReasonNativeVTableStillBlocked",
                "ReasonRuntimeProofStillBlocked", "CanAttemptRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "BlockedPrerequisites", "BlockedPrerequisiteCount", "Status",
                "Diagnostic"
            }
        },
        {
            "TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "TensorRtLine", "RuntimePackageKey", "OptInEnabled",
                "FullPackageConsumerReport", "AttachGuardReady", "NativeVTableReady",
                "BorrowedDebugTensorRuntimeReady", "CallbackInvocationReady", "AttachAttempted",
                "AttachSucceeded", "DetachAttempted", "DetachSucceeded", "RollbackAttempted", "RollbackSucceeded",
                "NativeVTableInstalled", "ProcessDebugTensorInvoked", "InvocationCount", "AllocationCount",
                "ReleaseCount", "FailureCount", "InFlightCallbackCount", "LastStatus", "LastDiagnostic",
                "ReportPointerFree", "CanPromoteRealCallbackRuntime", "CanAttemptRuntimeProof",
                "RuntimeProofBlocked", "DeferredRowsStillRequired", "ReasonRuntimeProofStillBlocked",
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
                     ("TensorRtDebugListenerRuntimeProofAttemptPreflightResult.cs",
                         "TensorRtDebugListenerRuntimeProofAttemptPreflightResult"),
                     ("TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.cs",
                         "TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult")
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
            "public readonly struct TensorRtDebugListenerRuntimeProofAttemptPreflightResult",
            ReadSource("TensorRtDebugListenerRuntimeProofAttemptPreflight.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult",
            ReadSource("TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFree()
    {
        foreach (string fileName in new[]
                 {
                     "TensorRtDebugListenerRuntimeProofAttemptPreflightResult.cs",
                     "TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.cs"
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
            "TensorRtDebugListenerRuntimeProofAttemptPreflight.cs",
            "TensorRtDebugListenerRuntimeProofAttemptPreflightResult.cs",
            "TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs",
            "TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "DebugListenerRealCallbackRuntimeProofPreflightBatchTests.cs",
                     "DebugListenerRealNonNullAttachRuntimeSmokeTests.cs"
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
            "debug-listener-real-callback-runtime-proof-preflight.md"));
        string smokeDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "debug-listener-real-non-null-attach-runtime-smoke.md"));

        Assert.Contains("TensorRtDebugListenerRuntimeProofAttemptPreflight.cs", preflightDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerRuntimeProofAttemptPreflightResult.cs",
            preflightDoc,
            StringComparison.Ordinal);
        Assert.Contains("TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs", smokeDoc, StringComparison.Ordinal);
        Assert.Contains(
            "TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.cs",
            smokeDoc,
            StringComparison.Ordinal);
    }

    [Fact]
    public void AttemptPreflightFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerRuntimeProofAttemptPreflight.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerRuntimeProofAttemptPreflightResult.cs"));
        Assert.Equal(AttemptPreflightOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void AttachSmokeFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.cs"));
        Assert.Equal(AttachSmokeOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
