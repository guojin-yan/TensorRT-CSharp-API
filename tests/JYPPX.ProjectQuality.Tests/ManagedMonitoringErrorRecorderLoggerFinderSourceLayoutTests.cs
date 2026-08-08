using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedMonitoringErrorRecorderLoggerFinderSourceLayoutTests
{
    private const string ErrorRecorderOriginalNormalizedSha256 =
        "cf5073560b8b5e2df9a131537c836503426ed193a6262555e606e4633d9d8668";
    private const string LoggerFinderGateOriginalNormalizedSha256 =
        "18c43b3a69885c448a1ba3558c3ef9c2dc664e45f6c57c49ac48cff8044cb9a3";

    public static TheoryData<string, string, string[]> ErrorRecorderModelOwners => new()
    {
        {
            "TensorRtErrorRecord.cs",
            "TensorRtErrorRecord",
            new[] { "Index", "Code", "Description" }
        },
        {
            "TensorRtErrorRecorderSnapshot.cs",
            "TensorRtErrorRecorderSnapshot",
            new[]
            {
                "Line", "HasRecorder", "ErrorCount", "HasOverflowed", "InterfaceInfoAvailable",
                "InterfaceInfo", "Records"
            }
        },
        {
            "TensorRtErrorRecorderSummary.cs",
            "TensorRtErrorRecorderSummary",
            new[]
            {
                "Line", "HasRecorder", "ErrorCount", "CopiedErrorRecordCount", "HasOverflowed",
                "InterfaceInfoAvailable", "InterfaceName", "InterfaceMajor", "InterfaceMinor", "FirstErrorCode",
                "FirstErrorDescriptionLength", "CopiedRecordCountMatchesErrorCount", "RuntimeEvidenceKind",
                "IsRuntimeExecutionEvidence", "IsRuntimeExecutionProof", "PointerFreeCopiedSummary",
                "CanPromoteRuntimeProof", "CanPromoteReleaseProof", "CanDeleteDeferredRecord"
            }
        }
    };

    [Theory]
    [MemberData(nameof(ErrorRecorderModelOwners))]
    public void ErrorRecorderModelsHaveDedicatedOwners(
        string fileName,
        string expectedType,
        string[] expectedProperties)
    {
        string source = ReadSource(fileName);
        Assert.Equal(
            new[] { expectedType },
            Regex.Matches(
                    source,
                    @"^public (?:sealed class|readonly struct) (?<name>[A-Za-z_][A-Za-z0-9_]*)$",
                    RegexOptions.Multiline)
                .Select(match => match.Groups["name"].Value)
                .ToArray());
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(source));
    }

    [Fact]
    public void ErrorRecorderModelsKeepExactConstructorsAndMethods()
    {
        string record = ReadSource("TensorRtErrorRecord.cs");
        string snapshot = ReadSource("TensorRtErrorRecorderSnapshot.cs");
        string summary = ReadSource("TensorRtErrorRecorderSummary.cs");

        Assert.Single(Regex.Matches(record, @"^    internal TensorRtErrorRecord\(", RegexOptions.Multiline));
        Assert.Equal(2, Regex.Matches(
            snapshot,
            @"^    internal TensorRtErrorRecorderSnapshot\(",
            RegexOptions.Multiline).Count);
        Assert.Single(Regex.Matches(summary, @"^    internal TensorRtErrorRecorderSummary\(", RegexOptions.Multiline));
        Assert.Equal(new[] { "ToString" }, EnumerateMethodNames(record));
        Assert.Equal(new[] { "ToSummary", "ToString" }, EnumerateMethodNames(snapshot));
        Assert.Equal(new[] { "ToString" }, EnumerateMethodNames(summary));
    }

    [Fact]
    public void LoggerFinderEvaluatorAndResultHaveDedicatedOwners()
    {
        string evaluator = ReadSource("TensorRtLoggerFinderMetadataDesignGate.cs");
        string result = ReadSource("TensorRtLoggerFinderMetadataDesignGateResult.cs");

        Assert.Equal(
            new[] { "TensorRtLoggerFinderMetadataDesignGate" },
            EnumerateTopLevelTypeNames(evaluator));
        Assert.Equal(
            new[] { "TensorRtLoggerFinderMetadataDesignGateResult" },
            EnumerateTopLevelTypeNames(result));
        Assert.Equal(new[] { "EvaluateKnownSurface", "Evaluate" }, EnumerateMethodNames(evaluator));
        Assert.Equal(new[] { "ToString" }, EnumerateMethodNames(result));
        Assert.Single(Regex.Matches(
            result,
            @"^    internal TensorRtLoggerFinderMetadataDesignGateResult\(",
            RegexOptions.Multiline));
        Assert.DoesNotContain(
            "public readonly struct TensorRtLoggerFinderMetadataDesignGateResult",
            evaluator,
            StringComparison.Ordinal);
    }

    [Fact]
    public void LoggerFinderResultKeepsExactPublicProperties()
    {
        Assert.Equal(
            new[]
            {
                "EvidenceKind", "DiagnosticsKind", "RuntimeEvidenceKind", "IsRuntimeExecutionEvidence",
                "IsRuntimeExecutionProof", "Line", "LineSupportsLoggerFinder", "CopiedInterfaceInfoMetadataReady",
                "FinderOwnerLifetimeModeled", "LoggerCallbackOwnershipModeled", "LoggerFinderPointerExposed",
                "LoggerCallbackPointerExposed", "LoggerFinderCallbackInvocationEnabled", "DirectLoggerFinderRowsDeferred",
                "CopiedMetadataShapeReady", "PointerFreeSurfaceReady", "DesignGateReady",
                "CanPromoteWithoutRuntimeProof", "CanPromoteRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "CandidateInterfaces", "CandidateMethods", "RequiredOutputMode",
                "NextSafeImplementationStep", "CandidateMethodCount", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            },
            EnumeratePublicPropertyNames(ReadSource("TensorRtLoggerFinderMetadataDesignGateResult.cs")));
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
            "TensorRtErrorRecord.cs",
            "TensorRtErrorRecorderSnapshot.cs",
            "TensorRtErrorRecorderSummary.cs",
            "TensorRtLoggerFinderMetadataDesignGate.cs",
            "TensorRtLoggerFinderMetadataDesignGateResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "CallbackAllocatorBoundaryTests.cs",
                     "DeferredReadonlyUpliftBatchTests.cs",
                     "ErrorRecorderDiagnosticsDesignGateTests.cs",
                     "ErrorRecorderSnapshotSummaryTests.cs",
                     "ReadonlyDiagnosticsCandidateImplementationEvidenceTests.cs"
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
    public void DocsReferenceDedicatedModelAndGateOwners()
    {
        string errorRecorderDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "error-recorder-diagnostics-design-gate.md"));
        string loggerFinderDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "logger-finder-metadata-design-gate.md"));

        foreach (string fileName in new[]
                 {
                     "TensorRtErrorRecord.cs",
                     "TensorRtErrorRecorderSnapshot.cs",
                     "TensorRtErrorRecorderSummary.cs"
                 })
        {
            Assert.Contains(fileName, errorRecorderDoc, StringComparison.Ordinal);
        }

        Assert.Contains("TensorRtLoggerFinderMetadataDesignGate.cs", loggerFinderDoc, StringComparison.Ordinal);
        Assert.Contains("TensorRtLoggerFinderMetadataDesignGateResult.cs", loggerFinderDoc, StringComparison.Ordinal);
    }

    [Fact]
    public void ErrorRecorderFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtErrorRecord.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtErrorRecorderSnapshot.cs"));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtErrorRecorderSummary.cs"));
        Assert.Equal(ErrorRecorderOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void LoggerFinderFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("TensorRtLoggerFinderMetadataDesignGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtLoggerFinderMetadataDesignGateResult.cs"));
        Assert.Equal(LoggerFinderGateOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string[] EnumerateTopLevelTypeNames(string source)
    {
        return Regex.Matches(
                source,
                @"^public (?:static class|sealed class|readonly struct) (?<name>[A-Za-z_][A-Za-z0-9_]*)$",
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
            "Monitoring",
            fileName));
    }
}
