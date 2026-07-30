using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedRuntimeDiagnosticEngineSummarySourceLayoutTests
{
    private const string RuntimeDiagnosticOriginalNormalizedSha256 =
        "744329112e0680d03a61b400c9de073d675d555882191f25dd9a3ba3186b25f5";
    private const string EngineOriginalNormalizedSha256 =
        "2803fd2737d166351134e43094c986aa0c6cb6484f2fe36b166d8dc95bbc004d";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "Execution",
            "TensorRtExecutionContextRuntimeDiagnosticSnapshot.cs",
            new[] { "TensorRtExecutionContextRuntimeDiagnosticSnapshot", "ToSummary", "ToString" }
        },
        {
            "Execution",
            "TensorRtExecutionContextRuntimeDiagnosticSummary.cs",
            new[] { "TensorRtExecutionContextRuntimeDiagnosticSummary", "ToString" }
        },
        {
            "Engine",
            "TensorRtEngineDeploymentSnapshot.cs",
            new[] { "TensorRtEngineDeploymentSnapshot", "ToSummary", "ToString" }
        },
        {
            "Engine",
            "TensorRtEngineDeploymentSummary.cs",
            new[] { "TensorRtEngineDeploymentSummary", "ToString" }
        }
    };

    public static TheoryData<string, string, string[]> SummaryProperties => new()
    {
        {
            "Execution",
            "TensorRtExecutionContextRuntimeDiagnosticSummary.cs",
            new[]
            {
                "Line", "HasOutputTensorName", "HasErrorRecorder", "IsInputConsumedEventSet",
                "HasInputConsumedEventAddressValue", "HasOutputAllocator", "IsOutputTensorAddressSet",
                "HasOutputTensorAddressValue", "HasTemporaryStorageAllocator", "HasDebugListener",
                "HasManagedProfiler", "HasNativeProfiler", "HasRuntimeConfig", "NvtxVerbosity",
                "UnfusedTensorsDebugState", "CallbackStateHasOutputAllocator",
                "CallbackStateHasTemporaryStorageAllocator", "CallbackStateHasDebugListener",
                "CallbackStateOutputAllocatorInterfaceInfoAvailable",
                "CallbackStateTemporaryStorageAllocatorInterfaceInfoAvailable",
                "CallbackStateDebugListenerInterfaceInfoAvailable", "CallbackStateLastStatus",
                "CallbackStateLastOperationAvailable", "DiagnosticCount"
            }
        },
        {
            "Engine",
            "TensorRtEngineDeploymentSummary.cs",
            new[]
            {
                "EngineName", "ProfileIndex", "IOTensorCount", "LayerCount", "OptimizationProfileCount",
                "DeviceMemorySizeInBytes", "DeviceMemorySizeV2InBytes", "ProfileDeviceMemorySizeInBytes",
                "ProfileDeviceMemorySizeV2InBytes", "AuxiliaryStreamCount", "IsRefittable",
                "StreamableWeightsSizeInBytes", "WeightStreamingBudgetV2InBytes",
                "WeightStreamingAutomaticBudgetInBytes", "WeightStreamingScratchMemorySizeInBytes",
                "TotalWeightsSizeInBytes", "StrippedWeightsSizeInBytes", "CopiedTensorCount",
                "CopiedProfileTensorValueCount", "DiagnosticCount",
                "CopiedTensorCountMatchesReportedIOTensorCount", "PointerFreeCopiedSummary",
                "CanPromoteRuntimeProof", "CanDeleteDeferredRecord"
            }
        }
    };

    [Theory]
    [MemberData(nameof(FileOwnersAndMethods))]
    public void FilesOwnExactTopLevelTypesAndMethods(string module, string fileName, string[] expected)
    {
        string source = ReadSource(module, fileName);
        Assert.Equal(new[] { expected[0] }, EnumerateTopLevelTypeNames(source));
        Assert.Equal(expected.Skip(1), EnumerateMethodNames(source));
    }

    [Theory]
    [MemberData(nameof(SummaryProperties))]
    public void SummariesKeepExactPublicProperties(string module, string fileName, string[] expectedProperties)
    {
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(ReadSource(module, fileName)));
    }

    [Fact]
    public void SummariesKeepOneInternalConstructorEach()
    {
        foreach ((string module, string fileName, string typeName) in new[]
                 {
                     ("Execution", "TensorRtExecutionContextRuntimeDiagnosticSummary.cs",
                         "TensorRtExecutionContextRuntimeDiagnosticSummary"),
                     ("Engine", "TensorRtEngineDeploymentSummary.cs", "TensorRtEngineDeploymentSummary")
                 })
        {
            Assert.Single(Regex.Matches(
                ReadSource(module, fileName),
                $@"^    internal {typeName}\(",
                RegexOptions.Multiline));
        }
    }

    [Fact]
    public void SnapshotsDoNotOwnSummaryDeclarations()
    {
        Assert.DoesNotContain(
            "public sealed class TensorRtExecutionContextRuntimeDiagnosticSummary",
            ReadSource("Execution", "TensorRtExecutionContextRuntimeDiagnosticSnapshot.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public sealed class TensorRtEngineDeploymentSummary",
            ReadSource("Engine", "TensorRtEngineDeploymentSnapshot.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedSummariesRemainPointerFree()
    {
        foreach ((string module, string fileName) in new[]
                 {
                     ("Execution", "TensorRtExecutionContextRuntimeDiagnosticSummary.cs"),
                     ("Engine", "TensorRtEngineDeploymentSummary.cs")
                 })
        {
            string source = ReadSource(module, fileName);
            Assert.DoesNotContain("public IntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public UIntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public nint", source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void SourceReaderCandidatesAndConsumersUseBothCompleteSourceSets()
    {
        string reader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));
        foreach (string sourceFile in new[]
                 {
                     "TensorRtExecutionContextRuntimeDiagnosticSnapshot.cs",
                     "TensorRtExecutionContextRuntimeDiagnosticSummary.cs",
                     "TensorRtEngineDeploymentSnapshot.cs",
                     "TensorRtEngineDeploymentSummary.cs"
                 })
        {
            Assert.Contains(sourceFile, reader, StringComparison.Ordinal);
        }

        string candidates = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "deferred-readonly-candidate-list.json"));
        Assert.Contains("TensorRtExecutionContextRuntimeDiagnosticSummary.cs", candidates, StringComparison.Ordinal);
        Assert.Contains("TensorRtEngineDeploymentSummary.cs", candidates, StringComparison.Ordinal);

        foreach (string consumerName in new[]
                 {
                     "ExecutionContextReadonlyControlsTests.cs",
                     "EngineAndRnnReadonlyDiagnosticsTests.cs",
                     "EngineBuilderReadonlyCandidateImplementationEvidenceTests.cs"
                 })
        {
            string consumer = File.ReadAllText(Path.Combine(
                RepositoryPaths.Root,
                "tests",
                "JYPPX.ProjectQuality.Tests",
                consumerName));
            Assert.Contains("RepositorySourceReader.Read", consumer, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void SourceOrganizationReferencesDedicatedSummaryOwners()
    {
        foreach (string relativePath in new[]
                 {
                     Path.Combine("docs", "articles", "en", "source-organization.md"),
                     Path.Combine("docs", "articles", "zh-cn", "source-organization.md"),
                     Path.Combine("docs", "articles", "zh-cn", "windows-api-completion.md")
                 })
        {
            string docs = File.ReadAllText(Path.Combine(RepositoryPaths.Root, relativePath));
            Assert.Contains("TensorRtExecutionContextRuntimeDiagnosticSummary", docs, StringComparison.Ordinal);
            Assert.Contains("TensorRtEngineDeploymentSummary", docs, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void RuntimeDiagnosticFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "Execution",
            "TensorRtExecutionContextRuntimeDiagnosticSnapshot.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "Execution",
            "TensorRtExecutionContextRuntimeDiagnosticSummary.cs"));
        Assert.Equal(RuntimeDiagnosticOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void EngineFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "Engine",
            "TensorRtEngineDeploymentSnapshot.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "Engine",
            "TensorRtEngineDeploymentSummary.cs"));
        Assert.Equal(EngineOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string[] EnumerateTopLevelTypeNames(string source)
    {
        return Regex.Matches(
                Normalize(source),
                @"^public sealed class (?<name>[A-Za-z_][A-Za-z0-9_]*)$",
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

    private static string ReadTopLevelSegment(string module, string fileName)
    {
        string source = Normalize(ReadSource(module, fileName));
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

    private static string ReadSource(string module, string fileName)
    {
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            module,
            fileName));
    }
}
