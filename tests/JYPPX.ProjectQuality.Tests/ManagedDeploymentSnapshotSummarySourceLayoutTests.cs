using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDeploymentSnapshotSummarySourceLayoutTests
{
    private const string ExecutionOriginalNormalizedSha256 =
        "b39b5f61a4b3da0da42cb1b2cd49bc28323ce5c86ea5bd97a56cfc3c6e895a24";
    private const string BuilderOriginalNormalizedSha256 =
        "101a16504baeef0c98da3dc784faa0368e04183a92c216890890503556aa839e";

    public static TheoryData<string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "Execution",
            "TensorRtExecutionContextDeploymentSnapshot.cs",
            new[] { "TensorRtExecutionContextDeploymentSnapshot", "ToSummary", "ToString" }
        },
        {
            "Execution",
            "TensorRtExecutionContextDeploymentSummary.cs",
            new[] { "TensorRtExecutionContextDeploymentSummary", "ToString" }
        },
        {
            "Builder",
            "TensorRtBuilderConfigDeploymentSnapshot.cs",
            new[] { "TensorRtBuilderConfigDeploymentSnapshot", "ToSummary", "ToString" }
        },
        {
            "Builder",
            "TensorRtBuilderConfigDeploymentSummary.cs",
            new[] { "TensorRtBuilderConfigDeploymentSummary", "ToString" }
        }
    };

    public static TheoryData<string, string, string[]> SummaryProperties => new()
    {
        {
            "Execution",
            "TensorRtExecutionContextDeploymentSummary.cs",
            new[]
            {
                "ContextName", "EngineName", "EngineIOTensorCount", "EngineLayerCount",
                "EngineOptimizationProfileCount", "OptimizationProfileIndex", "DebugSync",
                "AllInputDimensionsSpecified", "AllInputShapesSpecified", "DeviceMemorySizeInBytes",
                "PersistentCacheLimitInBytes", "EnqueueEmitsProfile", "IsInputConsumedEventSet",
                "HasTemporaryStorageAllocator", "HasDebugListener", "HasProfiler", "HasRuntimeConfig",
                "RuntimeConfigAllocationStrategy", "NvtxVerbosity", "UnfusedTensorsDebugState",
                "CopiedTensorStateCount", "CopiedRuntimeDiagnosticCount",
                "RuntimeDiagnosticsWithCallbackStateCount", "RuntimeDiagnosticsWithOutputAllocatorCount",
                "DiagnosticCount", "CopiedTensorStatesMatchEngineIOTensorCount", "PointerFreeCopiedSummary",
                "CanPromoteRuntimeProof", "CanDeleteDeferredRecord"
            }
        },
        {
            "Builder",
            "TensorRtBuilderConfigDeploymentSummary.cs",
            new[]
            {
                "OptimizationProfileCount", "IsProfileStreamSet", "HasCalibrationProfile", "Flags",
                "EngineCapability", "HardwareCompatibilityLevel", "RuntimePlatform",
                "WorkspaceMemoryPoolLimitInBytes", "OptimizationLevel", "ProfilingVerbosity", "MaxAuxStreams",
                "AverageTimingIterations", "TacticSources", "DefaultDeviceType", "DlaCore",
                "TilingOptimizationLevel", "L2LimitForTilingInBytes", "MaxTactics", "HasTimingCache",
                "PluginToSerializeCount", "CopiedSerializedPluginCount", "CopiedSerializedPluginPathCount",
                "HasProgressMonitor", "HasRemoteAutoTuningConfig", "DiagnosticCount",
                "CopiedPluginCountMatchesReportedCount", "PointerFreeCopiedSummary", "CanPromoteRuntimeProof",
                "CanDeleteDeferredRecord"
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
                     ("Execution", "TensorRtExecutionContextDeploymentSummary.cs",
                         "TensorRtExecutionContextDeploymentSummary"),
                     ("Builder", "TensorRtBuilderConfigDeploymentSummary.cs",
                         "TensorRtBuilderConfigDeploymentSummary")
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
            "public sealed class TensorRtExecutionContextDeploymentSummary",
            ReadSource("Execution", "TensorRtExecutionContextDeploymentSnapshot.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public sealed class TensorRtBuilderConfigDeploymentSummary",
            ReadSource("Builder", "TensorRtBuilderConfigDeploymentSnapshot.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedSummariesRemainPointerFree()
    {
        foreach ((string module, string fileName) in new[]
                 {
                     ("Execution", "TensorRtExecutionContextDeploymentSummary.cs"),
                     ("Builder", "TensorRtBuilderConfigDeploymentSummary.cs")
                 })
        {
            string source = ReadSource(module, fileName);
            Assert.DoesNotContain("public IntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public UIntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public nint", source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void SourceReaderAndConsumersUseBothCompleteSourceSets()
    {
        string reader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));
        foreach (string sourceFile in new[]
                 {
                     "TensorRtExecutionContextDeploymentSnapshot.cs",
                     "TensorRtExecutionContextDeploymentSummary.cs",
                     "TensorRtBuilderConfigDeploymentSnapshot.cs",
                     "TensorRtBuilderConfigDeploymentSummary.cs"
                 })
        {
            Assert.Contains(sourceFile, reader, StringComparison.Ordinal);
        }

        foreach (string consumerName in new[]
                 {
                     "ExecutionContextReadonlyControlsTests.cs",
                     "BuilderConfigScalarControlsTests.cs",
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
        string docs = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "source-organization.md"));
        Assert.Contains("TensorRtExecutionContextDeploymentSummary", docs, StringComparison.Ordinal);
        Assert.Contains("TensorRtBuilderConfigDeploymentSummary", docs, StringComparison.Ordinal);
    }

    [Fact]
    public void ExecutionFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "Execution",
            "TensorRtExecutionContextDeploymentSnapshot.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "Execution",
            "TensorRtExecutionContextDeploymentSummary.cs"));
        Assert.Equal(ExecutionOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void BuilderFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "Builder",
            "TensorRtBuilderConfigDeploymentSnapshot.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "Builder",
            "TensorRtBuilderConfigDeploymentSummary.cs"));
        Assert.Equal(BuilderOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
