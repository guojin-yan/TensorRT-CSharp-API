using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedOnnxEngineBuildLayoutTests
{
    private const string BuildServiceOriginalNormalizedSha256 =
        "433cd0e3ffdf2da39f8bb345eb96d39423885e5046119edb11b9f11c5bc4d4cb";
    private const string BuildResultOriginalNormalizedSha256 =
        "683fd2ce1579286a222cd61842b754b7731b94bf6b30440da4bfb653cd2dbc2b";

    public static TheoryData<string, string[], string[]> ServiceFeatureMembers => new()
    {
        {
            "OnnxEngineBuildService.cs",
            new[] { "Execute", "ExecuteCore" },
            Array.Empty<string>()
        },
        {
            "OnnxEngineBuildService.BuilderConfiguration.cs",
            new[] { "AddOptimizationProfile", "DryRunModelSource", "ApplyPrecisionFlags", "ApplyGlobalPrecisionFlag" },
            Array.Empty<string>()
        },
        {
            "OnnxEngineBuildService.TimingCache.cs",
            new[] { "CreateTimingCacheLease", "ExportTimingCache", "CreateTimingCacheBoundaryArtifact", "ComputeSha256" },
            new[] { "TimingCacheLease" }
        },
        {
            "OnnxEngineBuildService.ResultCreation.cs",
            new[] { "CreateResult", "RuntimeState" },
            Array.Empty<string>()
        },
        {
            "OnnxEngineBuildService.Refit.cs",
            new[] { "PersistAndReloadRefittedEngine", "RefitStrippedEngineFromOnnx", "CopyRefitEntries" },
            Array.Empty<string>()
        },
        {
            "OnnxEngineBuildService.Diagnostics.cs",
            new[]
            {
                "CaptureParserPreflightSnapshot",
                "TryGetBuilderConfigDeploymentSnapshot",
                "AppendCapabilityProbeLog",
                "ProbeCapabilities",
                "ProbeLoadedEngineDiagnostics",
                "CreateLoadedEngineReadbackFingerprint",
                "TryCollectLayerInformationFromSerializedEngine",
                "TryCollectLayerInformation",
                "ComputeSha256"
            },
            Array.Empty<string>()
        },
        {
            "OnnxEngineBuildService.RuntimeExecution.cs",
            new[]
            {
                "CanAttemptGenericExternalRuntime",
                "TryRunGenericFloatEngineFromFile",
                "TryRunGenericFloatEngine",
                "ApplyEngineRuntimePolicies",
                "ConfigureRuntimeForEnginePolicies"
            },
            new[] { "OnnxEngineRuntimeExecution", "OnnxEngineRuntimeInput", "OnnxEngineRuntimeOutputTensor" }
        },
        {
            "OnnxEngineBuildService.Benchmarking.cs",
            new[]
            {
                "RunBoundedBenchmark",
                "TryEnableCudaGraphs",
                "RunSingleThreadBoundedBenchmark",
                "RunThreadedBoundedBenchmark",
                "RunWorkerWarmUp",
                "RunWorkerMeasurement"
            },
            new[] { "OnnxEngineBenchmarkWorker", "OnnxEngineBenchmarkRun", "OnnxEngineWorkerWarmUp", "OnnxEngineWorkerRun" }
        },
        {
            "OnnxEngineBuildService.RuntimeInputs.cs",
            new[] { "ResolveRuntimeInputShape", "ShouldSetInputShape", "CreateRuntimeInputValues", "ParseLoadInputs", "ReadFloatInputData" },
            Array.Empty<string>()
        },
        {
            "OnnxEngineBuildService.ReferenceValidation.cs",
            new[]
            {
                "ValidateReferenceOutputs",
                "ParseReferenceOutputMappings",
                "ReadReferenceTensor",
                "CompareReferenceTensor",
                "ReferenceValuesMatch",
                "CreateReferenceMetadataMismatch",
                "CreateUnavailableReferenceComparison",
                "SanitizeReferenceDiagnostic",
                "CountElements",
                "CanEstimate",
                "ShapesEqual",
                "ValuesEqual",
                "FormatShape"
            },
            Array.Empty<string>()
        },
        {
            "OnnxEngineBuildService.DeploymentConfiguration.cs",
            new[]
            {
                "ApplyDeploymentOptions",
                "ApplyEnginePackagingOptions",
                "ApplyBuilderFlagWithReadback",
                "ShouldCreateStronglyTypedNetwork",
                "ApplyBuilderScalarDeploymentControls",
                "ParseProfilingVerbosity",
                "RuntimeOptionsLogLine"
            },
            Array.Empty<string>()
        }
    };

    public static TheoryData<string, string, string[]> ResultTypeProperties => new()
    {
        {
            "OnnxEngineBuildResult.cs",
            "OnnxEngineBuildResult",
            new[]
            {
                "Success", "Skipped", "State", "TensorRtLine", "ModelSource", "EnginePath", "Parsed", "EngineSaved",
                "EngineFileRoundTrip", "InferenceRan", "OutputMatch", "OutputValidated", "IdentityOutputMatch", "ProfileIndex",
                "ElapsedMilliseconds", "SkipReason", "NormalizedCommandLine", "NormalizedCommandSha256", "DeploymentOptions",
                "RuntimeOptions", "BenchmarkSummary", "PreflightMetadata", "LoadedEngineDiagnostics", "TimingCacheArtifact",
                "CapabilityProbe", "WorkspaceBytes", "BuilderConfigDeploymentSnapshot", "ParserPreflightSnapshot", "RefitSnapshot",
                "RefitPersistenceSnapshot", "IsRuntimeExecutionProof", "BuildEvidenceOnly", "ProofClassification",
                "EvidenceClassifications", "IsRealModelRuntimeProof", "IsPackageConsumerRuntimeProof", "StdoutSummary",
                "StderrSummary", "ModelEvidence", "EvidenceSidecar", "Diagnostics", "LogLines"
            }
        },
        {
            "OnnxEngineTimingCacheArtifact.cs",
            "OnnxEngineTimingCacheArtifact",
            new[]
            {
                "Empty", "InputRequested", "InputApplied", "InputPath", "InputLengthBytes", "InputSha256", "OutputRequested",
                "OutputWritten", "OutputPath", "OutputLengthBytes", "OutputSha256", "State", "EvidenceBoundary"
            }
        },
        {
            "OnnxEngineCapabilityProbe.cs",
            "OnnxEngineCapabilityProbe",
            new[]
            {
                "Empty", "Attempted", "ProbeState", "TensorRtLine", "TensorRtVersion", "CudaToolkitVersion", "RuntimeAvailable",
                "BuilderAvailable", "BuilderConfigAvailable", "EngineInspectorApiAvailable", "Fp8FlagRequested", "Fp8FlagKnown",
                "DebugTensorOptionsRequested", "DebugTensorApiKnown", "WeightStreamingRequested", "WeightStreamingApiKnown",
                "ProbeItems", "EvidenceBoundary"
            }
        },
        {
            "OnnxLoadedEngineDiagnostics.cs",
            "OnnxLoadedEngineDiagnostics",
            new[]
            {
                "Empty", "Attempted", "Succeeded", "DiagnosticsState", "FailureReason", "EngineName", "IOTensorCount",
                "LayerCount", "OptimizationProfileCount", "DeviceMemorySizeInBytes", "AuxiliaryStreamCount", "Capability",
                "ProfilingVerbosity", "InspectorInformationLength", "IOTensorSummaries", "ReadbackFingerprint", "ReadbackSha256",
                "EvidenceBoundary"
            }
        },
        {
            "OnnxEnginePreflightMetadata.cs",
            "OnnxEnginePreflightMetadata",
            new[] { "Empty", "Kind", "Path", "Exists", "LengthBytes", "Sha256", "PreflightState", "ProofClassification", "EvidenceBoundary" }
        },
        {
            "OnnxEngineBuildModelEvidence.cs",
            "OnnxEngineBuildModelEvidence",
            new[] { "ModelSource", "ModelSha256", "ModelLicense", "InputAssetName", "InputAssetSha256" }
        },
        {
            "OnnxEngineBenchmarkSummary.cs",
            "OnnxEngineBenchmarkSummary",
            new[]
            {
                "Empty", "TimingSamplesMilliseconds", "TimingSampleCount", "AveragedTimingSamplesMilliseconds",
                "AveragedTimingSampleCount", "AverageElapsedMilliseconds", "MinElapsedMilliseconds", "MaxElapsedMilliseconds",
                "AvgRunsRequested", "AvgRunsExecuted", "PercentileRequested", "PercentileElapsedMilliseconds", "ThreadsRequested",
                "ThreadsExecuted", "NoDataTransfersRequested", "NoDataTransfersApplied", "UseSpinWaitRequested",
                "UseSpinWaitApplied", "UseCudaGraphRequested", "UseCudaGraphApplied", "UseCudaGraphFallbackReason",
                "SleepTimeMillisecondsRequested", "SleepTimeMillisecondsApplied", "IdleTimeMillisecondsRequested",
                "IdleTimeMillisecondsApplied", "BenchmarkBoundary", "IterationsRequested", "MeasurementRoundsExecuted",
                "InferenceIterationsExecuted", "WarmUpMillisecondsRequested", "WarmUpElapsedMilliseconds",
                "WarmUpIterationsExecuted", "DurationSecondsRequested", "MeasurementElapsedMilliseconds", "StreamsRequested",
                "InfStreamsRequested", "ExecutionContextsCreated", "ConcurrentStreamsExecuted", "MeasurementRoundsPerContext"
            }
        }
    };

    [Theory]
    [MemberData(nameof(ServiceFeatureMembers))]
    public void ServiceFeaturePartialsOwnExactMethodsAndNestedTypes(
        string fileName,
        string[] expectedMethods,
        string[] expectedNestedTypes)
    {
        string source = ReadBuildSource(fileName);
        Assert.Equal(expectedMethods, EnumerateMethodNames(source));
        Assert.Equal(expectedNestedTypes, EnumerateNestedTypeNames(source));
    }

    [Theory]
    [MemberData(nameof(ResultTypeProperties))]
    public void ResultFilesOwnExactTypeAndProperties(
        string fileName,
        string expectedType,
        string[] expectedProperties)
    {
        string source = ReadBuildSource(fileName);
        Assert.Equal(new[] { expectedType }, EnumerateTopLevelTypeNames(source));
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(source));
    }

    [Fact]
    public void ReferenceJsonOptionsFollowsReferenceValidation()
    {
        string core = ReadBuildSource("OnnxEngineBuildService.cs");
        string reference = ReadBuildSource("OnnxEngineBuildService.ReferenceValidation.cs");

        Assert.Contains("public sealed partial class OnnxEngineBuildService", core, StringComparison.Ordinal);
        Assert.DoesNotContain("ReferenceJsonOptions", core, StringComparison.Ordinal);
        Assert.Contains("private static readonly JsonSerializerOptions ReferenceJsonOptions", reference, StringComparison.Ordinal);
    }

    [Fact]
    public void ArticlesAndExporterReferenceFeatureOwners()
    {
        string blog = ReadRepositorySource(
            "docs", "articles", "zh-cn", "blog-onnx-parser-engine-roundtrip.md");
        string builderArticle = ReadRepositorySource(
            "docs", "articles", "zh-cn", "publishing", "builder-config-readback-public-article.md");
        string inspectorArticle = ReadRepositorySource(
            "docs", "articles", "zh-cn", "publishing", "engine-inspector-public-article.md");
        string exporter = ReadRepositorySource(
            "eng", "Export-TechnicalArticleFoundationsSecondBatchAudit.ps1");

        Assert.Contains("OnnxEngineBuildService.DeploymentConfiguration.cs", blog + builderArticle + exporter, StringComparison.Ordinal);
        Assert.Contains("OnnxEngineBuildService.RuntimeExecution.cs", blog + exporter, StringComparison.Ordinal);
        Assert.Contains("OnnxEngineBuildService.Diagnostics.cs", builderArticle + inspectorArticle, StringComparison.Ordinal);
        Assert.Contains("OnnxLoadedEngineDiagnostics.cs", inspectorArticle, StringComparison.Ordinal);
    }

    [Fact]
    public void BuildServiceFilesRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadBuildSource("OnnxEngineBuildService.cs"));
        int declarationStart = core.IndexOf("public sealed partial class OnnxEngineBuildService", StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declarationStart);
        Assert.True(declarationStart >= 0 && bodyStart >= 0);

        string timing = ReadPartialBody("OnnxEngineBuildService.TimingCache.cs");
        string runtime = ReadPartialBody("OnnxEngineBuildService.RuntimeExecution.cs");
        string benchmark = ReadPartialBody("OnnxEngineBuildService.Benchmarking.cs");
        string reference = ReadPartialBody("OnnxEngineBuildService.ReferenceValidation.cs");

        int timingTypes = timing.IndexOf("    private sealed class TimingCacheLease", StringComparison.Ordinal);
        int runtimeTypes = runtime.IndexOf("    private sealed class OnnxEngineRuntimeExecution", StringComparison.Ordinal);
        int benchmarkTypes = benchmark.IndexOf("    private sealed class OnnxEngineBenchmarkWorker", StringComparison.Ordinal);
        int referenceMethods = reference.IndexOf(
            "    private static OnnxEngineReferenceValidationArtifact ValidateReferenceOutputs",
            StringComparison.Ordinal);
        Assert.True(timingTypes >= 0 && runtimeTypes >= 0 && benchmarkTypes >= 0 && referenceMethods >= 0);

        StringBuilder source = new();
        source.Append(core[..(bodyStart + 1)].Replace(
            "public sealed partial class OnnxEngineBuildService",
            "public sealed class OnnxEngineBuildService",
            StringComparison.Ordinal));
        source.Append('\n');
        source.Append(reference[..referenceMethods]);
        source.Append(ReadPartialBody("OnnxEngineBuildService.cs"));
        source.Append(ReadPartialBody("OnnxEngineBuildService.BuilderConfiguration.cs"));
        source.Append(timing[..timingTypes]);
        source.Append(ReadPartialBody("OnnxEngineBuildService.ResultCreation.cs"));
        source.Append(ReadPartialBody("OnnxEngineBuildService.Refit.cs"));
        source.Append(ReadPartialBody("OnnxEngineBuildService.Diagnostics.cs"));
        source.Append(runtime[..runtimeTypes]);
        source.Append(benchmark[..benchmarkTypes]);
        source.Append(ReadPartialBody("OnnxEngineBuildService.RuntimeInputs.cs"));
        source.Append(reference[referenceMethods..]);
        source.Append(timing[timingTypes..]);
        source.Append(benchmark[benchmarkTypes..]);
        source.Append(runtime[runtimeTypes..]);
        source.Append(ReadPartialBody("OnnxEngineBuildService.DeploymentConfiguration.cs"));
        source.Append("}\n");

        Assert.Equal(BuildServiceOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void BuildResultFilesRecomposeTheOriginalSource()
    {
        string[] typeFiles =
        {
            "OnnxEngineTimingCacheArtifact.cs",
            "OnnxEngineCapabilityProbe.cs",
            "OnnxLoadedEngineDiagnostics.cs",
            "OnnxEnginePreflightMetadata.cs",
            "OnnxEngineBuildModelEvidence.cs",
            "OnnxEngineBenchmarkSummary.cs"
        };

        StringBuilder source = new(Normalize(ReadBuildSource("OnnxEngineBuildResult.cs")).TrimEnd('\n'));
        foreach (string fileName in typeFiles)
        {
            source.Append("\n\n");
            source.Append(ReadTopLevelSegment(fileName).TrimEnd('\n'));
        }
        source.Append('\n');

        Assert.Equal(BuildResultOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string ReadPartialBody(string fileName)
    {
        string source = Normalize(ReadBuildSource(fileName));
        int declarationStart = source.IndexOf("OnnxEngineBuildService", StringComparison.Ordinal);
        int bodyStart = source.IndexOf('{', declarationStart);
        int bodyEnd = source.LastIndexOf('}');
        Assert.True(declarationStart >= 0 && bodyStart >= 0 && bodyEnd > bodyStart);
        string body = source[(bodyStart + 1)..bodyEnd];
        Assert.StartsWith("\n", body, StringComparison.Ordinal);
        return body[1..];
    }

    private static string ReadTopLevelSegment(string fileName)
    {
        string source = Normalize(ReadBuildSource(fileName));
        int start = source.IndexOf("public sealed class", StringComparison.Ordinal);
        Assert.True(start >= 0);
        return source[start..];
    }

    private static string[] EnumerateMethodNames(string source)
    {
        int nestedTypeStart = source.IndexOf("    private sealed class", StringComparison.Ordinal);
        if (nestedTypeStart >= 0)
        {
            source = source[..nestedTypeStart];
        }

        return Regex.Matches(
                source,
                @"^\s*(?:public|private|internal)\s+(?:static\s+)?[^\r\n=]+?\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
                RegexOptions.Multiline)
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumerateNestedTypeNames(string source)
    {
        return Regex.Matches(
                source,
                @"^\s+private\s+sealed\s+class\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)",
                RegexOptions.Multiline)
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumerateTopLevelTypeNames(string source)
    {
        return Regex.Matches(
                source,
                @"^public\s+sealed\s+class\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)",
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

    private static string ReadBuildSource(string fileName)
    {
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools",
            "Build",
            fileName));
    }

    private static string ReadRepositorySource(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(
            new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}
