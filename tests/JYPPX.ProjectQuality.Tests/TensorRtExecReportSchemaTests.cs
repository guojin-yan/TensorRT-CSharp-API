using System.Text.Json;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Tools;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecReportSchemaTests
{
    [Fact]
    public void TensorRtExecReportSchemaKeepsReportBoundaryExplicit()
    {
        string path = Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-report.schema.json");
        Assert.True(File.Exists(path), path);

        string text = File.ReadAllText(path);
        using JsonDocument document = JsonDocument.Parse(text);
        JsonElement root = document.RootElement;

        Assert.Equal("TensorRtExec report schema", root.GetProperty("title").GetString());
        Assert.Contains("not runtime proof", root.GetProperty("description").GetString(), StringComparison.OrdinalIgnoreCase);

        foreach (string term in new[]
                 {
                     "Success",
                     "State",
                     "TensorRtLine",
                     "NormalizedCommandLine",
                     "NormalizedCommandSha256",
                     "DeploymentOptions",
                     "RuntimeOptions",
                     "PreflightMetadata",
                     "LoadedEngineDiagnostics",
                     "LayerInfoArtifact",
                     "ContentKind",
                     "BindingMetadata",
                     "copied-pointer-free-TensorRtEngineBindingReport",
                     "PointerFreeCopiedSnapshot",
                     "CanPromoteRuntimeProof",
                     "CanPromoteReleaseProof",
                      "TimingCacheArtifact",
                      "InputRequested",
                      "InputApplied",
                      "OutputRequested",
                      "OutputWritten",
                      "EvidenceBoundary",
                     "ReadbackFingerprint",
                     "ReadbackSha256",
                     "CapabilityProbe",
                     "ProbeState",
                     "RuntimeAvailable",
                     "BuilderAvailable",
                     "BuilderConfigAvailable",
                     "EngineInspectorApiAvailable",
                     "Fp8FlagRequested",
                     "DebugTensorOptionsRequested",
                     "WeightStreamingRequested",
                     "capability-probe-only",
                     "WorkspaceBytes",
                     "BuilderConfigDeploymentSnapshot",
                     "MaxNbTactics",
                     "TilingOptimizationLevel",
                     "L2LimitForTilingBytes",
                     "QuantizationFlags",
                     "ParserPreflightSnapshot",
                     "copied-parser-preflight",
                     "Copied builder-config readback",
                     "RefitSnapshot",
                     "onnx-refit-complete",
                     "RefitPersistenceSnapshot",
                     "refitted-plan-persisted-and-reloaded",
                     "OriginalRefittedEngineDisposedBeforeReload",
                     "RefittableWeightsIncludedInSerialization",
                     "OptionImplementationStatus",
                     "ParsedOptions",
                     "AppliedOptions",
                     "ParseOnlyOptions",
                     "ProofClassification",
                     "BuildEvidenceOnly",
                     "IsRuntimeExecutionProof",
                     "ReportBoundary",
                     "IsRuntimeProof",
                     "IsBuildOnly",
                     "ForbiddenSubstituteReason",
                     "CopiedDiagnosticsBoundary",
                     "ParserDiagnosticsEvidenceKind",
                     "ParserRefitterDiagnosticsEvidenceKind",
                     "CanPromoteCopiedDiagnosticsToRuntimeProof",
                     "ParserDiagnosticsOwnerAction",
                     "TensorRtExec report",
                     "build-only",
                     "dry-run",
                     "template",
                     "local feed",
                     "ProjectReference",
                     "direct `.nupkg`",
                     "YoloVision matrix",
                     "OnnxToEngine report",
                     "readonly diagnostics",
                     "ONNX Parser diagnostic snapshot",
                     "ONNX ParserRefitter diagnostic snapshot",
                     "copied-parser-diagnostics",
                     "copied-parser-refitter-diagnostics",
                 })
        {
            Assert.Contains(term, text, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("YoloDet", text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canPublishPublicly=true", text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canCloseReleaseIssue=true", text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TensorRtExecReportSchemaMatchesCurrentDiagnosticsJsonContract()
    {
        string schemaPath = Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-report.schema.json");
        using JsonDocument schemaDocument = JsonDocument.Parse(File.ReadAllText(schemaPath));
        JsonElement schemaRoot = schemaDocument.RootElement;
        JsonElement schemaProperties = schemaRoot.GetProperty("properties");

        OnnxEngineBuildResult result = new OnnxEngineBuildResult(
            success: true,
            skipped: false,
            state: "build-only",
            tensorRtLine: TensorRtApiLine.TensorRt10,
            modelSource: "embedded-dynamic-identity",
            enginePath: "model.plan",
            parsed: true,
            engineSaved: true,
            engineFileRoundTrip: false,
            inferenceRan: false,
            outputMatch: false,
            profileIndex: 0,
            elapsedMilliseconds: null,
            skipReason: string.Empty,
            normalizedCommandLine: "--buildOnly --workspace 128",
            deploymentOptions: TrtexecLikeDeploymentOptions.Default,
            diagnostics: new[] { "report schema contract probe" },
            logLines: new[] { "TensorRtExec SchemaContract=True" },
            runtimeOptions: TrtexecLikeRuntimeOptions.Default,
            workspaceBytes: 128UL * 1024UL * 1024UL);

        using JsonDocument reportDocument = JsonDocument.Parse(OnnxEngineBuildDiagnostics.ToJson(result));
        JsonElement reportRoot = reportDocument.RootElement;

        foreach (JsonElement required in schemaRoot.GetProperty("required").EnumerateArray())
        {
            string name = required.GetString()!;
            Assert.True(reportRoot.TryGetProperty(name, out _), name);
            Assert.True(schemaProperties.TryGetProperty(name, out _), name);
        }

        foreach (JsonProperty reportProperty in reportRoot.EnumerateObject())
        {
            Assert.True(schemaProperties.TryGetProperty(reportProperty.Name, out _), reportProperty.Name);
        }

        JsonElement boundary = reportRoot.GetProperty("ReportBoundary");
        Assert.False(boundary.GetProperty("IsRuntimeProof").GetBoolean());
        Assert.True(boundary.GetProperty("IsBuildOnly").GetBoolean());
        Assert.Contains("diagnostic/build artifacts", boundary.GetProperty("ForbiddenSubstituteReason").GetString(), StringComparison.Ordinal);
        Assert.Contains("copied diagnostics", boundary.GetProperty("CopiedDiagnosticsBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("copied-parser-diagnostics", boundary.GetProperty("ParserDiagnosticsEvidenceKind").GetString());
        Assert.Equal("copied-parser-refitter-diagnostics", boundary.GetProperty("ParserRefitterDiagnosticsEvidenceKind").GetString());
        Assert.False(boundary.GetProperty("CanPromoteCopiedDiagnosticsToRuntimeProof").GetBoolean());
        Assert.Contains("real-model-runtime evidence", boundary.GetProperty("ParserDiagnosticsOwnerAction").GetString(), StringComparison.Ordinal);
        Assert.Contains(boundary.GetProperty("ForbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "TensorRtExec report");
        Assert.Contains(boundary.GetProperty("ForbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "OnnxToEngine report");
        Assert.Contains(boundary.GetProperty("ForbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "readonly diagnostics");
        Assert.Contains(boundary.GetProperty("ForbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "capability-probe-only");
        Assert.Contains(boundary.GetProperty("ForbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "ONNX Parser diagnostic snapshot");
        Assert.Contains(boundary.GetProperty("ForbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "ONNX ParserRefitter diagnostic snapshot");
        Assert.Contains(boundary.GetProperty("ForbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "copied-parser-diagnostics");
        Assert.Contains(boundary.GetProperty("ForbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "copied-parser-refitter-diagnostics");
        Assert.False(reportRoot.GetProperty("IsRuntimeExecutionProof").GetBoolean());
        Assert.False(reportRoot.GetProperty("IsRealModelRuntimeProof").GetBoolean());
        Assert.False(reportRoot.GetProperty("IsPackageConsumerRuntimeProof").GetBoolean());
        Assert.Equal("build-only", reportRoot.GetProperty("ProofClassification").GetString());
        Assert.Equal(64, reportRoot.GetProperty("NormalizedCommandSha256").GetString()!.Length);
        Assert.True(reportRoot.GetProperty("OptionImplementationStatus").TryGetProperty("ParsedOptions", out _));
        Assert.True(reportRoot.GetProperty("DeploymentOptions").TryGetProperty("BuilderOptimizationLevel", out _));
        Assert.True(reportRoot.GetProperty("RuntimeOptions").TryGetProperty("NoDataTransfers", out _));
        JsonElement benchmarkSummary = reportRoot.GetProperty("BenchmarkSummary");
        Assert.True(benchmarkSummary.TryGetProperty("UseSpinWaitApplied", out _));
        Assert.True(benchmarkSummary.TryGetProperty("UseCudaGraphRequested", out _));
        Assert.True(benchmarkSummary.TryGetProperty("UseCudaGraphApplied", out _));
        Assert.True(benchmarkSummary.TryGetProperty("UseCudaGraphFallbackReason", out _));
        Assert.True(benchmarkSummary.TryGetProperty("MeasurementRoundsPerContext", out _));
        Assert.True(reportRoot.GetProperty("PreflightMetadata").TryGetProperty("EvidenceBoundary", out _));
        Assert.True(reportRoot.GetProperty("LoadedEngineDiagnostics").TryGetProperty("EvidenceBoundary", out _));
        Assert.True(reportRoot.GetProperty("LoadedEngineDiagnostics").TryGetProperty("ReadbackFingerprint", out _));
        Assert.True(reportRoot.GetProperty("LoadedEngineDiagnostics").TryGetProperty("ReadbackSha256", out _));
        JsonElement layerInfoArtifact = reportRoot.GetProperty("LayerInfoArtifact");
        Assert.Equal("not-requested", layerInfoArtifact.GetProperty("State").GetString());
        Assert.False(layerInfoArtifact.GetProperty("Collected").GetBoolean());
        Assert.True(layerInfoArtifact.GetProperty("PointerFreeCopiedSnapshot").GetBoolean());
        Assert.False(layerInfoArtifact.GetProperty("CanPromoteRuntimeProof").GetBoolean());
        Assert.False(layerInfoArtifact.GetProperty("CanPromoteReleaseProof").GetBoolean());
        JsonElement bindingMetadata = reportRoot.GetProperty("BindingMetadata");
        Assert.Equal("not-attempted", bindingMetadata.GetProperty("State").GetString());
        Assert.True(bindingMetadata.GetProperty("PointerFreeCopiedSnapshot").GetBoolean());
        Assert.False(bindingMetadata.GetProperty("CanPromoteRuntimeProof").GetBoolean());
        Assert.False(bindingMetadata.GetProperty("CanPromoteReleaseProof").GetBoolean());
        JsonElement timingCacheArtifact = reportRoot.GetProperty("TimingCacheArtifact");
        Assert.False(timingCacheArtifact.GetProperty("InputRequested").GetBoolean());
        Assert.False(timingCacheArtifact.GetProperty("OutputRequested").GetBoolean());
        Assert.False(timingCacheArtifact.GetProperty("OutputWritten").GetBoolean());
        Assert.True(timingCacheArtifact.GetProperty("EvidenceBoundary").GetString()!.Contains("build-cache lifecycle", StringComparison.Ordinal));
        Assert.True(reportRoot.GetProperty("CapabilityProbe").TryGetProperty("EvidenceBoundary", out _));
        Assert.False(reportRoot.GetProperty("CapabilityProbe").GetProperty("Attempted").GetBoolean());
        Assert.Contains("capability-probe-only", reportRoot.GetProperty("OptionImplementationStatus").GetProperty("EvidenceBoundary").GetString(), StringComparison.Ordinal);
        JsonElement parserPreflight = reportRoot.GetProperty("ParserPreflightSnapshot");
        Assert.Equal("not-attempted", parserPreflight.GetProperty("DiagnosticsState").GetString());
        Assert.True(parserPreflight.GetProperty("PointerFreeCopiedSnapshot").GetBoolean());
        Assert.False(parserPreflight.GetProperty("CanPromoteRuntimeProof").GetBoolean());
        Assert.False(parserPreflight.GetProperty("CanPromoteReleaseProof").GetBoolean());
        Assert.False(parserPreflight.GetProperty("CanDeleteDeferredRecord").GetBoolean());
    }

    [Fact]
    public void ParserPreflightSnapshotKeepsCopiedDiagnosticsBoundary()
    {
        OnnxEngineParserPreflightSnapshot snapshot = new OnnxEngineParserPreflightSnapshot(
            TensorRtApiLine.TensorRt11,
            parseAttempted: true,
            parseSucceeded: true,
            diagnosticsState: "copied-readback",
            errorCount: 0,
            copiedDiagnosticCount: 0,
            diagnosticSummary: "ONNX parser reported no errors.",
            identityOperatorSupported: true,
            modelSupportAttempted: true,
            modelSupportState: "copied-readback",
            modelSupported: true,
            supportedSubgraphCount: 0,
            unsupportedSubgraphCount: 0,
            copiedSubgraphCount: 0,
            copiedSupportedSubgraphCount: 0,
            copiedUnsupportedSubgraphCount: 0,
            copiedNodeCount: 0);

        OnnxEngineBuildResult result = new OnnxEngineBuildResult(
            success: true,
            skipped: false,
            state: "build-only",
            tensorRtLine: TensorRtApiLine.TensorRt11,
            modelSource: "embedded-dynamic-identity",
            enginePath: "model.plan",
            parsed: true,
            engineSaved: true,
            engineFileRoundTrip: false,
            inferenceRan: false,
            outputMatch: false,
            profileIndex: 0,
            elapsedMilliseconds: null,
            skipReason: string.Empty,
            normalizedCommandLine: "--buildOnly",
            diagnostics: Array.Empty<string>(),
            logLines: Array.Empty<string>(),
            parserPreflightSnapshot: snapshot);

        using JsonDocument document = JsonDocument.Parse(OnnxEngineBuildDiagnostics.ToJson(result));
        JsonElement parser = document.RootElement.GetProperty("ParserPreflightSnapshot");
        Assert.Equal(11, parser.GetProperty("Line").GetInt32());
        Assert.True(parser.GetProperty("ParseSucceeded").GetBoolean());
        Assert.Equal("copied-readback", parser.GetProperty("ModelSupportState").GetString());
        Assert.Equal("copied-parser-preflight", parser.GetProperty("EvidenceKind").GetString());
        Assert.Contains("build/preflight evidence only", parser.GetProperty("EvidenceBoundary").GetString(), StringComparison.Ordinal);
        Assert.False(parser.GetProperty("CanPromoteRuntimeProof").GetBoolean());
    }

    [Fact]
    public void BuilderConfigReadbackProofArtifactMatchesReportBoundary()
    {
        string artifactPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "builder-config-deployment-readback-proof.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "builder-config-deployment-readback-proof.md");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(artifactPath));
        JsonElement root = document.RootElement;

        Assert.Equal("cross-version-builder-config-copied-readback", root.GetProperty("recordKind").GetString());
        Assert.True(root.GetProperty("pointerFree").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isRealModelRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.True(root.GetProperty("deferredHistoryRetained").GetBoolean());
        Assert.Contains("BuilderConfigDeploymentSnapshot", File.ReadAllText(markdownPath), StringComparison.Ordinal);
        Assert.Contains("dependency-probe-only", File.ReadAllText(markdownPath), StringComparison.Ordinal);
    }

    [Fact]
    public void CapabilityProbeEvidenceCannotPromoteTensorRtExecReportToRuntimeProof()
    {
        OnnxEngineCapabilityProbe capabilityProbe = new OnnxEngineCapabilityProbe(
            attempted: true,
            probeState: "capability-probe-only",
            tensorRtLine: TensorRtApiLine.TensorRt11,
            tensorRtVersion: "11.0-test",
            cudaToolkitVersion: "13.0-test",
            runtimeAvailable: true,
            builderAvailable: true,
            builderConfigAvailable: true,
            engineInspectorApiAvailable: true,
            fp8FlagRequested: true,
            fp8FlagKnown: true,
            debugTensorOptionsRequested: true,
            debugTensorApiKnown: true,
            weightStreamingRequested: true,
            weightStreamingApiKnown: true,
            probeItems: new[]
            {
                "runtime:create:true",
                "builder:create:true",
                "fp8-builder-flag:requested:True:known:True",
                "debug-tensor-options:requested:True:known:True",
                "weight-streaming-options:requested:True:known:True"
            },
            evidenceBoundary: "capability-probe-only records API availability; it does not enqueue inference or prove package-consumer-runtime.");

        TrtexecLikeDeploymentOptions deploymentOptions = new TrtexecLikeDeploymentOptions(
            builderOptimizationLevel: 5,
            maxAuxStreams: null,
            deviceOrdinal: null,
            dlaCore: null,
            allowGpuFallback: false,
            tacticSources: string.Empty,
            memoryPoolSizes: Array.Empty<TrtexecLikeMemoryPoolSize>(),
            inputIOFormats: string.Empty,
            outputIOFormats: string.Empty,
            calibrationCacheFile: string.Empty,
            directIO: false,
            sparsity: string.Empty,
            stronglyTyped: false,
            minTiming: null,
            avgTiming: null,
            precisionConstraints: string.Empty,
            layerPrecisions: string.Empty,
            layerOutputTypes: string.Empty,
            fp8: true,
            best: true,
            dumpRefit: false,
            allowWeightStreaming: true,
            markDebug: "features",
            dumpDebugTensors: true,
            versionCompatible: false,
            excludeLeanRuntime: false,
            stripWeights: false,
            refit: false,
            weightStreamingBudgetBytes: null,
            exportTimingCachePath: string.Empty,
            safe: false,
            consistency: false,
            builderCache: false,
            noBuilderCache: false);

        OnnxEngineBuildResult result = new OnnxEngineBuildResult(
            success: true,
            skipped: false,
            state: "build-only",
            tensorRtLine: TensorRtApiLine.TensorRt11,
            modelSource: "embedded-dynamic-identity",
            enginePath: "model.plan",
            parsed: true,
            engineSaved: true,
            engineFileRoundTrip: false,
            inferenceRan: false,
            outputMatch: false,
            profileIndex: 0,
            elapsedMilliseconds: null,
            skipReason: string.Empty,
            normalizedCommandLine: "--buildOnly --fp8 --best --allowWeightStreaming --markDebug features --dumpDebugTensors",
            deploymentOptions: deploymentOptions,
            diagnostics: new[] { "capability probe contract" },
            logLines: new[] { "CapabilityProbe State=capability-probe-only" },
            runtimeOptions: TrtexecLikeRuntimeOptions.Default,
            capabilityProbe: capabilityProbe,
            workspaceBytes: 128UL * 1024UL * 1024UL);

        using JsonDocument reportDocument = JsonDocument.Parse(OnnxEngineBuildDiagnostics.ToJson(result));
        JsonElement root = reportDocument.RootElement;
        JsonElement probe = root.GetProperty("CapabilityProbe");
        JsonElement status = root.GetProperty("OptionImplementationStatus");

        Assert.Equal("build-only", root.GetProperty("ProofClassification").GetString());
        Assert.False(root.GetProperty("IsRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("IsRealModelRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("IsPackageConsumerRuntimeProof").GetBoolean());
        Assert.True(probe.GetProperty("Attempted").GetBoolean());
        Assert.Equal("capability-probe-only", probe.GetProperty("ProbeState").GetString());
        Assert.True(probe.GetProperty("Fp8FlagRequested").GetBoolean());
        Assert.True(probe.GetProperty("DebugTensorOptionsRequested").GetBoolean());
        Assert.True(probe.GetProperty("WeightStreamingRequested").GetBoolean());
        Assert.Contains("capability-probe-only", probe.GetProperty("EvidenceBoundary").GetString(), StringComparison.Ordinal);
        Assert.Contains(status.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--fp8");
        Assert.Contains(status.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--best");
        Assert.Contains(status.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--allowWeightStreaming");
        Assert.Contains(status.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--markDebug");
        Assert.Contains(status.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--dumpDebugTensors");
        Assert.Contains(status.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "capability-probe-only");
        Assert.Contains(root.GetProperty("ReportBoundary").GetProperty("ForbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "capability-probe-only");
    }
}
