using System.Text.Json;
using JYPPX.Shared.Interop;
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
        Assert.Contains(boundary.GetProperty("ForbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "TensorRtExec report");
        Assert.Contains(boundary.GetProperty("ForbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "OnnxToEngine report");
        Assert.Contains(boundary.GetProperty("ForbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "readonly diagnostics");
        Assert.Contains(boundary.GetProperty("ForbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "capability-probe-only");
        Assert.False(reportRoot.GetProperty("IsRuntimeExecutionProof").GetBoolean());
        Assert.False(reportRoot.GetProperty("IsRealModelRuntimeProof").GetBoolean());
        Assert.False(reportRoot.GetProperty("IsPackageConsumerRuntimeProof").GetBoolean());
        Assert.Equal("build-only", reportRoot.GetProperty("ProofClassification").GetString());
        Assert.Equal(64, reportRoot.GetProperty("NormalizedCommandSha256").GetString()!.Length);
        Assert.True(reportRoot.GetProperty("OptionImplementationStatus").TryGetProperty("ParsedOptions", out _));
        Assert.True(reportRoot.GetProperty("DeploymentOptions").TryGetProperty("BuilderOptimizationLevel", out _));
        Assert.True(reportRoot.GetProperty("RuntimeOptions").TryGetProperty("NoDataTransfers", out _));
        Assert.True(reportRoot.GetProperty("PreflightMetadata").TryGetProperty("EvidenceBoundary", out _));
        Assert.True(reportRoot.GetProperty("LoadedEngineDiagnostics").TryGetProperty("EvidenceBoundary", out _));
        Assert.True(reportRoot.GetProperty("LoadedEngineDiagnostics").TryGetProperty("ReadbackFingerprint", out _));
        Assert.True(reportRoot.GetProperty("LoadedEngineDiagnostics").TryGetProperty("ReadbackSha256", out _));
        Assert.True(reportRoot.GetProperty("CapabilityProbe").TryGetProperty("EvidenceBoundary", out _));
        Assert.False(reportRoot.GetProperty("CapabilityProbe").GetProperty("Attempted").GetBoolean());
        Assert.Contains("capability-probe-only", reportRoot.GetProperty("OptionImplementationStatus").GetProperty("EvidenceBoundary").GetString(), StringComparison.Ordinal);
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
