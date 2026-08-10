using System;
using System.IO;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TrtexecEnginePackagingPolicyTests
{
    [Fact]
    public void CompactEvidenceProvesVersionGuardsAndNonZeroWeightedBudgetWithoutProofPromotion()
    {
        using JsonDocument evidence = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "trtexec-engine-packaging-runtime-evidence.json")));
        JsonElement root = evidence.RootElement;

        Assert.Equal("trtexec-engine-packaging-runtime-evidence.v1", root.GetProperty("schemaVersion").GetString());
        JsonElement trt8 = root.GetProperty("tensorRt8");
        Assert.True(trt8.GetProperty("versionCompatibleApplied").GetBoolean());
        Assert.True(trt8.GetProperty("excludeLeanRuntimeApplied").GetBoolean());
        Assert.True(trt8.GetProperty("refitVersionCompatibleConflictGuarded").GetBoolean());
        Assert.True(trt8.GetProperty("stripWeightsGuarded").GetBoolean());
        Assert.True(trt8.GetProperty("weightStreamingGuarded").GetBoolean());

        JsonElement trt10 = root.GetProperty("tensorRt10");
        JsonElement versionRefit = trt10.GetProperty("versionCompatibleRefit");
        Assert.True(versionRefit.GetProperty("inferenceRan").GetBoolean());
        Assert.True(versionRefit.GetProperty("outputMatch").GetBoolean());
        Assert.True(versionRefit.GetProperty("refittableEngineReadbackMatch").GetBoolean());
        Assert.True(versionRefit.GetProperty("runtimeHostCodeReadbackMatch").GetBoolean());
        Assert.Equal("RefitIdentical", trt10.GetProperty("stripWeights").GetProperty("defaultRefitMode").GetString());

        JsonElement streaming = trt10.GetProperty("weightStreaming");
        long streamable = streaming.GetProperty("streamableWeightsBytes").GetInt64();
        long resolved = streaming.GetProperty("resolvedBudgetBytes").GetInt64();
        Assert.True(streamable > 0);
        Assert.Equal(streamable / 2, resolved);
        Assert.Equal(resolved, streaming.GetProperty("readbackBudgetBytes").GetInt64());
        Assert.True(streaming.GetProperty("scratchBytes").GetInt64() > 0);
        Assert.True(streaming.GetProperty("contextCreatedAfterBudgetReadback").GetBoolean());
        Assert.Equal("captured-unverified", streaming.GetProperty("outputValidationState").GetString());

        JsonElement load = trt10.GetProperty("loadEngineAutomaticBudget");
        Assert.True(load.GetProperty("diagnosticsSucceeded").GetBoolean());
        Assert.True(load.GetProperty("inferenceRan").GetBoolean());
        Assert.Equal(load.GetProperty("automaticBudgetBytes").GetInt64(), load.GetProperty("readbackBudgetBytes").GetInt64());

        JsonElement trt11 = root.GetProperty("tensorRt11");
        Assert.Equal("external-onnx-refit-reload-reference-validated-runtime", trt11.GetProperty("state").GetString());
        Assert.True(trt11.GetProperty("refitConfigReadbackMatch").GetBoolean());
        Assert.True(trt11.GetProperty("stripPlanReadbackMatch").GetBoolean());
        Assert.True(trt11.GetProperty("parserRefitReturned").GetBoolean());
        Assert.True(trt11.GetProperty("engineRefitReturned").GetBoolean());
        Assert.True(trt11.GetProperty("persistedReloadSucceeded").GetBoolean());
        Assert.True(trt11.GetProperty("outputValidated").GetBoolean());
        Assert.True(trt11.GetProperty("historicalDependencyProbeRetained").GetBoolean());
        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("isBuilderAndEnginePolicyEvidence").GetBoolean());
        Assert.True(boundary.GetProperty("isWeightedModelEnqueueEvidence").GetBoolean());
        Assert.False(boundary.GetProperty("isModelAccuracyProof").GetBoolean());
        Assert.False(boundary.GetProperty("isCrossVersionLeanRuntimeProof").GetBoolean());
        Assert.True(boundary.GetProperty("isStrippedPlanRefitLifecycleProof").GetBoolean());
        Assert.False(boundary.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(boundary.GetProperty("publicReleaseSideEffectsExecuted").GetBoolean());
    }

    [Fact]
    public void StrictEvidenceValidatorAndDocumentationCoverTheFullPolicyBatch()
    {
        using JsonDocument validation = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "trtexec-engine-packaging-runtime-evidence-validation.json")));
        Assert.Equal("passed", validation.RootElement.GetProperty("validationState").GetString());
        Assert.Equal(21, validation.RootElement.GetProperty("checkCount").GetInt32());
        Assert.Equal(0, validation.RootElement.GetProperty("failureCount").GetInt32());

        string exporter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-TrtexecEnginePackagingRuntimeEvidence.ps1"));
        string validator = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-TrtexecEnginePackagingRuntimeEvidence.ps1"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "tensorrtexec-engine-packaging-weight-streaming.md"));
        string parity = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-trtexec-parity-matrix.json"));
        string checklist = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-TensorRtExecGuiCliParityChecklist.ps1"));

        Assert.Contains("StreamableWeightsBytes", exporter, StringComparison.Ordinal);
        Assert.Contains("ScratchBytes", exporter, StringComparison.Ordinal);
        Assert.Contains("trt10-percentage-budget", validator, StringComparison.Ordinal);
        Assert.Contains("trt10-context-order", validator, StringComparison.Ordinal);
        Assert.Contains("--weightStreamingBudget 50%", article, StringComparison.Ordinal);
        Assert.Contains("35,829,504", article, StringComparison.Ordinal);
        Assert.Contains("21 checks / 0 failures", article, StringComparison.Ordinal);
        Assert.Contains("engine-packaging-refit-weight-streaming", parity, StringComparison.Ordinal);
        Assert.Contains("implemented-build-runtime-readback-with-version-guards", parity, StringComparison.Ordinal);
        Assert.Contains("engine-packaging-refit-weight-streaming", checklist, StringComparison.Ordinal);
        Assert.DoesNotContain("IntPtr", File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools", "Trtexec", "TrtexecLikeWeightStreamingBudget.cs")), StringComparison.Ordinal);
    }
}
