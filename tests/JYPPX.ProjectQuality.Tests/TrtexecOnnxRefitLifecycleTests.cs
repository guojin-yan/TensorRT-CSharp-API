using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TrtexecOnnxRefitLifecycleTests
{
    [Fact]
    public void CompactEvidenceProvesParserLoadEngineCommitContextGateAndBaselineMatch()
    {
        using JsonDocument document = JsonDocument.Parse(ReadSource(
            "artifacts", "interface-coverage", "trtexec-onnx-refit-lifecycle-evidence.json"));
        JsonElement root = document.RootElement;
        JsonElement trt10 = root.GetProperty("tensorRt10");

        Assert.Equal("trtexec-onnx-refit-lifecycle-evidence.v1", root.GetProperty("schemaVersion").GetString());
        Assert.True(trt10.GetProperty("stripPlanApplied").GetBoolean());
        Assert.True(trt10.GetProperty("explicitRefitApplied").GetBoolean());
        Assert.True(trt10.GetProperty("engineRefittableBefore").GetBoolean());
        Assert.True(trt10.GetProperty("parserRefitReturned").GetBoolean());
        Assert.True(trt10.GetProperty("engineRefitReturned").GetBoolean());
        Assert.True(trt10.GetProperty("engineRefittableAfter").GetBoolean());
        Assert.Equal(0, trt10.GetProperty("missingWeightsAfter").GetInt32());
        Assert.True(trt10.GetProperty("allWeightsAfter").GetInt32() > 0);
        Assert.Equal(0, trt10.GetProperty("parserErrorCount").GetInt32());
        Assert.True(trt10.GetProperty("contextCreatedAfterRefitCommit").GetBoolean());
        Assert.True(trt10.GetProperty("inferenceRan").GetBoolean());
        Assert.True(trt10.GetProperty("outputExactMatch").GetBoolean());
        Assert.Equal(trt10.GetProperty("baselineOutputSha256").GetString(), trt10.GetProperty("refitOutputSha256").GetString());

        Assert.Equal("dry-run-precheck", root.GetProperty("tensorRt8").GetProperty("state").GetString());
        Assert.Equal("dependency-probe-only", root.GetProperty("tensorRt11").GetProperty("state").GetString());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("isRefittedPlanPersistenceProof").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("canPublishPublicly").GetBoolean());
    }

    [Fact]
    public void StrictValidatorPassesAllLifecycleChecks()
    {
        using JsonDocument document = JsonDocument.Parse(ReadSource(
            "artifacts", "interface-coverage", "trtexec-onnx-refit-lifecycle-validation.json"));
        JsonElement root = document.RootElement;
        Assert.True(root.GetProperty("strict").GetBoolean());
        Assert.Equal(24, root.GetProperty("checkCount").GetInt32());
        Assert.Equal(24, root.GetProperty("passedCount").GetInt32());
        Assert.Equal(0, root.GetProperty("failureCount").GetInt32());
    }

    [Fact]
    public void ApplicationCommitsRefitterBeforeAnyExecutionContextAndReportsCopiedSnapshot()
    {
        string service = ReadSource(
            "src", "JYPPX.TensorRtSharp.Tools", "Build", "OnnxEngineBuildService.Refit.cs");
        string snapshot = ReadSource("src", "JYPPX.TensorRtSharp.Tools", "Refit", "OnnxEngineRefitSnapshot.cs");
        string diagnostics = ReadSource("src", "JYPPX.TensorRtSharp.Tools", "Build", "OnnxEngineBuildDiagnostics.cs");
        string artifactWriter = ReadSource("src", "JYPPX.TensorRtSharp.Tools", "Artifacts", "OnnxEngineRuntimeArtifactWriter.cs");
        string schema = ReadSource("applications", "TensorRtExec", "tensor-rt-exec-report.schema.json");
        string form = ReadSource("applications", "TensorRtExec", "WinForms", "MainForm.cs");

        int parserRefit = service.IndexOf("parserRefitter.RefitFromFile(sourcePath)", StringComparison.Ordinal);
        int engineCommit = service.IndexOf("refitter.RefitCudaEngine()", parserRefit, StringComparison.Ordinal);
        int contextGate = service.IndexOf("ContextCreationAllowed={snapshot.ContextCreationAllowed}", engineCommit, StringComparison.Ordinal);
        Assert.True(parserRefit >= 0 && engineCommit > parserRefit && contextGate > engineCommit);
        Assert.Contains("missingAfter.Count == 0", service, StringComparison.Ordinal);
        Assert.Contains("parserSnapshot.ErrorCount == 0", service, StringComparison.Ordinal);
        Assert.Contains("public bool EngineRefitReturned", snapshot, StringComparison.Ordinal);
        Assert.Contains("result.RefitSnapshot", diagnostics, StringComparison.Ordinal);
        Assert.Contains("OutputComparisonSample", artifactWriter, StringComparison.Ordinal);
        Assert.Contains("OutputSha256", artifactWriter, StringComparison.Ordinal);
        Assert.Contains("\"RefitSnapshot\"", schema, StringComparison.Ordinal);
        Assert.Contains("_refitFromOnnxPath", form, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}
