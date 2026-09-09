using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecCurrentRuntimeEvidenceTests
{
    [Fact]
    public void ExternalYoloPrecisionPolicyEvidenceKeepsRuntimeAndReleaseBoundariesSeparate()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "tensorrtexec-yolov8n-cls-precision-policy-runtime-evidence.json");
        string text = File.ReadAllText(evidencePath);
        using JsonDocument document = JsonDocument.Parse(text);
        JsonElement root = document.RootElement;

        Assert.Equal("tensorrtexec-external-onnx-precision-policy-runtime-evidence", root.GetProperty("recordKind").GetString());
        Assert.Equal("external-onnx-reference-validated-runtime", root.GetProperty("state").GetString());
        Assert.Equal("synthetic-input-runtime", root.GetProperty("proofClassification").GetString());
        Assert.True(root.GetProperty("sourceTreeDirtyAtExecution").GetBoolean());

        JsonElement execution = root.GetProperty("execution");
        Assert.Equal(0, execution.GetProperty("processExitCode").GetInt32());
        Assert.True(execution.GetProperty("success").GetBoolean());
        Assert.True(execution.GetProperty("parsed").GetBoolean());
        Assert.True(execution.GetProperty("engineSaved").GetBoolean());
        Assert.True(execution.GetProperty("engineFileRoundTrip").GetBoolean());
        Assert.True(execution.GetProperty("inferenceRan").GetBoolean());
        Assert.True(execution.GetProperty("outputValidated").GetBoolean());
        Assert.Equal(0, execution.GetProperty("profileIndex").GetInt32());

        JsonElement policy = root.GetProperty("precisionPolicy");
        foreach (string name in new[] { "inputIOFormats", "outputIOFormats", "precisionConstraints", "layerPrecision", "layerOutputType" })
        {
            JsonElement item = policy.GetProperty(name);
            Assert.True(item.GetProperty("applied").GetBoolean(), name);
            Assert.True(item.GetProperty("readbackMatch").GetBoolean(), name);
        }
        Assert.True(policy.GetProperty("engineLayerImplementationMetadataObserved").GetBoolean());
        Assert.True(policy.GetProperty("engineLayerInputOutputDatatypeObserved").GetBoolean());
        Assert.True(policy.GetProperty("engineSelectedTacticObserved").GetBoolean());
        Assert.False(policy.GetProperty("standaloneImplementationPrecisionPropertyObserved").GetBoolean());
        Assert.False(policy.GetProperty("engineImplementationPrecisionObserved").GetBoolean());

        JsonElement inspector = policy.GetProperty("engineInspectorReadback");
        Assert.Equal("detailed", inspector.GetProperty("requestedProfilingVerbosity").GetString());
        Assert.Equal("Json", inspector.GetProperty("informationFormat").GetString());
        Assert.Equal("tensor-rt-engine-layer-information", inspector.GetProperty("artifactKind").GetString());
        Assert.Equal("json-document", inspector.GetProperty("contentKind").GetString());
        Assert.Equal("LoadEngine", inspector.GetProperty("source").GetString());
        Assert.Equal(87, inspector.GetProperty("layerCount").GetInt32());
        JsonElement targetLayer = inspector.GetProperty("targetLayer");
        Assert.Equal("/model.0/conv/Conv", targetLayer.GetProperty("name").GetString());
        Assert.Equal("CaskConvolution", targetLayer.GetProperty("layerType").GetString());
        Assert.Equal("Float", targetLayer.GetProperty("inputFormatDatatype").GetString());
        Assert.Equal("Float", targetLayer.GetProperty("outputFormatDatatype").GetString());
        Assert.Equal("Float", targetLayer.GetProperty("weightsType").GetString());
        Assert.Equal("Float", targetLayer.GetProperty("biasType").GetString());
        Assert.Equal("sm80_xmma_fprop_implicit_gemm_f32f32_f32f32_f32_nchwkcrs_nchw_tilesize128x16x8_stage3_warpsize4x1x1_g1_ffma_aligna4_alignc4", targetLayer.GetProperty("tacticName").GetString());
        Assert.Equal("0x9d9fdb5fd9945f64", targetLayer.GetProperty("tacticValue").GetString());

        JsonElement binding = root.GetProperty("bindingMetadata");
        Assert.Equal("runtime-binding-readback", binding.GetProperty("state").GetString());
        Assert.Equal(0, binding.GetProperty("profileIndex").GetInt32());
        Assert.True(binding.GetProperty("readyForEnqueue").GetBoolean());
        Assert.True(binding.GetProperty("pointerFreeCopiedSnapshot").GetBoolean());
        Assert.Equal(2, binding.GetProperty("tensorCount").GetInt32());

        JsonElement reference = root.GetProperty("referenceValidation");
        Assert.True(reference.GetProperty("passed").GetBoolean());
        Assert.Equal(1000, reference.GetProperty("comparedElementCount").GetInt32());
        Assert.Equal(0, reference.GetProperty("mismatchCount").GetInt32());

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("tensorRtExecRuntimeExecution").GetBoolean());
        Assert.True(boundary.GetProperty("externalRealOnnxAndInput").GetBoolean());
        Assert.True(boundary.GetProperty("engineLayerImplementationMetadataObserved").GetBoolean());
        Assert.True(boundary.GetProperty("engineLayerInputOutputDatatypeObserved").GetBoolean());
        Assert.True(boundary.GetProperty("engineSelectedTacticObserved").GetBoolean());
        Assert.False(boundary.GetProperty("standaloneImplementationPrecisionPropertyObserved").GetBoolean());
        Assert.False(boundary.GetProperty("engineImplementationPrecisionObserved").GetBoolean());
        Assert.False(boundary.GetProperty("sourceTreeRealModelRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("packageConsumerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("publicPackageProof").GetBoolean());
        Assert.False(boundary.GetProperty("postPublishProof").GetBoolean());
        Assert.False(boundary.GetProperty("releaseProof").GetBoolean());
        Assert.Empty(Regex.Matches(text, @"(?i)[A-Z]:\\"));

        string builderConfiguration = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools",
            "Build",
            "OnnxEngineBuildService.BuilderConfiguration.cs");
        JsonElement maintenance = root.GetProperty("currentMaintenanceValidation");
        Assert.Equal(
            ComputeSha256(builderConfiguration),
            maintenance.GetProperty("currentBuilderConfigurationSha256").GetString());

        string diagnostics = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools",
            "Build",
            "OnnxEngineBuildService.Diagnostics.cs");
        Assert.Equal(
            ComputeSha256(diagnostics),
            maintenance.GetProperty("currentDiagnosticsSha256").GetString());
        Assert.False(maintenance.GetProperty("gpuRuntimeScenarioRerun").GetBoolean());
        Assert.True(maintenance.GetProperty("historicalRuntimeEvidenceRetained").GetBoolean());

        JsonElement layerInfo = root.GetProperty("artifacts").GetProperty("layerInfo");
        Assert.Equal("Json", layerInfo.GetProperty("informationFormat").GetString());
        Assert.Equal("detailed", layerInfo.GetProperty("requestedProfilingVerbosity").GetString());
        Assert.Equal("tensor-rt-engine-layer-information", layerInfo.GetProperty("artifactKind").GetString());
        Assert.Equal("json-document", layerInfo.GetProperty("contentKind").GetString());
        Assert.Equal("LoadEngine", layerInfo.GetProperty("source").GetString());
        Assert.Equal(87, layerInfo.GetProperty("layerCount").GetInt32());
        Assert.Equal("05995dad0aedbee6cf6d9e50d2e2b970798fdebc230db08944f7c4cfbbef4319", layerInfo.GetProperty("sha256").GetString());

        JsonElement strictValidation = root.GetProperty("artifacts").GetProperty("strictValidation");
        Assert.Equal("tensor-rt-exec-report-ready", strictValidation.GetProperty("state").GetString());
        Assert.Equal(0, strictValidation.GetProperty("failedBlockers").GetInt32());

        using JsonDocument yoloEvidence = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-cls-real-model-runtime-evidence.json")));
        Assert.Equal(
            yoloEvidence.RootElement.GetProperty("assets").GetProperty("onnx").GetProperty("sha256").GetString(),
            root.GetProperty("sourceAssets").GetProperty("model").GetProperty("sha256").GetString());
    }

    [Fact]
    public void CurrentLocalPackageConsumerEvidenceIsStrictButNotPublicProof()
    {
        using JsonDocument evidence = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "trtexec-refitted-plan-package-consumer-evidence.json")));
        using JsonDocument validation = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "trtexec-refitted-plan-package-consumer-validation.json")));

        JsonElement root = evidence.RootElement;
        Assert.Equal("2026-08-09", root.GetProperty("generatedDate").GetString());
        Assert.Equal("4.0.0-local.staticprofile.20260809", root.GetProperty("packages").GetProperty("managed").GetProperty("version").GetString());
        Assert.Equal("4.0.0-l2", root.GetProperty("packages").GetProperty("bridge").GetProperty("version").GetString());
        Assert.True(root.GetProperty("execution").GetProperty("runtimePassed").GetBoolean());
        Assert.True(root.GetProperty("runtime").GetProperty("referenceValidationPassed").GetBoolean());
        Assert.True(root.GetProperty("proofBoundary").GetProperty("isLocalPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("packagesDownloadedFromPublicFeed").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("isPostPublishProof").GetBoolean());

        Assert.True(validation.RootElement.GetProperty("strict").GetBoolean());
        Assert.Equal(53, validation.RootElement.GetProperty("checkCount").GetInt32());
        Assert.Equal(53, validation.RootElement.GetProperty("passedCount").GetInt32());
        Assert.Equal(0, validation.RootElement.GetProperty("failureCount").GetInt32());
    }

    [Fact]
    public void CrossVersionEvidenceKeepsTrt8BlockerAndTrt11RuntimeResultsSeparate()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "tensorrtexec-yolov8n-cls-cross-version-runtime-evidence.json");
        string text = File.ReadAllText(evidencePath);
        using JsonDocument document = JsonDocument.Parse(text);
        JsonElement root = document.RootElement;

        Assert.Equal("tensorrtexec-cross-version-external-onnx-runtime-evidence", root.GetProperty("recordKind").GetString());
        Assert.True(root.GetProperty("sourceTreeDirtyAtExecution").GetBoolean());

        JsonElement trt8 = root.GetProperty("runs").GetProperty("tensorRt8");
        Assert.Equal("8.6.1", trt8.GetProperty("tensorRtVersion").GetString());
        Assert.Equal("dependency-probe-only", trt8.GetProperty("state").GetString());
        Assert.True(trt8.GetProperty("runtimeAvailable").GetBoolean());
        Assert.True(trt8.GetProperty("builderAvailable").GetBoolean());
        Assert.True(trt8.GetProperty("skipped").GetBoolean());
        Assert.False(trt8.GetProperty("parsed").GetBoolean());
        Assert.False(trt8.GetProperty("inferenceRan").GetBoolean());
        Assert.Equal("vendor-seh-parser-safety-guard", trt8.GetProperty("blockerKind").GetString());
        Assert.Equal(0, trt8.GetProperty("artifacts").GetProperty("strictValidation").GetProperty("failedBlockers").GetInt32());

        JsonElement trt11 = root.GetProperty("runs").GetProperty("tensorRt11");
        Assert.Equal("11.0.0", trt11.GetProperty("tensorRtVersion").GetString());
        Assert.Equal("external-onnx-reference-validated-runtime", trt11.GetProperty("state").GetString());
        Assert.True(trt11.GetProperty("parsed").GetBoolean());
        Assert.True(trt11.GetProperty("engineFileRoundTrip").GetBoolean());
        Assert.True(trt11.GetProperty("outputValidated").GetBoolean());
        Assert.Equal(0, trt11.GetProperty("referenceValidation").GetProperty("mismatchCount").GetInt32());
        Assert.Contains("--inputIOFormats", trt11.GetProperty("optionClassification").GetProperty("applied").EnumerateArray().Select(static item => item.GetString()));
        Assert.Contains("--precisionConstraints", trt11.GetProperty("optionClassification").GetProperty("parseOnly").EnumerateArray().Select(static item => item.GetString()));

        JsonElement layerInfo = trt11.GetProperty("layerInfoArtifact");
        Assert.Equal("tensor-rt-engine-layer-information", layerInfo.GetProperty("artifactKind").GetString());
        Assert.Equal("json-document", layerInfo.GetProperty("contentKind").GetString());
        Assert.Equal(88, layerInfo.GetProperty("layerCount").GetInt32());
        Assert.Matches("^[0-9a-f]{64}$", layerInfo.GetProperty("sha256").GetString());

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("tensorRt8DependencyProbeOnly").GetBoolean());
        Assert.True(boundary.GetProperty("tensorRt11TensorRtExecRuntimeExecution").GetBoolean());
        Assert.False(boundary.GetProperty("tensorRt11RemovedPrecisionSettersApplied").GetBoolean());
        Assert.False(boundary.GetProperty("sourceTreeRealModelRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("packageConsumerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("releaseProof").GetBoolean());
        Assert.Empty(Regex.Matches(text, @"(?i)[A-Z]:\\"));
    }

    private static string ComputeSha256(string path)
    {
        return Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    }
}
