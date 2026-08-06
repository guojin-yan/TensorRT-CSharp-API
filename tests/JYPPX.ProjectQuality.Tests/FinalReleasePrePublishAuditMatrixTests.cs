using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalReleasePrePublishAuditMatrixTests
{
    [Fact]
    public void PrePublishAuditMatrixKeepsTensorRtExecYoloVisionAndOnnxToEngineNonProofBoundaries()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseCandidateReadiness.ps1"),
            "-AllowRuntimeSmokeBlocked",
            "-WarnOnly");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseDryRun.ps1"), "-AllowRuntimeSmokeBlocked");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalReleaseCloseBlockerDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalReleasePrePublishAuditMatrix.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleasePrePublishAuditMatrix.ps1"), "-Strict");

        using JsonDocument matrixDocument = ReadFinalReleaseJson("final-release-pre-publish-audit-matrix.json");
        JsonElement matrix = matrixDocument.RootElement;

        Assert.Equal("final-release-pre-publish-audit-matrix", matrix.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-pre-publish-owner-proof-required", matrix.GetProperty("auditState").GetString());
        Assert.False(matrix.GetProperty("cleanOwnerInputReady").GetBoolean());
        Assert.False(matrix.GetProperty("ownerInputCanPromoteRuntimeProof").GetBoolean());
        Assert.False(matrix.GetProperty("externalRuntimeProofReady").GetBoolean());
        Assert.False(matrix.GetProperty("postPublishVerificationReady").GetBoolean());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment", matrix.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentState").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment-valid", matrix.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState").GetString());
        Assert.Equal("Smoke=not-requested", matrix.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus").GetString());
        Assert.True(matrix.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount").GetInt32() >= 30);
        Assert.Equal(0, matrix.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount").GetInt32());
        Assert.Equal(0, matrix.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount").GetInt32());
        Assert.False(matrix.GetProperty("performsPublish").GetBoolean());
        Assert.False(matrix.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(matrix.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(matrix.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(matrix.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(matrix.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(matrix.GetProperty("isPostPublishProof").GetBoolean());
        Assert.Equal(0, matrix.GetProperty("tensorRtExecRuntimeProofItems").GetInt32());
        Assert.Contains("not runtime proof", matrix.GetProperty("tensorRtExecBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package-consumer-runtime proof", matrix.GetProperty("yoloVisionBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("build/report", matrix.GetProperty("onnxToEngineBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not runtime proof", matrix.GetProperty("onnxToEngineBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.False(matrix.GetProperty("retiredSampleLiveProjectPathPresent").GetBoolean());
        Assert.False(matrix.GetProperty("retiredSampleLiveProjectFilePresent").GetBoolean());

        string[] sourceArtifacts = matrix.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-evidence-bundle.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-release-dry-run-summary.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-release-close-blocker-dashboard.json", sourceArtifacts);
        Assert.Contains("applications/TensorRtExec/tensor-rt-exec-feature-matrix.json", sourceArtifacts);
        Assert.Contains("applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json", sourceArtifacts);
        Assert.Contains("applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json", sourceArtifacts);
        Assert.Contains("applications/YoloVision/yolo-model-matrix.json", sourceArtifacts);
        Assert.Contains("applications/YoloVision/README.md", sourceArtifacts);
        Assert.Contains("applications/OnnxToEngine/trtexec-parity-matrix.json", sourceArtifacts);
        Assert.Contains("applications/OnnxToEngine/README.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json", sourceArtifacts);

        string[] families = matrix.GetProperty("yoloVisionFamilies").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string family in new[] { "yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov26", "custom" })
        {
            Assert.Contains(family, families);
        }

        string[] tasks = matrix.GetProperty("yoloVisionTasks").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string task in new[] { "det", "cls", "seg", "obb", "pose", "sem" })
        {
            Assert.Contains(task, tasks);
        }

        JsonElement signals = matrix.GetProperty("readmeBoundarySignals");
        Assert.True(signals.GetProperty("tensorRtExecMentionsBuildReportNotProof").GetBoolean());
        Assert.True(signals.GetProperty("yoloVisionMentionsRequiredFamilies").GetBoolean());
        Assert.True(signals.GetProperty("yoloVisionMentionsRequiredTasks").GetBoolean());
        Assert.True(signals.GetProperty("onnxToEngineMentionsBuildOnlyBoundary").GetBoolean());

        JsonElement[] auditItems = matrix.GetProperty("auditItems").EnumerateArray().ToArray();
        Assert.Contains(auditItems, item => item.GetProperty("id").GetString() == "owner-runtime-smoke-field-alignment");
        Assert.Contains(auditItems, item => item.GetProperty("id").GetString() == "tensorrtexec-cli-winforms-parity");
        Assert.Contains(auditItems, item => item.GetProperty("id").GetString() == "tensorrtexec-gap-list-non-proof");
        Assert.Contains(auditItems, item => item.GetProperty("id").GetString() == "yolovision-family-task-matrix");
        Assert.Contains(auditItems, item => item.GetProperty("id").GetString() == "onnxtoengine-trtexec-like-boundary");
        Assert.All(auditItems, item =>
        {
            Assert.False(item.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(item.GetProperty("isRuntimeProof").GetBoolean());
            Assert.False(item.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        });

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-release-pre-publish-audit-matrix-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-release-pre-publish-audit-matrix-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("final-release-pre-publish-audit-matrix-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Contains(validation.GetProperty("validationItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == "tensorrtexec-non-proof" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Contains(validation.GetProperty("validationItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == "yolovision-family-task-coverage" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Contains(validation.GetProperty("validationItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == "onnxtoengine-build-boundary" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Contains(validation.GetProperty("validationItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == "owner-runtime-smoke-field-alignment-projected" &&
            item.GetProperty("passed").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}
