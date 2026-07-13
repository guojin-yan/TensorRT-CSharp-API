using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RealCaseProofExecutionPackTests
{
    [Fact]
    public void RealCaseProofExecutionPackExportsBlockedAuditableCases()
    {
        string packOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealCaseProofExecutionPack.ps1"));
        string templateOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealCaseEvidenceRecordTemplate.ps1"));
        string validationOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealCaseEvidenceRecord.ps1"));

        Assert.Contains("Real case proof execution pack written", packOutput, StringComparison.Ordinal);
        Assert.Contains("Real case evidence record template written", templateOutput, StringComparison.Ordinal);
        Assert.Contains("ValidationState=blocked-owner-action-required", validationOutput, StringComparison.Ordinal);

        JsonElement pack = ReadJsonRoot(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "real-case-proof-execution-pack.json"));
        Assert.Equal("real-case-proof-execution-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", pack.GetProperty("packState").GetString());
        Assert.False(pack.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(pack.GetProperty("caseCount").GetInt32() >= 9);
        Assert.Equal(pack.GetProperty("caseCount").GetInt32(), pack.GetProperty("blockedCaseCount").GetInt32());
        Assert.True(pack.GetProperty("missingOwnerInputCount").GetInt32() >= 48);
        Assert.True(pack.GetProperty("yoloVisionCaseCount").GetInt32() >= 6);
        Assert.True(pack.GetProperty("onnxToEngineCaseCount").GetInt32() >= 1);
        Assert.True(pack.GetProperty("tensorRtExecCaseCount").GetInt32() >= 2);
        Assert.Contains("does not run models", pack.GetProperty("releaseProofBoundary").GetString()!, StringComparison.Ordinal);
        string[] requiredTasks = pack.GetProperty("requiredTaskCoverage").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string task in new[] { "det", "seg", "obb", "pose", "cls", "sem" })
        {
            Assert.Contains(task, requiredTasks);
        }

        JsonElement[] cases = pack.GetProperty("cases").EnumerateArray().ToArray();
        Assert.Contains(cases, static item => item.GetProperty("caseId").GetString() == "yolovision-detection");
        Assert.Contains(cases, static item => item.GetProperty("caseId").GetString() == "yolovision-segmentation");
        Assert.Contains(cases, static item => item.GetProperty("caseId").GetString() == "yolovision-obb");
        Assert.Contains(cases, static item => item.GetProperty("caseId").GetString() == "yolovision-pose");
        Assert.Contains(cases, static item => item.GetProperty("caseId").GetString() == "yolovision-classification");
        Assert.Contains(cases, static item => item.GetProperty("caseId").GetString() == "yolovision-semantic");
        Assert.Contains(cases, static item => item.GetProperty("caseId").GetString() == "onnx-to-engine-build");
        Assert.Contains(cases, static item => item.GetProperty("caseId").GetString() == "tensorrtexec-build-report");
        Assert.Contains(cases, static item => item.GetProperty("caseId").GetString() == "tensorrtexec-gui-workflow");

        foreach (JsonElement item in cases)
        {
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("caseId").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("sampleProject").GetString()));
            Assert.True(item.GetProperty("articleIds").GetArrayLength() >= 1);
            Assert.True(item.GetProperty("repoPaths").GetArrayLength() >= 1);
            Assert.True(item.GetProperty("requiredCommands").GetArrayLength() >= 1);
            Assert.True(item.GetProperty("expectedHashFields").GetArrayLength() >= 4);
            Assert.True(item.GetProperty("expectedHostMetadata").GetArrayLength() >= 6);
            Assert.True(item.GetProperty("missingOwnerInputs").GetArrayLength() >= 5);
            Assert.True(item.GetProperty("promotionBlockers").GetArrayLength() >= 2);
            Assert.True(item.GetProperty("ownerActionRequired").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.Contains("Test-RealCaseEvidenceRecord.ps1", item.GetProperty("validatorCommand").GetString()!, StringComparison.Ordinal);
            Assert.Contains(item.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static value => value.GetString() == "build-only");
            Assert.DoesNotContain("YoloDet", item.GetRawText(), StringComparison.Ordinal);
        }

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "real-case-proof-execution-pack.md"));
        Assert.Contains("Real Case Proof Execution Pack", markdown, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly", markdown, StringComparison.Ordinal);
        Assert.Contains("Build-only reports", markdown, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void RealCaseEvidenceTemplateAndValidatorDoNotPromoteTemplateToProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealCaseEvidenceRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealCaseEvidenceRecord.ps1"));

        JsonElement template = ReadJsonRoot(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "real-case-evidence-record-template.json"));
        Assert.Equal("real-case-evidence-record-template", template.GetProperty("recordKind").GetString());
        Assert.Equal("owner-action-required", template.GetProperty("templateState").GetString());
        Assert.True(template.GetProperty("templateOnly").GetBoolean());
        Assert.False(template.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(template.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] requiredOwnerFields = template.GetProperty("requiredOwnerFields").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string field in new[]
        {
            "caseId",
            "modelSource",
            "modelLicense",
            "onnxSha256",
            "engineSha256",
            "inputArtifactSha256",
            "outputArtifactSha256",
            "commandLine",
            "stdoutLogPath",
            "stderrLogPath",
            "screenshotPath",
            "hostOs",
            "gpuName",
            "nvidiaDriverVersion",
            "cudaVersion",
            "tensorRtVersion",
            "runtimePackageId",
            "runtimePackageVersion",
            "runtimePackageSource",
            "ownerReviewedBy",
            "ownerReviewedAtUtc",
        })
        {
            Assert.Contains(field, requiredOwnerFields);
        }

        JsonElement validation = ReadJsonRoot(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "real-case-evidence-record-validation.json"));
        Assert.Equal("real-case-evidence-record-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", validation.GetProperty("validationState").GetString());
        Assert.Equal("template-or-incomplete-not-proof", validation.GetProperty("proofClassification").GetString());
        Assert.True(validation.GetProperty("missingOwnerInputCount").GetInt32() >= requiredOwnerFields.Length);
        Assert.True(validation.GetProperty("invalidSha256FieldCount").GetInt32() >= 4);
        Assert.False(validation.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("never publishes", validation.GetProperty("boundary").GetString()!, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseMatricesAggregateRealCaseProofPackWithoutUnlockingRelease()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealCaseProofExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealCaseEvidenceRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealCaseEvidenceRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRuntimeProofExecutionMatrix.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseFreezeFinalVerification.ps1"));

        JsonElement matrix = ReadJsonRoot(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-runtime-proof-execution-matrix.json"));
        Assert.Equal("blocked-owner-action-required", matrix.GetProperty("realCaseProofPackState").GetString());
        Assert.True(matrix.GetProperty("realCaseProofCaseCount").GetInt32() >= 9);
        Assert.Equal(matrix.GetProperty("realCaseProofCaseCount").GetInt32(), matrix.GetProperty("realCaseProofBlockedCaseCount").GetInt32());
        Assert.False(matrix.GetProperty("realCaseProofCanPromote").GetBoolean());
        Assert.False(matrix.GetProperty("canPublishPublicly").GetBoolean());
        Assert.Contains("artifacts/final-release/real-case-proof-execution-pack.json", matrix.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains(matrix.GetProperty("proofItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "real-case-proof-pack");

        JsonElement freeze = ReadJsonRoot(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-freeze-final-verification.json"));
        Assert.Equal("blocked-owner-action-required", freeze.GetProperty("realCaseProofPackState").GetString());
        Assert.True(freeze.GetProperty("realCaseProofCaseCount").GetInt32() >= 9);
        Assert.Equal(freeze.GetProperty("realCaseProofCaseCount").GetInt32(), freeze.GetProperty("realCaseProofBlockedCaseCount").GetInt32());
        Assert.False(freeze.GetProperty("realCaseProofCanPromote").GetBoolean());
        Assert.False(freeze.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(freeze.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("artifacts/final-release/real-case-evidence-record-template.json", freeze.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains(freeze.GetProperty("releaseProofFinalAuditItems").EnumerateArray(), static item => item.GetProperty("proofId").GetString() == "real-case-proof-pack");

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-freeze-final-verification.md"));
        Assert.Contains("real case proof pack state", markdown, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", markdown, StringComparison.Ordinal);
        Assert.Contains("real-case-proof-pack", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void SampleAndToolReadmesLinkRealCaseProofPackAndKeepProofBoundary()
    {
        string[] readmePaths =
        {
            Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"),
            Path.Combine(RepositoryPaths.Root, "samples", "OnnxToEngine", "README.md"),
            Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md"),
        };

        foreach (string readmePath in readmePaths)
        {
            string readme = File.ReadAllText(readmePath);
            Assert.Contains("real-case-proof-execution-pack", readme, StringComparison.Ordinal);
            Assert.Contains("Export-RealCaseProofExecutionPack.ps1", readme, StringComparison.Ordinal);
            Assert.Contains("Export-RealCaseEvidenceRecordTemplate.ps1", readme, StringComparison.Ordinal);
            Assert.Contains("Test-RealCaseEvidenceRecord.ps1", readme, StringComparison.Ordinal);
            Assert.Contains("canPublishPublicly=false", readme, StringComparison.Ordinal);
            Assert.Contains("owner", readme, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("SHA256", readme, StringComparison.Ordinal);
            Assert.DoesNotContain("YoloDet", readme, StringComparison.Ordinal);
        }
    }

    private static JsonElement ReadJsonRoot(string path)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.Clone();
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);

        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)!;
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"PowerShell command failed with exit code {process.ExitCode}:{Environment.NewLine}{output}{error}");
        return output + error;
    }
}
