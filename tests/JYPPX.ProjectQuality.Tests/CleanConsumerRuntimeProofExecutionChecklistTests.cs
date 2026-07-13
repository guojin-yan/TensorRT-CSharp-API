using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class CleanConsumerRuntimeProofExecutionChecklistTests
{
    [Fact]
    public void CleanConsumerRuntimeProofExecutionChecklistCapturesOwnerExecutionWithoutPublishing()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanConsumerRuntimeProofExecutionChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-CleanConsumerRuntimeProofExecutionChecklist.ps1"), "-Strict");

        using JsonDocument checklistDocument = ReadFinalReleaseJson("clean-consumer-runtime-proof-execution-checklist.json");
        JsonElement checklist = checklistDocument.RootElement;

        Assert.Equal("clean-consumer-runtime-proof-execution-checklist", checklist.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-clean-consumer-runtime-proof-required", checklist.GetProperty("checklistState").GetString());
        Assert.True(checklist.GetProperty("executionStepCount").GetInt32() >= 14);
        Assert.False(checklist.GetProperty("performsPublish").GetBoolean());
        Assert.False(checklist.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(checklist.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(checklist.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(checklist.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(checklist.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());

        string[] sourceArtifacts = checklist.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-evidence-bundle.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-proof-owner-handoff-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-release-pre-publish-audit-matrix.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/tensor-rt-exec-gui-cli-parity-checklist.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-record-validation.json", sourceArtifacts);

        JsonElement[] steps = checklist.GetProperty("executionSteps").EnumerateArray().ToArray();
        foreach (string stepId in new[]
        {
            "create-repository-external-clean-consumer",
            "configure-public-package-source",
            "install-managed-and-runtime-packages",
            "restore-clean-consumer",
            "build-clean-consumer",
            "run-smoke-runtime-command",
            "persist-stdout-stderr-logs",
            "compute-log-sha256",
            "compute-nupkg-sha256",
            "fill-owner-input",
            "import-owner-input",
            "strict-validate-proof-record",
            "run-forbidden-substitute-scan",
            "refresh-release-evidence-and-dashboard"
        })
        {
            Assert.Contains(steps, item => item.GetProperty("id").GetString() == stepId);
        }

        Assert.All(steps, item =>
        {
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.DoesNotContain("dotnet nuget push", item.GetProperty("commandTemplate").GetString(), StringComparison.OrdinalIgnoreCase);
        });

        AssertIds(checklist, "requiredInputs", new[]
        {
            "public-package-source-url",
            "managed-package-id-version",
            "runtime-package-id-version",
            "repository-external-consumer-path",
            "restore-build-smoke-commands",
            "smoke-exit-code-zero",
            "startedAtUtc",
            "finishedAtUtc",
            "owner-review-identity"
        });
        AssertIds(checklist, "requiredHashes", new[]
        {
            "clean-consumer-project-hash",
            "managed-nupkg-sha256",
            "runtime-nupkg-sha256",
            "restore-log-sha256",
            "build-log-sha256",
            "stdout-log-sha256",
            "stderr-log-sha256"
        });
        AssertIds(checklist, "requiredLogs", new[]
        {
            "restore-log",
            "build-log",
            "smoke-stdout-log",
            "smoke-stderr-log",
            "owner-import-log",
            "strict-validator-log"
        });
        AssertIds(checklist, "requiredHostMetadata", new[]
        {
            "host-os-architecture",
            "gpu-name",
            "nvidia-driver-version",
            "cuda-version",
            "tensorrt-version",
            "cudnn-version"
        });
        AssertIds(checklist, "requiredPackageMetadata", new[]
        {
            "managed-package-id",
            "managed-package-version",
            "runtime-package-id",
            "runtime-package-version",
            "runtime-package-key",
            "public-package-source"
        });

        string[] validators = checklist.GetProperty("requiredValidators").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains(validators, item => item.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", StringComparison.Ordinal) && item.Contains("-RequireExistingLog", StringComparison.Ordinal) && item.Contains("-FailOnNotProof", StringComparison.Ordinal));
        Assert.Contains(validators, item => item.Contains("Import-PackageConsumerRuntimeProofOwnerInput.ps1", StringComparison.Ordinal));
        Assert.Contains(validators, item => item.Contains("Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1", StringComparison.Ordinal));
        Assert.Contains(validators, item => item.Contains("Export-ReleaseEvidenceBundle.ps1", StringComparison.Ordinal));

        string[] forbidden = checklist.GetProperty("forbiddenSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string forbiddenItem in new[]
        {
            "template",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "dry-run",
            "build-only",
            "preflight-only",
            "GUI screenshot",
            "TensorRtExec report",
            "YoloVision matrix",
            "OnnxToEngine report",
            "sample manifest",
            "sidecar-only",
            "readonly diagnostics",
            "dependency probe",
            "owner input without strict validator pass"
        })
        {
            Assert.Contains(forbiddenItem, forbidden);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("clean-consumer-runtime-proof-execution-checklist-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("clean-consumer-runtime-proof-execution-checklist-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("clean-consumer-runtime-proof-execution-checklist-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
    }

    private static void AssertIds(JsonElement root, string propertyName, string[] expectedIds)
    {
        string[] ids = root.GetProperty(propertyName).EnumerateArray().Select(static item => item.GetProperty("id").GetString()!).ToArray();
        foreach (string expectedId in expectedIds)
        {
            Assert.Contains(expectedId, ids);
        }
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
