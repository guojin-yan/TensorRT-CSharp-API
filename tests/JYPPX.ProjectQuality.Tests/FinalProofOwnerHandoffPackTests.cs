using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalProofOwnerHandoffPackTests
{
    [Fact]
    public void FinalProofOwnerHandoffPackKeepsOwnerProofRequiredAndNonPublishable()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseDryRun.ps1"), "-AllowRuntimeSmokeBlocked");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalReleaseCloseBlockerDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalReleasePrePublishAuditMatrix.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleasePrePublishAuditMatrix.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalProofOwnerHandoffPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalProofOwnerHandoffPack.ps1"), "-Strict");

        using JsonDocument packDocument = ReadFinalReleaseJson("final-proof-owner-handoff-pack.json");
        JsonElement pack = packDocument.RootElement;

        Assert.Equal("final-proof-owner-handoff-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-required", pack.GetProperty("handoffState").GetString());
        Assert.True(pack.GetProperty("prePublishAuditMatrixReady").GetBoolean());
        Assert.True(pack.GetProperty("releaseEvidenceBundleReady").GetBoolean());
        Assert.True(pack.GetProperty("finalDryRunReady").GetBoolean());
        Assert.True(pack.GetProperty("closeBlockerDashboardReady").GetBoolean());
        Assert.True(pack.GetProperty("ownerInputSchemaReady").GetBoolean());
        Assert.False(pack.GetProperty("cleanOwnerInputReady").GetBoolean());
        Assert.False(pack.GetProperty("ownerInputCanPromoteRuntimeProof").GetBoolean());
        Assert.False(pack.GetProperty("packageConsumerRuntimeProofRecordReady").GetBoolean());
        Assert.False(pack.GetProperty("postPublishVerificationReady").GetBoolean());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment", pack.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentState").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment-valid", pack.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState").GetString());
        Assert.Equal("Smoke=not-requested", pack.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus").GetString());
        Assert.True(pack.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount").GetInt32() >= 30);
        Assert.Equal(0, pack.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount").GetInt32());
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(pack.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(pack.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(pack.GetProperty("isPostPublishProof").GetBoolean());
        Assert.True(pack.GetProperty("remainingOwnerActionCount").GetInt32() >= 1);

        string[] sourceArtifacts = pack.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/final-release-pre-publish-audit-matrix.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-release-pre-publish-audit-matrix-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-evidence-bundle.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-release-dry-run-summary.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-release-close-blocker-dashboard.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json", sourceArtifacts);

        string[] forbiddenSubstitutes = pack.GetProperty("forbiddenSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string forbidden in new[]
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
            "readonly diagnostics"
        })
        {
            Assert.Contains(forbidden, forbiddenSubstitutes);
        }

        JsonElement[] requiredInputs = pack.GetProperty("requiredOwnerInputs").EnumerateArray().ToArray();
        foreach (string inputId in new[]
        {
            "public-package-source-url",
            "package-id-version",
            "package-sha256",
            "clean-consumer-repo",
            "restore-build-smoke-command",
            "smoke-exit-code",
            "smoke-log-hash",
            "host-os",
            "gpu-driver-cuda-trt",
            "runtime-package-metadata",
            "owner-review",
            "post-publish-verification"
        })
        {
            Assert.Contains(requiredInputs, item => item.GetProperty("id").GetString() == inputId);
        }

        JsonElement[] commands = pack.GetProperty("requiredCommands").EnumerateArray().ToArray();
        Assert.Contains(commands, item => item.GetProperty("command").GetString()!.Contains("Import-PackageConsumerRuntimeProofOwnerInput.ps1", StringComparison.Ordinal));
        Assert.Contains(commands, item => item.GetProperty("command").GetString()!.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", StringComparison.Ordinal));
        Assert.DoesNotContain(commands, item => item.GetProperty("command").GetString()!.Contains("dotnet nuget push", StringComparison.OrdinalIgnoreCase));

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-proof-owner-handoff-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-proof-owner-handoff-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("final-proof-owner-handoff-pack-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
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
