using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CompatibleHostRuntimeProofRunbookTests
{
    [Fact]
    public void CompatibleHostRuntimeProofRunbookScriptDefinesNoPublishOwnerHandoff()
    {
        string script = ReadSource("eng", "Export-CompatibleHostRuntimeProofRunbook.ps1");
        string doc = ReadSource("docs", "articles", "zh-cn", "compatible-host-runtime-proof-runbook.md");
        string toc = ReadSource("docs", "toc.yml");

        Assert.Contains("compatible-host-runtime-proof-runbook.json", script, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-runbook.md", script, StringComparison.Ordinal);
        Assert.Contains("recordKind = \"compatible-host-runtime-proof-runbook\"", script, StringComparison.Ordinal);
        Assert.Contains("runbookState", script, StringComparison.Ordinal);
        Assert.Contains("owner-action-required", script, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumer.ps1", script, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", script, StringComparison.Ordinal);
        Assert.Contains("RequireExistingLog", script, StringComparison.Ordinal);
        Assert.Contains("FailOnNotProof", script, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1", script, StringComparison.Ordinal);
        Assert.Contains("Import-PackageConsumerRuntimeProofOwnerInput.ps1", script, StringComparison.Ordinal);
        Assert.Contains("blocked-owner-compatible-host-runtime-smoke", script, StringComparison.Ordinal);
        Assert.Contains("Smoke=not-requested", script, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver is not smoke passed", script, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof", script, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionEvidence", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.Contains("approvesPublicRelease = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);
        Assert.Contains("release-candidate-package-inventory.json", script, StringComparison.Ordinal);
        Assert.Contains("release-package-proof-bundle.json", script, StringComparison.Ordinal);
        Assert.Contains("final-package-review-bundle.json", script, StringComparison.Ordinal);
        Assert.Contains("local-nuget-feed-consumer-summary.json", script, StringComparison.Ordinal);
        Assert.Contains("ownerInputArtifacts", script, StringComparison.Ordinal);
        Assert.Contains("owner inputs only and cannot promote runtime proof", script, StringComparison.Ordinal);

        Assert.Contains("兼容主机 Runtime Proof Runbook", doc, StringComparison.Ordinal);
        Assert.Contains("Export-CompatibleHostRuntimeProofRunbook.ps1", doc, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumer.ps1", doc, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", doc, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1", doc, StringComparison.Ordinal);
        Assert.Contains("-RequireExistingLog", doc, StringComparison.Ordinal);
        Assert.Contains("-FailOnNotProof", doc, StringComparison.Ordinal);
        Assert.Contains("blocked-owner-compatible-host-runtime-smoke", doc, StringComparison.Ordinal);
        Assert.Contains("Smoke=not-requested", doc, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", doc, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", doc, StringComparison.Ordinal);
        Assert.Contains("runbook 不是 runtime proof", doc, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-runbook.md", toc, StringComparison.Ordinal);
    }

    [Fact]
    public void CompatibleHostRuntimeProofRunbookExportsCurrentBlockedStateWithoutPublishing()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordDraft.ps1"));
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"),
            "-InputPath",
            "artifacts/final-release/external-runtime-proof-record.draft.json",
            "-RequireExistingLog");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofOwnerHandoff.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostRuntimeProofRunbook.ps1"));
        Assert.Contains("Compatible host runtime proof runbook written", output, StringComparison.Ordinal);
        string validationOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-CompatibleHostRuntimeProofRunbook.ps1"), "-Strict");
        Assert.Contains("ValidationState=blocked-owner-compatible-host-runtime-smoke", validationOutput, StringComparison.Ordinal);

        string recordPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "compatible-host-runtime-proof-runbook.json");
        using JsonDocument record = JsonDocument.Parse(File.ReadAllText(recordPath));
        JsonElement root = record.RootElement;

        Assert.Equal("compatible-host-runtime-proof-runbook", root.GetProperty("recordKind").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("runbookState").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke", root.GetProperty("ownerRuntimeSmokeRunbookState").GetString());
        Assert.True(root.GetProperty("compatibleHostRequired").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver is not smoke passed", root.GetProperty("promotionBlockedReason").GetString());
        Assert.NotEqual("runtime-proof-ready", root.GetProperty("externalRuntimeProofState").GetString());
        Assert.NotEqual("package-consumer-runtime", root.GetProperty("externalRuntimeProofClassification").GetString());
        Assert.True(root.GetProperty("externalRuntimeProofFailedProofItemCount").GetInt32() > 0);
        Assert.StartsWith("draft-", root.GetProperty("externalRuntimeProofDraftState").GetString(), StringComparison.Ordinal);
        Assert.False(root.GetProperty("externalRuntimeProofDraftCanPromoteRuntimeProof").GetBoolean());
        Assert.True(root.GetProperty("draftManagedNupkgSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("draftRuntimeNupkgSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("draftSmokeLogSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("draftNoProjectReference").GetBoolean());
        Assert.Contains(root.GetProperty("draftSmokeStatus").GetString(), new[] { "not-requested", "blocked-by-cuda-driver" });
        Assert.Equal("release-candidate-package-inventory", root.GetProperty("releaseCandidatePackageInventoryState").GetString());
        Assert.Equal(8, root.GetProperty("releaseCandidatePackageInventoryPackageCount").GetInt32());
        Assert.True(root.GetProperty("releaseCandidatePackageInventorySha256Ready").GetBoolean());
        Assert.False(root.GetProperty("releaseCandidatePackageInventoryCanUseAsPublicPackageProof").GetBoolean());
        Assert.False(root.GetProperty("releaseCandidatePackageInventoryCanCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("releasePackageProofCanPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("releasePackageProofCanUseAsPublicPackageProof").GetBoolean());
        Assert.False(root.GetProperty("releasePackageProofCanCloseReleaseIssue").GetBoolean());
        Assert.Equal(8, root.GetProperty("finalPackageReviewPackageCount").GetInt32());
        Assert.False(root.GetProperty("finalPackageReviewCanUseAsPublicPackageProof").GetBoolean());
        Assert.False(root.GetProperty("finalPackageReviewCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("dependency-probe-passed", root.GetProperty("localFeedConsumerRunStatus").GetString());
        Assert.Equal("local-feed-only", root.GetProperty("localFeedConsumerRestoreSourceMode").GetString());
        Assert.False(root.GetProperty("localFeedConsumerUsesProjectReference").GetBoolean());

        JsonElement commands = root.GetProperty("commands");
        Assert.Contains("Test-PackageConsumer.ps1", commands.GetProperty("runPackageConsumerSmoke").GetString(), StringComparison.Ordinal);
        Assert.Contains("Get-FileHash", commands.GetProperty("computeSmokeLogSha256").GetString(), StringComparison.Ordinal);
        Assert.Contains("Get-FileHash", commands.GetProperty("computeManagedNupkgSha256").GetString(), StringComparison.Ordinal);
        Assert.Contains("Get-FileHash", commands.GetProperty("computeRuntimeNupkgSha256").GetString(), StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", commands.GetProperty("validateFilledRecord").GetString(), StringComparison.Ordinal);
        Assert.Contains("RequireExistingLog", commands.GetProperty("validateFilledRecord").GetString(), StringComparison.Ordinal);
        Assert.Contains("FailOnNotProof", commands.GetProperty("validateFilledRecord").GetString(), StringComparison.Ordinal);
        Assert.Contains("Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1", commands.GetProperty("exportOwnerInputTemplate").GetString(), StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-proof-owner-input.template.json", commands.GetProperty("copyOwnerInputTemplateToRealInput").GetString(), StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1", commands.GetProperty("validateOwnerInputStrict").GetString(), StringComparison.Ordinal);
        Assert.Contains("-Strict", commands.GetProperty("validateOwnerInputStrict").GetString(), StringComparison.Ordinal);
        Assert.Contains("Import-PackageConsumerRuntimeProofOwnerInput.ps1", commands.GetProperty("importOwnerInputStrict").GetString(), StringComparison.Ordinal);

        Assert.Contains(root.GetProperty("steps").EnumerateArray(), static step =>
            step.GetProperty("id").GetString() == "run-package-consumer-smoke" &&
            step.GetProperty("command").GetString()!.Contains("Test-PackageConsumer.ps1", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("steps").EnumerateArray(), static step =>
            step.GetProperty("id").GetString() == "validate-owner-input-strict" &&
            step.GetProperty("command").GetString()!.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("requiredRecordFields").EnumerateArray(), static field =>
            field.GetProperty("path").GetString() == "proofClassification" &&
            field.GetProperty("expectedValue").GetString() == "package-consumer-runtime");
        string[] requiredOwnerInputFields = root.GetProperty("requiredOwnerInputFields").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("consumerProjectPath", requiredOwnerInputFields);
        Assert.Contains("publicPackageSource", requiredOwnerInputFields);
        Assert.Contains("managedNupkgSha256", requiredOwnerInputFields);
        Assert.Contains("runtimeNupkgSha256", requiredOwnerInputFields);
        Assert.Contains("smokeLogSha256", requiredOwnerInputFields);
        Assert.Contains("stdoutSummary", requiredOwnerInputFields);
        Assert.Contains("stderrSummary", requiredOwnerInputFields);
        Assert.Contains("gpuName", requiredOwnerInputFields);
        string[] forbiddenRuntimeSmokeSubstitutes = root.GetProperty("forbiddenRuntimeSmokeSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("local feed", forbiddenRuntimeSmokeSubstitutes);
        Assert.Contains("ProjectReference", forbiddenRuntimeSmokeSubstitutes);
        Assert.Contains("direct .nupkg", forbiddenRuntimeSmokeSubstitutes);
        Assert.Contains("Smoke=not-requested", forbiddenRuntimeSmokeSubstitutes);
        Assert.Contains("dependency-probe-only", forbiddenRuntimeSmokeSubstitutes);
        Assert.Contains("blocked-by-cuda-driver", forbiddenRuntimeSmokeSubstitutes);
        Assert.Contains(root.GetProperty("nonProofBoundaries").EnumerateArray(), static boundary =>
            boundary.GetString()!.Contains("does not publish packages", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("nonProofBoundaries").EnumerateArray(), static boundary =>
            boundary.GetString()!.Contains("blocked-by-cuda-driver is not smoke passed", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("nonProofBoundaries").EnumerateArray(), static boundary =>
            boundary.GetString()!.Contains("owner inputs only and cannot promote runtime proof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("ownerInputArtifacts").EnumerateArray(), static artifact =>
            artifact.GetProperty("id").GetString() == "release-candidate-package-inventory" &&
            artifact.GetProperty("boundary").GetString()!.Contains("not public channel proof, runtime proof, or post-publish proof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("ownerInputArtifacts").EnumerateArray(), static artifact =>
            artifact.GetProperty("id").GetString() == "local-feed-consumer-summary" &&
            artifact.GetProperty("boundary").GetString()!.Contains("not post-publish proof or runtime execution proof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-candidate-package-inventory.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/final-package-review-bundle.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json");

        using JsonDocument validation = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "compatible-host-runtime-proof-runbook-validation.json")));
        JsonElement validationRoot = validation.RootElement;
        Assert.Equal("compatible-host-runtime-proof-runbook-validation", validationRoot.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke", validationRoot.GetProperty("validationState").GetString());
        Assert.Equal(0, validationRoot.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validationRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validationRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validationRoot.GetProperty("canCloseReleaseIssue").GetBoolean());

        string markdownPath = Path.ChangeExtension(recordPath, ".md");
        string markdown = File.ReadAllText(markdownPath);
        Assert.Contains("Compatible Host Runtime Proof Runbook", markdown, StringComparison.Ordinal);
        Assert.Contains("blocked-owner-compatible-host-runtime-smoke", markdown, StringComparison.Ordinal);
        Assert.Contains("performs publish: `false`", markdown, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumer.ps1", markdown, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", markdown, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1", markdown, StringComparison.Ordinal);
        Assert.Contains("Smoke=not-requested", markdown, StringComparison.Ordinal);
        Assert.Contains("## Owner Input Artifacts", markdown, StringComparison.Ordinal);
        Assert.Contains("release-candidate-package-inventory", markdown, StringComparison.Ordinal);
        Assert.Contains("local-feed-consumer-summary", markdown, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
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
