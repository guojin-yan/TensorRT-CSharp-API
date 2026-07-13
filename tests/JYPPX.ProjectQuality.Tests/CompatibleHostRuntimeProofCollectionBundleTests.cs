using System;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CompatibleHostRuntimeProofCollectionBundleTests
{
    [Fact]
    public void ScriptDocsAndReleaseExportersCarryCollectionBundleBoundaries()
    {
        string script = ReadSource("eng", "Export-CompatibleHostRuntimeProofCollectionBundle.ps1");
        string doc = ReadSource("docs", "articles", "zh-cn", "compatible-host-runtime-proof-collection-bundle.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");
        string releaseEvidenceDoc = ReadSource("docs", "articles", "zh-cn", "release-evidence-bundle.md");

        Assert.Contains("compatible-host-runtime-proof-collection-bundle.json", script, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-collection-bundle.md", script, StringComparison.Ordinal);
        Assert.Contains("recordKind = \"compatible-host-runtime-proof-collection-bundle\"", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.Contains("approvesPublicRelease = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof = $false", script, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionEvidence = $false", script, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1", script, StringComparison.Ordinal);
        Assert.Contains("Import-PackageConsumerRuntimeProofOwnerInput.ps1", script, StringComparison.Ordinal);
        Assert.Contains("Smoke=not-requested", script, StringComparison.Ordinal);
        Assert.Contains("blocked-owner-compatible-host-runtime-smoke", script, StringComparison.Ordinal);
        Assert.Contains("release-candidate-package-inventory.json", script, StringComparison.Ordinal);
        Assert.Contains("release-package-proof-bundle.json", script, StringComparison.Ordinal);
        Assert.Contains("final-package-review-bundle.json", script, StringComparison.Ordinal);
        Assert.Contains("local-nuget-feed-consumer-summary.json", script, StringComparison.Ordinal);
        Assert.Contains("ownerInputArtifacts", script, StringComparison.Ordinal);

        Assert.Contains("compatible-host-runtime-proof-collection-bundle.md", doc, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-collection-bundle.md", index, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-collection-bundle.md", toc, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-collection-bundle.json", releaseEvidenceDoc, StringComparison.Ordinal);

        foreach (string exporter in new[]
        {
            "Export-ReleaseEvidenceBundle.ps1",
            "Export-ReleaseOwnerApprovalInputTemplate.ps1",
            "Export-ReleaseOwnerDecisionRecord.ps1",
            "Export-ReleasePublishExecutionChecklist.ps1",
            "Export-ReleasePromotionIssueRecord.ps1",
        })
        {
            string exporterSource = ReadSource("eng", exporter);
            Assert.Contains("compatible-host-runtime-proof-collection-bundle.json", exporterSource, StringComparison.Ordinal);
            Assert.Contains("compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof", exporterSource, StringComparison.Ordinal);
            Assert.Contains("compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence", exporterSource, StringComparison.Ordinal);
            Assert.Contains("compatibleHostRuntimeProofCollectionBundleOwnerInputArtifactCount", exporterSource, StringComparison.Ordinal);
            Assert.Contains("compatibleHostRuntimeProofCollectionBundleOwnerInputArtifacts", exporterSource, StringComparison.Ordinal);
            Assert.Contains("ownerInputArtifacts", exporterSource, StringComparison.Ordinal);
            Assert.Contains("not runtime proof", exporterSource, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ExporterEmitsNonProofNonPublishingCollectionBundle()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostRuntimeProofRunbook.ps1"));
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostRuntimeProofCollectionBundle.ps1"));
        Assert.Contains("Compatible host runtime proof collection bundle written", output, StringComparison.Ordinal);

        string recordPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "compatible-host-runtime-proof-collection-bundle.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "compatible-host-runtime-proof-collection-bundle.md");
        using JsonDocument record = JsonDocument.Parse(File.ReadAllText(recordPath));
        JsonElement root = record.RootElement;

        Assert.True(File.Exists(markdownPath));
        Assert.Equal("compatible-host-runtime-proof-collection-bundle", root.GetProperty("recordKind").GetString());
        Assert.True(root.GetProperty("compatibleHostRequired").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke", root.GetProperty("ownerRuntimeSmokeRunbookState").GetString());
        Assert.Equal("blocked-by-cuda-driver is not smoke passed", root.GetProperty("promotionBlockedReason").GetString());
        Assert.Contains("Test-PackageConsumer.ps1", root.GetProperty("commands").GetProperty("runPackageConsumerSmoke").GetString(), StringComparison.Ordinal);
        Assert.Contains("-FailOnNotProof", root.GetProperty("commands").GetProperty("validateFilledRecord").GetString(), StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1", root.GetProperty("commands").GetProperty("validateOwnerInputStrict").GetString(), StringComparison.Ordinal);
        Assert.Contains("Import-PackageConsumerRuntimeProofOwnerInput.ps1", root.GetProperty("commands").GetProperty("importOwnerInputStrict").GetString(), StringComparison.Ordinal);
        Assert.Contains(root.GetProperty("copyableExecutionOrder").EnumerateArray(), static item =>
            item.GetString()!.Contains("Test-PackageConsumer.ps1", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("copyableExecutionOrder").EnumerateArray(), static item =>
            item.GetString()!.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("operatorQuickStart").EnumerateArray(), static item => item.GetString()!.Contains("-FailOnNotProof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("operatorQuickStart").EnumerateArray(), static item => item.GetString()!.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("preflightChecklist").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "compatible-gpu-host" &&
            item.GetProperty("why").GetString()!.Contains("blocked-by-cuda-driver is not smoke passed", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("preflightChecklist").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "proof-validator" &&
            item.GetProperty("required").GetString()!.Contains("-RequireExistingLog -FailOnNotProof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("preflightChecklist").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "owner-input-strict-validator" &&
            item.GetProperty("required").GetString()!.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("not runtime proof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("Smoke=not-requested", StringComparison.Ordinal));
        Assert.Equal("release-candidate-package-inventory", root.GetProperty("releaseCandidatePackageInventoryState").GetString());
        Assert.Equal(8, root.GetProperty("releaseCandidatePackageInventoryPackageCount").GetInt32());
        Assert.True(root.GetProperty("releaseCandidatePackageInventorySha256Ready").GetBoolean());
        Assert.False(root.GetProperty("releaseCandidatePackageInventoryCanUseAsPublicPackageProof").GetBoolean());
        Assert.False(root.GetProperty("releasePackageProofCanPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("releasePackageProofCanUseAsPublicPackageProof").GetBoolean());
        Assert.Equal(8, root.GetProperty("finalPackageReviewPackageCount").GetInt32());
        Assert.False(root.GetProperty("finalPackageReviewCanUseAsPublicPackageProof").GetBoolean());
        Assert.Equal("dependency-probe-passed", root.GetProperty("localFeedConsumerRunStatus").GetString());
        Assert.Equal("local-feed-only", root.GetProperty("localFeedConsumerRestoreSourceMode").GetString());
        Assert.False(root.GetProperty("localFeedConsumerUsesProjectReference").GetBoolean());
        Assert.Contains(root.GetProperty("ownerInputArtifacts").EnumerateArray(), static artifact =>
            artifact.GetProperty("id").GetString() == "release-candidate-package-inventory" &&
            artifact.GetProperty("boundary").GetString()!.Contains("not public channel proof, runtime proof, or post-publish proof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("ownerInputArtifacts").EnumerateArray(), static artifact =>
            artifact.GetProperty("id").GetString() == "final-package-review-bundle" &&
            artifact.GetProperty("boundary").GetString()!.Contains("not owner authorization", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-candidate-package-inventory.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/final-package-review-bundle.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json");

        string markdown = File.ReadAllText(markdownPath);
        Assert.Contains("## Operator Quick Start", markdown, StringComparison.Ordinal);
        Assert.Contains("## Preflight Checklist", markdown, StringComparison.Ordinal);
        Assert.Contains("## Owner Input Artifacts", markdown, StringComparison.Ordinal);
        Assert.Contains("release-candidate-package-inventory", markdown, StringComparison.Ordinal);
        Assert.Contains("local-feed-consumer-summary", markdown, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1", markdown, StringComparison.Ordinal);
        Assert.Contains("blocked-owner-compatible-host-runtime-smoke", markdown, StringComparison.Ordinal);
        Assert.Contains("Stop on the first failure", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseFacingArtifactsKeepCollectionBundleAsNonProofInput()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostRuntimeProofRunbook.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostRuntimeProofCollectionBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerApprovalInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseOwnerApprovalInput.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerDecisionRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePublishExecutionChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePromotionIssueRecord.ps1"));

        using JsonDocument evidence = ReadArtifact("release-evidence-bundle.json");
        AssertCollectionBundleFields(evidence.RootElement);
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke", evidence.RootElement.GetProperty("compatibleHostRuntimeProofRunbookOwnerRuntimeSmokeState").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke", evidence.RootElement.GetProperty("compatibleHostRuntimeProofRunbookValidationState").GetString());
        Assert.Equal(0, evidence.RootElement.GetProperty("compatibleHostRuntimeProofRunbookValidationFailedBlockerCount").GetInt32());
        Assert.False(evidence.RootElement.GetProperty("compatibleHostRuntimeProofRunbookValidationCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.RootElement.GetProperty("compatibleHostRuntimeProofRunbookValidationCanPublishPublicly").GetBoolean());
        Assert.False(evidence.RootElement.GetProperty("compatibleHostRuntimeProofRunbookValidationCanCloseReleaseIssue").GetBoolean());
        Assert.Contains(evidence.RootElement.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "compatible-host-runtime-proof-collection-bundle" &&
            item.GetProperty("passed").GetBoolean() == false);
        Assert.Contains(evidence.RootElement.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "compatible-host-runtime-proof-runbook-validation" &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("state").GetString()!.Contains("blocked-owner-compatible-host-runtime-smoke", StringComparison.Ordinal));
        Assert.Contains(evidence.RootElement.GetProperty("sourceArtifacts").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/compatible-host-runtime-proof-runbook-validation.json");

        using JsonDocument approval = ReadArtifact("release-owner-approval-input-template.json");
        AssertCollectionBundleFields(approval.RootElement);
        AssertCollectionBundleExecutionGuidance(approval.RootElement);
        AssertCollectionBundleMarkdownExecutionGuidance("release-owner-approval-input-template.md");
        Assert.Contains(approval.RootElement.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json");
        Assert.Contains(approval.RootElement.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("collection-bundle is an external execution package", StringComparison.Ordinal));

        using JsonDocument decision = ReadArtifact("release-owner-decision-record.json");
        AssertCollectionBundleFields(decision.RootElement);
        AssertCollectionBundleExecutionGuidance(decision.RootElement);
        AssertCollectionBundleMarkdownExecutionGuidance("release-owner-decision-record.md");
        Assert.Contains(decision.RootElement.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetProperty("path").GetString() == "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json");
        Assert.Contains(decision.RootElement.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("collection-bundle is an external execution package", StringComparison.Ordinal));

        using JsonDocument checklist = ReadArtifact("release-publish-execution-checklist.json");
        AssertCollectionBundleFields(checklist.RootElement);
        AssertCollectionBundleExecutionGuidance(checklist.RootElement);
        AssertCollectionBundleMarkdownExecutionGuidance("release-publish-execution-checklist.md");
        Assert.Contains(checklist.RootElement.GetProperty("preflightItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "compatible-host-runtime-proof-collection-bundle");

        using JsonDocument promotion = ReadArtifact("release-promotion-issue-record.json");
        AssertCollectionBundleFields(promotion.RootElement);
        AssertCollectionBundleExecutionGuidance(promotion.RootElement);
        AssertCollectionBundleMarkdownExecutionGuidance("release-promotion-issue-record.md");
        Assert.Contains(promotion.RootElement.GetProperty("promotionItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "compatible-host-runtime-proof-collection-bundle");
    }

    private static void AssertCollectionBundleFields(JsonElement root)
    {
        Assert.Equal("owner-action-required", root.GetProperty("compatibleHostRuntimeProofCollectionBundleState").GetString());
        Assert.True(root.GetProperty("compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired").GetBoolean());
        Assert.False(root.GetProperty("compatibleHostRuntimeProofCollectionBundlePerformsPublish").GetBoolean());
        Assert.False(root.GetProperty("compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease").GetBoolean());
        Assert.False(root.GetProperty("compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver is not smoke passed", root.GetProperty("compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason").GetString());
        Assert.Contains("Test-PackageConsumer.ps1", root.GetProperty("compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("-FailOnNotProof", root.GetProperty("compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand").GetString(), StringComparison.Ordinal);
    }

    private static void AssertCollectionBundleExecutionGuidance(JsonElement root)
    {
        Assert.Equal(9, root.GetProperty("compatibleHostRuntimeProofCollectionBundleQuickStartCount").GetInt32());
        Assert.Equal(8, root.GetProperty("compatibleHostRuntimeProofCollectionBundlePreflightChecklistCount").GetInt32());
        Assert.Equal(14, root.GetProperty("compatibleHostRuntimeProofCollectionBundleCopyableExecutionOrderCount").GetInt32());
        Assert.Equal(4, root.GetProperty("compatibleHostRuntimeProofCollectionBundleOwnerInputArtifactCount").GetInt32());
        Assert.Contains(root.GetProperty("compatibleHostRuntimeProofCollectionBundleOperatorQuickStart").EnumerateArray(), static item =>
            item.GetString()!.Contains("-FailOnNotProof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("compatibleHostRuntimeProofCollectionBundleOperatorQuickStart").EnumerateArray(), static item =>
            item.GetString()!.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("compatibleHostRuntimeProofCollectionBundlePreflightChecklist").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "proof-validator" &&
            item.GetProperty("required").GetString()!.Contains("-RequireExistingLog -FailOnNotProof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("compatibleHostRuntimeProofCollectionBundlePreflightChecklist").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "owner-input-strict-validator" &&
            item.GetProperty("required").GetString()!.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("compatibleHostRuntimeProofCollectionBundleCopyableExecutionOrder").EnumerateArray(), static item =>
            item.GetString()!.Contains("Test-PackageConsumer.ps1", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("compatibleHostRuntimeProofCollectionBundleCopyableExecutionOrder").EnumerateArray(), static item =>
            item.GetString()!.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("compatibleHostRuntimeProofCollectionBundleOwnerInputArtifacts").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "release-candidate-package-inventory" &&
            item.GetProperty("boundary").GetString()!.Contains("not public channel proof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("compatibleHostRuntimeProofCollectionBundleOwnerInputArtifacts").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "local-feed-consumer-summary" &&
            item.GetProperty("boundary").GetString()!.Contains("not post-publish proof", StringComparison.Ordinal));
    }

    private static void AssertCollectionBundleMarkdownExecutionGuidance(string fileName)
    {
        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName));
        Assert.Contains("## Compatible Host Collection Bundle Execution", markdown, StringComparison.Ordinal);
        Assert.Contains("### Operator Quick Start", markdown, StringComparison.Ordinal);
        Assert.Contains("### Preflight Checklist", markdown, StringComparison.Ordinal);
        Assert.Contains("### Copyable Execution Order", markdown, StringComparison.Ordinal);
        Assert.Contains("### Owner Input Artifacts", markdown, StringComparison.Ordinal);
        Assert.Contains("release-candidate-package-inventory", markdown, StringComparison.Ordinal);
        Assert.Contains("local-feed-consumer-summary", markdown, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumer.ps1", markdown, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1", markdown, StringComparison.Ordinal);
        Assert.Contains("-RequireExistingLog -FailOnNotProof", markdown, StringComparison.Ordinal);
        Assert.Contains("not runtime proof, publication approval, or package push", markdown, StringComparison.Ordinal);
    }

    private static JsonDocument ReadArtifact(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static string ReadSource(params string[] segments)
    {
        return File.ReadAllText(Path.Combine(RepositoryPaths.Root, Path.Combine(segments)));
    }

    private static string RunPowerShell(string scriptPath)
    {
        using Process process = new()
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "pwsh",
                WorkingDirectory = RepositoryPaths.Root,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            },
        };

        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);

        process.Start();
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"PowerShell exited with code {process.ExitCode}.{Environment.NewLine}{output}{Environment.NewLine}{error}");
        }

        return output;
    }
}
