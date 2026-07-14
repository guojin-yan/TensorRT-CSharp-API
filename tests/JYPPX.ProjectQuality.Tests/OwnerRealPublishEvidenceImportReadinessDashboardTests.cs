using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealPublishEvidenceImportReadinessDashboardTests
{
    [Fact]
    public void OwnerImportReadinessAndFinalCloseCandidateAuditStayBlockedNonProofAndReachEvidenceBundle()
    {
        RunPowerShell("Export-OwnerRealPublishEvidenceAvailabilityLedger.ps1");
        RunPowerShell("Test-OwnerRealPublishEvidenceAvailabilityLedger.ps1", "-Strict");
        RunPowerShell("Export-PostPublishDocsAndSamplesFinalLandingPack.ps1");
        RunPowerShell("Test-PostPublishDocsAndSamplesFinalLandingPack.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealPublishEvidenceImportReadinessDashboard.ps1");
        RunPowerShell("Test-OwnerRealPublishEvidenceImportReadinessDashboard.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseFinalCandidateAuditPack.ps1");
        RunPowerShell("Test-ReleaseCloseFinalCandidateAuditPack.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument dashboardDocument = ReadFinalReleaseJson("owner-real-publish-evidence-import-readiness-dashboard.json");
        JsonElement dashboard = dashboardDocument.RootElement;
        Assert.Equal("owner-real-publish-evidence-import-readiness-dashboard", dashboard.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-publish-evidence-import-required", dashboard.GetProperty("dashboardState").GetString());
        Assert.True(dashboard.GetProperty("slotCount").GetInt32() >= 9);
        Assert.Equal(dashboard.GetProperty("slotCount").GetInt32(), dashboard.GetProperty("blockedSlotCount").GetInt32());
        Assert.Equal(0, dashboard.GetProperty("proofReadySlotCount").GetInt32());
        Assert.True(dashboard.GetProperty("expectedEvidenceFieldCount").GetInt32() >= 70);
        Assert.True(dashboard.GetProperty("validatorScriptCount").GetInt32() >= 15);
        AssertFalseProofPublishCloseFlags(dashboard);

        string[] slotIds = dashboard.GetProperty("slots").EnumerateArray()
            .Select(static slot => slot.GetProperty("id").GetString()!)
            .ToArray();
        foreach (string required in RequiredSlotIds)
        {
            Assert.Contains(required, slotIds);
        }

        foreach (JsonElement slot in dashboard.GetProperty("slots").EnumerateArray())
        {
            Assert.Equal("blocked-owner-real-evidence-import-required", slot.GetProperty("readinessState").GetString());
            Assert.True(slot.GetProperty("blocked").GetBoolean());
            Assert.False(slot.GetProperty("proofReady").GetBoolean());
            Assert.True(slot.GetProperty("expectedEvidenceFieldCount").GetInt32() >= 7);
            Assert.True(slot.GetProperty("validatorScriptCount").GetInt32() >= 1);
            Assert.Contains("not package push", slot.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument dashboardValidationDocument = ReadFinalReleaseJson("owner-real-publish-evidence-import-readiness-dashboard-validation.json");
        JsonElement dashboardValidation = dashboardValidationDocument.RootElement;
        Assert.Equal("owner-real-publish-evidence-import-readiness-dashboard-validation", dashboardValidation.GetProperty("recordKind").GetString());
        Assert.Equal("owner-real-publish-evidence-import-readiness-dashboard-ready-non-proof", dashboardValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, dashboardValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(dashboardValidation);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-close-final-candidate-audit-pack.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("release-close-final-candidate-audit-pack", audit.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-final-candidate-real-owner-evidence-required", audit.GetProperty("auditState").GetString());
        Assert.True(audit.GetProperty("checkCount").GetInt32() >= 9);
        Assert.Equal(audit.GetProperty("checkCount").GetInt32(), audit.GetProperty("blockedCheckCount").GetInt32());
        Assert.Equal(0, audit.GetProperty("passedCheckCount").GetInt32());
        Assert.Matches("^[0-9a-f]{64}$", audit.GetProperty("releaseEvidenceBundleSha256").GetString()!);
        Assert.Matches("^[0-9a-f]{64}$", audit.GetProperty("classificationAuditSha256").GetString()!);
        AssertFalseProofPublishCloseFlags(audit);

        string[] checkIds = audit.GetProperty("checks").EnumerateArray()
            .Select(static check => check.GetProperty("id").GetString()!)
            .ToArray();
        foreach (string required in RequiredFinalCloseCheckIds)
        {
            Assert.Contains(required, checkIds);
        }

        using JsonDocument auditValidationDocument = ReadFinalReleaseJson("release-close-final-candidate-audit-pack-validation.json");
        JsonElement auditValidation = auditValidationDocument.RootElement;
        Assert.Equal("release-close-final-candidate-audit-pack-validation", auditValidation.GetProperty("recordKind").GetString());
        Assert.Equal("release-close-final-candidate-audit-pack-ready-non-proof", auditValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, auditValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(auditValidation);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("owner-real-publish-evidence-import-readiness-dashboard-ready-non-proof", evidence.GetProperty("ownerRealPublishEvidenceImportReadinessDashboardValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("ownerRealPublishEvidenceImportReadinessDashboardProofReadySlotCount").GetInt32());
        Assert.False(evidence.GetProperty("ownerRealPublishEvidenceImportReadinessDashboardCanPublishPublicly").GetBoolean());
        Assert.Equal("release-close-final-candidate-audit-pack-ready-non-proof", evidence.GetProperty("releaseCloseFinalCandidateAuditPackValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("releaseCloseFinalCandidateAuditPackPassedCheckCount").GetInt32());
        Assert.False(evidence.GetProperty("releaseCloseFinalCandidateAuditPackCanCloseReleaseIssue").GetBoolean());

        JsonElement dashboardEvidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-real-publish-evidence-import-readiness-dashboard");
        Assert.False(dashboardEvidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("not publish approval", dashboardEvidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement auditEvidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-close-final-candidate-audit-pack");
        Assert.False(auditEvidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("not release close approval", auditEvidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    private static readonly string[] RequiredSlotIds =
    {
        "owner-public-publish-result",
        "github-actions-run-proof",
        "public-package-download-proof",
        "repository-external-clean-consumer-proof",
        "post-publish-clean-consumer-proof",
        "post-publish-user-verification",
        "release-issue-close-owner-decision",
        "release-issue-close-record",
        "strict-close-final-convergence",
    };

    private static readonly string[] RequiredFinalCloseCheckIds =
    {
        "release-evidence-bundle-hash-lock",
        "owner-import-readiness-no-proof-ready",
        "final-landing-pack-non-proof",
        "public-package-url-hash-proof",
        "external-clean-consumer-post-publish-proof",
        "strict-close-dashboard-and-bridge",
        "release-issue-close-owner-decision",
        "rollback-and-known-limitations",
        "forbidden-substitute-final-scan",
    };

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertFalseProofPublishCloseFlags(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
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
        process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}
