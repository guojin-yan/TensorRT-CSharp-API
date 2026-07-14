using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealPublishEvidenceAvailabilityLedgerTests
{
    [Fact]
    public void LedgerAggregatesOwnerPublishProofAvailabilityWithoutPromotingRelease()
    {
        RunPowerShell("Export-PublicPublishResultOwnerInputTemplate.ps1");
        RunPowerShell("Test-PublicPublishResultOwnerInput.ps1", "-Strict");
        RunPowerShell("Import-PublicPublishResultOwnerInput.ps1");
        RunPowerShell("Test-PublicPublishResultImport.ps1", "-Strict");
        RunPowerShell("Export-PublicPackageDownloadProofInputTemplate.ps1");
        RunPowerShell("Test-PublicPackageDownloadProofInput.ps1", "-Strict");
        RunPowerShell("Import-PublicPackageDownloadProofCandidate.ps1");
        RunPowerShell("Test-PublicPackageDownloadProofCandidate.ps1", "-Strict");
        RunPowerShell("Import-PostPublishCleanConsumerRealProofFromOwnerResult.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerRealProofFromOwnerResult.ps1", "-Strict");
        RunPowerShell("Export-ReleaseIssueCloseOwnerDecisionInput.ps1");
        RunPowerShell("Test-ReleaseIssueCloseOwnerDecisionInput.ps1", "-Strict");
        RunPowerShell("Export-StrictCloseReadyConvergenceDashboard.ps1");
        RunPowerShell("Test-StrictCloseReadyConvergenceDashboard.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealPublishEvidenceAvailabilityLedger.ps1");
        RunPowerShell("Test-OwnerRealPublishEvidenceAvailabilityLedger.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument ledgerDocument = ReadFinalReleaseJson("owner-real-publish-evidence-availability-ledger.json");
        JsonElement ledger = ledgerDocument.RootElement;

        Assert.Equal("owner-real-publish-evidence-availability-ledger", ledger.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-publish-evidence-required", ledger.GetProperty("ledgerState").GetString());
        Assert.True(ledger.GetProperty("slotCount").GetInt32() >= 9);
        Assert.Equal(0, ledger.GetProperty("proofReadySlotCount").GetInt32());
        Assert.Equal(ledger.GetProperty("slotCount").GetInt32(), ledger.GetProperty("blockedSlotCount").GetInt32());
        Assert.True(ledger.GetProperty("blockedValidationItemCount").GetInt32() >= 20);
        Assert.False(ledger.GetProperty("allRequiredOwnerProofAvailable").GetBoolean());
        Assert.True(ledger.GetProperty("ownerActionRequired").GetBoolean());
        AssertFalseProofPublishCloseFlags(ledger);

        string[] slotIds = ledger.GetProperty("slots").EnumerateArray()
            .Select(static slot => slot.GetProperty("id").GetString()!)
            .ToArray();

        foreach (string required in new[]
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
        })
        {
            Assert.Contains(required, slotIds);
        }

        foreach (JsonElement slot in ledger.GetProperty("slots").EnumerateArray())
        {
            Assert.Equal("blocked-owner-real-evidence-required", slot.GetProperty("slotState").GetString());
            Assert.True(slot.GetProperty("blocked").GetBoolean());
            Assert.False(slot.GetProperty("proofReady").GetBoolean());
            Assert.False(slot.GetProperty("performsPublish").GetBoolean());
            Assert.False(slot.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(slot.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.Contains("not package push", slot.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-real-publish-evidence-availability-ledger-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-real-publish-evidence-availability-ledger-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("owner-real-publish-evidence-availability-ledger-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("isValidOwnerRealPublishEvidenceAvailabilityLedger").GetBoolean());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("passed").GetBoolean());
        AssertFalseProofPublishCloseFlags(validation);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-owner-real-publish-evidence-required", evidence.GetProperty("ownerRealPublishEvidenceAvailabilityLedgerState").GetString());
        Assert.Equal("owner-real-publish-evidence-availability-ledger-ready-non-proof", evidence.GetProperty("ownerRealPublishEvidenceAvailabilityLedgerValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("ownerRealPublishEvidenceAvailabilityLedgerProofReadySlotCount").GetInt32());
        Assert.False(evidence.GetProperty("ownerRealPublishEvidenceAvailabilityLedgerCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("ownerRealPublishEvidenceAvailabilityLedgerCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-real-publish-evidence-availability-ledger");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("not release close approval", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

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
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
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
