using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerPublicPublishExecutionFinalIntakePackTests
{
    [Fact]
    public void FinalIntakePackAggregatesPublishCiDownloadCleanConsumerAndCloseEvidenceWithoutPromotion()
    {
        RunPipeline();

        using JsonDocument packDocument = ReadFinalReleaseJson("owner-public-publish-execution-final-intake-pack.json");
        JsonElement pack = packDocument.RootElement;
        Assert.Equal("owner-public-publish-execution-final-intake-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-publish-execution-final-intake-real-evidence-required", pack.GetProperty("intakeState").GetString());
        Assert.True(pack.GetProperty("phaseCount").GetInt32() >= 12);
        Assert.Equal(pack.GetProperty("phaseCount").GetInt32(), pack.GetProperty("blockedPhaseCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("proofReadyPhaseCount").GetInt32());
        Assert.True(pack.GetProperty("requiredEvidenceFieldCount").GetInt32() >= 110);
        Assert.True(pack.GetProperty("validatorScriptCount").GetInt32() >= 25);
        Assert.True(pack.GetProperty("ownerActionCount").GetInt32() >= 24);
        Assert.True(pack.GetProperty("forbiddenSubstituteCount").GetInt32() >= 120);
        AssertFalseProofPublishCloseFlags(pack);

        string[] phases = pack.GetProperty("phases").EnumerateArray()
            .Select(static phase => phase.GetProperty("id").GetString()!)
            .ToArray();
        foreach (string required in RequiredPhaseIds)
        {
            Assert.Contains(required, phases);
        }

        foreach (JsonElement phase in pack.GetProperty("phases").EnumerateArray())
        {
            Assert.Equal("blocked-owner-real-evidence-required", phase.GetProperty("phaseState").GetString());
            Assert.False(phase.GetProperty("proofReady").GetBoolean());
            Assert.True(phase.GetProperty("blocked").GetBoolean());
            Assert.True(phase.GetProperty("requiredFieldCount").GetInt32() >= 8);
            Assert.True(phase.GetProperty("validatorScriptCount").GetInt32() >= 2);
            Assert.Contains("not package push", phase.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-public-publish-execution-final-intake-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-public-publish-execution-final-intake-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("owner-public-publish-execution-final-intake-pack-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("finalCloseCandidateReady").GetBoolean());
        AssertFalseProofPublishCloseFlags(validation);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("owner-public-publish-execution-final-intake-pack-ready-non-proof", evidence.GetProperty("ownerPublicPublishExecutionFinalIntakePackValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("ownerPublicPublishExecutionFinalIntakePackProofReadyPhaseCount").GetInt32());
        Assert.False(evidence.GetProperty("ownerPublicPublishExecutionFinalIntakePackCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("ownerPublicPublishExecutionFinalIntakePackCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-public-publish-execution-final-intake-pack");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("not publish approval", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/owner-public-publish-execution-final-intake-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-public-publish-execution-final-intake-pack.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-public-publish-execution-final-intake-pack-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-public-publish-execution-final-intake-pack-validation.md", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        Assert.Contains(
            audit.GetProperty("auditedItems").EnumerateArray(),
            item => item.GetProperty("id").GetString() == "owner-public-publish-execution-final-intake-pack"
                && item.GetProperty("passed").GetBoolean() == false
                && item.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    private static readonly string[] RequiredPhaseIds =
    {
        "claim-boundary-preflight",
        "owner-public-publish-execution",
        "github-actions-run-proof",
        "public-managed-package-download-proof",
        "public-runtime-package-download-proof",
        "repository-external-clean-consumer",
        "post-publish-clean-consumer-proof",
        "post-publish-user-verification",
        "strict-close-convergence",
        "release-issue-close-owner-decision",
        "release-issue-close-record",
        "final-bundle-classification-lock",
    };

    private static void RunPipeline()
    {
        RunPowerShell("Export-PublicPublishFinalOwnerExecutionPack.ps1");
        RunPowerShell("Test-PublicPublishFinalOwnerExecutionPack.ps1", "-Strict");
        RunPowerShell("Export-PublicPackageDownloadProofOwnerExecutionPack.ps1");
        RunPowerShell("Test-PublicPackageDownloadProofOwnerExecutionPack.ps1", "-Strict");
        RunPowerShell("Export-GitHubPublishAndCiStatusSnapshot.ps1");
        RunPowerShell("Test-GitHubPublishAndCiStatusSnapshot.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealPublishEvidenceImportReadinessDashboard.ps1");
        RunPowerShell("Test-OwnerRealPublishEvidenceImportReadinessDashboard.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseFinalCandidateAuditPack.ps1");
        RunPowerShell("Test-ReleaseCloseFinalCandidateAuditPack.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublicPublishExecutionFinalIntakePack.ps1");
        RunPowerShell("Test-OwnerPublicPublishExecutionFinalIntakePack.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
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
