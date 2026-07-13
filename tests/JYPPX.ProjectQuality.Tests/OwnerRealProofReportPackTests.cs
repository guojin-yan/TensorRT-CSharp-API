using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealProofReportPackTests
{
    [Fact]
    public void ReportPackExportsBlockedItemsWithoutPromotingProof()
    {
        RunReportPackPipeline();

        using JsonDocument packDocument = ReadFinalReleaseJson("owner-real-proof-report-pack.json");
        JsonElement pack = packDocument.RootElement;

        Assert.Equal("owner-real-proof-report-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-report-input-required", pack.GetProperty("packState").GetString());
        Assert.Equal(6, pack.GetProperty("reportItemCount").GetInt32());
        Assert.Equal(6, pack.GetProperty("blockedReportItemCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("readyForOwnerReviewCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("readyForPromotionCount").GetInt32());
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(pack.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(pack.GetProperty("isReleaseCloseProof").GetBoolean());

        string[] lanes = pack.GetProperty("proofReportItems").EnumerateArray()
            .Select(static item => item.GetProperty("proofLane").GetString()!)
            .ToArray();
        Assert.Contains("package-consumer-runtime", lanes);
        Assert.Contains("post-publish-verification", lanes);
        Assert.Contains("linux-runner-proof", lanes);
        Assert.Contains("real-model-runtime", lanes);
        Assert.Contains("release-close-owner-input", lanes);
        Assert.Contains("strict-close-validation", lanes);

        JsonElement firstItem = pack.GetProperty("proofReportItems").EnumerateArray().First();
        Assert.True(firstItem.GetProperty("requiredOwnerInputs").GetArrayLength() >= 1);
        Assert.True(firstItem.GetProperty("requiredEvidenceFiles").GetArrayLength() >= 1);
        Assert.True(firstItem.GetProperty("requiredHashes").GetArrayLength() >= 1);
        Assert.True(firstItem.GetProperty("requiredCommands").GetArrayLength() >= 1);
        Assert.True(firstItem.GetProperty("requiredValidators").GetArrayLength() >= 1);
        Assert.True(firstItem.GetProperty("forbiddenSubstituteChecklist").GetArrayLength() >= 1);
        Assert.True(firstItem.GetProperty("reviewChecklist").GetArrayLength() >= 1);
        Assert.False(firstItem.GetProperty("readyForOwnerReview").GetBoolean());
        Assert.False(firstItem.GetProperty("readyForPromotion").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-real-proof-report-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("owner-real-proof-report-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-report-input-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("reportItemCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("blockedReportItemCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("readyForOwnerReviewCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("readyForPromotionCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceAndDocsIncludeReportPack()
    {
        RunReportPackPipeline();
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-owner-real-proof-report-input-required", evidence.GetProperty("ownerRealProofReportPackState").GetString());
        Assert.Equal("blocked-owner-real-proof-report-input-required", evidence.GetProperty("ownerRealProofReportPackValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("ownerRealProofReportPackReportItemCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("ownerRealProofReportPackBlockedReportItemCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("ownerRealProofReportPackReadyForOwnerReviewCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("ownerRealProofReportPackReadyForPromotionCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("ownerRealProofReportPackFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("ownerRealProofReportPackFailedActionRequiredCount").GetInt32() >= 1);
        Assert.False(evidence.GetProperty("ownerRealProofReportPackCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("ownerRealProofReportPackCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("ownerRealProofReportPackCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("ownerRealProofReportPackIsRuntimeExecutionProof").GetBoolean());
        Assert.False(evidence.GetProperty("ownerRealProofReportPackIsReleaseCloseProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-real-proof-report-pack");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("owner-fill", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/owner-real-proof-report-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-real-proof-report-pack.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-real-proof-report-pack-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-real-proof-report-pack-validation.md", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string reportPackDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-real-proof-report-pack.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("owner-real-proof-report-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-report-pack.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Export-OwnerRealProofReportPack.ps1", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("Test-OwnerRealProofReportPack.ps1 -Strict", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-report-pack", reportPackDoc, StringComparison.Ordinal);
        Assert.Contains("owner real proof report pack", evidenceMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    private static void RunReportPackPipeline()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealExternalProofBackfillExecutionBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofRunnerInputBackfill.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofRunnerInputBackfill.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofExecutionRecordProjection.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofExecutionRecordProjection.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealProofReportPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRealProofReportPack.ps1"), "-Strict");
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
