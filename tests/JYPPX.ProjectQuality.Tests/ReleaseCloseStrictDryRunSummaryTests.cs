using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCloseStrictDryRunSummaryTests
{
    [Fact]
    public void ReleaseCloseStrictDryRunSummaryExportsBlockedCloseGaps()
    {
        RunReleaseCloseStrictDryRunPipeline();

        using JsonDocument summaryDocument = ReadFinalReleaseJson("release-close-strict-dry-run-summary.json");
        JsonElement summary = summaryDocument.RootElement;

        Assert.Equal("release-close-strict-dry-run-summary", summary.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-real-proof-required", summary.GetProperty("closeDryRunState").GetString());
        Assert.Equal(6, summary.GetProperty("closeDryRunItemCount").GetInt32());
        Assert.Equal(6, summary.GetProperty("blockedCloseDryRunItemCount").GetInt32());
        Assert.Equal(0, summary.GetProperty("readyCloseDryRunItemCount").GetInt32());
        Assert.True(summary.GetProperty("remainingGapCount").GetInt32() >= 200);
        Assert.Equal("blocked-runtime-proof-real-evidence-required", summary.GetProperty("laneDryRunState").GetString());
        Assert.Equal("blocked-release-close-strict-validation-required", summary.GetProperty("releaseCloseStrictBridgeState").GetString());
        Assert.False(summary.GetProperty("performsPublish").GetBoolean());
        Assert.False(summary.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(summary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(summary.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(summary.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(summary.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement firstItem = summary.GetProperty("closeDryRunItems").EnumerateArray().First();
        Assert.Equal("blocked-release-close-real-proof-required", firstItem.GetProperty("closeDryRunItemState").GetString());
        Assert.True(firstItem.GetProperty("remainingGaps").GetArrayLength() >= 30);
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", firstItem.GetProperty("strictCloseCommand").GetString(), StringComparison.Ordinal);
        Assert.False(firstItem.GetProperty("readyForReleaseClose").GetBoolean());
        Assert.False(firstItem.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("release-close-strict-dry-run-summary-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("release-close-strict-dry-run-summary-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-real-proof-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("closeDryRunItemCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("blockedCloseDryRunItemCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceIncludesStrictDryRunAndAuditKeepsItFailed()
    {
        RunReleaseCloseStrictDryRunPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-release-close-real-proof-required", evidence.GetProperty("releaseCloseStrictDryRunSummaryState").GetString());
        Assert.Equal("blocked-release-close-real-proof-required", evidence.GetProperty("releaseCloseStrictDryRunSummaryValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("releaseCloseStrictDryRunSummaryCloseDryRunItemCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("releaseCloseStrictDryRunSummaryBlockedCloseDryRunItemCount").GetInt32());
        Assert.True(evidence.GetProperty("releaseCloseStrictDryRunSummaryRemainingGapCount").GetInt32() >= 200);
        Assert.False(evidence.GetProperty("releaseCloseStrictDryRunSummaryCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseStrictDryRunSummaryCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseStrictDryRunSummaryCanCloseReleaseIssue").GetBoolean());

        string[] evidenceIds = evidence.GetProperty("evidenceItems").EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("release-close-strict-dry-run-summary", evidenceIds);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/release-close-strict-dry-run-summary.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-close-strict-dry-run-summary-validation.json", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());

        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        Assert.Contains("release-close-strict-dry-run-summary", readme, StringComparison.Ordinal);
        Assert.Contains("release-close-strict-dry-run-summary", readmeZh, StringComparison.Ordinal);
        Assert.Contains("release-close-strict-dry-run-summary-validation.json", releaseEvidenceDoc, StringComparison.Ordinal);
    }

    internal static void RunReleaseCloseStrictDryRunPipeline()
    {
        RuntimeProofLaneDryRunSummaryTests.RunRuntimeProofLaneDryRunPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseStrictDryRunSummary.ps1"));
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseCloseStrictDryRunSummary.ps1"), "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }
}
