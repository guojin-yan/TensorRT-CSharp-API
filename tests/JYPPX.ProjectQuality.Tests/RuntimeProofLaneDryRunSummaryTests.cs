using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class RuntimeProofLaneDryRunSummaryTests
{
    [Fact]
    public void RuntimeProofLaneDryRunSummaryExportsBlockedLaneGaps()
    {
        RunRuntimeProofLaneDryRunPipeline();

        using JsonDocument summaryDocument = ReadFinalReleaseJson("runtime-proof-lane-dry-run-summary.json");
        JsonElement summary = summaryDocument.RootElement;

        Assert.Equal("runtime-proof-lane-dry-run-summary", summary.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-runtime-proof-real-evidence-required", summary.GetProperty("dryRunState").GetString());
        Assert.Equal(6, summary.GetProperty("laneItemCount").GetInt32());
        Assert.Equal(6, summary.GetProperty("blockedLaneItemCount").GetInt32());
        Assert.Equal(0, summary.GetProperty("readyLaneItemCount").GetInt32());
        Assert.True(summary.GetProperty("missingRealEvidenceFieldCount").GetInt32() >= 100);
        Assert.False(summary.GetProperty("performsPublish").GetBoolean());
        Assert.False(summary.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(summary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(summary.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(summary.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(summary.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement firstItem = summary.GetProperty("laneItems").EnumerateArray().First();
        Assert.Equal("blocked-runtime-proof-lane-real-evidence-required", firstItem.GetProperty("laneState").GetString());
        Assert.Equal("blocked-owner-runtime-proof-result-input-required", firstItem.GetProperty("ownerResultInputValidationState").GetString());
        Assert.True(firstItem.GetProperty("realEvidenceFieldsMissing").GetArrayLength() >= 20);
        Assert.True(firstItem.GetProperty("bridgeBlockedReasons").GetArrayLength() >= 4);
        Assert.False(firstItem.GetProperty("readyForRuntimeProof").GetBoolean());
        Assert.False(firstItem.GetProperty("canPromoteRuntimeProof").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("runtime-proof-lane-dry-run-summary-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("runtime-proof-lane-dry-run-summary-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-runtime-proof-real-evidence-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("laneItemCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("blockedLaneItemCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceIncludesRuntimeProofLaneDryRunAsNonProof()
    {
        RunRuntimeProofLaneDryRunPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-runtime-proof-real-evidence-required", evidence.GetProperty("runtimeProofLaneDryRunSummaryState").GetString());
        Assert.Equal("blocked-runtime-proof-real-evidence-required", evidence.GetProperty("runtimeProofLaneDryRunSummaryValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("runtimeProofLaneDryRunSummaryLaneItemCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("runtimeProofLaneDryRunSummaryBlockedLaneItemCount").GetInt32());
        Assert.True(evidence.GetProperty("runtimeProofLaneDryRunSummaryMissingRealEvidenceFieldCount").GetInt32() >= 100);
        Assert.False(evidence.GetProperty("runtimeProofLaneDryRunSummaryCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("runtimeProofLaneDryRunSummaryCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("runtimeProofLaneDryRunSummaryCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "runtime-proof-lane-dry-run-summary");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("dry-run analysis only", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/runtime-proof-lane-dry-run-summary.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/runtime-proof-lane-dry-run-summary-validation.json", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.Contains("runtime-proof-lane-dry-run-summary.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("runtime-proof-lane-dry-run-summary.md", docsToc, StringComparison.Ordinal);
    }

    internal static void RunRuntimeProofLaneDryRunPipeline()
    {
        OwnerRuntimeProofResultInputTests.RunOwnerRuntimeProofResultInputPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RuntimeProofLaneDryRunSummary.ps1"));
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RuntimeProofLaneDryRunSummary.ps1"), "-Strict");
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
