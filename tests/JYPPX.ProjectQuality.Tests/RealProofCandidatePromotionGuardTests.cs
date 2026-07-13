using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class RealProofCandidatePromotionGuardTests
{
    [Fact]
    public void PromotionGuardBlocksCandidatesWithoutPromotingProof()
    {
        RunPromotionGuardPipeline();

        using JsonDocument guardDocument = ReadFinalReleaseJson("real-proof-candidate-promotion-guard.json");
        JsonElement guard = guardDocument.RootElement;

        Assert.Equal("real-proof-candidate-promotion-guard", guard.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-candidate-promotion-not-allowed", guard.GetProperty("guardState").GetString());
        Assert.Equal(6, guard.GetProperty("candidateCount").GetInt32());
        Assert.Equal(0, guard.GetProperty("promotionAllowedCandidateCount").GetInt32());
        Assert.Equal(6, guard.GetProperty("blockedCandidateCount").GetInt32());
        Assert.False(guard.GetProperty("performsPublish").GetBoolean());
        Assert.False(guard.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(guard.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(guard.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(guard.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(guard.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement firstItem = guard.GetProperty("guardItems").EnumerateArray().First();
        Assert.Equal("blocked-real-proof-candidate-promotion-not-allowed", firstItem.GetProperty("guardState").GetString());
        Assert.False(firstItem.GetProperty("promotionAllowed").GetBoolean());
        Assert.True(firstItem.GetProperty("blockedRequirementCount").GetInt32() >= 1);
        Assert.True(firstItem.GetProperty("requirements").GetArrayLength() >= 4);

        using JsonDocument validationDocument = ReadFinalReleaseJson("real-proof-candidate-promotion-guard-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("real-proof-candidate-promotion-guard-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-candidate-promotion-not-allowed", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("candidateCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("promotionAllowedCandidateCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("blockedCandidateCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceAndDocsIncludePromotionGuard()
    {
        RunPromotionGuardPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-real-proof-candidate-promotion-not-allowed", evidence.GetProperty("realProofCandidatePromotionGuardState").GetString());
        Assert.Equal("blocked-real-proof-candidate-promotion-not-allowed", evidence.GetProperty("realProofCandidatePromotionGuardValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("realProofCandidatePromotionGuardCandidateCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("realProofCandidatePromotionGuardPromotionAllowedCandidateCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("realProofCandidatePromotionGuardBlockedCandidateCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("realProofCandidatePromotionGuardFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("realProofCandidatePromotionGuardFailedActionRequiredCount").GetInt32() >= 1);
        Assert.False(evidence.GetProperty("realProofCandidatePromotionGuardCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("realProofCandidatePromotionGuardCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("realProofCandidatePromotionGuardCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "real-proof-candidate-promotion-guard");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("blocks candidate promotion", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/real-proof-candidate-promotion-guard.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-candidate-promotion-guard.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-candidate-promotion-guard-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-candidate-promotion-guard-validation.md", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "real-proof-candidate-promotion-guard.md"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("real-proof-candidate-promotion-guard.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("real-proof-candidate-promotion-guard.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Export-RealProofCandidatePromotionGuard.ps1", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("Test-RealProofCandidatePromotionGuard.ps1 -Strict", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("real-proof-candidate-promotion-guard", article, StringComparison.Ordinal);
        Assert.Contains("real-proof-candidate-promotion-guard", readme, StringComparison.Ordinal);
        Assert.Contains("real-proof-candidate-promotion-guard", readmeZh, StringComparison.Ordinal);
        Assert.Contains("real proof candidate promotion guard", evidenceMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    private static void RunPromotionGuardPipeline()
    {
        OwnerRealProofFieldDeltaPackTests.RunDeltaPackPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofCandidatePromotionGuard.ps1"));
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofCandidatePromotionGuard.ps1"), "-Strict");
    }

    internal static void RunPromotionGuardPipelineForReuse()
    {
        RunPromotionGuardPipeline();
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
