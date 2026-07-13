using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealProofExecutionClosurePackTests
{
    [Fact]
    public void ClosurePackExportsBlockedExecutionItemsWithoutPromotingProof()
    {
        RunClosurePackPipeline();

        using JsonDocument packDocument = ReadFinalReleaseJson("owner-real-proof-execution-closure-pack.json");
        JsonElement pack = packDocument.RootElement;

        Assert.Equal("owner-real-proof-execution-closure-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-execution-closure-required", pack.GetProperty("closureState").GetString());
        Assert.Equal(6, pack.GetProperty("closureItemCount").GetInt32());
        Assert.Equal(6, pack.GetProperty("blockedClosureItemCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("readyClosureItemCount").GetInt32());
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(pack.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(pack.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement firstItem = pack.GetProperty("closureItems").EnumerateArray().First();
        Assert.Equal("blocked-owner-real-proof-execution-closure-required", firstItem.GetProperty("closureState").GetString());
        Assert.True(firstItem.GetProperty("ownerDeltaIds").GetArrayLength() > 0);
        Assert.False(string.IsNullOrWhiteSpace(firstItem.GetProperty("firstCommand").GetString()));
        Assert.True(firstItem.GetProperty("expectedArtifacts").GetArrayLength() > 0);
        Assert.True(firstItem.GetProperty("requiredLogs").GetArrayLength() > 0);
        Assert.True(firstItem.GetProperty("requiredSha256").GetArrayLength() > 0);
        Assert.True(firstItem.GetProperty("validatorCommands").GetArrayLength() > 0);
        Assert.True(firstItem.GetProperty("promotionGuardRequirements").GetArrayLength() > 0);
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", firstItem.GetProperty("releaseCloseFollowUp").GetString(), StringComparison.Ordinal);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-real-proof-execution-closure-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("owner-real-proof-execution-closure-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-execution-closure-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("closureItemCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("blockedClosureItemCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("readyClosureItemCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceAndDocsIncludeClosurePack()
    {
        RunClosurePackPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-owner-real-proof-execution-closure-required", evidence.GetProperty("ownerRealProofExecutionClosurePackState").GetString());
        Assert.Equal("blocked-owner-real-proof-execution-closure-required", evidence.GetProperty("ownerRealProofExecutionClosurePackValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("ownerRealProofExecutionClosurePackClosureItemCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("ownerRealProofExecutionClosurePackBlockedClosureItemCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("ownerRealProofExecutionClosurePackReadyClosureItemCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("ownerRealProofExecutionClosurePackFailedBlockerCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("ownerRealProofExecutionClosurePackFailedActionRequiredCount").GetInt32());
        Assert.False(evidence.GetProperty("ownerRealProofExecutionClosurePackCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("ownerRealProofExecutionClosurePackCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("ownerRealProofExecutionClosurePackCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-real-proof-execution-closure-pack");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("owner execution checklist only", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/owner-real-proof-execution-closure-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-real-proof-execution-closure-pack.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-real-proof-execution-closure-pack-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-real-proof-execution-closure-pack-validation.md", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("owner-real-proof-execution-closure-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-execution-closure-pack.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Export-OwnerRealProofExecutionClosurePack.ps1", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("Test-OwnerRealProofExecutionClosurePack.ps1 -Strict", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-execution-closure-pack", readme, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-execution-closure-pack", readmeZh, StringComparison.Ordinal);
        Assert.Contains("owner real proof execution closure pack", evidenceMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    private static void RunClosurePackPipeline()
    {
        RealProofRecordValidatorTests.RunRecordValidatorPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealProofExecutionClosurePack.ps1"));
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRealProofExecutionClosurePack.ps1"), "-Strict");
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
