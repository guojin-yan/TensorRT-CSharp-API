using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PublicProofClaimBoundaryAuditTests
{
    [Fact]
    public void PublicClaimAuditPassesWithoutPromotingProof()
    {
        RunPowerShell("Test-PublicProofClaimBoundaryAudit.ps1", "-Strict");

        using JsonDocument document = ReadFinalReleaseJson("public-proof-claim-boundary-audit.json");
        JsonElement audit = document.RootElement;

        Assert.Equal("public-proof-claim-boundary-audit", audit.GetProperty("recordKind").GetString());
        Assert.Equal("public-proof-claim-boundary-audit-passed", audit.GetProperty("auditState").GetString());
        Assert.Equal("public-docs-proof-boundary-freeze-passed", audit.GetProperty("publicFreezeState").GetString());
        Assert.True(audit.GetProperty("scannedFileCount").GetInt32() > 0);
        Assert.True(audit.GetProperty("publicFreezeRequiredCount").GetInt32() >= 5);
        Assert.Equal(0, audit.GetProperty("publicFreezeFindingCount").GetInt32());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        Assert.Equal(0, audit.GetProperty("blockedFindingCount").GetInt32());
        AssertFlagsStayNonProof(audit);
        AssertBoundary(audit.GetProperty("boundary").GetString()!);
    }

    [Fact]
    public void EvidenceBundleAndClassificationAuditCarryPublicClaimAuditAsNonProof()
    {
        FinalQualityFreezeDashboardTests.RunPipeline();
        RunPowerShell("Test-PublicProofClaimBoundaryAudit.ps1", "-Strict");
        RunPowerShell("Export-ArticleRoadmap30Plus.ps1");
        RunPowerShell("Test-ArticleRoadmap30Plus.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("public-proof-claim-boundary-audit-passed", evidence.GetProperty("publicProofClaimBoundaryAuditState").GetString());
        Assert.Equal("public-docs-proof-boundary-freeze-passed", evidence.GetProperty("publicProofClaimBoundaryAuditPublicFreezeState").GetString());
        Assert.True(evidence.GetProperty("publicProofClaimBoundaryAuditPublicFreezeRequiredCount").GetInt32() >= 5);
        Assert.Equal(0, evidence.GetProperty("publicProofClaimBoundaryAuditPublicFreezeFindingCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("publicProofClaimBoundaryAuditFindingCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("publicProofClaimBoundaryAuditBlockedFindingCount").GetInt32());
        Assert.False(evidence.GetProperty("publicProofClaimBoundaryAuditCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("publicProofClaimBoundaryAuditCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("publicProofClaimBoundaryAuditIsRuntimeExecutionProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "public-proof-claim-boundary-audit");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        AssertBoundary(evidenceItem.GetProperty("boundary").GetString()!);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement classificationAudit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", classificationAudit.GetProperty("auditState").GetString());
        Assert.Equal(0, classificationAudit.GetProperty("findingCount").GetInt32());
        Assert.Contains(classificationAudit.GetProperty("auditedItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "public-proof-claim-boundary-audit" &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void AssertFlagsStayNonProof(JsonElement element)
    {
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static void AssertBoundary(string boundary)
    {
        Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not publish approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", scriptName), arguments);
    }
}
