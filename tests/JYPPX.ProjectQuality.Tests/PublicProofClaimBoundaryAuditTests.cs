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
        Assert.Equal("inline-plus-markdown-heading-stack", audit.GetProperty("negationContextMode").GetString());
        AssertFlagsStayNonProof(audit);
        AssertBoundary(audit.GetProperty("boundary").GetString()!);
    }

    [Fact]
    public void MarkdownNegativeSectionsAreSafeButRealPromotionStillFailsClosed()
    {
        string fixtureRoot = Path.Combine(Path.GetTempPath(), $"public-proof-boundary-{Guid.NewGuid():N}");
        string docsRoot = Path.Combine(fixtureRoot, "docs");
        string outputRoot = Path.Combine(fixtureRoot, "output");
        Directory.CreateDirectory(docsRoot);
        Directory.CreateDirectory(outputRoot);

        try
        {
            File.WriteAllText(Path.Combine(docsRoot, "claims.md"), """
                # Claims

                ## Cannot claim

                - local feed is published.

                ## 以下材料不得替代发布证明

                - failedBlockerCount=0 means ready to publish.

                ## Current release status

                - candidate 可发布性 audit is active.
                - local feed is published.
                - failedBlockerCount=0 means ready to publish.
                """);

            RunPowerShell(
                "Test-PublicProofClaimBoundaryAudit.ps1",
                "-RepositoryRoot",
                fixtureRoot,
                "-OutputDirectory",
                outputRoot);

            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(
                Path.Combine(outputRoot, "public-proof-claim-boundary-audit.json")));
            JsonElement audit = document.RootElement;
            JsonElement[] findings = audit.GetProperty("findings").EnumerateArray().ToArray();

            Assert.Equal("inline-plus-markdown-heading-stack", audit.GetProperty("negationContextMode").GetString());
            Assert.Contains(findings, static item =>
                item.GetProperty("id").GetString() == "non-proof-artifact-promoted" &&
                item.GetProperty("line").GetInt32() == 14);
            Assert.Contains(findings, static item =>
                item.GetProperty("id").GetString() == "failed-blocker-zero-promoted" &&
                item.GetProperty("line").GetInt32() == 15);
            Assert.DoesNotContain(findings, static item =>
                item.GetProperty("path").GetString() == "docs/claims.md" &&
                item.GetProperty("line").GetInt32() is 5 or 9 or 13);
        }
        finally
        {
            Directory.Delete(fixtureRoot, recursive: true);
        }
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
