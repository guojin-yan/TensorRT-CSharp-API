using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class RealProofImportBoundaryAuditTests
{
    [Fact]
    public void RealProofImportBoundaryAuditPassesWithoutPromotingProof()
    {
        OwnerRealInputLandingPackTests.RunPowerShell("Test-RealProofImportBoundaryAudit.ps1", "-Strict");

        using JsonDocument auditDocument = OwnerRealInputLandingPackTests.ReadFinalReleaseJson("real-proof-import-boundary-audit.json");
        JsonElement audit = auditDocument.RootElement;

        Assert.Equal("real-proof-import-boundary-audit", audit.GetProperty("recordKind").GetString());
        Assert.Equal("real-proof-import-boundary-audit-passed", audit.GetProperty("auditState").GetString());
        Assert.True(audit.GetProperty("scannedFileCount").GetInt32() > 0);
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        Assert.Equal(0, audit.GetProperty("blockedFindingCount").GetInt32());
        OwnerRealInputLandingPackTests.AssertFlagsStayNonProof(audit);
        OwnerRealInputLandingPackTests.AssertBoundary(audit.GetProperty("boundary").GetString()!);
    }

    [Fact]
    public void ReleaseEvidenceCarriesImportAuditAsRequiredNonProofItem()
    {
        OwnerRealInputLandingPackTests.RunOwnerInputPipeline();

        using JsonDocument evidenceDocument = OwnerRealInputLandingPackTests.ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("real-proof-import-boundary-audit-passed", evidence.GetProperty("realProofImportBoundaryAuditState").GetString());
        Assert.Equal(0, evidence.GetProperty("realProofImportBoundaryAuditFindingCount").GetInt32());
        Assert.False(evidence.GetProperty("realProofImportBoundaryAuditCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("realProofImportBoundaryAuditCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("realProofImportBoundaryAuditIsRuntimeExecutionProof").GetBoolean());

        OwnerRealInputLandingPackTests.AssertBlockedEvidenceItem(evidence, "real-proof-import-boundary-audit");

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/real-proof-import-boundary-audit.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-import-boundary-audit.md", sourceArtifacts);

        using JsonDocument classificationDocument = OwnerRealInputLandingPackTests.ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement classification = classificationDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", classification.GetProperty("auditState").GetString());
        Assert.Equal(0, classification.GetProperty("findingCount").GetInt32());
        OwnerRealInputLandingPackTests.AssertAuditedNonProofItem(classification, "real-proof-import-boundary-audit");

        using JsonDocument finalGateDocument = OwnerRealInputLandingPackTests.ReadFinalReleaseJson("final-publish-proof-gate-report.json");
        JsonElement finalGate = finalGateDocument.RootElement;
        Assert.Equal("blocked-final-publish-real-proof-required", finalGate.GetProperty("validationState").GetString());
        Assert.False(finalGate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(finalGate.GetProperty("canCloseReleaseIssue").GetBoolean());
    }
}
