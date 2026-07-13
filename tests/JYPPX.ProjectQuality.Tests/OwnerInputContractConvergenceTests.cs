using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerInputContractConvergenceTests
{
    [Fact]
    public void OwnerInputContractConvergenceStaysBlockedAndNonProof()
    {
        RunPipeline();

        using JsonDocument convergenceDocument = ReadFinalReleaseJson("owner-input-contract-convergence.json");
        JsonElement convergence = convergenceDocument.RootElement;

        Assert.Equal("owner-input-contract-convergence", convergence.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-input-contract-convergence-real-owner-input-required", convergence.GetProperty("convergenceState").GetString());
        Assert.True(convergence.GetProperty("contractSurfaceCount").GetInt32() >= 5);
        Assert.True(convergence.GetProperty("canonicalFieldCount").GetInt32() >= 14);
        Assert.Equal(2, convergence.GetProperty("runbookInputCount").GetInt32());
        Assert.False(convergence.GetProperty("performsPublish").GetBoolean());
        Assert.False(convergence.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(convergence.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(convergence.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(convergence.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(convergence.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(convergence.GetProperty("isReleaseCloseProof").GetBoolean());
        AssertBoundary(convergence.GetProperty("boundary").GetString()!);

        string[] canonicalFields = convergence.GetProperty("canonicalFields")
            .EnumerateArray()
            .Select(static item => item.GetProperty("name").GetString()!)
            .ToArray();
        Assert.Contains("publicPackageSourceUrl", canonicalFields);
        Assert.Contains("downloadedNupkgSha256", canonicalFields);
        Assert.Contains("nonSubstituteConfirmations", canonicalFields);

        foreach (JsonElement surface in convergence.GetProperty("contractSurfaces").EnumerateArray())
        {
            Assert.False(surface.GetProperty("performsPublish").GetBoolean());
            Assert.False(surface.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(surface.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(surface.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(surface.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(surface.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(surface.GetProperty("isReleaseCloseProof").GetBoolean());
            AssertBoundary(surface.GetProperty("boundary").GetString()!);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-input-contract-convergence-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("blocked-owner-input-contract-convergence-real-owner-input-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("contractSurfaceCount").GetInt32() >= 5);
        Assert.True(validation.GetProperty("canonicalFieldCount").GetInt32() >= 14);
        Assert.Equal(2, validation.GetProperty("runbookInputCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());
        AssertBoundary(validation.GetProperty("boundary").GetString()!);
    }

    [Fact]
    public void EvidenceBundleAndClassificationAuditIncludeOwnerInputContractConvergence()
    {
        RunPipeline();

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.False(evidence.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal("blocked-owner-input-contract-convergence-real-owner-input-required", evidence.GetProperty("ownerInputContractConvergenceValidationState").GetString());
        Assert.True(evidence.GetProperty("ownerInputContractConvergenceContractSurfaceCount").GetInt32() >= 5);
        Assert.True(evidence.GetProperty("ownerInputContractConvergenceCanonicalFieldCount").GetInt32() >= 14);
        Assert.Equal(2, evidence.GetProperty("ownerInputContractConvergenceRunbookInputCount").GetInt32());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(item => item.GetProperty("id").GetString() == "owner-input-contract-convergence");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("contractSurfaces=", evidenceItem.GetProperty("state").GetString(), StringComparison.Ordinal);
        AssertBoundary(evidenceItem.GetProperty("boundary").GetString()!);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/owner-input-contract-convergence.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-input-contract-convergence.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-input-contract-convergence-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-input-contract-convergence-validation.md", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == "owner-input-contract-convergence" &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    private static void RunPipeline()
    {
        RunPowerShell("Export-OwnerInputContractConvergence.ps1");
        RunPowerShell("Test-OwnerInputContractConvergence.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
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
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}
