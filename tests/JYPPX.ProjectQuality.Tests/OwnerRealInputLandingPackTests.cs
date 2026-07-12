using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealInputLandingPackTests
{
    [Fact]
    public void LandingPackMapsFiveFinalBlockersAndStaysNonProof()
    {
        RunPowerShell("Export-OwnerRealInputLandingPack.ps1");
        RunPowerShell("Test-OwnerRealInputLandingPack.ps1", "-Strict");

        using JsonDocument packDocument = ReadFinalReleaseJson("owner-real-input-landing-pack.json");
        JsonElement pack = packDocument.RootElement;

        Assert.Equal("owner-real-input-landing-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-input-required", pack.GetProperty("landingState").GetString());
        Assert.Equal(5, pack.GetProperty("blockerCount").GetInt32());
        Assert.Equal(5, pack.GetProperty("blockedBlockerCount").GetInt32());
        AssertFlagsStayNonProof(pack);
        AssertBoundary(pack.GetProperty("boundary").GetString()!);

        string[] blockerIds = pack.GetProperty("blockerIds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string expected in new[]
        {
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish-verification",
        })
        {
            Assert.Contains(expected, blockerIds);
        }

        foreach (JsonElement blocker in pack.GetProperty("blockers").EnumerateArray())
        {
            Assert.True(blocker.GetProperty("requiredOwnerInputFiles").GetArrayLength() >= 2);
            Assert.True(blocker.GetProperty("requiredFields").GetArrayLength() >= 8);
            Assert.Contains("Test-", blocker.GetProperty("strictValidator").GetString(), StringComparison.Ordinal);
            Assert.Contains("validator", blocker.GetProperty("promotionBlockedUntil").GetString(), StringComparison.OrdinalIgnoreCase);
            AssertBoundary(blocker.GetProperty("nonProofBoundary").GetString()!);
            AssertFlagsStayNonProof(blocker);
        }

        string raw = pack.GetRawText();
        foreach (string marker in RequiredForbiddenSubstitutes)
        {
            Assert.Contains(marker, raw, StringComparison.Ordinal);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-real-input-landing-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("blocked-owner-real-input-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(5, validation.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFlagsStayNonProof(validation);
        AssertBoundary(validation.GetProperty("boundary").GetString()!);
    }

    [Fact]
    public void ReleaseEvidenceCarriesLandingPackAsRequiredNonProofItem()
    {
        RunOwnerInputPipeline();

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-owner-real-input-required", evidence.GetProperty("ownerRealInputLandingPackValidationState").GetString());
        Assert.Equal(5, evidence.GetProperty("ownerRealInputLandingPackBlockerCount").GetInt32());
        Assert.False(evidence.GetProperty("ownerRealInputLandingPackCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("ownerRealInputLandingPackCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("ownerRealInputLandingPackIsRuntimeExecutionProof").GetBoolean());

        AssertBlockedEvidenceItem(evidence, "owner-real-input-landing-pack");

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/owner-real-input-landing-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-real-input-landing-pack-validation.json", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        AssertAuditedNonProofItem(audit, "owner-real-input-landing-pack");
    }

    internal static void RunOwnerInputPipeline()
    {
        RunPowerShell("Export-OwnerRealInputLandingPack.ps1");
        RunPowerShell("Test-OwnerRealInputLandingPack.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerExecutionChecklist.ps1");
        RunPowerShell("Test-FinalOwnerExecutionChecklist.ps1", "-Strict");
        RunPowerShell("Test-RealProofImportBoundaryAudit.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
    }

    internal static readonly string[] RequiredForbiddenSubstitutes =
    {
        "local feed",
        "ProjectReference",
        "direct nupkg",
        "dry-run",
        "dashboard",
        "runbook",
        "candidate",
        "draft",
        "build-only",
        "parse-only",
        "sidecar-only",
        "template",
    };

    internal static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    internal static void AssertFlagsStayNonProof(JsonElement element)
    {
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    internal static void AssertBoundary(string boundary)
    {
        Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not publish approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
    }

    internal static void AssertBlockedEvidenceItem(JsonElement evidence, string id)
    {
        JsonElement item = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(item => item.GetProperty("id").GetString() == id);

        Assert.False(item.GetProperty("passed").GetBoolean());
        AssertBoundary(item.GetProperty("boundary").GetString()!);
    }

    internal static void AssertAuditedNonProofItem(JsonElement audit, string id)
    {
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    internal static void RunPowerShell(string scriptName, params string[] arguments)
    {
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", scriptName), arguments);
    }
}
