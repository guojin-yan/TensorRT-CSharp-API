using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCandidateFinalPublishabilityAuditTests
{
    private static readonly string[] EvidenceIds =
    [
        "release-candidate-final-publishability-audit",
        "release-candidate-owner-action-roadmap",
        "release-candidate-non-substitute-final-scan",
        "release-candidate-final-owner-checklist",
    ];

    private static readonly string[] SourceArtifacts =
    [
        "artifacts/final-release/release-candidate-final-publishability-audit.json",
        "artifacts/final-release/release-candidate-final-publishability-audit.md",
        "artifacts/final-release/release-candidate-final-publishability-audit-validation.json",
        "artifacts/final-release/release-candidate-final-publishability-audit-validation.md",
        "artifacts/final-release/release-candidate-owner-action-roadmap.json",
        "artifacts/final-release/release-candidate-owner-action-roadmap.md",
        "artifacts/final-release/release-candidate-owner-action-roadmap-validation.json",
        "artifacts/final-release/release-candidate-owner-action-roadmap-validation.md",
        "artifacts/final-release/release-candidate-non-substitute-final-scan.json",
        "artifacts/final-release/release-candidate-non-substitute-final-scan.md",
        "artifacts/final-release/release-candidate-non-substitute-final-scan-validation.json",
        "artifacts/final-release/release-candidate-non-substitute-final-scan-validation.md",
        "artifacts/final-release/release-candidate-final-owner-checklist.json",
        "artifacts/final-release/release-candidate-final-owner-checklist.md",
        "artifacts/final-release/release-candidate-final-owner-checklist-validation.json",
        "artifacts/final-release/release-candidate-final-owner-checklist-validation.md",
    ];

    [Fact]
    public void ReleaseCandidateFinalPublishabilityArtifactsStayBlockedNonProof()
    {
        RunPipeline();

        JsonElement publishability = ReadFinalReleaseJson("release-candidate-final-publishability-audit-validation.json").RootElement;
        Assert.Equal("blocked-release-candidate-final-publishability-owner-proof-required", publishability.GetProperty("validationState").GetString());
        Assert.Equal(17, publishability.GetProperty("publishabilityGateCount").GetInt32());
        Assert.Equal(17, publishability.GetProperty("blockedPublishabilityGateCount").GetInt32());
        Assert.Equal(0, publishability.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(1, publishability.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(publishability);

        JsonElement roadmap = ReadFinalReleaseJson("release-candidate-owner-action-roadmap-validation.json").RootElement;
        Assert.Equal("blocked-release-candidate-owner-action-roadmap-owner-proof-required", roadmap.GetProperty("validationState").GetString());
        Assert.Equal(12, roadmap.GetProperty("ownerActionCount").GetInt32());
        Assert.Equal(12, roadmap.GetProperty("blockedOwnerActionCount").GetInt32());
        Assert.Equal(0, roadmap.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(1, roadmap.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(roadmap);

        JsonElement nonSubstitute = ReadFinalReleaseJson("release-candidate-non-substitute-final-scan-validation.json").RootElement;
        Assert.Equal("blocked-release-candidate-non-substitute-final-scan-owner-proof-required", nonSubstitute.GetProperty("validationState").GetString());
        Assert.Equal(14, nonSubstitute.GetProperty("substituteCheckCount").GetInt32());
        Assert.Equal(14, nonSubstitute.GetProperty("blockedSubstituteCheckCount").GetInt32());
        Assert.Equal(0, nonSubstitute.GetProperty("promotedSubstituteCount").GetInt32());
        Assert.Equal(0, nonSubstitute.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(1, nonSubstitute.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(nonSubstitute);

        JsonElement checklist = ReadFinalReleaseJson("release-candidate-final-owner-checklist-validation.json").RootElement;
        Assert.Equal("blocked-release-candidate-final-owner-checklist-owner-proof-required", checklist.GetProperty("validationState").GetString());
        Assert.Equal(10, checklist.GetProperty("checklistItemCount").GetInt32());
        Assert.Equal(10, checklist.GetProperty("blockedChecklistItemCount").GetInt32());
        Assert.Equal(0, checklist.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(1, checklist.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(checklist);
    }

    [Fact]
    public void ReleaseEvidenceBundleAuditAndDocsIncludeFinalPublishabilityArtifacts()
    {
        RunPipeline();

        JsonElement evidence = ReadFinalReleaseJson("release-evidence-bundle.json").RootElement;
        Assert.Equal("blocked-release-candidate-final-publishability-owner-proof-required", evidence.GetProperty("releaseCandidateFinalPublishabilityAuditValidationState").GetString());
        Assert.Equal("blocked-release-candidate-owner-action-roadmap-owner-proof-required", evidence.GetProperty("releaseCandidateOwnerActionRoadmapValidationState").GetString());
        Assert.Equal("blocked-release-candidate-non-substitute-final-scan-owner-proof-required", evidence.GetProperty("releaseCandidateNonSubstituteFinalScanValidationState").GetString());
        Assert.Equal("blocked-release-candidate-final-owner-checklist-owner-proof-required", evidence.GetProperty("releaseCandidateFinalOwnerChecklistValidationState").GetString());

        Assert.Equal(17, evidence.GetProperty("releaseCandidateFinalPublishabilityAuditGateCount").GetInt32());
        Assert.Equal(17, evidence.GetProperty("releaseCandidateFinalPublishabilityAuditBlockedGateCount").GetInt32());
        Assert.Equal(12, evidence.GetProperty("releaseCandidateOwnerActionRoadmapActionCount").GetInt32());
        Assert.Equal(12, evidence.GetProperty("releaseCandidateOwnerActionRoadmapBlockedActionCount").GetInt32());
        Assert.Equal(14, evidence.GetProperty("releaseCandidateNonSubstituteFinalScanCheckCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("releaseCandidateNonSubstituteFinalScanPromotedSubstituteCount").GetInt32());
        Assert.Equal(10, evidence.GetProperty("releaseCandidateFinalOwnerChecklistItemCount").GetInt32());
        Assert.Equal(10, evidence.GetProperty("releaseCandidateFinalOwnerChecklistBlockedItemCount").GetInt32());

        foreach (string id in EvidenceIds)
        {
            AssertBlockedEvidenceItem(evidence, id);
        }

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string sourceArtifact in SourceArtifacts)
        {
            Assert.Contains(sourceArtifact, sourceArtifacts);
        }

        JsonElement audit = ReadFinalReleaseJson("release-evidence-classification-audit.json").RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        foreach (string id in EvidenceIds)
        {
            AssertAuditedNonProofItem(audit, id);
        }

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        foreach (string id in EvidenceIds)
        {
            Assert.Contains($"articles/zh-cn/{id}.md", docsIndex, StringComparison.Ordinal);
            Assert.Contains($"articles/zh-cn/{id}.md", docsToc, StringComparison.Ordinal);
            Assert.Contains(id, readme, StringComparison.Ordinal);
            Assert.Contains(id, readmeZh, StringComparison.Ordinal);
            Assert.Contains(id, releaseEvidenceDoc, StringComparison.Ordinal);
        }
    }

    private static void RunPipeline()
    {
        RunPowerShell("Export-ReleaseCandidateFinalPublishabilityAudit.ps1");
        RunPowerShell("Test-ReleaseCandidateFinalPublishabilityAudit.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCandidateOwnerActionRoadmap.ps1");
        RunPowerShell("Test-ReleaseCandidateOwnerActionRoadmap.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCandidateNonSubstituteFinalScan.ps1");
        RunPowerShell("Test-ReleaseCandidateNonSubstituteFinalScan.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCandidateFinalOwnerChecklist.ps1");
        RunPowerShell("Test-ReleaseCandidateFinalOwnerChecklist.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void AssertFalseProofPublishCloseFlags(JsonElement element)
    {
        Assert.True(element.GetProperty("notExecutedByAutomation").GetBoolean());
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseRecordProof").GetBoolean());
    }

    private static void AssertBlockedEvidenceItem(JsonElement evidence, string id)
    {
        JsonElement item = evidence.GetProperty("evidenceItems").EnumerateArray().Single(item => item.GetProperty("id").GetString() == id);
        string boundary = item.GetProperty("boundary").GetString()!;
        Assert.False(item.GetProperty("passed").GetBoolean());
        Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not publish approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertAuditedNonProofItem(JsonElement audit, string id)
    {
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}
