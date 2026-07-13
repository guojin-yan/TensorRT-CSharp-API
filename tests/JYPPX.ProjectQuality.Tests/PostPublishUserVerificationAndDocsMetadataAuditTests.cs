using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PostPublishUserVerificationAndDocsMetadataAuditTests
{
    [Fact]
    public void DocsMetadataAuditAndPostPublishUserVerificationPackStayNonPublishingAndBlocked()
    {
        RunPowerShell("Test-PublicDocsAndPackageMetadataGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseDocsAndNuGetMetadataAudit.ps1");
        RunPowerShell("Test-ReleaseDocsAndNuGetMetadataAudit.ps1", "-Strict");
        RunPowerShell("Export-PostPublishUserVerificationPack.ps1");
        RunPowerShell("Test-PostPublishUserVerificationPack.ps1", "-Strict");

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-docs-and-nuget-metadata-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("release-docs-and-nuget-metadata-audit", audit.GetProperty("recordKind").GetString());
        Assert.Equal("release-docs-and-nuget-metadata-audit-ready-non-proof", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(0, audit.GetProperty("yoloDetBlockedMatchCount").GetInt32());
        Assert.True(audit.GetProperty("auditItemCount").GetInt32() >= 18);
        Assert.True(audit.GetProperty("splitRuntimePackageCount").GetInt32() >= 10);
        AssertFalseProofPublishCloseFlags(audit);

        string auditRaw = audit.GetRawText();
        foreach (string marker in new[]
        {
            "JYPPX.TensorRT.CSharp.API",
            "YoloVision",
            "OnnxToEngine",
            "TensorRtExec",
            "yolov5",
            "yolov26",
            "custom",
            "det",
            "sem",
            "win-x64.trt11.0.cuda13.2",
            "public-docs-package-metadata-gate.json",
            "not public package download proof",
            "post-publish proof"
        })
        {
            Assert.Contains(marker, auditRaw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument auditValidationDocument = ReadFinalReleaseJson("release-docs-and-nuget-metadata-audit-validation.json");
        JsonElement auditValidation = auditValidationDocument.RootElement;
        Assert.Equal("release-docs-and-nuget-metadata-audit-validation", auditValidation.GetProperty("recordKind").GetString());
        Assert.Equal("release-docs-and-nuget-metadata-audit-ready-non-proof", auditValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, auditValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(auditValidation);

        using JsonDocument packDocument = ReadFinalReleaseJson("post-publish-user-verification-pack.json");
        JsonElement pack = packDocument.RootElement;
        Assert.Equal("post-publish-user-verification-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-post-publish-user-verification-required", pack.GetProperty("packState").GetString());
        Assert.Equal(9, pack.GetProperty("verificationLaneCount").GetInt32());
        Assert.True(pack.GetProperty("readyVerificationLaneCount").GetInt32() >= 2);
        Assert.True(pack.GetProperty("blockedVerificationLaneCount").GetInt32() >= 4);
        AssertFalseProofPublishCloseFlags(pack);

        JsonElement[] lanes = pack.GetProperty("lanes").EnumerateArray().ToArray();
        AssertLane(lanes, "release-docs-and-nuget-metadata-audit", ready: true);
        AssertLane(lanes, "public-docs-package-metadata-gate", ready: true);
        AssertLane(lanes, "public-package-download-proof", ready: false);
        AssertLane(lanes, "clean-external-consumer-smoke", ready: false);
        AssertLane(lanes, "post-publish-clean-consumer-proof", ready: false);
        AssertLane(lanes, "final-post-publish-audit-pack", ready: false);

        string packRaw = pack.GetRawText();
        foreach (string marker in new[]
        {
            "release-docs-and-nuget-metadata-audit-validation.json",
            "public-package-download-proof-candidate-validation.json",
            "clean-external-consumer-smoke-input-validation.json",
            "post-publish-clean-consumer-proof-result-validation.json",
            "local feed",
            "direct .nupkg",
            "ProjectReference",
            "does not execute dotnet nuget push",
            "post-publish user verification pack is an owner action aggregator only"
        })
        {
            Assert.Contains(marker, packRaw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument packValidationDocument = ReadFinalReleaseJson("post-publish-user-verification-pack-validation.json");
        JsonElement packValidation = packValidationDocument.RootElement;
        Assert.Equal("post-publish-user-verification-pack-validation", packValidation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-post-publish-user-verification-required", packValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, packValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(packValidation.GetProperty("failedActionRequiredCount").GetInt32() >= 4);
        AssertFalseProofPublishCloseFlags(packValidation);

        string auditMarkdown = ReadFinalReleaseText("release-docs-and-nuget-metadata-audit.md");
        string packMarkdown = ReadFinalReleaseText("post-publish-user-verification-pack.md");
        Assert.Contains("Release Docs And NuGet Metadata Audit", auditMarkdown, StringComparison.Ordinal);
        Assert.Contains("Post-Publish User Verification Pack", packMarkdown, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseEvidenceCarriesDocsAuditUserVerificationAndOwnerDownloadExecutionPackAsBlockedNonProof()
    {
        RunPowerShell("Test-PublicDocsAndPackageMetadataGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseDocsAndNuGetMetadataAudit.ps1");
        RunPowerShell("Test-ReleaseDocsAndNuGetMetadataAudit.ps1", "-Strict");
        RunPowerShell("Export-PublicPackageDownloadProofInputTemplate.ps1");
        RunPowerShell("Export-PostPublishUserVerificationPack.ps1");
        RunPowerShell("Test-PostPublishUserVerificationPack.ps1", "-Strict");
        RunPowerShell("Export-PublicPackageDownloadProofOwnerExecutionPack.ps1");
        RunPowerShell("Test-PublicPackageDownloadProofOwnerExecutionPack.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument executionPackDocument = ReadFinalReleaseJson("public-package-download-proof-owner-execution-pack.json");
        JsonElement executionPack = executionPackDocument.RootElement;
        Assert.Equal("public-package-download-proof-owner-execution-pack", executionPack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-package-download-proof-owner-execution-required", executionPack.GetProperty("packState").GetString());
        Assert.True(executionPack.GetProperty("ownerStepCount").GetInt32() >= 12);
        Assert.True(executionPack.GetProperty("blockedOwnerStepCount").GetInt32() >= 8);
        Assert.True(executionPack.GetProperty("manualCommandCount").GetInt32() >= 10);
        AssertFalseProofPublishCloseFlags(executionPack);

        string executionRaw = executionPack.GetRawText();
        foreach (string marker in new[]
        {
            "Invoke-WebRequest",
            "Get-FileHash -Algorithm SHA256",
            "Export-PublicPackageDownloadProofInputTemplate.ps1",
            "Test-PublicPackageDownloadProofInput.ps1 -Strict",
            "Import-PublicPackageDownloadProofCandidate.ps1",
            "Test-PublicPackageDownloadProofCandidate.ps1 -Strict",
            "Export-PostPublishUserVerificationPack.ps1",
            "Test-PostPublishUserVerificationPack.ps1 -Strict",
            "Export-ReleaseEvidenceBundle.ps1",
            "Test-ReleaseEvidenceClassificationAudit.ps1 -Strict",
            "manual owner guidance only"
        })
        {
            Assert.Contains(marker, executionRaw, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("dotnet nuget push", executionRaw, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("NUGET_API_KEY", executionRaw, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("--api-key", executionRaw, StringComparison.OrdinalIgnoreCase);

        using JsonDocument executionValidationDocument = ReadFinalReleaseJson("public-package-download-proof-owner-execution-pack-validation.json");
        JsonElement executionValidation = executionValidationDocument.RootElement;
        Assert.Equal("public-package-download-proof-owner-execution-pack-validation", executionValidation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-package-download-proof-owner-execution-required", executionValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, executionValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(executionValidation.GetProperty("failedActionRequiredCount").GetInt32() >= 8);
        AssertFalseProofPublishCloseFlags(executionValidation);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("release-docs-and-nuget-metadata-audit-ready-non-proof", evidence.GetProperty("releaseDocsAndNuGetMetadataAuditValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("releaseDocsAndNuGetMetadataAuditYoloDetBlockedMatchCount").GetInt32());
        Assert.Equal("blocked-post-publish-user-verification-required", evidence.GetProperty("postPublishUserVerificationPackValidationState").GetString());
        Assert.True(evidence.GetProperty("postPublishUserVerificationPackBlockedLaneCount").GetInt32() >= 4);
        Assert.Equal("blocked-public-package-download-proof-owner-execution-required", evidence.GetProperty("publicPackageDownloadProofOwnerExecutionPackValidationState").GetString());
        Assert.True(evidence.GetProperty("publicPackageDownloadProofOwnerExecutionPackBlockedStepCount").GetInt32() >= 8);

        AssertBlockedEvidenceItem(evidence, "release-docs-and-nuget-metadata-audit", "not public package download proof");
        AssertBlockedEvidenceItem(evidence, "post-publish-user-verification-pack", "owner action aggregation only");
        AssertBlockedEvidenceItem(evidence, "public-package-download-proof-owner-execution-pack", "manual owner guidance only");

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-docs-and-nuget-metadata-audit-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-user-verification-pack-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-package-download-proof-owner-execution-pack-validation.json", sourceArtifacts);

        using JsonDocument classificationDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement classification = classificationDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", classification.GetProperty("auditState").GetString());
        Assert.Equal(0, classification.GetProperty("findingCount").GetInt32());
        AssertAuditedNonProofItem(classification, "release-docs-and-nuget-metadata-audit");
        AssertAuditedNonProofItem(classification, "post-publish-user-verification-pack");
        AssertAuditedNonProofItem(classification, "public-package-download-proof-owner-execution-pack");
    }

    private static void AssertLane(JsonElement[] lanes, string laneId, bool ready)
    {
        JsonElement lane = lanes.Single(item => item.GetProperty("laneId").GetString() == laneId);
        Assert.Equal(ready, lane.GetProperty("ready").GetBoolean());
        Assert.False(lane.GetProperty("performsPublish").GetBoolean());
        Assert.False(lane.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(lane.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(lane.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(lane.GetProperty("isPostPublishProof").GetBoolean());
    }

    private static void AssertFalseProofPublishCloseFlags(JsonElement element)
    {
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(element);
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
    }

    private static void AssertBlockedEvidenceItem(JsonElement evidence, string id, string boundaryText)
    {
        JsonElement item = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(item => item.GetProperty("id").GetString() == id);

        Assert.False(item.GetProperty("passed").GetBoolean());
        Assert.Contains(boundaryText, item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertAuditedNonProofItem(JsonElement audit, string id)
    {
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(ReadFinalReleaseText(fileName));
    }

    private static string ReadFinalReleaseText(string fileName)
    {
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName));
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}
