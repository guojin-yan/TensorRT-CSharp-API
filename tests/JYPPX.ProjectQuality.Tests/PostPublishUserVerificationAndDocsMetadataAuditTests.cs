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
