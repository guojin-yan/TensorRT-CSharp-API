using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PostPublishPublicDocsArticleSampleReadinessPackTests
{
    private static readonly string[] RequiredSurfaceIds =
    {
        "onnx-to-engine",
        "classification",
        "dynamic-shape",
        "inference-bindings",
        "multi-stream",
        "yolovision",
        "tensorrt-exec",
        "plugin-registry-inventory-smoke",
    };

    private static readonly string[] RequiredYoloFamilies =
    {
        "yolov5",
        "yolov6",
        "yolov7",
        "yolov8",
        "yolov9",
        "yolov10",
        "yolov11",
        "yolov26",
        "yolox",
        "custom",
    };

    private static readonly string[] RequiredYoloTasks =
    {
        "det",
        "cls",
        "seg",
        "obb",
        "pose",
        "sem",
    };

    private static readonly string[] NewEvidenceIds =
    {
        "post-publish-public-docs-article-sample-readiness-pack",
        "public-claim-post-publish-proof-boundary-audit",
        "owner-post-publish-docs-article-sample-execution-pack",
    };

    [Fact]
    public void ReadinessPackCoversPublicSurfacesAndYoloVisionWithoutProofPromotion()
    {
        RunPostPublishDocsArticleSamplePipeline();

        using JsonDocument packDocument = ReadFinalReleaseJson("post-publish-public-docs-article-sample-readiness-pack.json");
        JsonElement pack = packDocument.RootElement;
        Assert.Equal("post-publish-public-docs-article-sample-readiness-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-publish-and-post-publish-proof-required", pack.GetProperty("readinessState").GetString());
        Assert.Equal(42, pack.GetProperty("articleCount").GetInt32());
        Assert.Equal(8, pack.GetProperty("sampleSurfaceCount").GetInt32());
        Assert.Equal(8, pack.GetProperty("readySampleSurfaceCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("missingSamplePathCount").GetInt32());
        Assert.True(pack.GetProperty("ownerActionFieldCount").GetInt32() >= 15);
        AssertFalsePublishCloseProofFlags(pack);

        string[] surfaceIds = pack.GetProperty("sampleSurfaces").EnumerateArray()
            .Select(static surface => surface.GetProperty("id").GetString()!)
            .ToArray();
        foreach (string required in RequiredSurfaceIds)
        {
            Assert.Contains(required, surfaceIds);
        }

        foreach (JsonElement surface in pack.GetProperty("sampleSurfaces").EnumerateArray())
        {
            Assert.True(surface.GetProperty("sourceReady").GetBoolean());
            Assert.Equal(0, surface.GetProperty("missingPathCount").GetInt32());
            Assert.Contains("blocked-owner", surface.GetProperty("runtimeProofStatus").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("blocked-owner", surface.GetProperty("postPublishProofStatus").GetString(), StringComparison.OrdinalIgnoreCase);
        }

        JsonElement yoloVision = pack.GetProperty("yoloVision");
        Assert.Equal("samples/YoloVision", yoloVision.GetProperty("samplePath").GetString());
        Assert.False(yoloVision.GetProperty("legacyYoloDetPathExists").GetBoolean());
        Assert.Equal(0, yoloVision.GetProperty("legacyYoloDetPublicPathMatchCount").GetInt32());
        Assert.False(Directory.Exists(Path.Combine(RepositoryPaths.Root, "samples", "YoloDet")));
        Assert.False(File.Exists(Path.Combine(RepositoryPaths.Root, "samples", "YoloDet", "YoloDet.csproj")));

        string[] families = yoloVision.GetProperty("families").EnumerateArray()
            .Select(static family => family.GetString()!)
            .ToArray();
        string[] tasks = yoloVision.GetProperty("tasks").EnumerateArray()
            .Select(static task => task.GetString()!)
            .ToArray();
        foreach (string required in RequiredYoloFamilies)
        {
            Assert.Contains(required, families);
        }

        foreach (string required in RequiredYoloTasks)
        {
            Assert.Contains(required, tasks);
        }

        Assert.True(pack.GetProperty("onnxToEngine").GetProperty("matrixEntryCount").GetInt32() >= 20);
        Assert.True(pack.GetProperty("tensorRtExec").GetProperty("featureCount").GetInt32() >= 15);

        using JsonDocument validationDocument = ReadFinalReleaseJson("post-publish-public-docs-article-sample-readiness-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("post-publish-public-docs-article-sample-readiness-pack-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(42, validation.GetProperty("articleCount").GetInt32());
        Assert.Equal(8, validation.GetProperty("sampleSurfaceCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("missingSamplePathCount").GetInt32());
        Assert.Equal(RequiredYoloFamilies.Length, validation.GetProperty("yoloFamilyCount").GetInt32());
        Assert.Equal(RequiredYoloTasks.Length, validation.GetProperty("yoloTaskCount").GetInt32());
        AssertFalsePublishCloseProofFlags(validation);
    }

    [Fact]
    public void ClaimAuditAndOwnerExecutionPackRemainBlockedOwnerInputSurfaces()
    {
        RunPostPublishDocsArticleSamplePipeline();

        using JsonDocument auditDocument = ReadFinalReleaseJson("public-claim-post-publish-proof-boundary-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("public-claim-post-publish-proof-boundary-audit", audit.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-proof-required-claim-boundary-audit-ready", audit.GetProperty("auditState").GetString());
        Assert.True(audit.GetProperty("scannedFileCount").GetInt32() >= 20);
        Assert.True(audit.GetProperty("claimRuleCount").GetInt32() >= 8);
        Assert.True(audit.GetProperty("claimCount").GetInt32() > 0);
        Assert.True(audit.GetProperty("ownerProofReviewClaimCount").GetInt32() > 0);
        Assert.Equal(0, audit.GetProperty("disallowedPostPublishProofClaimCount").GetInt32());
        AssertFalsePublishCloseProofFlags(audit);

        using JsonDocument auditValidationDocument = ReadFinalReleaseJson("public-claim-post-publish-proof-boundary-audit-validation.json");
        JsonElement auditValidation = auditValidationDocument.RootElement;
        Assert.Equal("public-claim-post-publish-proof-boundary-audit-ready-non-proof", auditValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, auditValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(0, auditValidation.GetProperty("disallowedPostPublishProofClaimCount").GetInt32());
        AssertFalsePublishCloseProofFlags(auditValidation);

        using JsonDocument ownerPackDocument = ReadFinalReleaseJson("owner-post-publish-docs-article-sample-execution-pack.json");
        JsonElement ownerPack = ownerPackDocument.RootElement;
        Assert.Equal("owner-post-publish-docs-article-sample-execution-pack", ownerPack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-post-publish-docs-article-sample-real-input-required", ownerPack.GetProperty("executionPackState").GetString());
        Assert.Equal(5, ownerPack.GetProperty("laneCount").GetInt32());
        Assert.Equal(5, ownerPack.GetProperty("blockedLaneCount").GetInt32());
        Assert.Equal(0, ownerPack.GetProperty("proofReadyLaneCount").GetInt32());
        Assert.True(ownerPack.GetProperty("requiredFieldCount").GetInt32() >= 37);
        Assert.True(ownerPack.GetProperty("copyableCommandCount").GetInt32() >= 9);
        Assert.True(ownerPack.GetProperty("forbiddenSubstituteCount").GetInt32() >= 50);
        AssertFalsePublishCloseProofFlags(ownerPack);

        string[] laneIds = ownerPack.GetProperty("lanes").EnumerateArray()
            .Select(static lane => lane.GetProperty("id").GetString()!)
            .ToArray();
        foreach (string expected in new[]
        {
            "public-package-urls-and-hashes",
            "external-clean-consumer-logs",
            "yolovision-real-model-assets",
            "article-publication-urls",
            "release-issue-close-material",
        })
        {
            Assert.Contains(expected, laneIds);
        }

        foreach (JsonElement lane in ownerPack.GetProperty("lanes").EnumerateArray())
        {
            Assert.Equal("blocked-owner-real-input-required", lane.GetProperty("laneState").GetString());
            Assert.False(lane.GetProperty("proofReady").GetBoolean());
            AssertFalsePublishCloseProofFlags(lane);
        }

        using JsonDocument ownerValidationDocument = ReadFinalReleaseJson("owner-post-publish-docs-article-sample-execution-pack-validation.json");
        JsonElement ownerValidation = ownerValidationDocument.RootElement;
        Assert.Equal("owner-post-publish-docs-article-sample-execution-pack-ready-non-proof", ownerValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, ownerValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(5, ownerValidation.GetProperty("blockedLaneCount").GetInt32());
        Assert.Equal(0, ownerValidation.GetProperty("proofReadyLaneCount").GetInt32());
        Assert.True(ownerValidation.GetProperty("requiredFieldCount").GetInt32() >= 37);
        AssertFalsePublishCloseProofFlags(ownerValidation);
    }

    [Fact]
    public void EvidenceBundleAndClassificationAuditCarryNewNonProofBoundaries()
    {
        RunPostPublishDocsArticleSamplePipeline();
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("post-publish-public-docs-article-sample-readiness-pack-ready-non-proof", evidence.GetProperty("postPublishPublicDocsArticleSampleReadinessPackValidationState").GetString());
        Assert.Equal(8, evidence.GetProperty("postPublishPublicDocsArticleSampleReadinessPackSampleSurfaceCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("postPublishPublicDocsArticleSampleReadinessPackMissingSamplePathCount").GetInt32());
        Assert.Equal("public-claim-post-publish-proof-boundary-audit-ready-non-proof", evidence.GetProperty("publicClaimPostPublishProofBoundaryAuditValidationState").GetString());
        Assert.True(evidence.GetProperty("publicClaimPostPublishProofBoundaryAuditClaimCount").GetInt32() > 0);
        Assert.Equal(0, evidence.GetProperty("publicClaimPostPublishProofBoundaryAuditDisallowedPostPublishProofClaimCount").GetInt32());
        Assert.Equal("owner-post-publish-docs-article-sample-execution-pack-ready-non-proof", evidence.GetProperty("ownerPostPublishDocsArticleSampleExecutionPackValidationState").GetString());
        Assert.Equal(5, evidence.GetProperty("ownerPostPublishDocsArticleSampleExecutionPackLaneCount").GetInt32());
        Assert.True(evidence.GetProperty("ownerPostPublishDocsArticleSampleExecutionPackRequiredFieldCount").GetInt32() >= 37);
        Assert.False(evidence.GetProperty("postPublishPublicDocsArticleSampleReadinessPackCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("publicClaimPostPublishProofBoundaryAuditCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("ownerPostPublishDocsArticleSampleExecutionPackCanCloseReleaseIssue").GetBoolean());

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string fileName in new[]
        {
            "post-publish-public-docs-article-sample-readiness-pack.json",
            "post-publish-public-docs-article-sample-readiness-pack-validation.json",
            "public-claim-post-publish-proof-boundary-audit.json",
            "public-claim-post-publish-proof-boundary-audit-validation.json",
            "owner-post-publish-docs-article-sample-execution-pack.json",
            "owner-post-publish-docs-article-sample-execution-pack-validation.json",
        })
        {
            Assert.Contains($"artifacts/final-release/{fileName}", sourceArtifacts);
        }

        string[] nonSubstituteMarkers = evidence.GetProperty("nonSubstituteProofKinds").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("post-publish public docs article sample readiness pack", nonSubstituteMarkers);
        Assert.Contains("public claim post-publish proof boundary audit", nonSubstituteMarkers);
        Assert.Contains("owner post-publish docs article sample execution pack", nonSubstituteMarkers);

        foreach (string id in NewEvidenceIds)
        {
            JsonElement item = evidence.GetProperty("evidenceItems").EnumerateArray()
                .Single(evidenceItem => evidenceItem.GetProperty("id").GetString() == id);
            Assert.False(item.GetProperty("passed").GetBoolean());
            string boundary = item.GetProperty("boundary").GetString()!;
            Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument classificationDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement classification = classificationDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", classification.GetProperty("auditState").GetString());
        Assert.Equal(0, classification.GetProperty("findingCount").GetInt32());
        foreach (string id in NewEvidenceIds)
        {
            Assert.Contains(
                classification.GetProperty("auditedItems").EnumerateArray(),
                item => item.GetProperty("id").GetString() == id
                    && !item.GetProperty("passed").GetBoolean()
                    && item.GetProperty("hasNonProofBoundary").GetBoolean());
        }
    }

    private static void RunPostPublishDocsArticleSamplePipeline()
    {
        RunPowerShell("Export-PostPublishPublicDocsArticleSampleReadinessPack.ps1");
        RunPowerShell("Test-PostPublishPublicDocsArticleSampleReadinessPack.ps1", "-Strict");
        RunPowerShell("Export-PublicClaimPostPublishProofBoundaryAudit.ps1");
        RunPowerShell("Test-PublicClaimPostPublishProofBoundaryAudit.ps1", "-Strict");
        RunPowerShell("Export-OwnerPostPublishDocsArticleSampleExecutionPack.ps1");
        RunPowerShell("Test-OwnerPostPublishDocsArticleSampleExecutionPack.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertFalsePublishCloseProofFlags(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());
        if (root.TryGetProperty("canPromoteRuntimeProof", out JsonElement canPromoteRuntimeProof))
        {
            Assert.False(canPromoteRuntimeProof.GetBoolean());
        }
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
    {
        using Process process = new()
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "pwsh",
                WorkingDirectory = RepositoryPaths.Root,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            },
        };

        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}
