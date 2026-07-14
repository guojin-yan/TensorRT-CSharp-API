using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PostPublishDocsAndSamplesFinalLandingPackTests
{
    [Fact]
    public void FinalLandingPackAggregatesDocsSamplesMetadataAndOwnerProofDependenciesWithoutPromotion()
    {
        RunPowerShell("Export-OwnerRealPublishEvidenceAvailabilityLedger.ps1");
        RunPowerShell("Test-OwnerRealPublishEvidenceAvailabilityLedger.ps1", "-Strict");
        RunPowerShell("Export-PostPublishDocsArticleAndSampleAssetPlan.ps1");
        RunPowerShell("Test-PostPublishDocsArticleAndSampleAssetPlan.ps1", "-Strict");
        RunPowerShell("Export-PostPublishDocsAndSamplesFinalLandingPack.ps1");
        RunPowerShell("Test-PostPublishDocsAndSamplesFinalLandingPack.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument packDocument = ReadFinalReleaseJson("post-publish-docs-and-samples-final-landing-pack.json");
        JsonElement pack = packDocument.RootElement;

        Assert.Equal("post-publish-docs-and-samples-final-landing-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-publish-evidence-required-final-landing-ready", pack.GetProperty("landingPackState").GetString());
        Assert.True(pack.GetProperty("landingLaneCount").GetInt32() >= 9);
        Assert.True(pack.GetProperty("blockedLandingLaneCount").GetInt32() >= 8);
        Assert.Equal(0, pack.GetProperty("proofReadyLandingLaneCount").GetInt32());
        Assert.True(pack.GetProperty("yoloVisionSampleReady").GetBoolean());
        Assert.Equal(0, pack.GetProperty("legacyYoloDetReferenceCount").GetInt32());
        AssertFalseProofPublishAndCloseFlags(pack);
        Assert.False(pack.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());

        string[] laneIds = pack.GetProperty("lanes").EnumerateArray()
            .Select(static lane => lane.GetProperty("id").GetString()!)
            .ToArray();

        foreach (string expected in new[]
        {
            "docs-site-final-links",
            "readme-frontpage-links",
            "nuget-metadata-owner-review",
            "release-notes-and-known-limitations",
            "yolovision-quickstart",
            "yolovision-real-asset-proof",
            "external-clean-install-guide",
            "article-case-asset-index",
            "owner-proof-dependency",
        })
        {
            Assert.Contains(expected, laneIds);
        }

        foreach (JsonElement lane in pack.GetProperty("lanes").EnumerateArray())
        {
            Assert.True(lane.GetProperty("blocked").GetBoolean());
            Assert.False(lane.GetProperty("proofReady").GetBoolean());
            Assert.False(lane.GetProperty("performsPublish").GetBoolean());
            Assert.False(lane.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(lane.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(lane.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(lane.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(lane.GetProperty("isReleaseCloseProof").GetBoolean());
        }

        JsonElement packageMetadata = pack.GetProperty("packageMetadata");
        Assert.True(packageMetadata.GetProperty("packageMetadataReady").GetBoolean());
        Assert.Equal("JYPPX.TensorRT.CSharp.API", packageMetadata.GetProperty("packPackageId").GetString());
        Assert.Equal("JYPPX.TensorRT.CSharp.API", packageMetadata.GetProperty("sourcePackageId").GetString());

        string raw = pack.GetRawText();
        foreach (string marker in new[]
        {
            "owner-real-publish-evidence-availability-ledger",
            "post-publish-docs-article-and-sample-asset-plan",
            "release-evidence-bundle.json",
            "strict-close-ready-convergence-dashboard.json",
            "final-public-release-closure-bridge.json",
            "samples/YoloVision/README.md",
            "samples/assets/yolovision-article-case-pack.json",
            "package-consumer-runtime-proof-clean-consumer-guide.md",
            "Owner public publish result",
            "Repository-external clean consumer",
        })
        {
            Assert.Contains(marker, raw, StringComparison.Ordinal);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("post-publish-docs-and-samples-final-landing-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("post-publish-docs-and-samples-final-landing-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("post-publish-docs-and-samples-final-landing-pack-passed-non-proof-boundaries-intact", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("packageMetadataReady").GetBoolean());
        Assert.True(validation.GetProperty("yoloVisionSampleReady").GetBoolean());
        Assert.Equal(0, validation.GetProperty("legacyYoloDetReferenceCount").GetInt32());
        AssertFalseProofPublishAndCloseFlags(validation);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("post-publish-docs-and-samples-final-landing-pack-passed-non-proof-boundaries-intact", evidence.GetProperty("postPublishDocsAndSamplesFinalLandingPackValidationState").GetString());
        Assert.True(evidence.GetProperty("postPublishDocsAndSamplesFinalLandingPackLaneCount").GetInt32() >= 9);
        Assert.True(evidence.GetProperty("postPublishDocsAndSamplesFinalLandingPackBlockedLaneCount").GetInt32() >= 8);
        Assert.Equal(0, evidence.GetProperty("postPublishDocsAndSamplesFinalLandingPackProofReadyLaneCount").GetInt32());
        Assert.True(evidence.GetProperty("postPublishDocsAndSamplesFinalLandingPackYoloVisionSampleReady").GetBoolean());
        Assert.Equal(0, evidence.GetProperty("postPublishDocsAndSamplesFinalLandingPackLegacyYoloDetReferenceCount").GetInt32());
        Assert.False(evidence.GetProperty("postPublishDocsAndSamplesFinalLandingPackCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("postPublishDocsAndSamplesFinalLandingPackCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "post-publish-docs-and-samples-final-landing-pack");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("not post-publish proof", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertFalseProofPublishAndCloseFlags(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());
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
