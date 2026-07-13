using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCloseStrictProofAndArticlePublishingReadinessTests
{
    [Fact]
    public void ReleaseCloseStrictProofExecutionOrderIncludesAllFinalActionRequiredLanes()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerDualRouteProofPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanExternalConsumerExecutionKit.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationIntakeMap.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublishActionRequiredEvidenceMap.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseStrictProofExecutionOrder.ps1"));

        using JsonDocument document = ReadFinalReleaseJson("release-close-strict-proof-execution-order.json");
        JsonElement root = document.RootElement;

        Assert.Equal("release-close-strict-proof-execution-order", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-execution-required", root.GetProperty("orderState").GetString());
        Assert.Equal(6, root.GetProperty("actionRequiredCount").GetInt32());
        Assert.Equal(6, root.GetProperty("executionStepCount").GetInt32());
        Assert.Equal(6, root.GetProperty("blockedStepCount").GetInt32());
        Assert.True(root.GetProperty("hasPackageConsumerPlanningLinks").GetBoolean());
        Assert.True(root.GetProperty("hasPostPublishIntakeLinks").GetBoolean());
        AssertFalsePublishAndCloseFlags(root);
        Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());

        string[] actionIds = root.GetProperty("actionIds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string expected in RequiredFinalActionIds)
        {
            Assert.Contains(expected, actionIds);
        }

        JsonElement[] steps = root.GetProperty("steps").EnumerateArray().OrderBy(static step => step.GetProperty("order").GetInt32()).ToArray();
        Assert.Equal(Enumerable.Range(1, 6).ToArray(), steps.Select(static step => step.GetProperty("order").GetInt32()).ToArray());

        JsonElement packageStep = steps.Single(static step => step.GetProperty("actionId").GetString() == "package-consumer-runtime-owner-proof-required");
        string[] packageLinks = packageStep.GetProperty("linkedArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/package-consumer-dual-route-proof-plan.json", packageLinks);
        Assert.Contains("artifacts/final-release/clean-external-consumer-execution-kit.json", packageLinks);

        JsonElement postPublishStep = steps.Single(static step => step.GetProperty("actionId").GetString() == "post-publish-verification-owner-proof-required");
        string[] postPublishLinks = postPublishStep.GetProperty("linkedArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/post-publish-verification-intake-map.json", postPublishLinks);

        foreach (JsonElement step in steps)
        {
            Assert.Equal("blocked-owner-action-required", step.GetProperty("stepState").GetString());
            Assert.False(step.GetProperty("performsPublish").GetBoolean());
            Assert.False(step.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(step.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(step.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
            Assert.True(step.GetProperty("failureRepairHints").GetArrayLength() >= 3);
            Assert.True(step.GetProperty("forbiddenSubstitutes").GetArrayLength() >= 5);
        }
    }

    [Fact]
    public void ArticlePublishingReadinessMapCoversRequiredPublicityAreasWithoutProofPromotion()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ArticleRoadmap30Plus.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-TechnicalArticlePublicationMatrix.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseStrictProofExecutionOrder.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ArticlePublishingReadinessMap.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ArticlePublishingReadinessMap.ps1"), "-Strict");

        using JsonDocument mapDocument = ReadFinalReleaseJson("article-publishing-readiness-map.json");
        JsonElement map = mapDocument.RootElement;

        Assert.Equal("article-publishing-readiness-map", map.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required-before-public-publication", map.GetProperty("readinessState").GetString());
        Assert.True(map.GetProperty("roadmapArticleCount").GetInt32() >= 30);
        Assert.True(map.GetProperty("technicalArticleCount").GetInt32() >= 30);
        Assert.Equal(7, map.GetProperty("focusedReadinessArticleCount").GetInt32());
        Assert.Equal(6, map.GetProperty("actionRequiredCount").GetInt32());
        Assert.True(map.GetProperty("strictExecutionStepCount").GetInt32() >= 6);
        AssertFalsePublishAndCloseFlags(map);
        Assert.False(map.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(map.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(map.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(map.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(map.GetProperty("isReleaseCloseProof").GetBoolean());

        string[] areas = map.GetProperty("coveredAreas").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string area in new[] { "YoloVision", "OnnxToEngine", "TensorRtExec", "RuntimePackages", "CleanConsumer", "PostPublish", "ReleaseClose" })
        {
            Assert.Contains(area, areas);
        }

        string raw = map.GetRawText();
        foreach (string marker in new[] { "template is proof", "dashboard is proof", "build report is proof", "ProjectReference is package consumer proof", "direct nupkg is package consumer proof" })
        {
            Assert.Contains(marker, raw, StringComparison.Ordinal);
        }

        JsonElement[] articles = map.GetProperty("articles").EnumerateArray().ToArray();
        foreach (JsonElement article in articles)
        {
            Assert.False(article.GetProperty("performsPublish").GetBoolean());
            Assert.False(article.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(article.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(article.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(article.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(article.GetProperty("isReleaseCloseProof").GetBoolean());
            Assert.True(article.GetProperty("requiredImagesOrTables").GetArrayLength() >= 1);
            Assert.True(article.GetProperty("codePaths").GetArrayLength() >= 1);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("article-publishing-readiness-map-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("article-publishing-readiness-map-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("article-publishing-readiness-map-passed-non-proof-boundaries-intact", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        AssertFalsePublishAndCloseFlags(validation);
    }

    [Fact]
    public void FinalActionMapLinksReleaseCloseAndArticleReadinessSurfaces()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseStrictProofExecutionOrder.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ArticlePublishingReadinessMap.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublishActionRequiredEvidenceMap.ps1"));

        using JsonDocument document = ReadFinalReleaseJson("final-publish-action-required-evidence-map.json");
        JsonElement root = document.RootElement;

        Assert.Equal("final-publish-action-required-evidence-map", root.GetProperty("recordKind").GetString());
        Assert.Equal("artifacts/final-release/release-close-strict-proof-execution-order.json", root.GetProperty("sourceReleaseCloseStrictProofExecutionOrder").GetString());
        Assert.Equal("artifacts/final-release/article-publishing-readiness-map.json", root.GetProperty("sourceArticlePublishingReadinessMap").GetString());
        Assert.True(root.GetProperty("releaseCloseStrictExecutionStepCount").GetInt32() >= 6);
        Assert.Equal(7, root.GetProperty("articlePublishingFocusedReadinessArticleCount").GetInt32());
        Assert.Equal("blocked-real-proof-required-before-public-publication", root.GetProperty("articlePublishingReadinessState").GetString());
        AssertFalsePublishAndCloseFlags(root);
        Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
    }

    [Fact]
    public void PublicDocsGateStillHasNoBlockedPublicationOrProofClaims()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicDocsAndPackageMetadataGate.ps1"));

        using JsonDocument document = ReadFinalReleaseJson("public-docs-package-metadata-gate.json");
        JsonElement root = document.RootElement;

        Assert.Equal("public-docs-package-metadata-gate", root.GetProperty("recordKind").GetString());
        Assert.Equal(0, root.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(0, root.GetProperty("blockedMatchCount").GetInt32());
        AssertFalsePublishAndCloseFlags(root);
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
    }

    private static readonly string[] RequiredFinalActionIds =
    {
        "real-model-runtime-owner-proof-required",
        "package-consumer-runtime-owner-proof-required",
        "post-publish-verification-owner-proof-required",
        "final-owner-real-input-template-pack-owner-input-required",
        "owner-external-proof-result-import-owner-proof-required",
        "owner-result-candidate-bridge-real-proof-required",
    };

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertFalsePublishAndCloseFlags(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}
