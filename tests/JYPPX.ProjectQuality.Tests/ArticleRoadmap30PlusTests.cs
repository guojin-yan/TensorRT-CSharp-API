using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ArticleRoadmap30PlusTests
{
    [Fact]
    public void ArticleRoadmapHasRequiredFieldsAndStaysNonProof()
    {
        RunPowerShell("Export-ArticleRoadmap30Plus.ps1");
        RunPowerShell("Test-ArticleRoadmap30Plus.ps1", "-Strict");

        string roadmapPath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "publishing", "article-roadmap-30plus.json");
        using JsonDocument roadmapDocument = JsonDocument.Parse(File.ReadAllText(roadmapPath));
        JsonElement roadmap = roadmapDocument.RootElement;

        Assert.Equal("article-roadmap-30plus", roadmap.GetProperty("roadmapId").GetString());
        Assert.Equal("release-readiness-planning", roadmap.GetProperty("roadmapState").GetString());
        Assert.True(roadmap.GetProperty("articleCount").GetInt32() >= 30);
        AssertFlagsStayNonProof(roadmap);
        AssertBoundary(roadmap.GetProperty("boundary").GetString()!);

        foreach (JsonElement article in roadmap.GetProperty("articles").EnumerateArray())
        {
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("title").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("audience").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("status").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("targetPath").GetString()));
            Assert.True(article.GetProperty("sourceArtifacts").GetArrayLength() >= 1);
            Assert.True(article.GetProperty("mustAvoidClaims").GetArrayLength() >= 1);
            AssertBoundary(article.GetProperty("proofBoundary").GetString()!);
        }

        JsonElement yoloV10Article = roadmap.GetProperty("articles").EnumerateArray()
            .Single(static article => article.GetProperty("id").GetInt32() == 44);
        Assert.Equal("ready", yoloV10Article.GetProperty("status").GetString());
        Assert.Contains("YoloEndToEndOutput.cs", yoloV10Article.GetProperty("sourceArtifacts").GetRawText(), StringComparison.Ordinal);
        Assert.Equal(
            "docs/articles/zh-cn/yolovision-yolov10-end-to-end-output-guide.md",
            yoloV10Article.GetProperty("targetPath").GetString());

        using JsonDocument validationDocument = ReadFinalReleaseJson("article-roadmap-30plus-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("article-roadmap-30plus-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("article-roadmap-30plus-validation-passed-non-proof-planning", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("articleCount").GetInt32() >= 30);
        Assert.Equal(0, validation.GetProperty("findingCount").GetInt32());
        AssertFlagsStayNonProof(validation);
        AssertBoundary(validation.GetProperty("boundary").GetString()!);
    }

    [Fact]
    public void EvidenceBundleAndClassificationAuditCarryRoadmapAsNonProof()
    {
        FinalQualityFreezeDashboardTests.RunPipeline();
        RunPowerShell("Test-PublicProofClaimBoundaryAudit.ps1", "-Strict");
        RunPowerShell("Export-ArticleRoadmap30Plus.ps1");
        RunPowerShell("Test-ArticleRoadmap30Plus.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("article-roadmap-30plus-validation-passed-non-proof-planning", evidence.GetProperty("articleRoadmap30PlusValidationState").GetString());
        Assert.True(evidence.GetProperty("articleRoadmap30PlusArticleCount").GetInt32() >= 30);
        Assert.Equal(0, evidence.GetProperty("articleRoadmap30PlusFindingCount").GetInt32());
        Assert.False(evidence.GetProperty("articleRoadmap30PlusCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("articleRoadmap30PlusCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("articleRoadmap30PlusIsRuntimeExecutionProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "article-roadmap-30plus");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        AssertBoundary(evidenceItem.GetProperty("boundary").GetString()!);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("docs/articles/zh-cn/publishing/article-roadmap-30plus.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/article-roadmap-30plus-validation.json", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "article-roadmap-30plus" &&
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
