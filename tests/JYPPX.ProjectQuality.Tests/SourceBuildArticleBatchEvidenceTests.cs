using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class SourceBuildArticleBatchEvidenceTests
{
    private static readonly string[] ReadyArticleIds = ["BLD-001", "BLD-002", "BLD-003", "BLD-004"];
    private static readonly string[] ReviewArticleIds = ["INS-001", "INS-002", "INS-003", "INS-004"];

    [Fact]
    public void EvidencePinsBuildGenerationPackagingAndNegativeBoundaries()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(EvidencePath()));
        JsonElement root = document.RootElement;

        Assert.Equal("source-build-article-batch-evidence", root.GetProperty("recordKind").GetString());
        Assert.Matches("^[a-f0-9]{40}$", root.GetProperty("baseCommit").GetString()!);

        JsonElement source = root.GetProperty("sourceState");
        Assert.Equal("base-commit-plus-local-tracked-source-changes", source.GetProperty("classification").GetString());
        Assert.False(source.GetProperty("cleanBaseCommitBuildProof").GetBoolean());
        Assert.Matches("^[a-f0-9]{64}$", source.GetProperty("sourceStateFingerprintSha256").GetString()!);

        JsonElement baseline = root.GetProperty("cleanBaselineComparison");
        Assert.True(baseline.GetProperty("bindingGeneratorValidationPassed").GetBoolean());
        Assert.False(baseline.GetProperty("solutionBuildPassed").GetBoolean());
        Assert.False(baseline.GetProperty("winX64DevBuildPassed").GetBoolean());

        JsonElement managed = root.GetProperty("managedBuild");
        Assert.True(managed.GetProperty("releaseSolutionBuildPassed").GetBoolean());
        Assert.Equal(0, managed.GetProperty("warningCount").GetInt32());
        Assert.Equal(0, managed.GetProperty("errorCount").GetInt32());
        Assert.True(managed.GetProperty("cleanConsumer").GetProperty("buildPassed").GetBoolean());
        Assert.False(managed.GetProperty("cleanConsumer").GetProperty("publicPackageProof").GetBoolean());

        JsonElement generation = root.GetProperty("bindingGeneration");
        Assert.Equal(214, generation.GetProperty("manifestCount").GetInt32());
        Assert.Equal(4046, generation.GetProperty("apiRecordCount").GetInt32());
        Assert.True(generation.GetProperty("deterministicDoubleGenerationPassed").GetBoolean());

        JsonElement native = root.GetProperty("nativeBuild");
        Assert.Equal("win-x64-trt10-cuda12-release", native.GetProperty("preset").GetString());
        Assert.True(native.GetProperty("buildPassed").GetBoolean());
        Assert.False(native.GetProperty("linuxBuildProof").GetBoolean());
        Assert.False(native.GetProperty("gpuInferenceProof").GetBoolean());

        JsonElement split = root.GetProperty("splitBridgePackage");
        Assert.True(split.GetProperty("packageConsumerValidationSucceeded").GetBoolean());
        Assert.True(split.GetProperty("environmentProbeSucceeded").GetBoolean());
        Assert.False(split.GetProperty("runtimeExecutionProof").GetBoolean());
        Assert.False(split.GetProperty("publicPackageProof").GetBoolean());
        Assert.False(split.GetProperty("postPublishProof").GetBoolean());

        JsonElement finalQuality = root.GetProperty("qualityValidation").GetProperty("currentWorkingTreeFinal");
        Assert.Equal(2230, finalQuality.GetProperty("total").GetInt32());
        Assert.Equal(2230, finalQuality.GetProperty("passed").GetInt32());
        Assert.Equal(0, finalQuality.GetProperty("failed").GetInt32());
        Assert.Equal(0, finalQuality.GetProperty("skipped").GetInt32());
        Assert.Matches("^[a-f0-9]{64}$", finalQuality.GetProperty("trxSha256").GetString()!);
    }

    [Fact]
    public void CanonicalIndexAndArticleHeadersMatchEvidenceDecisions()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root, "docs", "articles", "zh-cn", "article-index.json")));
        Dictionary<string, JsonElement> articles = document.RootElement.GetProperty("articles")
            .EnumerateArray()
            .ToDictionary(article => article.GetProperty("id").GetString()!, article => article);

        Assert.Equal(46, articles.Values.Count(article => article.GetProperty("status").GetString() == "ready"));
        Assert.Equal(4, articles.Values.Count(article => article.GetProperty("status").GetString() == "review"));

        foreach (string id in ReadyArticleIds)
        {
            Assert.Equal("ready", articles[id].GetProperty("status").GetString());
            Assert.Contains(
                $"文章编号：{id}；适用版本：4.0.0；当前状态：ready。",
                ReadArticle(articles[id]),
                StringComparison.Ordinal);
        }

        foreach (string id in ReviewArticleIds)
        {
            Assert.Equal("review", articles[id].GetProperty("status").GetString());
            string article = ReadArticle(articles[id]);
            Assert.Contains(
                $"文章编号：{id}；适用版本：4.0.0；当前状态：review。",
                article,
                StringComparison.Ordinal);
            Assert.Contains("review", article, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void EvidenceIsIncludedInDocfxAndDoesNotClaimPublication()
    {
        string docfx = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "docfx.json"));
        Assert.Contains("articles/zh-cn/06-source-build/source-build-evidence-20260813.json", docfx, StringComparison.Ordinal);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(EvidencePath()));
        JsonElement boundary = document.RootElement.GetProperty("proofBoundary");
        Assert.False(boundary.GetProperty("performsPublish").GetBoolean());
        Assert.False(boundary.GetProperty("releaseClosureProof").GetBoolean());
        Assert.False(boundary.GetProperty("linuxGpuRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("wslGpuRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("containerTensorRtRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("gpuCiRuntimeProof").GetBoolean());
    }

    private static string EvidencePath() => Path.Combine(
        RepositoryPaths.Root,
        "docs",
        "articles",
        "zh-cn",
        "06-source-build",
        "source-build-evidence-20260813.json");

    private static string ReadArticle(JsonElement article)
    {
        string relativePath = article.GetProperty("sourcePath").GetString()!;
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            relativePath.Replace('/', Path.DirectorySeparatorChar)));
    }
}
