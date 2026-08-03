using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class EngineeringAndArticleCatalogTests
{
    [Fact]
    public void PublicationCatalogRequiresRealResultsImagesAndModelReproductionWithoutAuthorization()
    {
        string catalogPath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "publication-catalog.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(catalogPath));
        JsonElement root = document.RootElement;

        Assert.Equal("technical-article-publication-catalog", root.GetProperty("recordKind").GetString());
        Assert.Equal("project-documentation-not-publication-ready", root.GetProperty("defaultClassification").GetString());
        Assert.False(root.GetProperty("publicPublicationAuthorized").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());

        JsonElement rule = root.GetProperty("completenessRule");
        Assert.True(rule.GetProperty("requiresRealExecutionResult").GetBoolean());
        Assert.True(rule.GetProperty("requiresImageSourceStatement").GetBoolean());
        Assert.True(rule.GetProperty("modelArticleRequiresAcquisitionAndConversion").GetBoolean());
        Assert.True(rule.GetProperty("requiresProofBoundary").GetBoolean());
        Assert.True(rule.GetProperty("minimumResultImageCount").GetInt32() >= 1);

        JsonElement[] articles = root.GetProperty("articles").EnumerateArray().ToArray();
        Assert.NotEmpty(articles);
        foreach (JsonElement article in articles)
        {
            Assert.Equal("complete-technical-article", article.GetProperty("classification").GetString());
            Assert.Equal("complete-with-runtime-results-and-images", article.GetProperty("status").GetString());
            Assert.False(article.GetProperty("publicationAuthorized").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("proofBoundary").GetString()));

            string articlePath = Resolve(article.GetProperty("path").GetString()!);
            string articleText = File.ReadAllText(articlePath);
            Assert.Contains("## 已验证结果", articleText, StringComparison.Ordinal);
            Assert.Contains("上图由本次真实运行报告生成", articleText, StringComparison.Ordinal);

            string evidencePath = Resolve(article.GetProperty("realExecutionEvidence").GetString()!);
            using JsonDocument evidence = JsonDocument.Parse(File.ReadAllText(evidencePath));
            Assert.Contains(
                evidence.RootElement.GetProperty("proofClassification").GetString(),
                new[] { "real-model-runtime", "local-package-consumer-runtime", "package-consumer-runtime", "post-publish-runtime" });

            JsonElement model = article.GetProperty("model");
            foreach (string propertyName in new[] { "sourceUrl", "pinnedRevision", "license", "conversionCommand", "onnxWorkspacePath", "onnxSha256" })
            {
                Assert.False(string.IsNullOrWhiteSpace(model.GetProperty(propertyName).GetString()));
            }
            Assert.Contains(model.GetProperty("sourceUrl").GetString()!, articleText, StringComparison.Ordinal);
            Assert.Contains("yolo export", articleText, StringComparison.Ordinal);
            Assert.Contains(model.GetProperty("onnxSha256").GetString()!, articleText, StringComparison.Ordinal);
            Assert.False(model.GetProperty("trackedByGit").GetBoolean());
            Assert.False(model.GetProperty("uploadsModelFiles").GetBoolean());

            JsonElement[] images = article.GetProperty("resultImages").EnumerateArray().ToArray();
            Assert.True(images.Length >= rule.GetProperty("minimumResultImageCount").GetInt32());
            foreach (JsonElement image in images)
            {
                string relativePath = image.GetProperty("path").GetString()!;
                string imagePath = Resolve(relativePath);
                Assert.True(File.Exists(imagePath), relativePath);
                Assert.Equal(image.GetProperty("sha256").GetString(), Sha256(imagePath));
                Assert.False(string.IsNullOrWhiteSpace(image.GetProperty("source").GetString()));
                Assert.Contains(Path.GetFileName(relativePath), articleText, StringComparison.Ordinal);
            }
        }
    }

    [Fact]
    public void CompletenessValidatorFailsClosedAndDoesNotPublish()
    {
        string script = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-TechnicalArticleCompleteness.ps1"));
        string articleReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "README.md"));

        foreach (string term in new[]
        {
            "minimumResultImageCount", "result-image-missing", "result-image-sha256-mismatch",
            "result-image-not-referenced-by-article", "missing-result-image-source-statement",
            "real-execution-evidence-missing", "model-contract-not-documented", "outer-model-sha256-mismatch",
            "model-git-or-upload-boundary-invalid", "article-must-not-self-authorize-publication",
            "PublicPublicationAuthorized=False", "performsPublish = $false"
        })
        {
            Assert.Contains(term, script, StringComparison.Ordinal);
        }
        Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release", script, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("complete-technical-article", articleReadme, StringComparison.Ordinal);
        Assert.Contains("project-documentation-not-publication-ready", articleReadme, StringComparison.Ordinal);
        Assert.Contains("执行结果", articleReadme, StringComparison.Ordinal);
        Assert.Contains("配图", articleReadme, StringComparison.Ordinal);
    }

    [Fact]
    public void EngineeringReadmeSeparatesSupportedEntrypointsFromInternalEvidencePipelines()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "README.md"));
        foreach (string entrypoint in new[]
        {
            "Invoke-LocalReleaseBundle.ps1",
            "Test-RuntimePackageReadiness.ps1",
            "Sync-DemoOnnxModels.ps1",
            "Acquire-YoloV8DetectionOfficialAssets.ps1",
            "Test-YoloVisionDetectionLocalPackageConsumer.ps1",
            "Test-YoloVisionLocalPackageConsumer.ps1",
            "Test-TechnicalArticleCompleteness.ps1"
        })
        {
            Assert.Contains(entrypoint, readme, StringComparison.Ordinal);
            Assert.True(File.Exists(Path.Combine(RepositoryPaths.Root, "eng", entrypoint)), entrypoint);
        }
        Assert.Contains("不是面向最终用户的命令集合", readme, StringComparison.Ordinal);
        Assert.Contains("内部工程脚本", readme, StringComparison.Ordinal);
        Assert.Contains("不能仅凭“没有字面引用”判定无用", readme, StringComparison.Ordinal);
        Assert.Contains("不会因为“质量门通过”而自动获得", readme, StringComparison.Ordinal);
    }

    private static string Resolve(string relativePath) =>
        Path.Combine(RepositoryPaths.Root, relativePath.Replace('/', Path.DirectorySeparatorChar));

    private static string Sha256(string path) =>
        Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
}
