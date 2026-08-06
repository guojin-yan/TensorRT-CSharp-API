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

        Assert.Equal(2, root.GetProperty("schemaVersion").GetInt32());
        Assert.Equal("technical-article-publication-catalog", root.GetProperty("recordKind").GetString());
        Assert.Equal("project-documentation-not-publication-ready", root.GetProperty("defaultClassification").GetString());
        Assert.False(root.GetProperty("publicPublicationAuthorized").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());

        JsonElement rule = root.GetProperty("completenessRule");
        Assert.True(rule.GetProperty("requiresRealExecutionResult").GetBoolean());
        Assert.True(rule.GetProperty("requiresImageSourceStatement").GetBoolean());
        Assert.True(rule.GetProperty("modelArticleRequiresAcquisitionAndConversion").GetBoolean());
        Assert.True(rule.GetProperty("requiresProofBoundary").GetBoolean());
        Assert.True(rule.GetProperty("requiresProjectAndLibraryIntroduction").GetBoolean());
        Assert.True(rule.GetProperty("requiresStepByStepProjectFlow").GetBoolean());
        Assert.True(rule.GetProperty("minimumResultImageCount").GetInt32() >= 2);
        Assert.Equal(0, rule.GetProperty("maximumMachineSpecificAbsolutePathCount").GetInt32());
        string[] requiredImageRoles = rule.GetProperty("requiredResultImageRoles")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Equal(new[] { "annotated-inference-result", "program-runtime-screenshot" }, requiredImageRoles);

        JsonElement[] articles = root.GetProperty("articles").EnumerateArray().ToArray();
        Assert.NotEmpty(articles);
        string articleReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "README.md"));
        int markdownCount = Directory.GetFiles(
            Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn"),
            "*.md",
            SearchOption.AllDirectories).Length;
        Assert.Contains($"盘点到 {markdownCount} 个 Markdown 文件", articleReadme, StringComparison.Ordinal);
        Assert.Contains($"严格目录 {articles.Length}/{articles.Length} 通过", articleReadme, StringComparison.Ordinal);
        foreach (JsonElement article in articles)
        {
            Assert.Equal("complete-technical-article", article.GetProperty("classification").GetString());
            Assert.Equal("complete-with-runtime-screenshot-and-annotated-result", article.GetProperty("status").GetString());
            Assert.False(article.GetProperty("publicationAuthorized").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("proofBoundary").GetString()));

            string articlePath = Resolve(article.GetProperty("path").GetString()!);
            string articleText = File.ReadAllText(articlePath);
            string[][] requiredHeadingGroups =
            {
                new[] { "## 本文使用的项目与库" },
                new[] { "## 模型获取与许可证" },
                new[] { "## ONNX 转换与暂存" },
                new[] { "## 使用公开包准备应用", "## 使用公开包准备案例", "## 创建本地包消费项目" },
                new[] { "## 编写程序入口" },
                new[] { "## 编译并运行" },
                new[] { "## 已验证结果" },
                new[] { "## 复查与边界" },
            };
            foreach (string[] headingGroup in requiredHeadingGroups)
            {
                Assert.Contains(headingGroup, heading => articleText.Contains(heading, StringComparison.Ordinal));
            }
            Assert.Contains("终端截图来自", articleText, StringComparison.Ordinal);
            Assert.Contains("真实运行", articleText, StringComparison.Ordinal);
            Assert.Contains("stdout", articleText, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("真实 TensorRT", articleText, StringComparison.OrdinalIgnoreCase);
            Assert.True(
                articleText.Contains("两张图都来自同一次", StringComparison.Ordinal) ||
                articleText.Contains("同次运行输出", StringComparison.Ordinal) ||
                articleText.Contains("结果图使用同一个", StringComparison.Ordinal));
            Assert.Empty(Regex.Matches(articleText, @"(?im)(?:[A-Z]:\\|/Users/[^/\s]+/|/home/[^/\s]+/)"));

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
            Assert.Contains(model.GetProperty("conversionCommand").GetString()!, articleText, StringComparison.Ordinal);
            Assert.Contains(model.GetProperty("onnxSha256").GetString()!, articleText, StringComparison.Ordinal);
            Assert.False(model.GetProperty("trackedByGit").GetBoolean());
            Assert.False(model.GetProperty("uploadsModelFiles").GetBoolean());

            JsonElement[] images = article.GetProperty("resultImages").EnumerateArray().ToArray();
            Assert.True(images.Length >= rule.GetProperty("minimumResultImageCount").GetInt32());
            Assert.Equal(
                requiredImageRoles.OrderBy(static item => item, StringComparer.Ordinal),
                images.Select(static image => image.GetProperty("role").GetString()!).OrderBy(static item => item, StringComparer.Ordinal));
            foreach (JsonElement image in images)
            {
                string relativePath = image.GetProperty("path").GetString()!;
                string imagePath = Resolve(relativePath);
                Assert.True(File.Exists(imagePath), relativePath);
                Assert.Equal(image.GetProperty("sha256").GetString(), Sha256(imagePath));
                Assert.False(string.IsNullOrWhiteSpace(image.GetProperty("source").GetString()));
                Assert.Contains(Path.GetFileName(relativePath), articleText, StringComparison.Ordinal);
            }

            string visualAssetEvidencePath = Resolve(article.GetProperty("visualAssetEvidence").GetString()!);
            using JsonDocument visualAssetEvidence = JsonDocument.Parse(File.ReadAllText(visualAssetEvidencePath));
            Assert.Equal("technical-article-visual-assets", visualAssetEvidence.RootElement.GetProperty("recordKind").GetString());
            Assert.Equal(article.GetProperty("id").GetString(), visualAssetEvidence.RootElement.GetProperty("articleId").GetString());
            Assert.True(visualAssetEvidence.RootElement.GetProperty("sourceImage").GetProperty("publicRedistributionPermittedByLicense").GetBoolean());
            Assert.False(visualAssetEvidence.RootElement.GetProperty("modelFilesTrackedByGit").GetBoolean());
            Assert.False(visualAssetEvidence.RootElement.GetProperty("uploadsModelFiles").GetBoolean());
            Assert.False(visualAssetEvidence.RootElement.GetProperty("performsPublish").GetBoolean());
        }
    }

    [Fact]
    public void CompletenessValidatorFailsClosedAndDoesNotPublish()
    {
        string script = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-TechnicalArticleCompleteness.ps1"));
        string articleReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "README.md"));

        foreach (string term in new[]
        {
            "minimumResultImageCount", "required-result-image-role-missing", "result-image-role-not-unique",
            "result-image-missing", "result-image-sha256-mismatch", "result-image-not-referenced-by-article",
            "missing-runtime-and-annotated-image-source-statement", "too-many-machine-specific-absolute-paths",
            "visual-asset-evidence-missing", "visual-asset-evidence-image-mismatch",
            "visual-source-public-redistribution-not-permitted", "visual-asset-evidence-boundary-invalid",
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
        Assert.Contains("原图叠加识别结果", articleReadme, StringComparison.Ordinal);
        Assert.Contains("真实终端或软件运行页面截图", articleReadme, StringComparison.Ordinal);
    }

    [Fact]
    public void EngineeringReadmeSeparatesSupportedEntrypointsFromInternalEvidencePipelines()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "README.md"));
        int powershellScriptCount = Directory.GetFiles(
            Path.Combine(RepositoryPaths.Root, "eng"),
            "*.ps1",
            SearchOption.TopDirectoryOnly).Length;
        Assert.Contains($"保留 {powershellScriptCount} 个 PowerShell 脚本", readme, StringComparison.Ordinal);
        foreach (string entrypoint in new[]
        {
            "Invoke-LocalReleaseBundle.ps1",
            "Invoke-WindowsBridgePackageMatrix.ps1",
            "Test-RuntimePackageReadiness.ps1",
            "Sync-DemoOnnxModels.ps1",
            "Acquire-YoloV8DetectionOfficialAssets.ps1",
            "Test-TechnicalArticleCompleteness.ps1"
        })
        {
            Assert.Contains(entrypoint, readme, StringComparison.Ordinal);
            Assert.True(File.Exists(Path.Combine(RepositoryPaths.Root, "eng", entrypoint)), entrypoint);
        }
        Assert.Contains("不是面向最终用户的命令集合", readme, StringComparison.Ordinal);
        Assert.Contains("内部工程脚本", readme, StringComparison.Ordinal);
        Assert.Contains("已退役的案例包工具", readme, StringComparison.Ordinal);
        Assert.Contains("Test-YoloVisionLocalPackageConsumer.ps1", readme, StringComparison.Ordinal);
        Assert.Contains("不能仅凭“没有字面引用”判定无用", readme, StringComparison.Ordinal);
        Assert.Contains("不会因为“质量门通过”而自动获得", readme, StringComparison.Ordinal);
    }

    private static string Resolve(string relativePath) =>
        Path.Combine(RepositoryPaths.Root, relativePath.Replace('/', Path.DirectorySeparatorChar));

    private static string Sha256(string path) =>
        Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
}
