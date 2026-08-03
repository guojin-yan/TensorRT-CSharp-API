using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TechnicalArticleClosureLedgerTests
{
    [Fact]
    public void ExporterProducesDeterministicFullRoadmapClosureLedger()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-TechnicalArticleClosureLedger.ps1");
        string output = RunPowerShell(script);
        Assert.Contains("ArticleCount=103 Supplemental=1", output, StringComparison.Ordinal);
        Assert.Contains("PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False", output, StringComparison.Ordinal);

        string jsonPath = Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "technical-article-closure-ledger.json");
        string markdownPath = Path.ChangeExtension(jsonPath, ".md");
        string firstJsonSha256 = Sha256(jsonPath);
        string firstMarkdownSha256 = Sha256(markdownPath);

        RunPowerShell(script);
        Assert.Equal(firstJsonSha256, Sha256(jsonPath));
        Assert.Equal(firstMarkdownSha256, Sha256(markdownPath));

        JsonElement root = ReadJsonRoot(jsonPath);
        Assert.Equal("technical-article-closure-ledger", root.GetProperty("recordKind").GetString());
        Assert.Equal("content-closure-audited-release-frozen", root.GetProperty("ledgerState").GetString());
        Assert.Equal(103, root.GetProperty("articleCount").GetInt32());
        Assert.Equal(103, root.GetProperty("expectedArticleCount").GetInt32());
        Assert.Equal(1, root.GetProperty("supplementalArticleCount").GetInt32());
        Assert.Equal("7.1", root.GetProperty("supplementalArticles")[0].GetProperty("articleId").GetString());
        Assert.Equal(0, root.GetProperty("missingArticleIds").GetArrayLength());
        Assert.Equal(0, root.GetProperty("duplicateArticleIds").GetArrayLength());
        Assert.Equal(10, root.GetProperty("canonicalMappingCount").GetInt32());
        Assert.Equal(0, root.GetProperty("targetForbiddenMarkerCount").GetInt32());
        Assert.True(root.GetProperty("contentAndProofStateAreIndependent").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.Equal(64, root.GetProperty("sourceRoadmapSha256").GetString()!.Length);

        string boundary = root.GetProperty("proofBoundary").GetString()!;
        foreach (string marker in new[]
        {
            "not runtime proof",
            "not post-publish proof",
            "not publish approval",
            "not package push",
            "not release close approval"
        })
        {
            Assert.Contains(marker, boundary, StringComparison.OrdinalIgnoreCase);
        }

        JsonElement[] articles = root.GetProperty("articles").EnumerateArray().ToArray();
        Assert.Equal(Enumerable.Range(1, 103), articles.Select(static article => article.GetProperty("articleId").GetInt32()));
        Assert.Equal(
            articles.Count(static article => article.GetProperty("contentComplete").GetBoolean()),
            root.GetProperty("contentCompleteCount").GetInt32());
        Assert.Equal(
            articles.Count(static article => article.GetProperty("contentState").GetString() == "complete-long-form"),
            root.GetProperty("completeLongFormCount").GetInt32());
        Assert.Equal(
            articles.Count(static article => article.GetProperty("contentState").GetString() == "canonical-covered"),
            root.GetProperty("canonicalCoveredCount").GetInt32());
        Assert.Equal(
            articles.Count(static article => article.GetProperty("proofRequired").GetBoolean()),
            root.GetProperty("ownerOrRuntimeProofRequiredCount").GetInt32());

        foreach (JsonElement article in articles)
        {
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("title").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("contentState").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("proofState").GetString()));
            Assert.True(article.GetProperty("proofDependencies").GetArrayLength() >= 1);
            Assert.False(article.GetProperty("performsPublish").GetBoolean());
            Assert.False(article.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(article.GetProperty("canCloseReleaseIssue").GetBoolean());
        }
    }

    [Fact]
    public void CanonicalMappingsAndLongFormTargetsStayAuditableAndProofBounded()
    {
        JsonElement root = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "technical-article-closure-ledger.json"));
        JsonElement[] articles = root.GetProperty("articles").EnumerateArray().ToArray();

        Dictionary<int, (int CanonicalId, string Path)> mappings = new()
        {
            [25] = (74, "docs/articles/zh-cn/yolovision-detection-tutorial.md"),
            [26] = (74, "docs/articles/zh-cn/yolovision-detection-tutorial.md"),
            [34] = (79, "docs/articles/zh-cn/callback-allocator-safety-bridge-roadmap.md"),
            [63] = (81, "docs/articles/zh-cn/project-release-story-and-boundaries.md"),
            [64] = (72, "docs/articles/zh-cn/tensorrtexec-option-layering-deep-dive.md"),
            [65] = (73, "docs/articles/zh-cn/yolovision-all-task-overview.md"),
            [66] = (69, "docs/articles/zh-cn/package-consumer-runtime-proof-playbook.md"),
            [67] = (70, "docs/articles/zh-cn/post-publish-verification-proof-playbook.md"),
            [68] = (79, "docs/articles/zh-cn/callback-allocator-safety-bridge-roadmap.md"),
            [71] = (80, "docs/articles/zh-cn/external-model-evidence-case-study.md")
        };

        foreach (KeyValuePair<int, (int CanonicalId, string Path)> mapping in mappings)
        {
            int articleId = mapping.Key;
            int canonicalId = mapping.Value.CanonicalId;
            string path = mapping.Value.Path;
            JsonElement article = FindArticle(articles, articleId);
            Assert.Equal("canonical-covered", article.GetProperty("contentState").GetString());
            Assert.Equal(canonicalId, article.GetProperty("canonicalArticleId").GetInt32());
            Assert.Equal(path, article.GetProperty("canonicalArticlePath").GetString());
            Assert.True(article.GetProperty("canonicalArticleExists").GetBoolean());
        }

        AssertTarget(
            FindArticle(articles, 69),
            "docs/articles/zh-cn/package-consumer-runtime-proof-playbook.md",
            "package-consumer-owner-proof-required",
            "Test-PackageConsumer.ps1",
            "Test-ExternalRuntimeProofRecord.ps1");
        AssertTarget(
            FindArticle(articles, 70),
            "docs/articles/zh-cn/post-publish-verification-proof-playbook.md",
            "post-publish-owner-proof-required",
            "Test-PostPublishCleanConsumerProject.ps1",
            "Test-PostPublishVerificationRecord.ps1");
        AssertTarget(
            FindArticle(articles, 79),
            "docs/articles/zh-cn/callback-allocator-safety-bridge-roadmap.md",
            "callback-runtime-proof-required",
            "Test-CallbackRuntimeProofExecutionPack.ps1");

        JsonElement optionLayering = FindArticle(articles, 64);
        Assert.Equal("external-model-asset-required-not-runtime-proof", optionLayering.GetProperty("proofState").GetString());
        Assert.False(optionLayering.GetProperty("proofRequired").GetBoolean());
        Assert.True(optionLayering.GetProperty("externalAssetRequired").GetBoolean());

        JsonElement projectStory = FindArticle(articles, 81);
        Assert.Equal("not-required-for-content-closure", projectStory.GetProperty("proofState").GetString());
        Assert.False(projectStory.GetProperty("proofRequired").GetBoolean());
    }

    [Fact]
    public void ClosureLedgerAndCanonicalArticlesAreLinkedAndUseCurrentCommands()
    {
        string ledgerRelativePath = "articles/zh-cn/publishing/technical-article-closure-ledger.md";
        string repositoryLedgerPath = "docs/articles/zh-cn/publishing/technical-article-closure-ledger.md";
        Assert.Contains(ledgerRelativePath, ReadSource("docs", "index.md"), StringComparison.Ordinal);
        Assert.Contains(ledgerRelativePath, ReadSource("docs", "toc.yml"), StringComparison.Ordinal);
        Assert.Contains(repositoryLedgerPath, ReadSource("README.md"), StringComparison.Ordinal);
        Assert.Contains(repositoryLedgerPath, ReadSource("README.zh-CN.md"), StringComparison.Ordinal);

        string roadmap = ReadSource("docs", "articles", "zh-cn", "technical-article-roadmap.md");
        foreach (string marker in new[]
        {
            "完整教程已由 81 收口",
            "完整教程已由 72 收口",
            "完整系列已由 73-78 收口",
            "完整教程已由 69 收口",
            "完整教程已由 70 收口",
            "完整教程已由 79 收口",
            "完整教程已由 80 收口",
            "完整教程已由 74 收口",
            "contentState",
            "proofState",
            "eng/Export-TechnicalArticleClosureLedger.ps1"
        })
        {
            Assert.Contains(marker, roadmap, StringComparison.Ordinal);
        }

        string packageArticle = ReadSource("docs", "articles", "zh-cn", "package-consumer-runtime-proof-playbook.md");
        string postPublishArticle = ReadSource("docs", "articles", "zh-cn", "post-publish-verification-proof-playbook.md");
        string callbackArticle = ReadSource("docs", "articles", "zh-cn", "callback-allocator-safety-bridge-roadmap.md");
        Assert.True(packageArticle.Length >= 12_000);
        Assert.True(postPublishArticle.Length >= 12_000);
        Assert.True(callbackArticle.Length >= 12_000);

        foreach (string marker in new[]
        {
            "..\\proof\\package-consumer",
            "New-PackageConsumerExternalSmokeScaffold.ps1",
            "RuntimeProofPreflight",
            "Test-PackageConsumerRuntimeProofOwnerInput.ps1",
            "Test-ExternalRuntimeProofRecord.ps1",
            "readonly summary",
            "strict validator"
        })
        {
            Assert.Contains(marker, packageArticle, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "..\\proof\\post-publish",
            "Test-PostPublishCleanConsumerProject.ps1",
            "-ProjectPath $consumerProject",
            "Export-PostPublishVerificationRecordFromOwnerInput.ps1",
            "-OwnerInputPath",
            "-OutputPath",
            "Test-PostPublishVerificationRecord.ps1",
            "noLocalPackageSource=true"
        })
        {
            Assert.Contains(marker, postPublishArticle, StringComparison.Ordinal);
        }
        Assert.DoesNotContain("-ConsumerProject", postPublishArticle, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "TensorRtCallbackAllocatorReadinessSnapshot",
            "TensorRtExecutionContextCallbackAllocatorSafeControlSummary",
            "TensorRtCallbackOwnerClosureMatrixResult",
            "GpuAllocator",
            "GpuAsyncAllocator",
            "OutputAllocator",
            "DebugListener",
            "StreamReaderWriter",
            "DeferredRowsStillRequired=true",
            "仍缺 14",
            "Test-CallbackRuntimeProofExecutionPack.ps1"
        })
        {
            Assert.Contains(marker, callbackArticle, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string article in new[] { packageArticle, postPublishArticle, callbackArticle })
        {
            Assert.Contains("## 适用读者", article, StringComparison.Ordinal);
            Assert.Contains("## 解决问题", article, StringComparison.Ordinal);
            Assert.Contains("## 边界说明", article, StringComparison.Ordinal);
            Assert.Contains("## 下一步", article, StringComparison.Ordinal);
            Assert.DoesNotContain(@"C:\", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canPublishPublicly=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canCloseReleaseIssue=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("performsPublish=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("YoloDet", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static void AssertTarget(JsonElement article, string expectedPath, string expectedProofState, params string[] validators)
    {
        Assert.Equal("complete-long-form", article.GetProperty("contentState").GetString());
        Assert.Equal(expectedPath, article.GetProperty("canonicalArticlePath").GetString());
        Assert.True(article.GetProperty("canonicalArticleExists").GetBoolean());
        Assert.True(article.GetProperty("canonicalCharacterCount").GetInt32() >= 12_000);
        Assert.True(article.GetProperty("canonicalHeadingCount").GetInt32() >= 12);
        Assert.True(article.GetProperty("canonicalCodeBlockCount").GetInt32() >= 2);
        Assert.Equal(expectedProofState, article.GetProperty("proofState").GetString());
        Assert.True(article.GetProperty("proofRequired").GetBoolean());

        string[] actualValidators = article.GetProperty("validatorCommands")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string validator in validators)
        {
            Assert.Contains(validator, actualValidators);
        }
    }

    private static JsonElement FindArticle(IEnumerable<JsonElement> articles, int articleId)
    {
        return articles.Single(article => article.GetProperty("articleId").GetInt32() == articleId);
    }

    private static JsonElement ReadJsonRoot(string path)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.Clone();
    }

    private static string ReadSource(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }

    private static string Sha256(string path)
    {
        return Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path)));
    }

    private static string RunPowerShell(string scriptPath)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };
        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);

        using Process process = Process.Start(startInfo)!;
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        Assert.True(process.WaitForExit(120_000), $"Exporter timed out.{Environment.NewLine}{output}{error}");
        Assert.Equal(0, process.ExitCode);
        return output + error;
    }
}
