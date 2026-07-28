using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TechnicalArticleFoundationsSecondBatchTests
{
    private static readonly int[] RoadmapEntryIds = { 28, 29, 30, 31, 32, 38, 39, 40, 41, 42, 43, 44, 45, 103 };

    [Fact]
    public void ExporterAuditsFourteenEntriesAndTwelveUniqueCanonicalBodiesDeterministically()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-TechnicalArticleFoundationsSecondBatchAudit.ps1");
        string output = RunPowerShell(script);
        Assert.Contains("RoadmapEntries=14 UniqueCanonical=12 ContentComplete=12 MissingReferences=0 MissingLinks=0 Forbidden=0", output, StringComparison.Ordinal);
        Assert.Contains("PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False", output, StringComparison.Ordinal);

        string jsonPath = SourcePath("docs", "articles", "zh-cn", "publishing", "technical-article-foundations-second-batch-audit.json");
        string markdownPath = Path.ChangeExtension(jsonPath, ".md");
        string firstJsonHash = Sha256(jsonPath);
        string firstMarkdownHash = Sha256(markdownPath);
        RunPowerShell(script);
        Assert.Equal(firstJsonHash, Sha256(jsonPath));
        Assert.Equal(firstMarkdownHash, Sha256(markdownPath));

        JsonElement root = ReadJsonRoot(jsonPath);
        Assert.Equal("technical-article-foundations-second-batch-audit", root.GetProperty("recordKind").GetString());
        Assert.Equal("content-expanded-source-quality-audited", root.GetProperty("auditState").GetString());
        Assert.Equal(RoadmapEntryIds, root.GetProperty("roadmapEntryIds").EnumerateArray().Select(static item => item.GetInt32()));
        Assert.Equal(14, root.GetProperty("roadmapEntryCount").GetInt32());
        Assert.Equal(12, root.GetProperty("uniqueCanonicalArticleCount").GetInt32());
        Assert.Equal(2, root.GetProperty("sharedCanonicalMappingCount").GetInt32());
        Assert.Equal(14, root.GetProperty("roadmapStatusCompleteCount").GetInt32());
        Assert.Equal(89, root.GetProperty("baselineRoadmapContentCompleteCount").GetInt32());
        Assert.Equal(14, root.GetProperty("baselineRoadmapNeedsExpansionCount").GetInt32());
        Assert.Equal(103, root.GetProperty("expectedPostExpansionContentCompleteCount").GetInt32());
        Assert.Equal(0, root.GetProperty("expectedPostExpansionNeedsExpansionCount").GetInt32());
        Assert.Equal(12, root.GetProperty("batchCanonicalContentCompleteCount").GetInt32());
        Assert.True(root.GetProperty("totalCurrentCharacterCount").GetInt32() >= 70_000);
        Assert.True(root.GetProperty("totalHeadingCount").GetInt32() >= 130);
        Assert.True(root.GetProperty("totalCodeBlockCount").GetInt32() >= 55);
        Assert.True(root.GetProperty("totalMermaidDiagramCount").GetInt32() >= 18);
        Assert.Equal(0, root.GetProperty("missingMarkerCount").GetInt32());
        Assert.Equal(0, root.GetProperty("missingAnchorCount").GetInt32());
        Assert.Equal(0, root.GetProperty("missingRepositoryReferenceCount").GetInt32());
        Assert.Equal(0, root.GetProperty("missingMarkdownLinkCount").GetInt32());
        Assert.Equal(0, root.GetProperty("forbiddenFindingCount").GetInt32());
        Assert.True(root.GetProperty("contentAndProofStateAreIndependent").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());

        JsonElement[] mappings = root.GetProperty("sharedCanonicalMappings").EnumerateArray().ToArray();
        Assert.Contains(mappings, item => item.GetProperty("canonicalArticlePath").GetString() == "docs/articles/zh-cn/blog-refit-weights-guide.md" &&
            item.GetProperty("roadmapEntryIds").EnumerateArray().Select(static id => id.GetInt32()).SequenceEqual(new[] { 28, 44 }));
        Assert.Contains(mappings, item => item.GetProperty("canonicalArticlePath").GetString() == "docs/articles/zh-cn/blog-network-layer-coverage-guide.md" &&
            item.GetProperty("roadmapEntryIds").EnumerateArray().Select(static id => id.GetInt32()).SequenceEqual(new[] { 30, 45 }));

        string ledgerPath = SourcePath("docs", "articles", "zh-cn", "publishing", "technical-article-closure-ledger.json");
        Assert.Equal(Sha256(ledgerPath).ToLowerInvariant(), root.GetProperty("sourceClosureLedgerSha256").GetString());
    }

    [Fact]
    public void CanonicalArticlesKeepOperationalStructureAndAuthoritativeMarkers()
    {
        JsonElement root = ReadJsonRoot(SourcePath("docs", "articles", "zh-cn", "publishing", "technical-article-foundations-second-batch-audit.json"));
        foreach (JsonElement article in root.GetProperty("canonicalArticles").EnumerateArray())
        {
            Assert.True(article.GetProperty("contentComplete").GetBoolean());
            Assert.True(article.GetProperty("currentCharacterCount").GetInt32() >= 4_500);
            Assert.True(article.GetProperty("headingCount").GetInt32() >= 10);
            Assert.True(article.GetProperty("codeBlockCount").GetInt32() >= 3);
            Assert.True(article.GetProperty("mermaidDiagramCount").GetInt32() >= 1);
            Assert.Equal(0, article.GetProperty("missingMarkers").GetArrayLength());
            Assert.Equal(0, article.GetProperty("missingAnchors").GetArrayLength());
            Assert.Equal(0, article.GetProperty("missingRepositoryReferences").GetArrayLength());
            Assert.Equal(0, article.GetProperty("missingMarkdownLinks").GetArrayLength());
            Assert.Equal(0, article.GetProperty("forbiddenFindings").GetArrayLength());

            string text = ReadSource(article.GetProperty("canonicalArticlePath").GetString()!.Split('/'));
            Assert.Contains("performsPublish=false", text, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canPublishPublicly=false", text, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canCloseReleaseIssue=false", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canPublishPublicly=true", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canCloseReleaseIssue=true", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("performsPublish=true", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("dotnet nuget push", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("YoloDet", text, StringComparison.OrdinalIgnoreCase);
        }

        string refit = ReadSource("docs", "articles", "zh-cn", "blog-refit-weights-guide.md");
        Assert.Contains("GetAllEntries", refit, StringComparison.Ordinal);
        Assert.Contains("RefitCudaEngine", ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRefitter.cs"), StringComparison.Ordinal);

        string modern = ReadSource("docs", "articles", "zh-cn", "trt11-modern-layers-guide.md");
        Assert.Contains("Dims64Evidence", modern, StringComparison.Ordinal);
        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11", modern, StringComparison.Ordinal);

        string callbacks = ReadSource("docs", "articles", "zh-cn", "managed-logger-profiler-progress-monitor.md");
        Assert.Contains("CallbackInvocationCount", callbacks, StringComparison.Ordinal);
        Assert.Contains("ManagedProgressMonitorAttach", callbacks, StringComparison.Ordinal);

        string streamCapture = ReadSource("docs", "articles", "zh-cn", "cuda-stream-capture-to-graph-owner-safety.md");
        Assert.Contains("CudaStreamCaptureToGraphSession", streamCapture, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 12030", streamCapture, StringComparison.Ordinal);
    }

    [Fact]
    public void RoadmapLedgerAndNavigationProjectFinalContentClosure()
    {
        string roadmap = ReadSource("docs", "articles", "zh-cn", "technical-article-roadmap.md");
        foreach (int articleId in RoadmapEntryIds)
        {
            Assert.Matches($@"(?m)^\|\s*{articleId}\s*\|.*\|\s*完整教程已收口\s*\|$", roadmap);
        }

        JsonElement ledger = ReadJsonRoot(SourcePath("docs", "articles", "zh-cn", "publishing", "technical-article-closure-ledger.json"));
        Assert.Equal(103, ledger.GetProperty("contentCompleteCount").GetInt32());
        Assert.Equal(0, ledger.GetProperty("needsExpansionCount").GetInt32());
        Assert.Equal(42, ledger.GetProperty("ownerOrRuntimeProofRequiredCount").GetInt32());
        Assert.False(ledger.GetProperty("performsPublish").GetBoolean());
        Assert.False(ledger.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(ledger.GetProperty("canCloseReleaseIssue").GetBoolean());

        string auditRelative = "articles/zh-cn/publishing/technical-article-foundations-second-batch-audit.md";
        string auditRepositoryPath = "docs/articles/zh-cn/publishing/technical-article-foundations-second-batch-audit.md";
        Assert.Contains(auditRepositoryPath, ReadSource("README.md"), StringComparison.Ordinal);
        Assert.Contains(auditRepositoryPath, ReadSource("README.zh-CN.md"), StringComparison.Ordinal);
        Assert.Contains(auditRelative, ReadSource("docs", "index.md"), StringComparison.Ordinal);
        Assert.Contains(auditRelative, ReadSource("docs", "toc.yml"), StringComparison.Ordinal);
    }

    private static JsonElement ReadJsonRoot(string path)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.Clone();
    }

    private static string ReadSource(params string[] pathParts)
    {
        return File.ReadAllText(SourcePath(pathParts));
    }

    private static string SourcePath(params string[] pathParts)
    {
        return Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
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
