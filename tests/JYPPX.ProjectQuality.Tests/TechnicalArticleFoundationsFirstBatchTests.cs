using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TechnicalArticleFoundationsFirstBatchTests
{
    private static readonly int[] ArticleIds = { 2, 3, 4, 5, 6, 10, 15, 16, 19 };

    private static readonly string[] ArticleNames =
    {
        "why-not-plain-pinvoke.md",
        "interface-zero-to-deferred-boundary.md",
        "trt-cross-version-strategy.md",
        "windows-local-dev-environment.md",
        "runtime-package-selection.md",
        "tensorrt-object-model.md",
        "plugin-serialization-paths.md",
        "cuda-memory-wrapper.md",
        "cuda-memory-range-apis.md"
    };

    [Fact]
    public void ExporterProducesDeterministicAuditedNineArticleBatch()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-TechnicalArticleFoundationsFirstBatchAudit.ps1");
        string output = RunPowerShell(script);
        Assert.Contains("ArticleCount=9 ContentComplete=9 MissingReferences=0 MissingLinks=0 Forbidden=0", output, StringComparison.Ordinal);
        Assert.Contains("PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False", output, StringComparison.Ordinal);

        string jsonPath = SourcePath("docs", "articles", "zh-cn", "publishing", "technical-article-foundations-first-batch-audit.json");
        string markdownPath = Path.ChangeExtension(jsonPath, ".md");
        string firstJsonHash = Sha256(jsonPath);
        string firstMarkdownHash = Sha256(markdownPath);
        RunPowerShell(script);
        Assert.Equal(firstJsonHash, Sha256(jsonPath));
        Assert.Equal(firstMarkdownHash, Sha256(markdownPath));

        JsonElement root = ReadJsonRoot(jsonPath);
        Assert.Equal("technical-article-foundations-first-batch-audit", root.GetProperty("recordKind").GetString());
        Assert.Equal("content-expanded-source-quality-audited", root.GetProperty("auditState").GetString());
        Assert.Equal(ArticleIds, root.GetProperty("articleIds").EnumerateArray().Select(static item => item.GetInt32()));
        Assert.Equal(9, root.GetProperty("articleCount").GetInt32());
        Assert.Equal(9, root.GetProperty("batchContentCompleteCount").GetInt32());
        Assert.Equal(80, root.GetProperty("baselineRoadmapContentCompleteCount").GetInt32());
        Assert.Equal(23, root.GetProperty("baselineRoadmapNeedsExpansionCount").GetInt32());
        Assert.Equal(89, root.GetProperty("expectedPostExpansionContentCompleteCount").GetInt32());
        Assert.Equal(14, root.GetProperty("expectedPostExpansionNeedsExpansionCount").GetInt32());
        Assert.True(root.GetProperty("totalCharacterGrowth").GetInt32() > 60_000);
        Assert.True(root.GetProperty("totalHeadingCount").GetInt32() >= 200);
        Assert.True(root.GetProperty("totalCodeBlockCount").GetInt32() >= 80);
        Assert.True(root.GetProperty("totalMermaidDiagramCount").GetInt32() >= 15);
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

        string ledgerPath = SourcePath("docs", "articles", "zh-cn", "publishing", "technical-article-closure-ledger.json");
        Assert.Equal(Sha256(ledgerPath).ToLowerInvariant(), root.GetProperty("sourceClosureLedgerSha256").GetString());

        JsonElement[] articles = root.GetProperty("articles").EnumerateArray().ToArray();
        Assert.Equal(9, articles.Length);
        foreach (JsonElement article in articles)
        {
            Assert.True(article.GetProperty("contentComplete").GetBoolean());
            Assert.True(article.GetProperty("currentCharacterCount").GetInt32() >= 4_500);
            Assert.True(article.GetProperty("characterGrowth").GetInt32() > 4_000);
            Assert.True(article.GetProperty("headingCount").GetInt32() >= 15);
            Assert.True(article.GetProperty("codeBlockCount").GetInt32() >= 4);
            Assert.True(article.GetProperty("mermaidDiagramCount").GetInt32() >= 1);
            Assert.Equal(0, article.GetProperty("missingMarkers").GetArrayLength());
            Assert.Equal(0, article.GetProperty("missingAnchors").GetArrayLength());
            Assert.Equal(0, article.GetProperty("missingRepositoryReferences").GetArrayLength());
            Assert.Equal(0, article.GetProperty("missingMarkdownLinks").GetArrayLength());
            Assert.Equal(0, article.GetProperty("forbiddenFindings").GetArrayLength());
        }
    }

    [Fact]
    public void ArticlesKeepCompleteBlogStructureAndProofBoundaries()
    {
        foreach (string articleName in ArticleNames)
        {
            string article = ReadSource("docs", "articles", "zh-cn", articleName);
            Assert.True(article.Length >= 4_500, $"{articleName} must remain a substantive standalone article.");
            Assert.Contains("## 适用读者", article, StringComparison.Ordinal);
            Assert.Contains("```mermaid", article, StringComparison.Ordinal);
            Assert.Matches(@"(?m)^## (?:常见|排障)", article);
            Assert.Contains("## 边界说明", article, StringComparison.Ordinal);
            Assert.Contains("## 下一步", article, StringComparison.Ordinal);
            Assert.Contains("performsPublish=false", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canPublishPublicly=false", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canCloseReleaseIssue=false", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canPublishPublicly=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canCloseReleaseIssue=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("performsPublish=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("YoloDet", article, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ArchitectureVersionAndPackageFactsTrackAuthoritativeSources()
    {
        string coverageExporter = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        Assert.Contains("machine-specific paths omitted", coverageExporter, StringComparison.Ordinal);
        Assert.Contains("Manifest file count: $manifestFileCount", coverageExporter, StringComparison.Ordinal);
        Assert.DoesNotContain("TensorRT package root: ``$TensorRtPackageRoot``", coverageExporter, StringComparison.Ordinal);
        Assert.DoesNotContain("CUDA toolkit root: ``$CudaToolkitRoot``", coverageExporter, StringComparison.Ordinal);

        string coverage = ReadSource("artifacts", "interface-coverage", "interface-coverage-summary.md");
        string interfaceArticle = ReadSource("docs", "articles", "zh-cn", "interface-zero-to-deferred-boundary.md");
        Match manifestCount = Regex.Match(coverage, @"Manifest API count:\s*(?<count>\d+)", RegexOptions.CultureInvariant);
        Assert.True(manifestCount.Success);
        string[] currentManifestPaths = Directory.GetFiles(
            Path.Combine(RepositoryPaths.Root, "native", "manifests"),
            "*.manifest.json",
            SearchOption.AllDirectories);
        int currentManifestApiCount = 0;
        foreach (string manifestPath in currentManifestPaths)
        {
            using JsonDocument manifest = JsonDocument.Parse(File.ReadAllText(manifestPath));
            currentManifestApiCount += manifest.RootElement.GetProperty("apis").GetArrayLength();
        }
        Assert.Contains($"{currentManifestPaths.Length} 个 manifest 文件", interfaceArticle, StringComparison.Ordinal);
        Assert.Contains($"{currentManifestApiCount} 条 API", interfaceArticle, StringComparison.Ordinal);
        Assert.Contains("逐版本 coverage summary 只代表生成时实际可见的", interfaceArticle, StringComparison.Ordinal);
        Assert.Contains("完整 SDK 矩阵主机仍需", interfaceArticle, StringComparison.Ordinal);

        foreach (string line in new[] { "8.6", "10.11", "11.0" })
        {
            string[] summaryLines = coverage
                .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Where(item => item.Contains($"TensorRT-{line}", StringComparison.Ordinal))
                .ToArray();
            Assert.NotEmpty(summaryLines);
            Match[] summaries = summaryLines
                .Select(item => Regex.Match(
                    item,
                    @"official interfaces scanned=(?<official>\d+).*implemented=(?<implemented>\d+), deferred-only=(?<deferred>\d+)",
                    RegexOptions.CultureInvariant))
                .ToArray();
            Assert.All(summaries, summary => Assert.True(summary.Success, $"Coverage summary for TensorRT {line} was not found."));
            Assert.Single(summaries.Select(static summary => summary.Value[summary.Value.IndexOf("official interfaces", StringComparison.Ordinal)..]).Distinct(StringComparer.Ordinal));
            Match summary = summaries[0];
            Assert.Contains(summary.Groups["official"].Value, interfaceArticle, StringComparison.Ordinal);
            Assert.Contains(summary.Groups["implemented"].Value, interfaceArticle, StringComparison.Ordinal);
            Assert.Contains(summary.Groups["deferred"].Value, interfaceArticle, StringComparison.Ordinal);
        }

        using JsonDocument runtimeManifest = JsonDocument.Parse(ReadSource("pack", "runtime", "runtime-packages.manifest.json"));
        JsonElement[] packages = runtimeManifest.RootElement.GetProperty("packages").EnumerateArray().ToArray();
        Assert.Equal(18, packages.Length);
        Assert.Equal(6, packages.Count(static item => item.GetProperty("platform").GetString() == "windows"));
        Assert.Equal(12, packages.Count(static item => item.GetProperty("platform").GetString() == "linux"));

        string crossVersion = ReadSource("docs", "articles", "zh-cn", "trt-cross-version-strategy.md");
        string packageSelection = ReadSource("docs", "articles", "zh-cn", "runtime-package-selection.md");
        Assert.Contains("18 个 runtime key", crossVersion, StringComparison.Ordinal);
        Assert.Contains("当前 manifest 共 18 个 key", packageSelection, StringComparison.Ordinal);
        foreach (JsonElement package in packages.Where(static item => item.GetProperty("platform").GetString() == "windows"))
        {
            string key = package.GetProperty("key").GetString()!;
            string preset = package.GetProperty("buildPreset").GetString()!;
            Assert.Contains(key, crossVersion, StringComparison.Ordinal);
            Assert.Contains(preset, crossVersion, StringComparison.Ordinal);
            Assert.Contains(key, packageSelection, StringComparison.Ordinal);
        }

        using JsonDocument presets = JsonDocument.Parse(ReadSource("CMakePresets.json"));
        string[] configurePresetNames = presets.RootElement.GetProperty("configurePresets")
            .EnumerateArray()
            .Select(static item => item.GetProperty("name").GetString()!)
            .ToArray();
        string windowsEnvironment = ReadSource("docs", "articles", "zh-cn", "windows-local-dev-environment.md");
        foreach (string preset in packages
                     .Where(static item => item.GetProperty("platform").GetString() == "windows")
                     .Select(static item => item.GetProperty("buildPreset").GetString()!)
                     .Distinct(StringComparer.Ordinal))
        {
            Assert.Contains(preset, configurePresetNames);
            Assert.Contains(preset, windowsEnvironment, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ObjectPluginAndCudaArticlesTrackWrappersSmokesAndNavigation()
    {
        string objectArticle = ReadSource("docs", "articles", "zh-cn", "tensorrt-object-model.md");
        string inferenceBindings = string.Join(
            '\n',
            ReadSource("src", "JYPPX.TensorRtSharp", "Inference", "TensorRtInferenceBindings.TensorGeometry.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Inference", "TensorRtInferenceBindings.Buffers.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Inference", "TensorRtInferenceBindings.HostTransfers.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Inference", "TensorRtInferenceBindings.AddressBinding.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Inference", "TensorRtInferenceBindings.Execution.cs"));
        string sample = ReadSource("samples", "InferenceBindings", "Program.cs");
        foreach (string marker in new[]
        {
            "SetInputShape", "CopyInputFromHost", "AllocateDeviceBuffer", "BindAll", "GetReadiness", "EnqueueAsync", "ReadOutputSingles"
        })
        {
            Assert.Contains(marker, inferenceBindings, StringComparison.Ordinal);
            Assert.Contains(marker, sample, StringComparison.Ordinal);
            Assert.Contains(marker, objectArticle, StringComparison.Ordinal);
        }

        string pluginArticle = ReadSource("docs", "articles", "zh-cn", "plugin-serialization-paths.md");
        string pluginControl = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11PluginSerialization.cs");
        string pluginDiagnostics = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11Diagnostics.cs");
        string pluginSmoke = ReadSource("smoke", "PluginSerializationPathsSmokeRunner", "Program.cs");
        foreach (string marker in new[] { "SetPluginsToSerialize", "GetPluginsToSerialize", "GetSerializedPluginSnapshot", "ClearPluginsToSerialize" })
        {
            Assert.Contains(marker, pluginControl + pluginDiagnostics, StringComparison.Ordinal);
            Assert.Contains(marker, pluginSmoke, StringComparison.Ordinal);
            Assert.Contains(marker, pluginArticle, StringComparison.Ordinal);
        }
        Assert.Contains("IPluginV2::serialize", pluginArticle, StringComparison.Ordinal);
        Assert.Contains("plugin-v2-serialize-deferred", ReadSource("native", "manifests", "tensorrt", "v10", "trt10-cross-version-second-batch-plugin-deferred.manifest.json"), StringComparison.Ordinal);

        string cudaWrapper = ReadSource("docs", "articles", "zh-cn", "cuda-memory-wrapper.md");
        string cudaRange = ReadSource("docs", "articles", "zh-cn", "cuda-memory-range-apis.md");
        string cudaMemory = ReadSource(
            "src", "JYPPX.CudaSharp", "Memory", "CudaMemory.RangeDiagnostics.cs");
        string cudaBatch = ReadSource("src", "JYPPX.CudaSharp", "Memory", "CudaManagedMemoryBatch.cs");
        string cudaSmoke = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");
        foreach (string marker in new[] { "GetRangeAttribute", "GetRangeAttributes", "GetRangeAccessedByDevices", "GetRangeDiagnosticSummary" })
        {
            Assert.Contains(marker, cudaMemory, StringComparison.Ordinal);
            Assert.Contains(marker, cudaRange, StringComparison.Ordinal);
        }
        foreach (string marker in new[] { "PrefetchAsync", "DiscardAsync", "DiscardAndPrefetchAsync" })
        {
            Assert.Contains(marker, cudaBatch, StringComparison.Ordinal);
            Assert.Contains(marker, cudaRange, StringComparison.Ordinal);
        }
        foreach (string marker in new[] { "PinnedAsyncRoundTrip=", "MemoryRangeSummary=", "CudaManagedMemoryBatch" })
        {
            Assert.Contains(marker, cudaSmoke, StringComparison.Ordinal);
            Assert.Contains(marker, cudaWrapper + cudaRange, StringComparison.Ordinal);
        }
        Assert.Contains("PinnedAsyncRoundTrip=True", cudaWrapper, StringComparison.Ordinal);

        string auditRelative = "articles/zh-cn/publishing/technical-article-foundations-first-batch-audit.md";
        string auditRepositoryPath = "docs/articles/zh-cn/publishing/technical-article-foundations-first-batch-audit.md";
        Assert.Contains(auditRepositoryPath, ReadSource("README.md"), StringComparison.Ordinal);
        Assert.Contains(auditRepositoryPath, ReadSource("README.zh-CN.md"), StringComparison.Ordinal);
        Assert.Contains(auditRelative, ReadSource("docs", "index.md"), StringComparison.Ordinal);
        Assert.Contains(auditRelative, ReadSource("docs", "toc.yml"), StringComparison.Ordinal);

        string roadmap = ReadSource("docs", "articles", "zh-cn", "technical-article-roadmap.md");
        foreach (int articleId in ArticleIds)
        {
            Assert.Matches($@"(?m)^\|\s*{articleId}\s*\|.*\|\s*完整教程已收口\s*\|$", roadmap);
        }

        JsonElement ledger = ReadJsonRoot(SourcePath("docs", "articles", "zh-cn", "publishing", "technical-article-closure-ledger.json"));
        Assert.True(ledger.GetProperty("contentCompleteCount").GetInt32() >= 89);
        Assert.Equal(0, ledger.GetProperty("needsExpansionCount").GetInt32());
        Assert.Empty(
            ledger.GetProperty("articles").EnumerateArray()
                .Where(static item => new[] { 2, 3, 4, 5, 6, 10, 15, 16, 19 }.Contains(item.GetProperty("articleId").GetInt32()))
                .Where(static item => item.GetProperty("contentState").GetString() == "needs-expansion"));
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
