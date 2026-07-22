using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseReadinessSamplesAndApplicationsTests
{
    [Fact]
    public void YoloVisionModelMatrixCoversFamiliesTasksAndProofBoundary()
    {
        using JsonDocument document = ReadJson("samples", "YoloVision", "yolo-model-matrix.json");
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-model-matrix", root.GetProperty("matrixId").GetString());
        Assert.Contains("not real-model-runtime proof", root.GetProperty("proofBoundary").GetString()!, StringComparison.OrdinalIgnoreCase);

        string[] families = root.GetProperty("families").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string family in new[] { "yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov26" })
        {
            Assert.Contains(family, families);
        }

        string[] tasks = root.GetProperty("tasks").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string task in new[] { "det", "cls", "seg", "obb", "pose", "sem" })
        {
            Assert.Contains(task, tasks);
        }

        string markdown = ReadText("samples", "YoloVision", "yolo-model-matrix.md");
        Assert.Contains("YOLOv26", markdown, StringComparison.Ordinal);
        Assert.Contains("sample-run-evidence", markdown, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void OnnxToEngineAndTensorRtExecMatricesCoverTrtexecReadiness()
    {
        using JsonDocument parity = ReadJson("samples", "OnnxToEngine", "trtexec-parity-matrix.json");
        string[] options = parity.RootElement.GetProperty("entries").EnumerateArray()
            .Select(static item => item.GetProperty("option").GetString()!)
            .ToArray();

        foreach (string option in new[] { "--onnx", "--saveEngine", "--minShapes/--optShapes/--maxShapes", "--fp16", "--fp8/--best", "--int8/--calib", "--workspace", "--memPoolSize", "--tacticSources", "--timingCacheFile/--timingCache", "--exportTimingCache", "--profilingVerbosity/--verbose", "--previewOnly/--dryRun", "--sparsity", "--refit", "--dumpRefit/--markDebug/--dumpDebugTensors", "--allowWeightStreaming/--weightStreamingBudget", "--versionCompatible/--excludeLeanRuntime/--stripWeights", "--device/--useDLACore/--allowGPUFallback", "--loadInputs", "--dumpOutput", "--dumpRawBindingsToFile", "--exportOutput/--exportTimes/--exportProfile/--saveProfile" })
        {
            Assert.Contains(option, options);
        }

        using JsonDocument app = ReadJson("applications", "TensorRtExec", "tensor-rt-exec-feature-matrix.json");
        string[] modes = app.RootElement.GetProperty("modes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("CLI", modes);
        Assert.Contains("WinForms", modes);

        string[] statuses = app.RootElement.GetProperty("features").EnumerateArray()
            .Select(static item => item.GetProperty("status").GetString()!)
            .ToArray();
        Assert.Contains("implemented", statuses);
        Assert.Contains(statuses, static status => status.StartsWith("wrapper-ready", StringComparison.Ordinal));
        Assert.Contains("parse-report-only", statuses);
        Assert.Contains("planned", statuses);
        Assert.Contains("blocked by runtime proof", statuses);
    }

    [Fact]
    public void ArticleRoadmapHasThirtyPlusAuditableArticles()
    {
        using JsonDocument document = ReadJson("docs", "articles", "zh-cn", "publishing", "article-roadmap-30plus.json");
        JsonElement root = document.RootElement;
        JsonElement.ArrayEnumerator articles = root.GetProperty("articles").EnumerateArray();
        JsonElement[] articleArray = articles.ToArray();

        Assert.True(articleArray.Length >= 30);

        foreach (JsonElement article in articleArray)
        {
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("title").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("audience").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("type").GetString()));
            Assert.True(article.GetProperty("outline").GetArrayLength() >= 3);
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("sampleOrCodePath").GetString()));
            Assert.True(article.GetProperty("visualAssets").GetArrayLength() >= 1);
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("status").GetString()));
        }

        string roadmap = ReadText("docs", "articles", "zh-cn", "publishing", "article-roadmap-30plus.md");
        int articleCount = root.GetProperty("articleCount").GetInt32();
        Assert.Equal(articleArray.Length, articleCount);
        Assert.Contains($"articleCount | `{articleCount}`", roadmap, StringComparison.Ordinal);
        Assert.Contains("YoloVision", roadmap, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec", roadmap, StringComparison.Ordinal);
        Assert.Contains("ONNX Parser 与 ParserRefitter 诊断", roadmap, StringComparison.Ordinal);
    }

    [Fact]
    public void ApiReadinessAuditAndDocsExposeReleaseReadinessMatrices()
    {
        using JsonDocument audit = ReadJson("artifacts", "interface-coverage", "release-api-readiness-audit.json");
        JsonElement root = audit.RootElement;
        Assert.Equal("release-api-readiness-audit", root.GetProperty("artifactId").GetString());
        Assert.Contains("not runtime proof", root.GetProperty("proofBoundary").GetString()!, StringComparison.OrdinalIgnoreCase);
        Assert.True(root.GetProperty("safeReadOnlyCandidates").GetArrayLength() >= 5);
        Assert.True(root.GetProperty("blockedOwnershipRiskApis").GetArrayLength() >= 5);
        Assert.True(root.GetProperty("versionGuardNotes").GetArrayLength() >= 3);

        string rootReadme = ReadText("README.md");
        string rootReadmeZh = ReadText("README.zh-CN.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        foreach (string marker in new[]
        {
            "samples/YoloVision/yolo-model-matrix.json",
            "samples/OnnxToEngine/trtexec-parity-matrix.json",
            "applications/TensorRtExec/tensor-rt-exec-feature-matrix.json",
            "artifacts/interface-coverage/release-api-readiness-audit.json",
            "article-roadmap-30plus"
        })
        {
            Assert.Contains(marker, rootReadme, StringComparison.Ordinal);
            Assert.Contains(marker, rootReadmeZh, StringComparison.Ordinal);
        }

        Assert.Contains("release-api-readiness-audit.json", docsIndex, StringComparison.Ordinal);
        Assert.Contains("article-roadmap-30plus.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("article-roadmap-30plus.md", docsToc, StringComparison.Ordinal);
    }

    private static JsonDocument ReadJson(params string[] pathParts)
    {
        return JsonDocument.Parse(ReadText(pathParts));
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}
