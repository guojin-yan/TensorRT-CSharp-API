using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class EvidenceLadderAndArticleMatrixTests
{
    [Fact]
    public void EvidenceLadderExplainsOnnxToEngineTensorRtExecYoloVisionAndProofNonSubstitutes()
    {
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "onnxtoengine-tensorrtexec-yolovision-evidence-ladder.md");
        string article = File.ReadAllText(articlePath);
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.True(File.Exists(articlePath));
        Assert.Contains("articles/zh-cn/onnxtoengine-tensorrtexec-yolovision-evidence-ladder.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/onnxtoengine-tensorrtexec-yolovision-evidence-ladder.md", docsToc, StringComparison.Ordinal);

        foreach (string required in new[]
        {
            "applications/OnnxToEngine",
            "applications/TensorRtExec",
            "applications/YoloVision",
            "real-model-runtime",
            "package-consumer-runtime",
            "template",
            "dry-run",
            "build-only",
            "dependency-probe-only",
            "local feed",
            "ProjectReference",
            "direct `.nupkg`",
            "GUI 截图",
            "TensorRtExec build report"
        })
        {
            Assert.Contains(required, article, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void TechnicalAndPromoArticleMatrixHasThirtyPlusHighQualityRowsAndPublishingBoundaries()
    {
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "publishing", "technical-and-promo-article-matrix-30plus.md");
        string article = File.ReadAllText(articlePath);
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.True(File.Exists(articlePath));
        Assert.Contains("articles/zh-cn/publishing/technical-and-promo-article-matrix-30plus.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/publishing/technical-and-promo-article-matrix-30plus.md", docsToc, StringComparison.Ordinal);

        int articleRows = article
            .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
            .Count(static line => line.StartsWith("| ", StringComparison.Ordinal) && line.Contains("| P", StringComparison.Ordinal));
        Assert.True(articleRows >= 35, $"Expected at least 35 article rows, found {articleRows}.");

        foreach (string required in new[]
        {
            "目标读者",
            "类型",
            "样例/代码路径",
            "真实资产需求",
            "Proof 边界",
            "优先级",
            "项目总览",
            "NuGet",
            "TensorRtExec",
            "YoloVision",
            "YOLOv8 Detection",
            "YOLOv8 Segmentation",
            "Plugin Registry Inventory",
            "ONNX Parser",
            "ParserRefitter",
            "copied diagnostics",
            "Package Consumer Runtime Proof",
            "Troubleshooting",
            "证据梯度图"
        })
        {
            Assert.Contains(required, article, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string boundary in new[]
        {
            "local feed",
            "ProjectReference",
            "direct `.nupkg`",
            "template",
            "dry-run",
            "build-only",
            "不是 runtime proof"
        })
        {
            Assert.Contains(boundary, article, StringComparison.OrdinalIgnoreCase);
        }
    }
}
