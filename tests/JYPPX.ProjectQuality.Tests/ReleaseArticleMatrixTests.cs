using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseArticleMatrixTests
{
    [Fact]
    public void ReleaseArticleMatrixContainsDirectYoloOnnxAndTensorRtExecEntrypoints()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string rootReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string zhReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string samplesReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "README.md"));

        string[] articleFiles =
        {
            "yolovision-sample-overview.md",
            "yolovision-preprocess-postprocess.md",
            "yolovision-engine-build-and-run.md",
            "yolovision-troubleshooting.md",
            "onnx-to-engine-quickstart.md",
            "onnx-to-engine-dynamic-shape-profile.md",
            "onnx-to-engine-fp16-int8-boundary.md",
            "onnx-to-engine-output-artifacts.md",
            "tensorrtexec-cli-parameter-map.md",
            "tensorrtexec-winforms-guide.md",
        };

        foreach (string articleFile in articleFiles)
        {
            string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string href = "articles/zh-cn/" + articleFile;
            string article = File.ReadAllText(articlePath);

            Assert.True(File.Exists(articlePath));
            Assert.Contains(href, docsIndex, StringComparison.Ordinal);
            Assert.Contains(href, docsToc, StringComparison.Ordinal);
            Assert.Contains("边界说明", article, StringComparison.Ordinal);
            Assert.Contains("build-only", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("proof", article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("yolovision-sample-overview.md", rootReadme, StringComparison.Ordinal);
        Assert.Contains("onnx-to-engine-quickstart.md", rootReadme, StringComparison.Ordinal);
        Assert.Contains("tensorrtexec-cli-parameter-map.md", rootReadme, StringComparison.Ordinal);
        Assert.Contains("yolovision-sample-overview.md", zhReadme, StringComparison.Ordinal);
        Assert.Contains("onnx-to-engine-quickstart.md", zhReadme, StringComparison.Ordinal);
        Assert.Contains("tensorrtexec-cli-parameter-map.md", zhReadme, StringComparison.Ordinal);
        Assert.Contains("yolovision-sample-overview.md", samplesReadme, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet", docsIndex, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet", docsToc, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ReleaseArticleMatrixGuardsProofBoundariesAcrossNewGuides()
    {
        string[] articleFiles =
        {
            "yolovision-sample-overview.md",
            "yolovision-engine-build-and-run.md",
            "yolovision-troubleshooting.md",
            "onnx-to-engine-quickstart.md",
            "onnx-to-engine-output-artifacts.md",
            "tensorrtexec-cli-parameter-map.md",
            "tensorrtexec-winforms-guide.md",
        };

        foreach (string articleFile in articleFiles)
        {
            string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile));

            Assert.Contains("not", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("public package proof", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("post-publish", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canPublishPublicly=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canCloseReleaseIssue=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ArticleMatrixKeepsAtLeastThirtyChineseGuidesForPublishableAdoption()
    {
        string docsRoot = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn");
        string[] requiredArticleFiles =
        {
            "release-article-index-and-publishing-order.md",
            "release-candidate-article-matrix-summary.md",
            "project-overview.md",
            "project-release-story-and-boundaries.md",
            "known-limitations-4.0.0-rc.md",
            "readiness-summary-guide.md",
            "yolovision-sample-overview.md",
            "yolovision-model-assets.md",
            "yolovision-preprocess-postprocess.md",
            "yolovision-engine-build-and-run.md",
            "yolovision-troubleshooting.md",
            "onnx-to-engine-quickstart.md",
            "onnx-to-engine-dynamic-shape-profile.md",
            "onnx-to-engine-fp16-int8-boundary.md",
            "onnx-to-engine-output-artifacts.md",
            "onnx-to-engine-trtexec-conversion-guide.md",
            "onnxtoengine-and-tensorrtexec-boundary.md",
            "tensorrtexec-cli-parameter-map.md",
            "tensorrtexec-winforms-guide.md",
            "tensorrtexec-tool-getting-started.md",
            "tensorrtexec-gui-user-guide.md",
            "classification-real-asset-walkthrough.md",
            "classification-model-assets.md",
            "dynamic-shape-optimization-profile-tutorial.md",
            "cuda-stream-event-multistream-tutorial.md",
            "inference-bindings-tutorial.md",
            "onnx-parser-to-serialized-engine-tutorial.md",
            "refit-weights-guide.md",
            "plugin-inventory-readonly-api.md",
            "cuda-memory-wrapper.md",
            "cuda-memory-range-apis.md",
        };

        Assert.True(requiredArticleFiles.Length >= 30);
        foreach (string articleFile in requiredArticleFiles)
        {
            Assert.True(File.Exists(Path.Combine(docsRoot, articleFile)), articleFile);
        }
    }
}
