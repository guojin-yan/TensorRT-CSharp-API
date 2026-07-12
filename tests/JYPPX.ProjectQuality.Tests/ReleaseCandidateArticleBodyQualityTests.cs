using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseCandidateArticleBodyQualityTests
{
    [Fact]
    public void FirstWaveChineseArticlesHavePublishableBodyShape()
    {
        string[] articleFiles =
        {
            "project-overview.md",
            "onnx-to-engine-quickstart.md",
            "tensorrtexec-cli-parameter-map.md",
            "yolovision-sample-overview.md",
            "plugin-inventory-readonly-api.md"
        };

        foreach (string articleFile in articleFiles)
        {
            string article = ReadArticle(articleFile);

            foreach (string marker in new[]
            {
                "## 目标读者",
                "## 可复制命令",
                "## 截图与图示建议",
                "## 下一步",
                "Boundary keywords: not public package proof, not post-publish proof, not package push, not release close approval"
            })
            {
                Assert.Contains(marker, article, StringComparison.Ordinal);
            }

            Assert.Contains("proof", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canPublishPublicly=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canCloseReleaseIssue=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("YoloDet", article, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void FirstWaveArticlesPointToRealProjectFilesAndEvidenceBoundaries()
    {
        (string Article, string[] RequiredPaths)[] expectations =
        {
            ("project-overview.md", new[] { "src/JYPPX.TensorRtSharp", "samples/OnnxToEngine", "samples/YoloVision", "artifacts/interface-coverage/release-api-readiness-audit.json" }),
            ("onnx-to-engine-quickstart.md", new[] { "samples/OnnxToEngine", "applications/TensorRtExec", "samples/YoloVision", "samples/Classification" }),
            ("tensorrtexec-cli-parameter-map.md", new[] { "applications/TensorRtExec", "src/JYPPX.TensorRtSharp.Tools", "samples/OnnxToEngine/trtexec-parity-matrix.json", "artifacts/user-acceptance/trtexec-option-coverage.md" }),
            ("yolovision-sample-overview.md", new[] { "samples/YoloVision", "samples/assets/yolovision-assets.template.json", "artifacts/user-acceptance/sample-run-evidence-record.yolovision.template.json", "applications/TensorRtExec" }),
            ("plugin-inventory-readonly-api.md", new[] { "smoke/PluginRegistryInventorySmokeRunner", "src/JYPPX.TensorRtSharp/TensorRtPluginRegistryInventory.cs", "native/src/tensorrt/common/plugin_registry_inventory.inc", "artifacts/interface-coverage/tensorrt-interface-comparison.csv" })
        };

        foreach ((string articleFile, string[] requiredPaths) in expectations)
        {
            string article = ReadArticle(articleFile);

            foreach (string requiredPath in requiredPaths)
            {
                Assert.Contains(requiredPath, article, StringComparison.Ordinal);
            }

            Assert.Contains("not runtime proof", article, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static string ReadArticle(string articleFile)
    {
        return File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile));
    }
}
