using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TechnicalArticleRoadmapTests
{
    [Fact]
    public void RoadmapKeepsASubstantialUniqueArticleSeries()
    {
        string roadmap = ReadRoadmap();
        int[] articleIds = ExtractArticleIds(roadmap);

        Assert.True(articleIds.Length >= 55, "The roadmap must remain a substantial article series.");
        Assert.Equal(articleIds.Length, articleIds.Distinct().Count());
        Assert.All(Enumerable.Range(81, 18), id => Assert.Contains(id, articleIds));
    }

    [Fact]
    public void RoadmapDefinesModelAcquisitionConversionAndVisualResultRequirements()
    {
        string roadmap = ReadRoadmap();

        foreach (string marker in new[]
        {
            "对应 sample/application",
            "模型/资产",
            "模型获取方式",
            "license/hash 要求",
            "命令",
            "输出",
            "截图/图示需求",
            "proof boundary",
            "TensorRtExec",
            "OnnxToEngine",
            "YoloVision",
            "Classification",
            "C++",
            "CMake",
            "GitHub Release",
            "NuGet managed + bridge-only",
            "build-only",
            "real-model-runtime",
            "package-consumer-runtime"
        })
        {
            Assert.Contains(marker, roadmap, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("samples/YoloDet", roadmap, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet.csproj", roadmap, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void CurrentTechnicalGuidesCoverSourceBuildAndPackageConsumption()
    {
        string sourceBuildGuide = ReadText("docs", "articles", "zh-cn", "tensorrtsharp-source-build-cpp-guide.md");
        string packageGuide = ReadText("docs", "articles", "zh-cn", "nuget-github-dual-package-strategy.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");
        string combined = string.Join('\n', sourceBuildGuide, packageGuide, docsIndex, docsToc);

        foreach (string marker in new[]
        {
            "Visual Studio 2022",
            "CMake 3.27",
            "cmake --preset",
            "cmake --build",
            "Generate-Bindings.ps1",
            "Test-BindingGeneratorOutputs.ps1",
            "Test-BridgePackageConsumer.ps1",
            "JYPPX_TENSORRT_ROOT",
            "JYPPX_CUDA_ROOT",
            "JYPPX_CUDNN_ROOT",
            "GitHub Release",
            "NuGet-compatible source",
            "bridge-only",
            "ProjectReference",
            "direct `.nupkg`"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void PublicDocumentationKeepsCurrentApplicationsAndProofBoundariesVisible()
    {
        string combined = string.Join('\n',
            ReadText("README.md"),
            ReadText("README.zh-CN.md"),
            ReadText("docs", "index.md"),
            ReadText("docs", "toc.yml"),
            ReadText("samples", "README.md"),
            ReadText("applications", "README.md"));

        foreach (string marker in new[]
        {
            "applications/YoloVision",
            "applications/OnnxToEngine",
            "applications/TensorRtExec",
            "samples",
            "NuGet"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("samples/YoloDet", combined, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet.csproj", combined, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ProjectReleaseStoryUsesCurrentManifestCountsWithoutClaimingNewPublication()
    {
        string article = ReadText("docs", "articles", "zh-cn", "project-release-story-and-boundaries.md");
        string[] manifestPaths = Directory.GetFiles(
            Path.Combine(RepositoryPaths.Root, "native", "manifests"),
            "*.manifest.json",
            SearchOption.AllDirectories);
        int manifestApiCount = 0;

        foreach (string manifestPath in manifestPaths)
        {
            using JsonDocument manifest = JsonDocument.Parse(File.ReadAllText(manifestPath));
            manifestApiCount += manifest.RootElement.GetProperty("apis").GetArrayLength();
        }

        Assert.True(article.Length >= 12_000, "The release story must remain a complete long-form article.");
        Assert.Contains($"当前 tracked manifest API count：{manifestApiCount}", article, StringComparison.Ordinal);
        Assert.Contains($"当前 tracked manifest file count：{manifestPaths.Length}", article, StringComparison.Ordinal);
        Assert.Contains("manifest/source 匹配不等于 100% 可用", article, StringComparison.Ordinal);
        Assert.Contains("用户自行安装匹配的 TensorRT、CUDA、cuDNN", article, StringComparison.Ordinal);

        foreach (string forbidden in new[]
        {
            "canPublishPublicly=true",
            "performsPublish=true",
            "canCloseReleaseIssue=true",
            "已完成公开发布",
            "已经发布到 NuGet",
            "已经发布到 GitHub Release"
        })
        {
            Assert.DoesNotContain(forbidden, article, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static string ReadRoadmap()
    {
        return ReadText("docs", "articles", "zh-cn", "technical-article-roadmap.md");
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }

    private static int[] ExtractArticleIds(string roadmap)
    {
        return roadmap
            .Split(new[] { "\r\n", "\n" }, StringSplitOptions.None)
            .Select(static line => Regex.Match(line, @"^\|\s*(\d+)\s*\|"))
            .Where(static match => match.Success)
            .Select(static match => int.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture))
            .ToArray();
    }
}
