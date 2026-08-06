using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class UserFacingReadmeLocalizationTests
{
    public static TheoryData<string, string, string> ReadmePairs => new()
    {
        { string.Empty, "README.md", "README.zh-CN.md" },
        { "samples", "README.md", "README.zh-CN.md" },
        { Path.Combine("samples", "_shared"), "README.md", "README.zh-CN.md" },
        { Path.Combine("samples", "assets"), "README.md", "README.zh-CN.md" },
        { Path.Combine("samples", "Cuda", "01.RuntimeCompilation"), "README.md", "README.zh-CN.md" },
        { Path.Combine("samples", "Inference", "01.Bindings"), "README.md", "README.zh-CN.md" },
        { Path.Combine("samples", "Inference", "02.DynamicShapes"), "README.md", "README.zh-CN.md" },
        { Path.Combine("samples", "Performance", "01.MultiStream"), "README.md", "README.zh-CN.md" },
        { Path.Combine("samples", "ComputerVision", "01.Classification"), "README.md", "README.zh-CN.md" },
        { "applications", "README.md", "README.zh-CN.md" },
        { Path.Combine("applications", "OnnxToEngine"), "README.md", "README.zh-CN.md" },
        { Path.Combine("applications", "YoloVision"), "README.md", "README.zh-CN.md" },
        { Path.Combine("applications", "TensorRtExec"), "README.md", "README.en.md" },
    };

    [Theory]
    [MemberData(nameof(ReadmePairs))]
    public void UserFacingReadmesHaveLinkedEnglishAndChinesePages(
        string relativeDirectory,
        string primaryFileName,
        string counterpartFileName)
    {
        string directory = Path.Combine(RepositoryPaths.Root, relativeDirectory);
        string primaryPath = Path.Combine(directory, primaryFileName);
        string counterpartPath = Path.Combine(directory, counterpartFileName);

        Assert.True(File.Exists(primaryPath), $"Missing README: {primaryPath}");
        Assert.True(File.Exists(counterpartPath), $"Missing localized README: {counterpartPath}");

        string primary = File.ReadAllText(primaryPath);
        string counterpart = File.ReadAllText(counterpartPath);

        Assert.Contains(counterpartFileName, primary, StringComparison.Ordinal);
        Assert.Contains(primaryFileName, counterpart, StringComparison.Ordinal);
        Assert.Contains("English", primary + counterpart, StringComparison.Ordinal);
        Assert.Contains("简体中文", primary + counterpart, StringComparison.Ordinal);
    }

    [Fact]
    public void SampleSeriesDocumentsCompleteModelArticleAndResultRequirements()
    {
        string chineseReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "README.zh-CN.md"));
        string chineseSeries = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "sample-series-overview.md"));
        string combined = chineseReadme + chineseSeries;

        foreach (string marker in new[]
        {
            "<workspace-root>/models",
            "ONNX",
            "许可证",
            "SHA256",
            "转换命令",
            "真实程序运行页面",
            "绘制回原图",
            "不上传 GitHub",
        })
        {
            Assert.Contains(marker, combined, StringComparison.Ordinal);
        }
    }
}
