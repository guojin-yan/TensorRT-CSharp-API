using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ProjectCompletionReviewTests
{
    [Fact]
    public void ReviewRemainsABoundedCurrentStateIndex()
    {
        string path = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "project-completion-review.md");
        FileInfo file = new(path);
        string content = File.ReadAllText(path);

        Assert.True(file.Length < 64 * 1024, $"Completion review grew to {file.Length} bytes.");
        Assert.Contains("当前状态索引", content, StringComparison.Ordinal);
        Assert.Contains("publication-catalog.json", content, StringComparison.Ordinal);
        Assert.Contains("JYPPX_DEMO_IMAGE_ROOT", content, StringComparison.Ordinal);
        Assert.Contains("CUDA、cuDNN、TensorRT 和 NVRTC 由用户自行安装", content, StringComparison.Ordinal);
        Assert.Contains("NuGet push", content, StringComparison.Ordinal);
        Assert.Contains("不再追加每次提交的实现日记", content, StringComparison.Ordinal);
    }
}
