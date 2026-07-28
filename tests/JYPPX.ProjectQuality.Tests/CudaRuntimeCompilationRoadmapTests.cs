using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaRuntimeCompilationRoadmapTests
{
    [Fact]
    public void RoadmapsPreserveOwnerPackagingAndProofBoundaries()
    {
        string chinese = ReadSource("docs", "articles", "zh-cn", "cuda-runtime-compilation-roadmap.md");
        string english = ReadSource("docs", "articles", "en", "cuda-runtime-compilation-roadmap.md");
        string combined = chinese + english;

        string[] requiredTokens =
        {
            "JYPPX_CudaRtcProgram",
            "CudaRtcCompiler",
            "CudaRtcProgram",
            "CudaRtcCompilationResult",
            "CudaRtcArtifact",
            "CudaKernelLibrary.Load(byte[])",
            "CUDA 11.8",
            "CUDA 12.1",
            "CUDA 12.9",
            "CUDA 13.2",
            "PTX",
            "CUBIN",
            "LTO IR",
            "nvrtc-builtins",
            "caller-buffer",
            "compile-to-launch",
            "post-publish"
        };

        foreach (string token in requiredTokens)
        {
            Assert.Contains(token, combined, StringComparison.Ordinal);
        }

        Assert.Contains("IntPtr", combined, StringComparison.Ordinal);
        Assert.Contains("SafeHandle", combined, StringComparison.Ordinal);
        Assert.Contains("does not bundle NVRTC", english, StringComparison.Ordinal);
        Assert.Contains("does not use RTC must not fail to load", english, StringComparison.Ordinal);
        Assert.Contains("not kernel-output correctness", english, StringComparison.Ordinal);
    }

    [Fact]
    public void RoadmapsAreLinkedFromPublicNavigation()
    {
        string readme = ReadSource("README.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");

        Assert.Contains("docs/articles/en/cuda-runtime-compilation-roadmap.md", readme, StringComparison.Ordinal);
        Assert.Contains("docs/articles/zh-cn/cuda-runtime-compilation-roadmap.md", readme, StringComparison.Ordinal);
        Assert.Contains("articles/en/cuda-runtime-compilation-roadmap.md", index, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/cuda-runtime-compilation-roadmap.md", index, StringComparison.Ordinal);
        Assert.Contains("articles/en/cuda-runtime-compilation-roadmap.md", toc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/cuda-runtime-compilation-roadmap.md", toc, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
