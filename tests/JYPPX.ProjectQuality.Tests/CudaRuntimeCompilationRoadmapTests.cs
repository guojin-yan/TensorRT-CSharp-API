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

    [Fact]
    public void TechnicalArticleUsesRealRuntimeEvidenceAndNoModelClaims()
    {
        string article = ReadSource("docs", "articles", "zh-cn", "cuda-runtime-compilation-technical-article.md");
        string sampleReadme = ReadSource("samples", "Cuda", "01.RuntimeCompilation", "README.md");
        string evidenceText = ReadSource("samples", "assets", "cuda-rtc-article-runtime-evidence.json");
        using System.Text.Json.JsonDocument evidence = System.Text.Json.JsonDocument.Parse(evidenceText);
        System.Text.Json.JsonElement root = evidence.RootElement;

        Assert.Contains("本文使用的项目与库", article, StringComparison.Ordinal);
        Assert.Contains("本案例不使用深度学习模型，也不读取或生成 ONNX", article, StringComparison.Ordinal);
        Assert.Contains("CudaRtcCompiler", article, StringComparison.Ordinal);
        Assert.Contains("CudaKernelLibrary", article, StringComparison.Ordinal);
        Assert.Contains("CudaDriverModule", article, StringComparison.Ordinal);
        Assert.Contains("终端截图来自本次真实运行的 stdout", article, StringComparison.Ordinal);
        Assert.Contains("../../images/cuda-rtc-runtime-terminal.png", article, StringComparison.Ordinal);
        Assert.Contains("ProcessExitCode=0", article, StringComparison.Ordinal);
        Assert.Contains("不创建 Tag、GitHub Release", article, StringComparison.Ordinal);
        Assert.DoesNotContain("E:\\GitSpace", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("C:\\Users", article, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("-notlike '*.alt.dll'", ReadSource("eng", "Invoke-CudaRtcLocalSmoke.ps1"), StringComparison.Ordinal);
        Assert.Contains("excludes `.alt.dll`", sampleReadme, StringComparison.Ordinal);

        Assert.Equal("local-toolkit-kernel-runtime-readback", root.GetProperty("proofClassification").GetString());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("modelOrOnnxRequired").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("performsPublish").GetBoolean());
        Assert.True(root.GetProperty("runtimeValidation").GetProperty("passed").GetBoolean());

        string imagePath = Path.Combine(RepositoryPaths.Root, "docs", "images", "cuda-rtc-runtime-terminal.png");
        Assert.True(File.Exists(imagePath));
        string actualHash = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(imagePath))).ToLowerInvariant();
        Assert.Equal(root.GetProperty("assets").GetProperty("runtimeScreenshotSha256").GetString(), actualHash);

        string rootReadme = ReadSource("README.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");
        Assert.Contains("docs/articles/zh-cn/cuda-runtime-compilation-technical-article.md", rootReadme, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/cuda-runtime-compilation-technical-article.md", index, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/cuda-runtime-compilation-technical-article.md", toc, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
