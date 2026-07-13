using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class SourceBuildCmakeWindowsGuideTests
{
    [Fact]
    public void WindowsSourceBuildGuideDocumentsNativeCmakeAndDependencyMatrix()
    {
        string article = ReadSource("docs", "articles", "zh-cn", "source-build-cmake-windows-guide.md");

        Assert.Contains("Windows 源码编译教程", article);
        Assert.Contains("C++ bridge", article);
        Assert.Contains("GitHub 全依赖包", article);
        Assert.Contains("NuGet 小包", article);
        Assert.Contains("Visual Studio 2022", article);
        Assert.Contains("CMake 3.27", article);
        Assert.Contains("CUDA Toolkit", article);
        Assert.Contains("TensorRT", article);
        Assert.Contains("cuDNN", article);
        Assert.Contains("win-x64-trt11-cuda13-release", article);
        Assert.Contains("win-x64-trt10-cuda12-release", article);
        Assert.Contains("pwsh -NoProfile -ExecutionPolicy Bypass -File .\\eng\\Generate-Bindings.ps1", article);
        Assert.Contains("pwsh -NoProfile -ExecutionPolicy Bypass -File .\\eng\\Test-BindingGeneratorOutputs.ps1", article);
        Assert.Contains("cmake --preset win-x64-trt11-cuda13-release", article);
        Assert.Contains("cmake --build --preset win-x64-trt11-cuda13-release --parallel", article);
        Assert.Contains("dotnet test .\\tests\\JYPPX.ProjectQuality.Tests\\JYPPX.ProjectQuality.Tests.csproj", article);
        Assert.Contains("DLL 加载失败", article);
        Assert.Contains("CUDA error 35", article);
        Assert.Contains("samples/YoloVision", article);
        Assert.Contains("samples/OnnxToEngine", article);
        Assert.Contains("applications/TensorRtExec", article);
    }

    [Fact]
    public void WindowsSourceBuildGuideIsLinkedFromDocsIndexAndToc()
    {
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");

        Assert.Contains("source-build-cmake-windows-guide.md", index);
        Assert.Contains("Windows Source Build CMake Guide", index);
        Assert.Contains("source-build-cmake-windows-guide.md", toc);
        Assert.Contains("Windows Source Build CMake Guide", toc);
    }

    [Fact]
    public void CmakePresetsStillExposeDocumentedWindowsReleaseLanes()
    {
        string presets = ReadSource("CMakePresets.json");

        foreach (string preset in new[]
        {
            "win-x64-trt8-cuda11-release",
            "win-x64-trt8-cuda12-release",
            "win-x64-trt10-cuda11-release",
            "win-x64-trt10-cuda12-release",
            "win-x64-trt11-cuda12-release",
            "win-x64-trt11-cuda13-release"
        })
        {
            Assert.Contains($"\"name\": \"{preset}\"", presets);
        }
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
