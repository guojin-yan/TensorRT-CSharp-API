using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CompatibleHostSourceRuntimeTests
{
    [Fact]
    public void CMakeInvalidatesTensorRtLibrariesThatDoNotBelongToTheSelectedRoot()
    {
        string cmake = ReadSource("CMakeLists.txt");

        Assert.Contains("function(jyppx_invalidate_stale_tensorrt_library_cache cache_var)", cmake, StringComparison.Ordinal);
        Assert.Contains("cmake_path(IS_PREFIX _jyppx_normalized_tensorrt_root", cmake, StringComparison.Ordinal);
        Assert.Contains("unset(${cache_var} CACHE)", cmake, StringComparison.Ordinal);
        foreach (string cacheName in new[]
        {
            "TensorRT_NVINFER_LIBRARY",
            "TensorRT_NVINFER_PLUGIN_LIBRARY",
            "TensorRT_NVONNXPARSER_LIBRARY",
            "TensorRT_NVPARSERS_LIBRARY"
        })
        {
            Assert.Contains(cacheName, cmake, StringComparison.Ordinal);
        }

        int invalidation = cmake.IndexOf("jyppx_invalidate_stale_tensorrt_library_cache", StringComparison.Ordinal);
        int discovery = cmake.IndexOf("find_package(TensorRT QUIET)", StringComparison.Ordinal);
        Assert.True(invalidation >= 0 && invalidation < discovery);
    }

    [Fact]
    public void HarnessBuildsCurrentSourceRunsRealSmokeAndCannotPublish()
    {
        string script = ReadSource("eng", "Test-CompatibleHostSourceRuntime.ps1");

        foreach (string required in new[]
        {
            "Resolve-RequiredDirectory",
            "include\\NvInfer.h",
            "bin\\nvcc.exe",
            "Get-CMakeCacheValue",
            "Test-PathInside",
            "TensorRT_NVINFER_LIBRARY",
            "cmake\" -ArgumentList @(\"--build\", \"--preset\"",
            "TensorRtSmokeRunner\\TensorRtSmokeRunner.csproj",
            "TryCreateRuntime$TensorRtLine=True:",
            "TryCreateBuilder$TensorRtLine=True:",
            "TryBuildSerializedNetwork$TensorRtLine=True:",
            "TryRunMinimalBuildChain$TensorRtLine=True:",
            "HighLevelChain$TensorRtLine=True:",
            "Enqueue=True",
            "ImportLibrariesInsideTensorRtRoot=True",
            "ExactPackageMatrix=$exactPackageMatrix",
            "performsPublish = $false",
            "Write-TerminalScreenshot"
        })
        {
            Assert.Contains(required, script, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("dotnet pack", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("nuget push", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release create", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("git tag", script, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RealEvidenceIsPathFreeHashBoundAndKeepsTheCompatibilityBoundary()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts", "real-case", "tensorrt10-compatible-host-source-runtime",
            "compatible-host-source-runtime-evidence.json");
        string evidenceText = File.ReadAllText(evidencePath);
        using JsonDocument document = JsonDocument.Parse(evidenceText);
        JsonElement root = document.RootElement;

        Assert.Equal("compatible-host-source-runtime", root.GetProperty("recordKind").GetString());
        Assert.Equal("passed-compatible-host-source-runtime", root.GetProperty("validationState").GetString());
        Assert.Equal(10, root.GetProperty("tensorRtLine").GetInt32());
        Assert.Equal("10.13.0", root.GetProperty("tensorRtVersion").GetString());
        Assert.Equal("11.8", root.GetProperty("cudaToolkitVersion").GetString());
        Assert.Equal("TensorRT-10.13.0.35-cu11", root.GetProperty("tensorRtSdkLabel").GetString());
        Assert.Equal("v11.8", root.GetProperty("cudaSdkLabel").GetString());

        JsonElement build = root.GetProperty("build");
        Assert.True(build.GetProperty("configured").GetBoolean());
        Assert.True(build.GetProperty("built").GetBoolean());
        Assert.True(build.GetProperty("importLibrariesInsideSelectedTensorRtRoot").GetBoolean());
        JsonElement[] libraries = build.GetProperty("importLibraries").EnumerateArray().ToArray();
        Assert.Equal(3, libraries.Length);
        Assert.All(libraries, library =>
        {
            Assert.EndsWith(".lib", library.GetProperty("fileName").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Matches("^[0-9a-f]{64}$", library.GetProperty("sha256").GetString()!);
        });
        Assert.Equal("jyppxtrtbridge.dll", build.GetProperty("bridgeFileName").GetString());
        Assert.Matches("^[0-9a-f]{64}$", build.GetProperty("bridgeSha256").GetString()!);

        JsonElement runtime = root.GetProperty("runtime");
        Assert.True(runtime.GetProperty("runtimeCreated").GetBoolean());
        Assert.True(runtime.GetProperty("builderCreated").GetBoolean());
        Assert.True(runtime.GetProperty("serializedNetworkBuilt").GetBoolean());
        Assert.True(runtime.GetProperty("minimalBuildChainCompleted").GetBoolean());
        Assert.True(runtime.GetProperty("highLevelChainCompleted").GetBoolean());
        Assert.True(runtime.GetProperty("enqueueCompleted").GetBoolean());
        Assert.Empty(runtime.GetProperty("missingMarkers").EnumerateArray());

        JsonElement model = root.GetProperty("model");
        Assert.Equal("not-applicable", model.GetProperty("acquisition").GetString());
        Assert.Equal("not-applicable", model.GetProperty("conversion").GetString());
        Assert.False(model.GetProperty("onnxRequired").GetBoolean());
        Assert.False(model.GetProperty("externalModelRequired").GetBoolean());
        Assert.False(model.GetProperty("outerModelsDirectoryUsed").GetBoolean());

        foreach (JsonProperty artifact in root.GetProperty("artifacts").EnumerateObject())
        {
            string relativePath = artifact.Value.GetProperty("path").GetString()!;
            string path = Path.Combine(
                RepositoryPaths.Root,
                relativePath.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(path), $"Missing compatible-host artifact: {relativePath}");
            Assert.Equal(artifact.Value.GetProperty("length").GetInt64(), new FileInfo(path).Length);
            string hash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
            Assert.Equal(artifact.Value.GetProperty("sha256").GetString(), hash);
        }

        JsonElement boundary = root.GetProperty("boundary");
        Assert.True(boundary.GetProperty("isCurrentSourceBridgeRuntimeEvidence").GetBoolean());
        Assert.True(boundary.GetProperty("isSameMajorCompatibleHostEvidence").GetBoolean());
        Assert.False(boundary.GetProperty("isExactPackageMatrixEvidence").GetBoolean());
        Assert.False(boundary.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("isPublicPackageProof").GetBoolean());
        Assert.False(boundary.GetProperty("isReleaseProof").GetBoolean());
        Assert.False(boundary.GetProperty("performsPack").GetBoolean());
        Assert.False(boundary.GetProperty("performsPublish").GetBoolean());
        Assert.False(boundary.GetProperty("createsTag").GetBoolean());
        Assert.False(boundary.GetProperty("createsRelease").GetBoolean());

        Assert.DoesNotMatch(@"[A-Za-z]:\\", evidenceText);
        Assert.DoesNotContain("GitSpace", evidenceText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Program Files", evidenceText, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ArticleExplainsTheWholeFlowUsesTheRealScreenshotAndAvoidsLocalPaths()
    {
        string article = ReadSource(
            "docs", "articles", "zh-cn", "tensorrt10-compatible-host-source-runtime.md");
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");

        foreach (string required in new[]
        {
            "## 1. 项目与验证目标",
            "## 2. 使用到的库与职责",
            "## 3. 模型获取与转换说明",
            "权重获取方式：不适用",
            "ONNX 转换命令：不适用",
            "## 4. 环境准备",
            "## 5. 为什么要校验 CMake 缓存",
            "## 6. 一键执行源码兼容宿主验证",
            "Test-CompatibleHostSourceRuntime.ps1",
            "## 7. 程序内部执行流程",
            "## 8. 通过条件与失败条件",
            "## 9. 本机真实执行结果",
            "tensorrt10-compatible-host-source-runtime-terminal.png",
            "TryCreateRuntime10=True",
            "HighLevelChain10=True",
            "Enqueue=True",
            "## 10. 如何复查证据",
            "## 11. 10.13 结果不能证明什么",
            "不创建 tag、不创建 GitHub Release、不发布新包"
        })
        {
            Assert.Contains(required, article, StringComparison.Ordinal);
        }

        Assert.Contains("tensorrt10-compatible-host-source-runtime.md", toc, StringComparison.Ordinal);
        Assert.Contains("tensorrt10-compatible-host-source-runtime.md", index, StringComparison.Ordinal);
        Assert.DoesNotMatch(@"[A-Za-z]:\\", article);
        Assert.DoesNotContain("GitSpace", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Program Files", article, StringComparison.OrdinalIgnoreCase);

        string screenshot = Path.Combine(
            RepositoryPaths.Root,
            "docs", "images", "tensorrt10-compatible-host-source-runtime-terminal.png");
        Assert.True(File.Exists(screenshot));
        Assert.True(new FileInfo(screenshot).Length > 10_000);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}
