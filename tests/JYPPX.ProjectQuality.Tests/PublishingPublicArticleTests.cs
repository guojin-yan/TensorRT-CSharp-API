using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PublishingPublicArticleTests
{
    [Fact]
    public void PublishingPublicArticlesAreLinkedAndKeepProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        foreach (ArticleExpectation article in Articles)
        {
            string href = article.Href;
            string path = Path.Combine(RepositoryPaths.Root, "docs", href.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(path), href);

            string content = File.ReadAllText(path);
            Assert.Contains(href, docsIndex, StringComparison.Ordinal);
            Assert.Contains(href, docsToc, StringComparison.Ordinal);
            Assert.Contains("## 适合", content, StringComparison.Ordinal);
            Assert.Contains("## 配图建议", content, StringComparison.Ordinal);
            Assert.Contains("## 下一步", content, StringComparison.Ordinal);
            Assert.Contains(article.RequiredPath, content, StringComparison.Ordinal);
            Assert.Contains(article.RequiredBoundary, content, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void PublishingPublicArticlesDoNotClaimRuntimeProofFromTemplatesOrBuildOnlyOutputs()
    {
        foreach (ArticleExpectation article in Articles)
        {
            string path = Path.Combine(RepositoryPaths.Root, "docs", article.Href.Replace('/', Path.DirectorySeparatorChar));
            string content = File.ReadAllText(path);

            Assert.DoesNotContain("已经证明 package-consumer-runtime", content, StringComparison.Ordinal);
            Assert.DoesNotContain("模板就是 runtime proof", content, StringComparison.Ordinal);
            Assert.DoesNotContain("build-only 就是 runtime proof", content, StringComparison.Ordinal);
            Assert.Contains("proof", content, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void SourceBuildPublicArticleCoversCppBridgeEnvironmentPresetsPackagesAndTroubleshooting()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "source-build-windows-public-article.md"));

        foreach (string marker in new[]
        {
            "C++ bridge DLL",
            "CUDA_PATH",
            "NvInfer.h",
            "NvOnnxParser.h",
            "cuDNN",
            "win-x64-trt8-cuda11-release",
            "win-x64-trt10-cuda12-release",
            "win-x64-trt11-cuda13-release",
            "Generate-Bindings.ps1",
            "Test-BindingGeneratorOutputs.ps1",
            "Export-InterfaceCoverageMatrix.ps1",
            "TensorRtNativeAbiSurfaceParityTests",
            "PublicApiHandleExposureAuditTests",
            "dumpbin /dependents",
            "CUDA error 35",
            "GitHub full runtime 包",
            "NuGet 小包",
            "docs/articles/zh-cn/source-build-cmake-windows-guide.md",
            "docs/articles/zh-cn/tensorrtsharp-source-build-cpp-guide.md",
            "package-consumer-runtime proof"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("```mermaid", content, StringComparison.Ordinal);
        Assert.Contains("CMake 找不到 CUDA", content, StringComparison.Ordinal);
        Assert.Contains("TensorRT 头文件和 lib 不匹配", content, StringComparison.Ordinal);
        Assert.Contains("DLL 加载失败", content, StringComparison.Ordinal);
        Assert.Contains("不要把系统目录污染", content, StringComparison.Ordinal);
    }

    [Fact]
    public void PackageStrategyPublicArticleCoversDualRoutesRuntimeKeysAndProofBoundaries()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "package-strategy-public-article.md"));

        foreach (string marker in new[]
        {
            "GitHub full runtime 包",
            "NuGet small bridge/core 包",
            "managed API",
            "C++ bridge DLL",
            "Bridge",
            "CudaCudnn",
            "TensorRt",
            "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge",
            "win-x64-trt11.0-cuda13.2-cudnn9.22",
            "Export-DualPackagePublishPreflightMatrix.ps1",
            "Export-FinalOwnerExecutionChecklist.ps1",
            "release-docs-and-nuget-metadata-audit.json",
            "release-candidate-package-inventory.md",
            "clean external consumer",
            "post-publish verification",
            "strict validator",
            "nonSubstituteConfirmations",
            "local feed consumer",
            "ProjectReference consumer",
            "direct `.nupkg` install",
            "build-only report",
            "dependency-probe-only report",
            "failedBlockerCount=0"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不能作为 package-consumer-runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不能说“公开发布已经完成”", content, StringComparison.Ordinal);
        Assert.Contains("“runtime proof 已完成”", content, StringComparison.Ordinal);
    }

    private static readonly ArticleExpectation[] Articles =
    {
        new(
            "articles/zh-cn/publishing/nuget-install-runtime-package-public-article.md",
            "JYPPX.TensorRT.CSharp.API",
            "package-consumer-runtime proof"),
        new(
            "articles/zh-cn/publishing/yolovision-overview-public-article.md",
            "samples/assets/yolovision-yolov8-det-candidate.template.json",
            "owner-action-required"),
        new(
            "articles/zh-cn/publishing/onnxtoengine-trtexec-parity-public-article.md",
            "applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json",
            "build-only"),
        new(
            "articles/zh-cn/publishing/cuda-tensorrt-dll-troubleshooting-public-article.md",
            "dotnet --info",
            "package-consumer-runtime proof"),
        new(
            "articles/zh-cn/publishing/package-consumer-proof-public-article.md",
            "ProjectReference",
            "Package Consumer Runtime Proof"),
        new(
            "articles/zh-cn/publishing/plugin-inventory-public-article.md",
            "src/JYPPX.TensorRtSharp/TensorRtPluginRegistryInventory.cs",
            "package-consumer-runtime proof"),
        new(
            "articles/zh-cn/publishing/engine-inspector-public-article.md",
            "applications/TensorRtExec/Core/TensorRtExecReport.cs",
            "readonly diagnostics"),
        new(
            "articles/zh-cn/publishing/deferred-boundary-public-article.md",
            "artifacts/interface-coverage/project-completion-review.md",
            "manifest/source"),
        new(
            "articles/zh-cn/publishing/release-evidence-ladder-public-article.md",
            "artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json",
            "package-consumer-runtime proof"),
        new(
            "articles/zh-cn/publishing/builder-config-readback-public-article.md",
            "applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json",
            "build-only"),
        new(
            "articles/zh-cn/publishing/source-build-windows-public-article.md",
            "cmake --preset win-x64-trt11-cuda13-release",
            "build-only"),
        new(
            "articles/zh-cn/publishing/native-bridge-build-public-article.md",
            "native/generated/bridge_entrypoints.g.h",
            "readonly diagnostics"),
        new(
            "articles/zh-cn/publishing/package-strategy-public-article.md",
            "artifacts/final-release/owner-external-proof-execution-result.input.json",
            "package-consumer-runtime proof"),
        new(
            "articles/zh-cn/publishing/onnx-to-engine-public-article.md",
            "samples/OnnxToEngine/Program.cs",
            "build-only")
    };

    private sealed record ArticleExpectation(string Href, string RequiredPath, string RequiredBoundary);
}
