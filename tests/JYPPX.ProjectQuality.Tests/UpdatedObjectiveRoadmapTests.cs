using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class UpdatedObjectiveRoadmapTests
{
    [Fact]
    public void ProjectRoadmapPinsRealDeferredApiWorkAndStopsProofDashboardDrift()
    {
        string roadmap = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "project-roadmap-to-public-release.md"));

        foreach (string marker in new[]
        {
            "真实 deferred 接口提升 + 可发布项目完整度",
            "no-arg placeholder",
            "native source",
            "C# interop",
            "高层 wrapper",
            "manifest/source 100%",
            "100% runtime 可用",
            "非 deferred 实现",
            "TensorRtExec report",
            "OnnxToEngine report",
            "YoloVision matrix",
            "不要继续扩 proof dashboard"
        })
        {
            Assert.Contains(marker, roadmap, StringComparison.Ordinal);
        }

        foreach (string forbidden in new[]
        {
            "canPromoteReleaseProof=true",
            "canDeleteDeferredRecord=true",
            "public IntPtr",
            "public nint"
        })
        {
            Assert.DoesNotContain(forbidden, roadmap, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void RoadmapCapturesSourceBuildDualPackageSamplesAppsAndArticleObjectives()
    {
        string roadmap = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "project-roadmap-to-public-release.md"));

        foreach (string marker in new[]
        {
            "源码编译教程路线",
            "Windows Visual Studio C++ toolchain",
            "cmake --preset",
            "cmake --build --preset",
            "Generate-Bindings.ps1",
            "Test-BindingGeneratorOutputs.ps1",
            "JYPPX_TENSORRT_ROOT",
            "JYPPX_CUDA_ROOT",
            "GitHub 全依赖包",
            "NuGet 小包",
            "C# 核心 API",
            "中间 C++ bridge",
            "用户自行安装 CUDA / TensorRT / cuDNN",
            "samples/YoloVision",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det、cls、seg、obb、pose、sem",
            "samples/OnnxToEngine",
            "applications/TensorRtExec",
            "CLI 和 WinForms GUI",
            "微信公众号",
            "博客",
            "不少于 30 篇"
        })
        {
            Assert.Contains(marker, roadmap, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("samples/YoloDet", roadmap, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void NextReadonlyQueueAvoidsKnownOwnerRiskAndNamesImplementableBatches()
    {
        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "deferred-next-readonly-candidate-list.md"));

        foreach (string marker in new[]
        {
            "ONNX parser copied diagnostics / layer-output presence",
            "LayerOutputTensorExists",
            "subgraph count/copy",
            "CUDA graph / memory range copied query",
            "count/copy",
            "caller-owned output struct",
            "IDimensionExpr::isConstant/getConstantValue/isSizeTensor",
            "降级为 design gate",
            "不得新增 public `IDimensionExpr` wrapper",
            "IStreamReader/IStreamWriter::getInterfaceInfo",
            "不得暴露 native reader/writer pointer",
            "plugin instance create/clone/enqueue",
            "registry register/deregister/load library"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("优先做 `IDimensionExpr` copied scalar snapshot", article, StringComparison.Ordinal);
    }
}
