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

    [Fact]
    public void BuilderConfigReadbackPublicArticleCoversTrtexecControlsVersionGuardsAndProofBoundaries()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "builder-config-readback-public-article.md"));

        foreach (string marker in new[]
        {
            "TensorRtBuilderConfig.cs",
            "TensorRtBuilderConfig.Trt11Diagnostics.cs",
            "TrtexecLikeDeploymentOptions.cs",
            "OnnxEngineBuildService.cs",
            "OnnxEngineBuildDiagnostics.cs",
            "tensor-rt-exec-release-candidate-gap-list.json",
            "TrtexecMemoryPool",
            "TrtexecTiming",
            "TrtexecDeploymentControl",
            "Requested",
            "Readback",
            "ReadbackMatch",
            "EvidenceBoundary=builder-config-readback-only",
            "--workspace",
            "--memPoolSize",
            "--avgTiming",
            "--minTiming",
            "--tacticSources",
            "--profilingVerbosity",
            "--exportTimingCache",
            "SetMemoryPoolLimit",
            "GetMemoryPoolLimit",
            "SetAverageTimingIterations",
            "GetAverageTimingIterations",
            "MaxWorkspaceSizeCompatibilityInBytes",
            "MinTimingIterationsCompatibility",
            "TRT8",
            "TRT10/11",
            "TRT11",
            "dependency-probe-only",
            "build-only report",
            "parse-only report",
            "TensorRtExec GUI screenshot",
            "local feed package consumer",
            "package-consumer-runtime proof",
            "clean external consumer",
            "owner input validator",
            "post-publish verification"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不能单独晋级", content, StringComparison.Ordinal);
        Assert.Contains("不能用 presence probe 替代生命周期设计", content, StringComparison.Ordinal);
    }

    [Fact]
    public void DeferredBoundaryPublicArticleCoversRiskTiersRuntimeDeserializationAndNoSubstituteProof()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "deferred-boundary-public-article.md"));

        foreach (string marker in new[]
        {
            "manifest/source match",
            "non-deferred native bridge",
            "typed C# wrapper",
            "package-consumer-runtime proof",
            "deferred-boundary-risk-tier-gate.md",
            "deferred-manual-design-groups.md",
            "runtime-deserialization-deferred-boundary-audit.md",
            "A-tier copied value",
            "B-tier safe alternative",
            "C-tier design-gate-required",
            "D-tier keep-deferred",
            "默认低风险 deferred 候选已经为 `0`",
            "algorithm selector borrowed objects",
            "IGpuAllocator",
            "IGpuAsyncAllocator",
            "IOutputAllocator",
            "IDebugListener",
            "registerCreator",
            "deregisterCreator",
            "loadLibrary",
            "executeV2",
            "enqueueV2",
            "IDimensionExpr",
            "IExprBuilder",
            "deserializeCudaEngineV2",
            "loadRuntime",
            "TensorRtRuntime.Deserialize(byte[])",
            "TensorRtRuntimeDeserializationBoundaryPrecheck.EvaluateKnownSurface",
            "TensorRtRuntimeDeserializationDependencyDiagnostics.EvaluateKnownSurface",
            "TensorRtStreamIoInterfaceInfoDesignGate.EvaluateKnownSurface",
            "trt10-runtime-deserialize-cuda-engine-v2-deferred",
            "trt11-runtime-deserialize-cuda-engine-v2-deferred",
            "trt8-runtime-load-runtime-deferred",
            "trt10-runtime-load-runtime-deferred",
            "trt11-runtime-load-runtime-deferred",
            "dependency-probe-only",
            "owner-action-required",
            "local feed package consumer",
            "ProjectReference consumer",
            "direct `.nupkg` install"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("manifest/source 100% 只能证明", content, StringComparison.Ordinal);
        Assert.Contains("不能证明“用户可以安全调用这个接口”", content, StringComparison.Ordinal);
        Assert.Contains("不要删除 deferred manifest 来制造完成度", content, StringComparison.Ordinal);
        Assert.Contains("不暴露 `IRuntime*`、`ICudaEngine*`", content, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseEvidenceLadderPublicArticleCoversOwnerInputForbiddenSubstitutesAndCloseBoundaries()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "release-evidence-ladder-public-article.md"));

        foreach (string marker in new[]
        {
            "package-consumer-runtime-proof-owner-input.template.json",
            "package-consumer-runtime-proof-owner-input.schema.json",
            "package-consumer-runtime-proof-forbidden-substitute-scan.json",
            "public-docs-package-metadata-gate.json",
            "Test-PackageConsumerRuntimeProofOwnerInput.ps1",
            "Import-PackageConsumerRuntimeProofOwnerInput.ps1",
            "Test-PackageConsumerRuntimeProofRecord.ps1",
            "Test-ReleaseIssueCloseRecord.ps1",
            "final-owner-execution-package.json",
            "fieldCount=80",
            "requiredFieldCount=49",
            "cleanExternalConsumerRoot",
            "publicPackageSourceKind",
            "managedNupkgSha256",
            "runtimePackageKey",
            "smokeLogSha256",
            "stdoutSummary",
            "stderrSummary",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "canPromoteProof=false",
            "blocked-forbidden-substitute-detected",
            "repository path leakage",
            "template placeholder",
            "local feed",
            "ProjectReference",
            "direct `.nupkg`",
            "build-only",
            "dry-run",
            "queued GitHub Actions run",
            "missing self-hosted runner",
            "GitHub Actions dry-run `.nupkg`",
            "dashboard",
            "GUI screenshot",
            "TensorRtExec build report only",
            "isDryRunOnly=true",
            "isPublishedPackageProof=false",
            "isPackageConsumerRuntimeProof=false",
            "manualWorkflowDispatchNotPerformed=true",
            "post-publish verification",
            "Linux runner proof",
            "real-model-runtime proof",
            "rollback review",
            "final close decision",
            "blocked-owner-public-postpublish-proof-required",
            "failedBlockerCount=0"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 ready-to-publish", content, StringComparison.Ordinal);
        Assert.Contains("不是 release close", content, StringComparison.Ordinal);
        Assert.Contains("不要 workflow dispatch", content, StringComparison.Ordinal);
        Assert.Contains("不能推动 release close", content, StringComparison.Ordinal);
    }

    [Fact]
    public void EngineInspectorPublicArticleCoversReadbackArtifactsAndReadonlyProofBoundary()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "engine-inspector-public-article.md"));

        foreach (string marker in new[]
        {
            "TensorRtEngineInspector.Trt11Diagnostics.cs",
            "OnnxEngineBuildResult.cs",
            "OnnxEngineRuntimeArtifactWriter.cs",
            "OnnxEngineBuildDiagnostics.cs",
            "TensorRtExecReport.cs",
            "TensorRtExecCommand.cs",
            "MainForm.cs",
            "tensor-rt-exec-trtexec-parity-matrix.json",
            "GetLayerInformation",
            "HasExecutionContext",
            "TryGetErrorRecorderSnapshot",
            "OnnxLoadedEngineDiagnostics",
            "EngineName",
            "IOTensorCount",
            "LayerCount",
            "OptimizationProfileCount",
            "DeviceMemorySizeInBytes",
            "AuxiliaryStreamCount",
            "ProfilingVerbosity",
            "ReadbackFingerprint",
            "ReadbackSha256",
            "EvidenceBoundary",
            "--dumpLayerInfo",
            "--exportLayerInfo",
            "--profilingVerbosity",
            "trtexec-like-engine-readback",
            "trtexec-like-engine-readback-skipped",
            "LoadEngineDiagnosticsState",
            "LoadEngineDiagnosticsAttempted",
            "LoadEngineDiagnosticsSucceeded",
            "LoadEngineDiagnosticsBoundary",
            "ProofClassification",
            "BuildEvidenceOnly",
            "InferenceRan",
            "NormalizedCommandSha256",
            "IsRuntimeExecutionProof = false",
            "IsRealModelRuntimeProof = false",
            "IsPackageConsumerRuntimeProof = false",
            "load-engine readonly diagnostics",
            "does not create execution bindings",
            "enqueue inference",
            "validate outputs",
            "local feed package consumer",
            "ProjectReference consumer",
            "direct `.nupkg` install",
            "clean external consumer",
            "post-publish verification"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 real-model-runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不是 package-consumer-runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不伪造 per-layer timing", content, StringComparison.Ordinal);
        Assert.Contains("GUI 能显示 report", content, StringComparison.Ordinal);
    }

    [Fact]
    public void PluginInventoryPublicArticleCoversSourcesCopiedMetadataSmokeAndProofBoundary()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "plugin-inventory-public-article.md"));

        foreach (string marker in new[]
        {
            "TensorRtPluginRegistryInventory.cs",
            "TensorRtBuilder.PluginRegistryInventory.cs",
            "TensorRtRuntime.PluginRegistryInventory.cs",
            "TensorRtEnvironmentProbe.PluginRegistryInventory.cs",
            "plugin_registry_inventory.inc",
            "NativeBridgeApi.PluginRegistryInventory.cs",
            "NativeBridgeApi.RuntimePluginRegistryInventory.cs",
            "PluginRegistryInventorySmokeRunner",
            "tensorrt-interface-comparison.csv",
            "PluginRegistryInventoryTests",
            "PluginInventorySourceOnlySmokeTests",
            "PluginCreatorApiLanguageReadonlyTests",
            "PluginCreatorV3MetadataDesignGateTests",
            "TensorRtPluginRegistrySource.Builder",
            "TensorRtPluginRegistrySource.Global",
            "TensorRtPluginRegistrySource.BuilderCapability",
            "TensorRtPluginRegistrySource.Runtime",
            "HasErrorRecorder",
            "ParentSearchEnabled",
            "CreatorCount",
            "RecursiveCreatorCount",
            "FindCreator",
            "TryFindCreator",
            "GetCreatorSummaries",
            "GetFieldSummaries",
            "GetDiagnostics",
            "InterfaceKind",
            "InterfaceMajor",
            "InterfaceMinor",
            "ApiLanguage",
            "TensorRtVersion",
            "FieldType",
            "Length",
            "HasData",
            "TensorRtPluginRegistryInventoryDiagnostics",
            "EmptyFieldNameCount",
            "NegativeFieldLengthCount",
            "getAllCreators",
            "SEH guard",
            "vendor mismatch",
            "getAllCreatorsRecursive",
            "createPlugin",
            "clone",
            "serialize",
            "deserializePlugin",
            "attachToContext",
            "enqueue",
            "registerCreator",
            "deregisterCreator",
            "loadLibrary",
            "Skipped=True",
            "DependencyProbeOnly",
            "local feed package consumer",
            "ProjectReference consumer",
            "direct `.nupkg` install",
            "clean external consumer",
            "forbidden substitute scan",
            "not package-consumer-runtime proof"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 real-model-runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不是 package-consumer-runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不暴露 borrowed pointer", content, StringComparison.Ordinal);
        Assert.Contains("不会重新拿 native pointer", content, StringComparison.Ordinal);
    }

    [Fact]
    public void OnnxToEnginePublicArticleCoversSharedParserReportsYoloVisionAndProofBoundary()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "onnx-to-engine-public-article.md"));

        foreach (string marker in new[]
        {
            "samples/OnnxToEngine/Program.cs",
            "TrtexecLikeParser.cs",
            "TrtexecLikeOptions.cs",
            "OnnxEngineBuildOptions.cs",
            "OnnxEngineBuildService.cs",
            "OnnxEngineBuildResult.cs",
            "OnnxEngineBuildDiagnostics.cs",
            "OnnxEngineBuildReportWriter.cs",
            "OnnxEngineBuildEvidenceSidecar.cs",
            "TensorRtExecCommand.cs",
            "MainForm.cs",
            "tensor-rt-exec-trtexec-parity-matrix.json",
            "tensor-rt-exec-release-candidate-gap-list.json",
            "yolovision-article-case-pack.json",
            "yolovision-family-task-real-asset-roadmap.json",
            "--onnx",
            "--model",
            "--onnxFile",
            "--saveEngine",
            "--loadEngine",
            "--minShapes",
            "--optShapes",
            "--maxShapes",
            "--fp16",
            "--int8",
            "--bf16",
            "--workspace",
            "--memPoolSize",
            "--timingCache",
            "--exportTimingCache",
            "--profilingVerbosity",
            "--dumpLayerInfo",
            "--exportLayerInfo",
            "--refitFromOnnx",
            "--saveRefittedEngine",
            "--allowWeightStreaming",
            "--weightStreamingBudget",
            "--useCudaGraph",
            "--loadInputs",
            "--dumpOutput",
            "--dumpRawBindingsToFile",
            "--exportOutput",
            "--exportTimes",
            "--exportProfile",
            "--saveProfile",
            "--dryRun",
            "--previewOnly",
            "--buildOnly",
            "--skipInference",
            "--exportReport",
            "--evidenceSidecar",
            "Success",
            "Skipped",
            "State",
            "ModelSource",
            "EnginePath",
            "Parsed",
            "EngineSaved",
            "EngineFileRoundTrip",
            "InferenceRan",
            "OutputMatch",
            "ProofClassification",
            "BuildEvidenceOnly",
            "IsRuntimeExecutionProof",
            "IsRealModelRuntimeProof",
            "IsPackageConsumerRuntimeProof",
            "NormalizedCommandSha256",
            "PreflightMetadata",
            "LoadedEngineDiagnostics",
            "BuilderConfigDeploymentSnapshot",
            "ParserPreflightSnapshot",
            "TimingCacheArtifact",
            "CapabilityProbe",
            "EvidenceSidecar",
            ".engine-readback.json",
            "MnistOnnxRuntimeService",
            "Expected",
            "Predicted",
            "Confidence",
            "samples/YoloVision",
            "YOLOv5",
            "YOLOv6",
            "YOLOv7",
            "YOLOv8",
            "YOLOv9",
            "YOLOv10",
            "YOLO11",
            "YOLO26",
            "detection",
            "classification",
            "segmentation",
            "OBB",
            "pose",
            "semantic segmentation",
            "clean external consumer",
            "post-publish verification",
            "release close"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 package-consumer-runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不能证明公开包可被用户消费", content, StringComparison.Ordinal);
        Assert.Contains("不要把 dry-run 写成已经构建 engine", content, StringComparison.Ordinal);
        Assert.Contains("不要把模型、engine、runtime package 或 NuGet 临时包下载到 C 盘", content, StringComparison.Ordinal);
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
