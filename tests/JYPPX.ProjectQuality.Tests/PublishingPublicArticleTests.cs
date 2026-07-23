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
    public void NuGetInstallRuntimePackagePublicArticleCoversRuntimeKeysSplitPackagesCleanConsumerAndProofBoundary()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "nuget-install-runtime-package-public-article.md"));

        foreach (string marker in new[]
        {
            "JYPPX.TensorRT.CSharp.API",
            "pack/runtime/runtime-packages.manifest.json",
            "pack/runtime-split/split-runtime-packages.manifest.json",
            "pack/runtime/runtime-package-smoke-command-template.json",
            "pack/runtime/README.md",
            "pack/runtime-split/README.md",
            "docs/articles/zh-cn/tensorrtsharp-nuget-runtime-package-guide.md",
            "docs/articles/zh-cn/runtime-package-selection.md",
            "docs/articles/zh-cn/runtime-package-matrix-reading-guide.md",
            "docs/articles/zh-cn/runtime-package-installation-deep-dive.md",
            "docs/articles/zh-cn/package-consumer-runtime-proof-clean-consumer-guide.md",
            "win-x64-trt8.6-cuda11.8-cudnn8.9",
            "win-x64-trt10.11-cuda12.9-cudnn9.22",
            "win-x64-trt11.0-cuda13.2-cudnn9.22",
            "linux-x64-trt10.11-cuda12.9-cudnn9.22",
            "role = bridge",
            "role = cuda-cudnn",
            "role = tensorrt",
            "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge",
            "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.CudaCudnn",
            "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.TensorRt",
            "key",
            "packageId",
            "rid",
            "platform",
            "tensorRtLine",
            "tensorRtVersion",
            "cudaLine",
            "cudaVersion",
            "cudnnMajor",
            "cudnnVersion",
            "distributionTier",
            "validationState",
            "buildPreset",
            "bridgeFile",
            "tensorRtFiles",
            "cudaFiles",
            "cudnnFiles",
            "dotnet --info",
            "dotnet restore --force-evaluate",
            "dotnet build -c Release",
            "jyppxtrtbridge.dll",
            "nvinfer_10.dll",
            "nvonnxparser_10.dll",
            "cudart64_12.dll",
            "cudnn64_9.dll",
            "public package source URL",
            "managed package id/version",
            "runtime package id/version/runtime key",
            "managed nupkg SHA256",
            "runtime nupkg SHA256",
            "native asset listing",
            "dependency probe log",
            "runtime smoke log",
            "exitCode = 0",
            "OS / architecture / GPU / driver",
            "strict validator result",
            "eng/Test-ExternalRuntimeProofRecord.ps1",
            "eng/Test-PackageConsumerRuntimeProofRecord.ps1",
            "eng/Test-PostPublishVerificationRecord.ps1",
            "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json",
            "artifacts/final-release/package-consumer-runtime-proof-record-validation.md",
            "artifacts/final-release/post-publish-verification-record.json",
            "local feed package consumer",
            "ProjectReference consumer",
            "direct `.nupkg` install",
            "GitHub Actions dry-run",
            "GitHub full runtime collection package",
            "owner execution package",
            "package id/version template",
            "release issue close record template",
            "post-publish verification input draft",
            "blocked-by-cuda-driver",
            "package-consumer-runtime",
            "Linux runner proof",
            "real-model-runtime",
            "owner authorization",
            "post-publish verification"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 package-consumer-runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不能授权发布", content, StringComparison.Ordinal);
        Assert.Contains("不能发布", content, StringComparison.Ordinal);
        Assert.Contains("不能关闭 release issue", content, StringComparison.Ordinal);
        Assert.Contains("不要把 runtime deserialization ownership", content, StringComparison.Ordinal);
        Assert.Contains("伪装成低风险安装问题", content, StringComparison.Ordinal);
    }

    [Fact]
    public void CudaTensorRtDllTroubleshootingPublicArticleCoversNativeLoadDecisionTreeAndProofBoundary()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "cuda-tensorrt-dll-troubleshooting-public-article.md"));

        foreach (string marker in new[]
        {
            "DllNotFoundException",
            "BadImageFormatException",
            "CUDA error 35",
            "CUDA driver/runtime mismatch",
            "native initialization failed",
            "blocked-by-cuda-driver",
            "jyppxtrtbridge.dll",
            "nvinfer_10.dll",
            "nvinfer_plugin_10.dll",
            "nvonnxparser_10.dll",
            "cudart64_12.dll",
            "cudnn64_9.dll",
            "src/JYPPX.Shared/Interop/NativeBridgePathResolver.cs",
            "src/JYPPX.Shared/Interop/NativeBridgeLibraryLoader.cs",
            "src/JYPPX.Shared/BridgeConstants.cs",
            "src/JYPPX.CudaSharp/CudaEnvironmentProbe.cs",
            "src/JYPPX.TensorRtSharp/TensorRtEnvironmentProbe.cs",
            "src/JYPPX.TensorRtSharp.Tools/TensorRtToolSupport.cs",
            "src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildService.cs",
            "NativeLibrary.SetDllImportResolver",
            "NativeLibrary.TryLoad",
            "BridgeConstants.NativeBridgeLibraryName",
            "pack/runtime/runtime-packages.manifest.json",
            "pack/runtime-split/split-runtime-packages.manifest.json",
            "docs/articles/zh-cn/runtime-package-native-load-troubleshooting.md",
            "docs/articles/zh-cn/windows-installation-and-troubleshooting-guide.md",
            "docs/articles/zh-cn/cuda-error-35-troubleshooting.md",
            "docs/articles/zh-cn/runtime-package-minimal-smoke-commands.md",
            "docs/articles/zh-cn/package-consumer-runtime-proof-clean-consumer-guide.md",
            "dotnet --info",
            "nvidia-smi",
            "dotnet list package",
            "dotnet restore --force-evaluate",
            "dotnet build -c Release",
            "Get-ChildItem .\\bin\\Release\\net8.0 -Filter *.dll",
            "$env:PATH -split ';'",
            "key",
            "packageId",
            "rid",
            "tensorRtLine",
            "tensorRtVersion",
            "cudaLine",
            "cudaVersion",
            "cudnnMajor",
            "cudnnVersion",
            "bridgeFile",
            "tensorRtFiles",
            "cudaFiles",
            "cudnnFiles",
            "role = bridge",
            "role = cuda-cudnn",
            "role = tensorrt",
            "native asset listing",
            "dependency probe log",
            "runtime smoke log",
            "stdout/stderr SHA256",
            "managed/runtime package SHA256",
            "OS / architecture / GPU / driver / CUDA / TensorRT / cuDNN metadata",
            "eng/Test-ExternalRuntimeProofRecord.ps1",
            "eng/Test-PackageConsumerRuntimeProofRecord.ps1",
            "eng/Test-PostPublishVerificationRecord.ps1",
            "local feed",
            "ProjectReference",
            "direct `.nupkg` install",
            "GitHub Actions dry-run",
            "dependency-probe-only log",
            "build-only 的 TensorRtExec report",
            "OnnxToEngine report",
            "YoloVision matrix",
            "sidecar-only metadata",
            "GUI screenshot",
            "command preview"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 package-consumer-runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不能授权发布", content, StringComparison.Ordinal);
        Assert.Contains("不能关闭 release issue", content, StringComparison.Ordinal);
        Assert.Contains("不要把 runtime deserialization ownership", content, StringComparison.Ordinal);
        Assert.Contains("伪装成 DLL 加载问题", content, StringComparison.Ordinal);
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

    [Fact]
    public void OnnxParserParserRefitterCopiedDiagnosticsArticleCoversSnapshotsReportsAndReleaseBoundary()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "onnx-parser-parserrefitter-诊断-copied-diagnostics-release-gate.md"));

        foreach (string marker in new[]
        {
            "TensorRtOnnxParser.cs",
            "TensorRtOnnxParserDiagnosticSnapshot.cs",
            "TensorRtOnnxParserDiagnostic.cs",
            "TensorRtOnnxParser.ModelSupport.cs",
            "TensorRtOnnxParserRefitter.cs",
            "TensorRtOnnxParserRefitterDiagnosticSnapshot.cs",
            "NativeBridgeApi.ParserRefitterDiagnostics.cs",
            "OnnxEngineBuildDiagnostics.cs",
            "OnnxEngineParserPreflightSnapshot.cs",
            "OnnxEngineBuildService.cs",
            "Test-BridgePackageConsumer.ps1",
            "Test-RuntimePackageReadiness.ps1",
            "Test-ReleaseQualityGate.ps1",
            "BridgePackageConsumerTests",
            "RuntimePackageReadinessTests",
            "RuntimeSerializationOnnxSupportTests",
            "ParserRefitterBoundaryTests",
            "TensorRtExecReportSchemaTests",
            "TensorRtOnnxParser.GetDiagnosticSnapshot()",
            "TensorRtOnnxParser.GetDiagnosticSummary()",
            "TensorRtOnnxParser.GetDiagnostics()",
            "TensorRtOnnxParser.TryParse",
            "TensorRtOnnxParser.GetUsedVCPluginLibraries()",
            "TensorRtOnnxParser.CheckModelSupport",
            "TensorRtOnnxParserRefitter.GetDiagnosticSnapshot()",
            "TensorRtOnnxParserRefitter.GetDiagnosticSummary()",
            "TensorRtOnnxParserRefitter.GetDiagnostics()",
            "TensorRtOnnxParserRefitter.RefitFromBytes",
            "TensorRtOnnxParserRefitter.RefitLoadedModel",
            "hasOnnxParserDiagnosticReadiness",
            "hasOnnxParserRefitterDiagnosticReadiness",
            "Line",
            "ErrorCount",
            "Diagnostics",
            "DiagnosticSummary",
            "UsedVCPluginLibraries",
            "IdentityOperatorSupported",
            "CopiedDiagnosticCount",
            "DiagnosticSummaryLength",
            "UsedVCPluginLibraryCount",
            "RuntimeEvidenceKind = copied-readonly-summary",
            "IsRuntimeExecutionEvidence = false",
            "IsRuntimeExecutionProof = false",
            "PointerFreeCopiedSummary = true",
            "CanPromoteRuntimeProof = false",
            "CanPromoteReleaseProof = false",
            "CanDeleteDeferredRecord = false",
            "ParserPreflightSnapshot",
            "DiagnosticsState",
            "ModelSupportState",
            "ModelSupported",
            "CopiedSubgraphCount",
            "CopiedUnsupportedSubgraphCount",
            "CopiedNodeCount",
            "ParserDiagnosticsEvidenceKind = copied-parser-diagnostics",
            "ParserRefitterDiagnosticsEvidenceKind = copied-parser-refitter-diagnostics",
            "CopiedDiagnosticsBoundary",
            "ForbiddenSubstitutes",
            "CanPromoteCopiedDiagnosticsToRuntimeProof = False",
            "onnx-parser-diagnostic-readiness",
            "onnx-parser-refitter-diagnostic-readiness",
            "compile-surface-proof",
            "proof=false",
            "local feed package consumer",
            "ProjectReference consumer",
            "direct `.nupkg` install",
            "GitHub Actions dry-run",
            "failedBlockerCount=0"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不是 package-consumer-runtime", content, StringComparison.Ordinal);
        Assert.Contains("不暴露 native borrowed pointer", content, StringComparison.Ordinal);
        Assert.Contains("不要 workflow dispatch", content, StringComparison.Ordinal);
    }

    [Fact]
    public void NativeBridgeBuildPublicArticleCoversAbiGenerationPresetsPackagesAndProofBoundary()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "native-bridge-build-public-article.md"));

        foreach (string marker in new[]
        {
            "Visual Studio 2022",
            "CMake >= 3.27",
            ".NET SDK 8",
            "CUDA_PATH",
            "TensorRT include/lib/bin",
            "cuDNN include/lib/bin",
            "native/manifests/tensorrt/v8",
            "native/manifests/tensorrt/v10",
            "native/manifests/tensorrt/v11",
            "native/manifests/cuda",
            "module",
            "versionLine",
            "entryPoint",
            "ownership",
            "manualOverride",
            "native/generated/bridge_api_catalog.g.h",
            "native/generated/bridge_entrypoints.g.h",
            "GeneratedApiCatalog.g.cs",
            "GeneratedEntryPointNames.g.cs",
            "GeneratedNativeMethods.g.cs",
            "GeneratedTensorRtManifestNativeMethods.g.cs",
            "GeneratedCudaManifestNativeMethods.g.cs",
            "NativeMethodsTensorRt.Generated.g.cs",
            "NativeMethodsCuda.Generated.g.cs",
            "NativeBridgeApi.TensorRtBindings.Generated.g.cs",
            "NativeBridgeApi.TensorRtHelpers.Generated.g.cs",
            "NativeCudaApi.Generated.g.cs",
            "Generate-Bindings.ps1",
            "Test-BindingGeneratorOutputs.ps1",
            "Export-InterfaceCoverageMatrix.ps1",
            "Export-NativeMethodsComparison.ps1",
            "Export-WrapperLiftCandidates.ps1",
            "Export-GeneratedApiCoverage.ps1",
            "SHA256",
            "win-x64-dev",
            "win-x64-trt8-cuda11-release",
            "win-x64-trt8-cuda12-release",
            "win-x64-trt10-cuda11-release",
            "win-x64-trt10-cuda12-release",
            "win-x64-trt11-cuda12-release",
            "win-x64-trt11-cuda13-release",
            "linux-x64-trt8-cuda11-release",
            "linux-x64-trt8-cuda12-release",
            "linux-x64-trt10-cuda11-release",
            "linux-x64-trt10-cuda12-release",
            "linux-x64-trt11-cuda12-release",
            "linux-x64-trt11-cuda13-release",
            "JYPPX_ENABLE_TENSORRT_BINDINGS",
            "JYPPX_ENABLE_CUDA_BINDINGS",
            "JYPPX_TENSORRT_LINE",
            "JYPPX_CUDA_LINE",
            "JYPPX_CUDA_VERSION",
            "JYPPX_TENSORRT_CUDA_VERSION",
            "JYPPX_CUDNN_MAJOR",
            "JYPPX_StatusCode",
            "SafeTensorRtObjectHandle",
            "SafeCudaObjectHandle",
            "NativeBridgePathResolver",
            "NativeBridgeLibraryLoader",
            "TensorRtNativeAbiSurfaceParityTests",
            "PublicApiHandleExposureAuditTests",
            "NativeBridgePathResolverTests",
            "NativeVendorBoundaryGuardTests",
            "SourceBuildCmakeWindowsGuideTests",
            "JYPPX.TensorRT.CSharp.API",
            "JYPPX.TensorRT.CSharp.API.NativeBridge",
            "dumpbin /dependents",
            "local feed package consumer",
            "ProjectReference consumer",
            "direct `.nupkg` install",
            "GitHub Actions dry-run",
            "post-publish verification",
            "release close"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 package-consumer-runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不要把 TensorRT、CUDA、cuDNN、ONNX、engine、runtime package 或 NuGet 临时包下载到 C 盘", content, StringComparison.Ordinal);
        Assert.Contains("不要混用 TRT10 header 和 TRT11 runtime DLL", content, StringComparison.Ordinal);
        Assert.Contains("不用删除记录制造完成度", content, StringComparison.Ordinal);
    }

    [Fact]
    public void OnnxToEngineTrtexecParityPublicArticleCoversMatrixStatusesGuiYoloAndProofBoundary()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "onnxtoengine-trtexec-parity-public-article.md"));

        foreach (string marker in new[]
        {
            "applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json",
            "applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json",
            "samples/OnnxToEngine/Program.cs",
            "TensorRtExecOptions.cs",
            "TensorRtExecService.cs",
            "TensorRtExecReport.cs",
            "TensorRtExecCommand.cs",
            "MainForm.cs",
            "TrtexecLikeParser",
            "TrtexecLikeOptions",
            "OnnxEngineBuildOptions.FromTrtexecLikeOptions",
            "OnnxEngineBuildService",
            "matrixId",
            "matrixState = release-readiness-planning",
            "tensorRtExecStatus",
            "implemented",
            "implemented-report",
            "implemented-build-readback",
            "implemented-bounded-runtime",
            "parse-report-only",
            "diagnostic-alias-compatible",
            "checklist-backed-command-preview",
            "runtimeProofItems = 0",
            "packageConsumerRuntimeProofItems = 0",
            "--onnx",
            "--model",
            "--onnxFile",
            "--saveEngine",
            "--save-engine",
            "--engine",
            "--plan",
            "--engineFile",
            "--loadEngine",
            "--minShapes",
            "--optShapes",
            "--maxShapes",
            "--shapes",
            "--inputShapes",
            "--batch",
            "--fp16",
            "--int8",
            "--calib",
            "--fp8",
            "--best",
            "--dumpRefit",
            "--markDebug",
            "--dumpDebugTensors",
            "--workspace",
            "--memPoolSize",
            "--avgTiming",
            "--minTiming",
            "--device",
            "--useDLACore",
            "--allowGPUFallback",
            "--tacticSources",
            "--directIO",
            "--sparsity",
            "--stronglyTyped",
            "--inputIOFormats",
            "--outputIOFormats",
            "--precisionConstraints",
            "--layerPrecisions",
            "--layerOutputTypes",
            "--versionCompatible",
            "--excludeLeanRuntime",
            "--stripWeights",
            "--refit",
            "--refitFromOnnx",
            "--saveRefittedEngine",
            "--allowWeightStreaming",
            "--weightStreamingBudget",
            "--timingCacheFile",
            "--exportTimingCache",
            "--profilingVerbosity",
            "--dumpProfile",
            "--exportProfile",
            "--saveProfile",
            "--dumpLayerInfo",
            "--exportLayerInfo",
            "--iterations",
            "--warmUp",
            "--duration",
            "--streams",
            "--useCudaGraph",
            "--noDataTransfers",
            "--loadInputs",
            "--dumpOutput",
            "--dumpRawBindingsToFile",
            "--exportOutput",
            "--exportTimes",
            "--sleepTime",
            "tensor-rt-exec-gui-cli-field-map.json",
            "runtime-output-captured-unverified",
            "ProofClassification",
            "BuildEvidenceOnly",
            "NormalizedCommandSha256",
            "OptionImplementationStatus",
            "ReportBoundary.ForbiddenSubstitutes",
            "samples/YoloVision/yolovision-task-output-contract.json",
            "yolovision-real-asset-owner-backfill-pack.json",
            "yolovision-article-case-pack.json",
            "Test-YoloVisionRealAssetCandidate.ps1",
            "Test-YoloVisionRealAssetOwnerBackfillPack.ps1",
            "Test-SampleRunEvidenceRecord.ps1",
            "YOLOv5",
            "YOLOv6",
            "YOLOv7",
            "YOLOv8",
            "YOLOv9",
            "YOLOv10",
            "YOLO11",
            "YOLO26",
            "det/cls/seg/obb/pose/sem",
            "local feed package consumer",
            "ProjectReference consumer",
            "direct `.nupkg` install",
            "GitHub Actions dry-run",
            "post-publish verification"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 package-consumer-runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不能授权发布", content, StringComparison.Ordinal);
        Assert.Contains("不能证明 runtime output", content, StringComparison.Ordinal);
        Assert.Contains("不要伪装成低风险实现", content, StringComparison.Ordinal);
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
