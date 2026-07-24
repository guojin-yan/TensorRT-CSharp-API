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
    public void CudaTensorRtDllTroubleshootingPublicArticleCoversWindowsLoaderResolverPathContaminationAndTemporaryCopyBoundary()
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
            "Windows native loader 检查表",
            "where jyppxtrtbridge.dll",
            "where nvinfer.dll",
            "where nvinfer_10.dll",
            "where nvinfer_plugin.dll",
            "where nvonnxparser.dll",
            "where cudart64_12.dll",
            "where cudnn64_9.dll",
            "Get-Command jyppxtrtbridge.dll",
            "dumpbin /dependents .\\\\bin\\\\Release\\\\net8.0\\\\jyppxtrtbridge.dll",
            "NativeBridgePathResolver candidate paths",
            "NativeBridgeLibraryLoader load result",
            "NativeBridgeLoadException message",
            "NativeDependencyProbeStatus",
            "ResolvedBridgePath",
            "ResolvedVendorDllDirectory",
            "ProcessArchitecture",
            "RuntimeIdentifier",
            "VC++ runtime installed",
            "temporary-local-diagnostic-copy",
            "PATH 污染与版本漂移",
            "path-contamination",
            "MinimalReproProjectOutsideRepository",
            "PackageReferenceOnly",
            "UsesLocalFeed",
            "UsesProjectReference",
            "UsesDirectNupkg",
            "NativeAssetsCopied",
            "DependencyProbeOnly",
            "RuntimeSmokeAttempted",
            "RuntimeSmokePassed",
            "DriverBlocked",
            "PathContaminationSuspected",
            "TemporaryLocalDiagnosticCopyUsed",
            "dependency probe passed 但 runtime smoke 未执行"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("只能说明 bridge import table 需要什么", content, StringComparison.Ordinal);
        Assert.Contains("不能作为 package content proof", content, StringComparison.Ordinal);
        Assert.Contains("不是 package bug，也不是 runtime proof", content, StringComparison.Ordinal);
    }

    [Fact]
    public void YoloVisionOverviewPublicArticleCoversFamiliesTasksReportsOwnerEvidenceAndProofBoundary()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "yolovision-overview-public-article.md"));

        foreach (string marker in new[]
        {
            "samples/YoloVision",
            "YOLOv5",
            "YOLOv6",
            "YOLOv7",
            "YOLOv8",
            "YOLOv9",
            "YOLOv10",
            "YOLO11",
            "YOLO26",
            "YOLOX",
            "det / cls / seg / obb / pose / sem",
            "samples/YoloVision/yolo-model-matrix.json",
            "samples/YoloVision/yolo-model-matrix.md",
            "samples/YoloVision/yolovision-task-output-contract.json",
            "samples/YoloVision/yolovision-output.schema.json",
            "samples/YoloVision/yolovision-preflight.schema.json",
            "samples/assets/yolovision-family-task-real-asset-roadmap.json",
            "samples/assets/yolovision-article-case-pack.json",
            "samples/assets/yolovision-real-asset-owner-backfill-pack.json",
            "samples/YoloVision/Program.cs",
            "samples/YoloVision/YoloVision.csproj",
            "samples/YoloVision/YoloSampleRunner.cs",
            "samples/YoloVision/YoloVisionResult.cs",
            "samples/YoloVision/YoloVisionOutputReport.cs",
            "samples/YoloVision/YoloVisionPreflightReport.cs",
            "samples/YoloVision/YoloImagePreprocessor.cs",
            "samples/YoloVision/YoloRuntimeOutputSet.cs",
            "samples/YoloVision/YoloRuntimeOutputTensor.cs",
            "samples/YoloVision/YoloRuntimeOutputRoleResolver.cs",
            "samples/YoloVision/YoloMultiOutputMetadata.cs",
            "samples/YoloVision/YoloVisionVisualizationWriter.cs",
            "YoloDetectionDecoder",
            "YoloClassificationPrediction",
            "YoloMaskComposer",
            "YoloSegmentationPrediction",
            "YoloObbDecoder",
            "YoloObbDetection",
            "YoloPoseDecoder",
            "YoloPosePrediction",
            "YoloSemanticMap",
            "YoloEndToEndOutput",
            "YoloXOutputDecoder",
            "--self-test-end2end",
            "ManagedSmoke=YOLOv10EndToEnd Passed=True",
            "ManagedSmokeBoundary=managed-array-decode-only",
            "--preflight",
            "proofClassification=precheck",
            "canPromoteRealModelRuntime=false",
            "--preprocess-only",
            "ImagePreprocess",
            "source/tensor SHA256",
            "--layout end2end",
            "--classification-output logits",
            "--mask-coefficient-count 32",
            "--obb-angle-output angles",
            "--pose-keypoint-count 17",
            "--semantic-output semantic",
            "--detection-output",
            "--classification-output",
            "--semantic-output",
            "--mask-prototypes-output",
            "--pose-keypoints-output",
            "--obb-angle-output",
            "yolovision-output.v1",
            "bindingMetadata",
            "boundary.isRuntimeProof=false",
            "samples/YoloVision/examples/yolovision-output-det.example.json",
            "samples/YoloVision/examples/yolovision-output-cls.example.json",
            "samples/YoloVision/examples/yolovision-output-seg.example.json",
            "samples/YoloVision/examples/yolovision-output-obb.example.json",
            "samples/YoloVision/examples/yolovision-output-pose.example.json",
            "samples/YoloVision/examples/yolovision-output-sem.example.json",
            "eng/Test-YoloVisionOutputReport.ps1",
            "eng/Acquire-YoloV10OfficialAssets.ps1",
            "samples/assets/yolovision-yolov10-official-assets.json",
            "docs/articles/zh-cn/yolovision-yolov10-end-to-end-output-guide.md",
            "artifacts/interface-coverage/yolov10-official-runtime-proof-closure.json",
            "[1,300,6]",
            "x1,y1,x2,y2,score,classId",
            "eng/Acquire-YoloXOfficialAssets.ps1",
            "samples/assets/yolovision-yolox-official-assets.json",
            "docs/articles/zh-cn/yolovision-yolox-official-runtime-tutorial.md",
            "docs/articles/zh-cn/yolovision-yolox-local-package-consumer-tutorial.md",
            "[1,8400,85]",
            "samples/assets/yolovision-yolov8-det-candidate.template.json",
            "samples/assets/yolovision-yolov8-seg-candidate.template.json",
            "samples/assets/yolovision-yolov8-pose-candidate.template.json",
            "samples/assets/yolovision-yolov8-obb-candidate.template.json",
            "samples/assets/yolovision-yolov8-cls-candidate.template.json",
            "samples/assets/yolovision-yolov8-sem-candidate.template.json",
            "owner-action-required",
            "YoloVision Passed=True",
            "sample-run-evidence record",
            "eng/Export-YoloVisionRealAssetOwnerBackfillPack.ps1",
            "eng/Test-YoloVisionRealAssetOwnerBackfillPack.ps1",
            "eng/Export-YoloVisionRealAssetOwnerProofInputTemplate.ps1",
            "eng/Test-YoloVisionRealAssetOwnerProofInput.ps1",
            "eng/Import-YoloVisionRealAssetOwnerProofInput.ps1",
            "eng/Test-YoloVisionRealAssetCandidate.ps1",
            "eng/Test-SampleRunEvidenceRecord.ps1",
            "JYPPX.TensorRT.CSharp.API.YoloVision",
            "YoloVisionCommand.Run(string[] args)",
            "samples/YoloVision.PackageConsumer",
            "eng/Test-YoloVisionLocalPackageConsumer.ps1",
            "eng/Test-YoloVisionLocalPackageConsumerMatrix.ps1",
            "eng/Test-YoloVisionPublicPackageProof.ps1",
            "eng/Test-YoloVisionPublicPackageConsumer.ps1",
            "local-package-consumer-runtime",
            "support matrix",
            "preflight report",
            "SVG visualization",
            "local feed",
            "ProjectReference",
            "direct `.nupkg`",
            "GitHub Actions dry-run",
            "TensorRtExec build-only report",
            "OnnxToEngine report",
            "dependency-probe-only",
            "GUI screenshot",
            "command preview",
            "package-consumer-runtime string in sample evidence"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 package-consumer-runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不能授权发布", content, StringComparison.Ordinal);
        Assert.Contains("不能关闭 release issue", content, StringComparison.Ordinal);
    }

    [Fact]
    public void ProjectOverviewPublicArticleCoversArchitectureSamplesPackagesArticleMatrixAndReleaseBoundary()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "project-overview-public-article.md"));

        foreach (string marker in new[]
        {
            "src/JYPPX.TensorRtSharp",
            "src/JYPPX.CudaSharp",
            "src/JYPPX.TensorRtSharp.Tools",
            "src/JYPPX.Shared",
            "native/src/tensorrt",
            "native/src/cuda",
            "native/manifests/tensorrt/v8",
            "native/manifests/tensorrt/v10",
            "native/manifests/tensorrt/v11",
            "native/manifests/cuda",
            "samples/OnnxToEngine",
            "samples/YoloVision",
            "applications/TensorRtExec",
            "pack/runtime",
            "pack/runtime-split",
            "artifacts/final-release",
            "TensorRtBuilder",
            "TensorRtBuilderConfig",
            "TensorRtRuntime",
            "TensorRtEngine",
            "TensorRtExecutionContext",
            "TensorRtOnnxParser",
            "TensorRtOnnxParserRefitter",
            "TensorRtPluginRegistryInventory",
            "TensorRtEngineInspector",
            "CudaDevice",
            "CudaEnvironmentProbe",
            "TensorRtEnvironmentProbe",
            "native/generated/bridge_api_catalog.g.h",
            "native/generated/bridge_entrypoints.g.h",
            "GeneratedApiCatalog.g.cs",
            "GeneratedEntryPointNames.g.cs",
            "GeneratedNativeMethods.g.cs",
            "GeneratedTensorRtManifestNativeMethods.g.cs",
            "GeneratedCudaManifestNativeMethods.g.cs",
            "artifacts/interface-coverage/tensorrt-interface-comparison.csv",
            "artifacts/interface-coverage/project-completion-review.md",
            "TensorRtExecCommand.cs",
            "TensorRtExecService.cs",
            "TensorRtExecReport.cs",
            "MainForm.cs",
            "TrtexecLikeParser",
            "TrtexecLikeOptions",
            "YOLOv5",
            "YOLOv6",
            "YOLOv7",
            "YOLOv8",
            "YOLOv9",
            "YOLOv10",
            "YOLO11",
            "YOLO26",
            "YOLOX",
            "det、cls、seg、obb、pose、sem",
            "GitHub full runtime packages",
            "NuGet small managed/core + bridge package route",
            "pack/runtime/runtime-packages.manifest.json",
            "pack/runtime-split/split-runtime-packages.manifest.json",
            "docs/articles/zh-cn/publishing/package-strategy-public-article.md",
            "docs/articles/zh-cn/publishing/nuget-install-runtime-package-public-article.md",
            "docs/articles/zh-cn/publishing/native-bridge-build-public-article.md",
            "docs/articles/zh-cn/publishing/source-build-windows-public-article.md",
            "docs/articles/zh-cn/publishing/cuda-tensorrt-dll-troubleshooting-public-article.md",
            "artifacts/final-release/release-evidence-bundle.json",
            "artifacts/final-release/release-close-preflight.json",
            "artifacts/final-release/final-release-close-blocker-dashboard.md",
            "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json",
            "artifacts/final-release/post-publish-verification-record.json",
            "artifacts/final-release/technical-article-publication-matrix.md",
            "artifacts/final-release/technical-article-campaign-matrix.md",
            "README.md",
            "README.zh-CN.md",
            "docs/index.md",
            "docs/toc.yml",
            "docs/articles/zh-cn/project-overview.md",
            "docs/articles/zh-cn/tensorrtsharp-4-project-overview-campaign.md",
            "docs/articles/zh-cn/project-release-story-and-boundaries.md",
            "docs/articles/zh-cn/publishing/onnx-to-engine-public-article.md",
            "docs/articles/zh-cn/publishing/onnxtoengine-trtexec-parity-public-article.md",
            "applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json",
            "applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json",
            "samples/YoloVision/yolo-model-matrix.json",
            "samples/YoloVision/yolovision-task-output-contract.json",
            "samples/assets/yolovision-article-case-pack.json",
            "samples/assets/yolovision-real-asset-owner-backfill-pack.json",
            "dotnet build .\\TensorRtSharp.sln -c Debug --no-restore",
            "dotnet run --project .\\applications\\TensorRtExec\\TensorRtExec.csproj -- --help",
            "dotnet run --project .\\samples\\OnnxToEngine\\OnnxToEngine.csproj -- --help",
            "dotnet run --project .\\samples\\YoloVision\\YoloVision.csproj -- --help",
            "eng/Test-YoloVisionRealAssetCandidate.ps1",
            "eng/Test-SampleRunEvidenceRecord.ps1",
            "eng/Test-ExternalRuntimeProofRecord.ps1",
            "technical-and-promo-article-matrix-30plus.md",
            "article-roadmap-30plus.md",
            "article-roadmap-30plus.json",
            "manifest/source match != runtime proof",
            "generated interop != high-level wrapper",
            "build-only report != package-consumer-runtime proof",
            "local feed != public package source",
            "ProjectReference != package consumer",
            "direct .nupkg install != post-publish verification",
            "YoloVision matrix != real-model-runtime proof",
            "TensorRtExec report != release close approval",
            "package-consumer-runtime",
            "Linux runner proof",
            "real-model-runtime",
            "owner authorization",
            "post-publish verification",
            "blocked-real-proof-required"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 package-consumer-runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不能把 guidance 写成发布完成声明", content, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", content, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", content, StringComparison.Ordinal);
    }

    [Fact]
    public void PackageConsumerProofPublicArticleCoversCleanConsumerOwnerInputValidatorsAndForbiddenSubstitutes()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "package-consumer-proof-public-article.md"));

        foreach (string marker in new[]
        {
            "clean consumer project",
            "public package source",
            "runtime smoke",
            "logSha256Matches=true",
            "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json",
            "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.md",
            "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json",
            "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
            "artifacts/final-release/package-consumer-runtime-proof-record.json",
            "artifacts/final-release/package-consumer-runtime-proof-record-validation.json",
            "artifacts/final-release/package-consumer-runtime-proof-record-validation.md",
            "artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json",
            "artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.md",
            "artifacts/final-release/package-consumer-runtime-proof-execution-pack.md",
            "artifacts/final-release/clean-consumer-proof-owner-execution-pack.md",
            "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.md",
            "artifacts/final-release/external-runtime-proof-record.json",
            "artifacts/final-release/external-runtime-proof-validation.json",
            "artifacts/final-release/external-clean-consumer-proof-kit.md",
            "artifacts/final-release/post-publish-verification-record.json",
            "artifacts/final-release/release-close-preflight.json",
            "artifacts/final-release/final-release-close-blocker-dashboard.md",
            "eng/Export-CleanConsumerProofOwnerExecutionPack.ps1",
            "eng/Test-CleanConsumerProofOwnerExecutionPack.ps1",
            "eng/Import-PackageConsumerRuntimeProofOwnerInput.ps1",
            "eng/Test-PackageConsumerRuntimeProofRecord.ps1",
            "eng/Test-ExternalRuntimeProofRecord.ps1",
            "eng/Test-PostPublishVerificationRecord.ps1",
            "eng/Test-ReleaseIssueCloseRecord.ps1",
            "eng/Test-PackageConsumerRuntimeProofCandidate.ps1",
            "eng/Export-PackageConsumerRuntimeProofCandidate.ps1",
            "cleanExternalConsumerRoot",
            "consumerProjectPath",
            "publicPackageSourceKind",
            "publicPackageSource",
            "publicPackageFeedUrl",
            "managedPackageUrl",
            "managedPackageId",
            "managedPackageVersion",
            "managedNupkgSha256",
            "runtimePackageUrl",
            "runtimePackageId",
            "runtimePackageVersion",
            "runtimePackageKey",
            "runtimeNupkgSha256",
            "ownerName",
            "machineName",
            "hostOs",
            "hostArchitecture",
            "gpuName",
            "cudaDriverVersion",
            "cudaDriverSupportedRuntime",
            "cudaRuntimeVersion",
            "cudnnVersion",
            "tensorRtVersion",
            "tensorRtLine",
            "restoreCommand",
            "buildCommand",
            "smokeCommand",
            "exitCode",
            "startedAtUtc",
            "finishedAtUtc",
            "dependencyProbeStatus",
            "smokeStatus",
            "nativeAssetsCopied",
            "smokeLogPath",
            "smokeLogSha256",
            "stdoutSummary",
            "stderrSummary",
            "failureDiagnostic",
            "no-stderr-emitted",
            "dotnet new console -n TensorRtSharpConsumerProof",
            "dotnet add package JYPPX.TensorRT.CSharp.API",
            "--runtime-package-key",
            "-RequireExistingLog",
            "-FailOnNotProof",
            "package-consumer-runtime-proof-owner-input.imported.json",
            "package-consumer-runtime-proof-owner-input-import.json",
            "package-consumer-runtime-proof-owner-input-import.md",
            "proofClassification",
            "package-consumer-runtime",
            "isPackageConsumerRuntimeProof=false",
            "canPromoteProof=false",
            "local feed package consumer",
            "ProjectReference consumer",
            "direct `.nupkg` install",
            "owner execution package",
            "collection package",
            "GitHub Actions dry-run",
            "queued GitHub Actions run",
            "missing self-hosted runner",
            "dependency-probe-only log",
            "bridge-only diagnostics",
            "compile-surface proof",
            "TensorRtExec report",
            "TensorRtExec GUI screenshot",
            "command preview",
            "OnnxToEngine report",
            "YoloVision candidate template",
            "YoloVision matrix",
            "sample-run evidence",
            "real-model-runtime",
            "post-publish input draft",
            "release close preflight",
            "stale claim audit",
            "owner authorization",
            "Linux runner proof",
            "post-publish verification",
            "release issue close record",
            "blocked-real-proof-required",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "performsPublish=false",
            "不要 workflow dispatch",
            "不要 NuGet/GitHub Packages 发布",
            "不要上传 GitHub Release"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 package consumer release proof", content, StringComparison.Ordinal);
        Assert.Contains("不会执行 `dotnet nuget push`", content, StringComparison.Ordinal);
        Assert.Contains("不会关闭 release issue", content, StringComparison.Ordinal);
        Assert.Contains("不应替 owner 做公开发布决定", content, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtExecCliPublicArticleCoversCliGuiParityReportsRuntimeArtifactsAndProofBoundary()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "tensorrtexec-cli-public-article.md"));

        foreach (string marker in new[]
        {
            "applications/TensorRtExec/README.md",
            "applications/TensorRtExec/TensorRtExec.csproj",
            "applications/TensorRtExec/Console/TensorRtExecCommand.cs",
            "applications/TensorRtExec/Core/TensorRtExecOptions.cs",
            "applications/TensorRtExec/Core/TensorRtExecService.cs",
            "applications/TensorRtExec/Core/TensorRtExecReport.cs",
            "applications/TensorRtExec/WinForms/MainForm.cs",
            "applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json",
            "applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json",
            "applications/TensorRtExec/tensor-rt-exec-gui-cli-field-map.json",
            "applications/TensorRtExec/tensor-rt-exec-report.schema.json",
            "src/JYPPX.TensorRtSharp.Tools/TrtexecLikeParser.cs",
            "src/JYPPX.TensorRtSharp.Tools/TrtexecLikeOptions.cs",
            "src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildOptions.cs",
            "src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildService.cs",
            "src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildDiagnostics.cs",
            "src/JYPPX.TensorRtSharp.Tools/OnnxEngineRuntimeArtifactWriter.cs",
            "TensorRtExecCommand",
            "TensorRtExecOptions",
            "TensorRtExecService",
            "TensorRtExecReport",
            "MainForm",
            "--onnx",
            "--model",
            "--onnxFile",
            "--saveEngine",
            "--save-engine",
            "--engine",
            "--plan",
            "--engineFile",
            "--loadEngine",
            "--load-engine",
            "--buildOnly",
            "--dryRun",
            "--previewOnly",
            "--exportReport",
            "--report",
            "--evidenceSidecar",
            "--minShapes",
            "--optShapes",
            "--maxShapes",
            "--shapes",
            "--inputShapes",
            "--batch",
            "--fp16",
            "--bf16",
            "--noTF32",
            "--int8",
            "--calib",
            "--fp8",
            "--best",
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
            "--iterations",
            "--warmUp",
            "--duration",
            "--streams",
            "--infStreams",
            "--avgRuns",
            "--percentile",
            "--threads",
            "--useSpinWait",
            "--useCudaGraph",
            "--noDataTransfers",
            "--loadInputs",
            "--dumpOutput",
            "--dumpRawBindingsToFile",
            "--exportOutput",
            "--exportTimes",
            "--sleepTime",
            "--idleTime",
            "--timingCacheFile",
            "--timingCache",
            "--exportTimingCache",
            "--profilingVerbosity",
            "--verbose",
            "--dumpProfile",
            "--separateProfileRun",
            "--exportProfile",
            "--saveProfile",
            "--dumpLayerInfo",
            "--exportLayerInfo",
            "--plugins",
            "--plugin",
            "--dynamicPlugins",
            "--setPluginsToSerialize",
            "--safe",
            "--consistency",
            "--builderCache",
            "--noBuilderCache",
            "OptionImplementationStatus",
            "ParsedOptions",
            "AppliedOptions",
            "ParseOnlyOptions",
            "implemented",
            "implemented-report",
            "implemented-build-readback",
            "implemented-bounded-runtime",
            "parse-report-only",
            "diagnostic-alias-compatible",
            "checklist-backed-command-preview",
            "matrixState = release-readiness-planning",
            "runtimeProofItems = 0",
            "packageConsumerRuntimeProofItems = 0",
            "ProofClassification",
            "BuildEvidenceOnly",
            "DryRun",
            "NormalizedCommandLine",
            "NormalizedCommandSha256",
            "DeploymentOptions",
            "BuilderConfigDeploymentSnapshot",
            "ParserPreflightSnapshot",
            "RuntimeOptions",
            "InferenceRan",
            "OutputMatch",
            "IsRuntimeExecutionProof",
            "IsRealModelRuntimeProof",
            "IsPackageConsumerRuntimeProof",
            "PreflightMetadata",
            "LoadedEngineDiagnostics",
            "CapabilityProbe",
            "WorkspaceBytes",
            "ReportBoundary.ForbiddenSubstitutes",
            "ReportBoundary.CopiedDiagnosticsBoundary",
            "eng/Test-TensorRtExecReport.ps1",
            "ArtifactProofBoundary",
            "RuntimeProofClass",
            "HasTensorOutputProof",
            "HasRawBindingProof",
            "IsBuildOnlyEvidence",
            "IsDependencyProbeOnly",
            "IsSyntheticRuntime",
            "*.engine-readback.json",
            "ReadbackFingerprint",
            "ReadbackSha256",
            "runtime-output-captured-unverified",
            "synthetic-input-runtime is not real-model-runtime",
            "eng/Export-TensorRtExecGuiCliParityChecklist.ps1",
            "eng/Test-TensorRtExecGuiCliParityChecklist.ps1",
            "GUI screenshot",
            "command preview",
            "samples/OnnxToEngine",
            "samples/YoloVision",
            "YOLOv5",
            "YOLOv6",
            "YOLOv7",
            "YOLOv8",
            "YOLOv9",
            "YOLOv10",
            "YOLO11",
            "YOLO26",
            "det/cls/seg/obb/pose/sem",
            "samples/assets/yolovision-real-asset-owner-backfill-pack.json",
            "samples/YoloVision/yolovision-task-output-contract.json",
            "real-case-proof-execution-pack.json",
            "tensorrtexec-build-report",
            "tensorrtexec-gui-workflow",
            "eng/Export-RealCaseProofExecutionPack.ps1",
            "eng/Export-RealCaseEvidenceRecordTemplate.ps1",
            "eng/Test-RealCaseEvidenceRecord.ps1",
            "precheck",
            "dependency-probe-only",
            "capability-probe-only",
            "sidecar-only",
            "blocked-by-cuda-driver",
            "Linux runner proof",
            "owner authorization",
            "post-publish verification",
            "canPublishPublicly=false"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 package-consumer-runtime proof", content, StringComparison.Ordinal);
        Assert.Contains("不能授权发布", content, StringComparison.Ordinal);
        Assert.Contains("不能关闭 release issue", content, StringComparison.Ordinal);
        Assert.Contains("不要把 callback、allocator、plugin lifecycle", content, StringComparison.Ordinal);
        Assert.Contains("伪装成低风险实现", content, StringComparison.Ordinal);
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
    public void SourceBuildPublicArticleCoversGeneratedAbiVersionGuardsPackageLayoutAndForbiddenProofSubstitutes()
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
            "Developer PowerShell for VS 2022",
            "VCToolsVersion",
            "WindowsSDKVersion",
            "native/manifests/tensorrt",
            "native/manifests/cuda",
            "GeneratedTensorRtManifestNativeMethods.g.cs",
            "NativeBridgeApi.TensorRtBindings.Generated.g.cs",
            "NativeMethodsTensorRt.Generated.g.cs",
            "NativeMethodsCuda.Generated.g.cs",
            "native/generated/bridge_api_catalog.g.h",
            "native/generated/bridge_entrypoints.g.h",
            "Test-TensorRtNativeAbiSurface.ps1",
            "build-out/win-x64-trt11-cuda13-release",
            "Version Guard 核对",
            "legacy parser / network flags",
            "strongly typed network",
            "precision constraints / layer precision",
            "refit / stripped plan",
            "debug listener / callback",
            "borrowed pointer",
            "callback、allocator、plugin lifecycle",
            "external resource",
            "runtime deserialization ownership",
            "Test-ManagedPackageContent.ps1",
            "Test-RuntimePackageReadiness.ps1",
            "Validate-SplitRuntimePackages.ps1",
            "cmake configure log",
            "cmake build log",
            "binding generator validation log",
            "targeted dotnet test log",
            "jyppxtrtbridge.dll",
            "nvinfer_plugin.dll",
            "nvonnxparser.dll",
            "cudart64_*.dll",
            "public package URL",
            "clean consumer root",
            "merged transcript SHA256",
            "不能升级为 Proof",
            "dependency-probe-only",
            "blocked-by-cuda-driver",
            "local feed consumer",
            "ProjectReference consumer",
            "direct `.nupkg` install",
            "package inventory ready",
            "runtime readiness ready",
            "public package download template",
            "GitHub Actions dry-run",
            "queued workflow",
            "Linux runner proof",
            "real-model-runtime proof",
            "post-publish verification",
            "owner authorization",
            "release issue close record"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("manifest 是契约源", content, StringComparison.Ordinal);
        Assert.Contains("构建即使偶然通过，也不应进入发布候选", content, StringComparison.Ordinal);
        Assert.Contains("不把 TRT8 raw enum 发送给 TRT11", content, StringComparison.Ordinal);
        Assert.Contains("不能证明包已经从 NuGet.org、GitHub Packages 或 GitHub Release", content, StringComparison.Ordinal);
        Assert.Contains("不能替代 package-consumer-runtime proof", content, StringComparison.Ordinal);
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
    public void PackageStrategyPublicArticleCoversRepositoryPackEntrypointsSplitPackagesHashProofAndReleaseCloseBoundary()
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
            "pack/JYPPX.TensorRT.CSharp.API/JYPPX.TensorRT.CSharp.API.csproj",
            "pack/JYPPX.TensorRT.CSharp.API/README.md",
            "pack/runtime/Directory.Build.props",
            "pack/runtime/README.md",
            "pack/runtime/runtime-packages.manifest.json",
            "pack/runtime/linux-runtime-targets.manifest.json",
            "pack/runtime/runtime-package-smoke-command-template.json",
            "pack/runtime/runtime-packages.local.example.json",
            "pack/runtime-split/Directory.Build.props",
            "pack/runtime-split/README.md",
            "pack/runtime-split/split-runtime-packages.manifest.json",
            "TensorRtRuntime",
            "TensorRtBuilder.Sm75Sm86",
            "TensorRtBuilder.Sm89Sm90",
            "TensorRtBuilder.Sm100Sm120Ptx",
            "Invoke-LocalRuntimePackage.ps1",
            "Invoke-LocalSplitRuntimePackage.ps1",
            "Resolve-SplitPackagePins.ps1",
            "Validate-SplitRuntimePackages.ps1",
            "Test-RuntimePackageReadiness.ps1",
            "Export-ReleaseCandidatePackageInventory.ps1",
            "Export-PreReleasePackageProofReadinessMatrix.ps1",
            "Export-PackageConsumerDualRouteProofPlan.ps1",
            "dual-package-publish-preflight-matrix.json",
            "pre-release-package-proof-readiness-matrix.json",
            "package-consumer-dual-route-proof-plan.json",
            "release-package-proof-bundle.json",
            "public-package-url-hash-verification-candidate.json",
            "publicPackageSource",
            "publicPackageUrl",
            "downloadedNupkgSha256",
            "expectedNupkgSha256",
            "packageHashMatch",
            "packageReferenceOnly",
            "cleanConsumerRoot",
            "smokeExitCode",
            "nativeAssetsCopied",
            "mergedTranscriptSha256",
            "publicPackageUrlHashVerification",
            "publicPackageDownloadProof",
            "Linux runner proof",
            "real-model-runtime proof",
            "owner authorization",
            "release issue close record",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不能写入公开文章作为下载来源", content, StringComparison.Ordinal);
        Assert.Contains("模板本身不是执行日志", content, StringComparison.Ordinal);
        Assert.Contains("不能只记录 meta package 名称", content, StringComparison.Ordinal);
        Assert.Contains("不等于 clean consumer proof", content, StringComparison.Ordinal);
        Assert.Contains("不能被文章", content, StringComparison.Ordinal);
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
    public void EngineInspectorPublicArticleCoversInspectorBoundaryLevelsArtifactsBoundedRuntimeAndPromotionFlags()
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
            "Inspector 边界分层",
            "Engine-level metadata",
            "Layer-level metadata",
            "Association state",
            "GetEngineInformation",
            "TensorRtLayerInformationFormat.Oneline",
            "TensorRtLayerInformationFormat.Json",
            "SetEngineInspectorExecutionContext",
            "binding address",
            "device buffer",
            "stream synchronization",
            "TensorRtErrorRecorderSnapshot",
            "TensorRtErrorRecorderSummary",
            "TensorRtErrorRecord",
            "Readback artifact 字段",
            "ArtifactKind",
            "ArtifactBoundary",
            "EngineInspectorApiAvailable",
            "SkippedReason",
            "RuntimeOutputCaptured",
            "OutputValidationPerformed",
            "capability-probe-only",
            "trtexec-like-engine-readback-skipped",
            "Bounded runtime 与 inspector 的分界",
            "LoadEngineReadonlyDiagnostics Attempted=True Succeeded=True",
            "LoadEngineBoundedRuntime Attempted=True Succeeded=True",
            "CapabilityProbe State=capability-probe-only",
            "EngineInspectorReadonlyMetadata = true",
            "EngineInspectorCreatedExecutionBindings = false",
            "EngineInspectorEnqueuedInference = false",
            "EngineInspectorValidatedOutputs = false",
            "EngineInspectorCanPromoteRuntimeProof = false",
            "EngineInspectorCanPromotePackageConsumerProof = false"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不是 runtime log hash", content, StringComparison.Ordinal);
        Assert.Contains("不是 proof 升级器", content, StringComparison.Ordinal);
        Assert.Contains("ReadbackSha256` 不是 smoke log SHA256", content, StringComparison.Ordinal);
        Assert.Contains("InspectorInformationLength` 不是 per-layer timing", content, StringComparison.Ordinal);
        Assert.Contains("EngineInspectorApiAvailable` 不是 TensorRT runtime 已经执行", content, StringComparison.Ordinal);
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
    public void PluginInventoryPublicArticleCoversV3MetadataLayerMetadataSourceOnlyReportsAndLifecycleBoundary()
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
            "V3 metadata 与 layer metadata",
            "TensorRtPluginCreatorInfo",
            "TensorRtPluginCreatorSummary",
            "TensorRtPluginFieldInfo",
            "TensorRtPluginFieldSummary",
            "TensorRtPluginCreatorV3MetadataDesignGate",
            "TensorRtPluginCreatorV3MetadataDesignGateResult",
            "TensorRtVersionedInterfaceMetadata",
            "TensorRtPluginV2LayerMetadata",
            "TensorRtPluginV3LayerMetadata",
            "TensorRtPluginV3SerializationFieldInventory",
            "IVersionedInterface metadata",
            "getInterfaceInfo",
            "metadata inspection",
            "plugin lifecycle",
            "Source-only smoke 与报告字段",
            "plugin-inventory-source-only",
            "sourceOnly=True",
            "RuntimeEvidenceKind=source-only-readonly-diagnostics",
            "IsRuntimeExecutionProof=false",
            "IsPackageConsumerRuntimeProof=false",
            "CanPromoteRuntimeProof=false",
            "CanPromoteReleaseProof=false",
            "CanDeleteDeferredRecord=false",
            "EvidenceKind = plugin-inventory-readonly-diagnostics",
            "RuntimeEvidenceKind = readonly-metadata",
            "PluginLifecycleProof = false",
            "PluginCreateProof = false",
            "PluginEnqueueProof = false",
            "PackageConsumerRuntimeProof = false",
            "ForbiddenSubstitutes = local-feed, ProjectReference, direct-nupkg, dashboard, dry-run"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("不回答 creator 能否真正创建可 enqueue", content, StringComparison.Ordinal);
        Assert.Contains("不把 field data pointer 交给 C#", content, StringComparison.Ordinal);
        Assert.Contains("不是 plugin enqueue proof", content, StringComparison.Ordinal);
        Assert.Contains("不是 plugin lifecycle 的放行证", content, StringComparison.Ordinal);
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
    public void NativeBridgeBuildPublicArticleCoversImplementationAcceptanceWrapperUpliftLoaderAndNonProofBridgeConsumers()
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
            "Bridge 实现验收表",
            "native/src/tensorrt/v8/api.cpp",
            "native/src/tensorrt/v10/api.cpp",
            "native/src/tensorrt/v11/api.cpp",
            "native/src/cuda/api.cpp",
            "NativeBridgeApi helper",
            "public wrapper",
            "smoke/quality gate",
            "plugin registry inventory",
            "engine inspector layer information",
            "parser diagnostics",
            "builder config readback",
            "runtime dependency diagnostics",
            "CUDA device/memory/stream 状态",
            "TensorRtPluginRegistryInventory",
            "TensorRtEngineInspectorReport",
            "TensorRtOnnxParserDiagnosticSnapshot",
            "TensorRtBuilderConfigReadback",
            "CudaDeviceInfo",
            "CudaMemoryInfo",
            "callback trampoline",
            "exception-to-status",
            "keep-alive",
            "in-flight drain",
            "detach-before-release",
            "allocator ownership",
            "owner ledger",
            "plugin lifecycle",
            "creator metadata",
            "field metadata copied snapshot",
            "runtime deserialization ownership",
            "serialized buffer copied-before-interop",
            "Engine handle owned by wrapper",
            "plugin library dependency diagnostics",
            "loadRuntime ownership 模型",
            "Loader 与运行时解析",
            "NativeBridgePathResolver.EnumerateCandidatePaths",
            "NativeBridgeLoadException",
            "runtimes/<rid>/native",
            "jyppxtrtbridge.dll",
            "jyppxcudabridge.dll",
            "nvinfer_plugin.dll",
            "nvonnxparser.dll",
            "cudart64_*.dll",
            "Test-TensorRtNativeAbiSurface.ps1",
            "FullyQualifiedName~TensorRtNativeAbiSurfaceParityTests",
            "Bridge 组件成功",
            "downloaded nupkg SHA256",
            "native asset copy",
            "host metadata",
            "runtime smoke",
            "queued workflow",
            "package inventory ready",
            "runtime readiness ready",
            "public package download template",
            "Bridge package local consumer passed"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }

        Assert.Contains("没有 public wrapper 的 native entrypoint 仍然只是低层能力", content, StringComparison.Ordinal);
        Assert.Contains("没有 smoke/quality gate 的 wrapper 只能算实验入口", content, StringComparison.Ordinal);
        Assert.Contains("没有 release proof 的 runtime path 不能推动发布", content, StringComparison.Ordinal);
        Assert.Contains("文章测试替代", content, StringComparison.Ordinal);
        Assert.Contains("Bridge 包通过不能替代 TensorRT runtime 真实执行", content, StringComparison.Ordinal);
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
            "articles/zh-cn/publishing/project-overview-public-article.md",
            "artifacts/interface-coverage/project-completion-review.md",
            "package-consumer-runtime proof"),
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
            "articles/zh-cn/publishing/tensorrtexec-cli-public-article.md",
            "applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json",
            "build-only"),
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
