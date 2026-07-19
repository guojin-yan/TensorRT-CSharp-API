using JYPPX.TensorRtSharp.Tools;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecApplicationTests
{
    [Fact]
    public void TensorRtExecProjectIsInApplicationsFolderAndSolution()
    {
        string project = Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "TensorRtExec.csproj");
        string appReadmePath = Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md");
        string solution = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "TensorRtSharp.sln"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "README.md"));
        string projectText = File.ReadAllText(project);
        string appReadme = File.ReadAllText(appReadmePath);

        Assert.True(File.Exists(project));
        Assert.True(File.Exists(appReadmePath));
        Assert.Contains("applications\\TensorRtExec\\TensorRtExec.csproj", solution, StringComparison.Ordinal);
        Assert.Contains("`TensorRtExec`", readme, StringComparison.Ordinal);
        Assert.Contains("WinForms", appReadme, StringComparison.Ordinal);
        Assert.Contains("ProofClassification", appReadme, StringComparison.Ordinal);
        Assert.Contains("ArtifactProofBoundary", appReadme, StringComparison.Ordinal);
        Assert.Contains("RuntimeProofClass", appReadme, StringComparison.Ordinal);
        Assert.Contains("HasTensorOutputProof", appReadme, StringComparison.Ordinal);
        Assert.Contains("HasRawBindingProof", appReadme, StringComparison.Ordinal);
        Assert.Contains("synthetic-input-runtime is not real-model-runtime", appReadme, StringComparison.Ordinal);
        Assert.Contains("官方 trtexec 对齐状态表", appReadme, StringComparison.Ordinal);
        Assert.Contains("TrtexecAlignmentStatus=parse-only", appReadme, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime belongs to release proof records", appReadme, StringComparison.Ordinal);
        Assert.Contains("Layer/profile diagnostic switch parity", appReadme, StringComparison.Ordinal);
        Assert.Contains("--dumpLayerInfo", appReadme, StringComparison.Ordinal);
        Assert.Contains("--dumpProfile", appReadme, StringComparison.Ordinal);
        Assert.Contains("--separateProfileRun", appReadme, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", appReadme, StringComparison.Ordinal);
        Assert.Contains("src\\JYPPX.TensorRtSharp.Tools\\JYPPX.TensorRtSharp.Tools.csproj", projectText.Replace("/", "\\"), StringComparison.Ordinal);
        Assert.DoesNotContain("samples\\OnnxToEngine\\OnnxToEngine.csproj", projectText.Replace("/", "\\"), StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtExecOptionsBuildArgumentLinePreservesCoreSwitches()
    {
        string optionsSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Core", "TensorRtExecOptions.cs"));
        string toolsOptionsSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "TrtexecLikeOptions.cs"));
        string toolsParserSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "TrtexecLikeParser.cs"));

        Assert.Contains("public string ToArgumentLine()", optionsSource, StringComparison.Ordinal);
        Assert.Contains("TrtexecLikeOptions", optionsSource, StringComparison.Ordinal);
        Assert.Contains("TrtexecLikeParser.Parse", optionsSource, StringComparison.Ordinal);
        Assert.Contains("return TrtexecOptions.ToArgumentLine()", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--onnx\"", toolsOptionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--saveEngine\"", toolsOptionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--model\"", toolsParserSource, StringComparison.Ordinal);
        Assert.Contains("\"--onnxFile\"", toolsParserSource, StringComparison.Ordinal);
        Assert.Contains("\"--plan\"", toolsParserSource, StringComparison.Ordinal);
        Assert.Contains("\"--engineFile\"", toolsParserSource, StringComparison.Ordinal);
        Assert.Contains("ShouldTreatEngineAliasAsLoad", toolsParserSource, StringComparison.Ordinal);
        Assert.Contains("\"--fp16\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--buildOnly\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--skipInference\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--dryRun\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--plugins\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--profilingVerbosity\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--builderOptimizationLevel\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--maxAuxStreams\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--useDLACore\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--allowGPUFallback\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--tacticSources\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--memPoolSize\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--inputIOFormats\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--outputIOFormats\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--calib\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--directIO\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--sparsity\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--stronglyTyped\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("BuilderOptimizationLevel", optionsSource, StringComparison.Ordinal);
        Assert.Contains("public bool DryRun", optionsSource, StringComparison.Ordinal);
        Assert.Contains("public string ReportPath", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Core", "TensorRtExecReport.cs")), StringComparison.Ordinal);
        Assert.Contains("public string ProofClassification", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Core", "TensorRtExecReport.cs")), StringComparison.Ordinal);
        Assert.Contains("public string NormalizedCommandSha256", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Core", "TensorRtExecReport.cs")), StringComparison.Ordinal);
        Assert.Contains("public string LoadEngineDiagnosticsState", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Core", "TensorRtExecReport.cs")), StringComparison.Ordinal);
        Assert.Contains("public ulong WorkspaceBytes", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Core", "TensorRtExecReport.cs")), StringComparison.Ordinal);
        Assert.Contains("BuilderConfigDeploymentSnapshotState", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Core", "TensorRtExecReport.cs")), StringComparison.Ordinal);
        Assert.Contains("DeploymentOptions", toolsOptionsSource, StringComparison.Ordinal);
        Assert.Contains("RuntimeOptions", toolsOptionsSource, StringComparison.Ordinal);
        Assert.Contains("PreflightMetadata", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildDiagnostics.cs")), StringComparison.Ordinal);
        Assert.Contains("LoadedEngineDiagnostics", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildDiagnostics.cs")), StringComparison.Ordinal);
        Assert.Contains("Loaded engine readback fingerprint", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildDiagnostics.cs")), StringComparison.Ordinal);
        Assert.Contains("Loaded engine readback SHA256", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildDiagnostics.cs")), StringComparison.Ordinal);
        Assert.Contains("\"--exportLayerInfo\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--exportReport\"", toolsOptionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--report\"", toolsParserSource, StringComparison.Ordinal);
        Assert.Contains("\"--evidenceSidecar\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--evidenceSidecar\"", toolsOptionsSource, StringComparison.Ordinal);
        Assert.Contains("EvidenceSidecarPath", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--minTiming\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--avgTiming\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--precisionConstraints\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--layerPrecisions\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--layerOutputTypes\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--fp8\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--best\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--dumpRefit\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--allowWeightStreaming\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--markDebug\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--dumpDebugTensors\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--versionCompatible\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--excludeLeanRuntime\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--stripWeights\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--refit\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--weightStreamingBudget\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--exportTimingCache\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--safe\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--consistency\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--builderCache\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--noBuilderCache\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--dumpLayerInfo\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--dumpProfile\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--separateProfileRun\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("public bool DumpLayerInfo", optionsSource, StringComparison.Ordinal);
        Assert.Contains("public bool DumpProfile", optionsSource, StringComparison.Ordinal);
        Assert.Contains("public bool SeparateProfileRun", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--infStreams\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("TrtexecAlignmentStatus=parse-only", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildOptions.cs")), StringComparison.Ordinal);
        Assert.Contains("\"--iterations\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--warmUp\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--duration\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--streams\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--useCudaGraph\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--noDataTransfers\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--dumpRawBindingsToFile\"", optionsSource, StringComparison.Ordinal);
        Assert.Contains("\"--exportTimes\"", optionsSource, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtExecParserNormalizesModelAndEngineFileAliasesToCanonicalOptions()
    {
        string tempPath = Path.GetTempPath();
        TrtexecLikeOptions buildOptions = TrtexecLikeParser.Parse(new[]
        {
            "--dryRun",
            "--model", Path.Combine(tempPath, "owner-model.onnx"),
            "--plan", Path.Combine(tempPath, "owner-model.plan"),
            "--buildOnly"
        });

        string buildLine = buildOptions.ToArgumentLine();
        Assert.EndsWith("owner-model.onnx", buildOptions.OnnxPath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("owner-model.plan", buildOptions.SaveEnginePath, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(buildOptions.LoadEnginePath);
        Assert.Contains("--onnx", buildLine, StringComparison.Ordinal);
        Assert.Contains("--saveEngine", buildLine, StringComparison.Ordinal);
        Assert.DoesNotContain("--model ", buildLine, StringComparison.Ordinal);
        Assert.DoesNotContain("--plan ", buildLine, StringComparison.Ordinal);

        TrtexecLikeOptions loadOptions = TrtexecLikeParser.Parse(new[]
        {
            "--dryRun",
            "--engineFile", Path.Combine(tempPath, "existing.plan"),
            "--skipInference"
        });

        string loadLine = loadOptions.ToArgumentLine();
        Assert.Empty(loadOptions.OnnxPath);
        Assert.Empty(loadOptions.SaveEnginePath);
        Assert.EndsWith("existing.plan", loadOptions.LoadEnginePath, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("--loadEngine", loadLine, StringComparison.Ordinal);
        Assert.DoesNotContain("--engineFile", loadLine, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtExecLayerAndProfileDiagnosticsFlowThroughCliAndGuiOptionModel()
    {
        string optionsSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Core", "TensorRtExecOptions.cs"));
        string formSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "WinForms", "MainForm.cs"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md"));
        string featureMatrix = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-feature-matrix.json"));
        string parityMatrix = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-trtexec-parity-matrix.json"));

        foreach (string marker in new[]
        {
            "bool dumpLayerInfo",
            "bool dumpProfile",
            "bool separateProfileRun",
            "bool fp8",
            "bool best",
            "bool dumpRefit",
            "bool allowWeightStreaming",
            "string markDebug",
            "bool dumpDebugTensors",
            "AddSwitch(args, \"--dumpLayerInfo\", dumpLayerInfo)",
            "AddSwitch(args, \"--dumpProfile\", dumpProfile)",
            "AddSwitch(args, \"--separateProfileRun\", separateProfileRun)",
            "AddSwitch(args, \"--fp8\", fp8)",
            "AddSwitch(args, \"--best\", best)",
            "AddSwitch(args, \"--dumpRefit\", dumpRefit)",
            "AddSwitch(args, \"--allowWeightStreaming\", allowWeightStreaming)",
            "Add(args, \"--markDebug\", markDebug)",
            "AddSwitch(args, \"--dumpDebugTensors\", dumpDebugTensors)",
            "public bool DumpLayerInfo",
            "public bool DumpProfile",
            "public bool SeparateProfileRun",
            "public bool Fp8",
            "public bool Best",
            "public bool DumpRefit",
            "public bool AllowWeightStreaming",
            "public string MarkDebug",
            "public bool DumpDebugTensors"
        })
        {
            Assert.Contains(marker, optionsSource, StringComparison.Ordinal);
        }

        foreach (string marker in new[]
        {
            "_dumpLayerInfo",
            "_dumpProfile",
            "_separateProfileRun",
            "_fp8",
            "_best",
            "_dumpRefit",
            "_allowWeightStreaming",
            "_markDebug",
            "_dumpDebugTensors",
            "ConfigureCheck(_dumpLayerInfo, \"Dump layer info\")",
            "ConfigureCheck(_dumpProfile, \"Dump profile\")",
            "ConfigureCheck(_separateProfileRun, \"Separate profile\")",
            "ConfigureCheck(_fp8, \"FP8\")",
            "ConfigureCheck(_best, \"Best\")",
            "ConfigureCheck(_dumpRefit, \"Dump refit\")",
            "ConfigureCheck(_allowWeightStreaming, \"Weight streaming\")",
            "ConfigureCheck(_dumpDebugTensors, \"Debug tensors\")",
            "_dumpLayerInfo.Checked",
            "_dumpProfile.Checked",
            "_separateProfileRun.Checked",
            "_fp8.Checked",
            "_best.Checked",
            "_dumpRefit.Checked",
            "_allowWeightStreaming.Checked",
            "_markDebug.Text",
            "_dumpDebugTensors.Checked"
        })
        {
            Assert.Contains(marker, formSource, StringComparison.Ordinal);
        }

        Assert.Contains("Layer/profile diagnostic switch parity", readme, StringComparison.Ordinal);
        Assert.Contains("Layer/profile diagnostic switches", featureMatrix, StringComparison.Ordinal);
        Assert.Contains("GUI/CLI field map", featureMatrix, StringComparison.Ordinal);
        Assert.Contains("checklist-backed", featureMatrix + parityMatrix, StringComparison.Ordinal);
        Assert.Contains("\"DumpProfile\"", parityMatrix, StringComparison.Ordinal);
        Assert.Contains("\"SeparateProfileRun\"", parityMatrix, StringComparison.Ordinal);
        Assert.Contains("\"DumpLayerInfo\"", parityMatrix, StringComparison.Ordinal);
        Assert.Contains("precision-shortcuts-debug-boundary", parityMatrix, StringComparison.Ordinal);
        Assert.Contains("--fp8 --best", parityMatrix, StringComparison.Ordinal);
        Assert.Contains("--dumpRefit --allowWeightStreaming --markDebug --dumpDebugTensors", parityMatrix, StringComparison.Ordinal);
        Assert.Contains("parse-report-only", featureMatrix + parityMatrix, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("gui-cli-field-map", parityMatrix, StringComparison.Ordinal);
        Assert.Contains("not runtime proof", featureMatrix + parityMatrix, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TensorRtExecDocumentsYoloVisionOwnerBackfillShapeProfilesWithoutProofPromotion()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md"));
        string featureMatrix = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-feature-matrix.json"));
        string parityMatrix = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-trtexec-parity-matrix.json"));
        string ownerBackfillPack = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-real-asset-owner-backfill-pack.json"));
        string combined = readme + featureMatrix + parityMatrix + ownerBackfillPack;

        foreach (string marker in new[]
        {
            "YoloVision owner backfill profiles",
            "yolovision-real-asset-owner-backfill-sample-run-evidence.template.json",
            "1x3x224x224",
            "1x3x640x640",
            "1x3x1024x1024",
            "--dumpLayerInfo",
            "--exportLayerInfo",
            "--dumpProfile",
            "--separateProfileRun",
            "--exportProfile",
            "--saveProfile",
            "real run logs",
            "not runtime proof",
            "real-model-runtime requires YoloVision run log"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("yolovision-owner-backfill-shape-profiles", parityMatrix, StringComparison.Ordinal);
        Assert.Contains("YoloVision owner backfill shape profiles", featureMatrix, StringComparison.Ordinal);
        Assert.Contains("\"canPromoteRealModelRuntime\": false", ownerBackfillPack, StringComparison.Ordinal);
        Assert.Contains("\"canPromotePackageConsumerRuntime\": false", ownerBackfillPack, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtExecDocumentsTimingCacheInt8AndGuiCliFieldMapAsNonProofBoundaries()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md"));
        string featureMatrix = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-feature-matrix.json"));
        string parityMatrix = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-trtexec-parity-matrix.json"));
        string guiCliExporter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-TensorRtExecGuiCliParityChecklist.ps1"));
        string guiCliValidator = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-TensorRtExecGuiCliParityChecklist.ps1"));
        string diagnostics = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildDiagnostics.cs"));
        string combined = readme + featureMatrix + parityMatrix + guiCliExporter + guiCliValidator + diagnostics;

        foreach (string marker in new[]
        {
            "--timingCacheFile",
            "--timingCache",
            "--exportTimingCache",
            "--int8",
            "--calib",
            "CalibrationCacheFile",
            "ExportTimingCachePath",
            "OptionImplementationStatus",
            "ParseOnlyOptions",
            "GUI/CLI field map",
            "gui-cli-field-map",
            "Export-TensorRtExecGuiCliParityChecklist.ps1",
            "not package-consumer-runtime proof"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("parse-report-only", featureMatrix + parityMatrix, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("cache import/export lifecycle proof", parityMatrix + readme, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("calibrator ownership", parityMatrix + readme, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("\"isRuntimeProof\": true", parityMatrix, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TensorRtExecServiceReportsBuildPreflightWithoutClaimingRuntimeProof()
    {
        string serviceSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Core", "TensorRtExecService.cs"));
        string commandSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Console", "TensorRtExecCommand.cs"));
        string formSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "WinForms", "MainForm.cs"));
        string diagnosticsSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildDiagnostics.cs"));

        Assert.Contains("new OnnxEngineBuildService().Execute", serviceSource, StringComparison.Ordinal);
        Assert.Contains("OnnxEngineBuildOptions.FromTrtexecLikeOptions(options.TrtexecOptions)", serviceSource, StringComparison.Ordinal);
        Assert.Contains("ProofClassification", diagnosticsSource, StringComparison.Ordinal);
        Assert.Contains("PreflightMetadata", diagnosticsSource, StringComparison.Ordinal);
        Assert.Contains("LoadedEngineDiagnostics", diagnosticsSource, StringComparison.Ordinal);
        Assert.Contains("Loaded engine readback fingerprint", diagnosticsSource, StringComparison.Ordinal);
        Assert.Contains("Loaded engine readback SHA256", diagnosticsSource, StringComparison.Ordinal);
        Assert.Contains("Workspace bytes:", diagnosticsSource, StringComparison.Ordinal);
        Assert.Contains("Preflight SHA256", diagnosticsSource, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime is tracked by release proof records", diagnosticsSource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExecOptions.Parse(args)", commandSource, StringComparison.Ordinal);
        Assert.Contains("--builderOptimizationLevel <0..5>", commandSource, StringComparison.Ordinal);
        Assert.Contains("--minTiming <n> --avgTiming <n> --precisionConstraints <none|prefer|obey>", commandSource, StringComparison.Ordinal);
        Assert.Contains("--layerPrecisions <spec> --layerOutputTypes <spec>", commandSource, StringComparison.Ordinal);
        Assert.Contains("--versionCompatible --excludeLeanRuntime --stripWeights --refit --weightStreamingBudget <MiB>", commandSource, StringComparison.Ordinal);
        Assert.Contains("--safe --consistency --builderCache|--noBuilderCache", commandSource, StringComparison.Ordinal);
        Assert.Contains("--memPoolSize workspace:512,tacticDram:1024", commandSource, StringComparison.Ordinal);
        Assert.Contains("--shapes|--inputShapes input:1x3x640x640[,other:...] --batch <n>", commandSource, StringComparison.Ordinal);
        Assert.Contains("--fp16 --int8 --bf16 --fp8 --best --noTF32 --workspace <MiB>", commandSource, StringComparison.Ordinal);
        Assert.Contains("--dumpRefit --allowWeightStreaming --markDebug <names> --dumpDebugTensors", commandSource, StringComparison.Ordinal);
        Assert.Contains("--dryRun|--previewOnly", commandSource, StringComparison.Ordinal);
        Assert.Contains("--plugins|--plugin|--dynamicPlugins|--setPluginsToSerialize", commandSource, StringComparison.Ordinal);
        Assert.Contains("--profilingVerbosity <none|layer_names_only|detailed> --verbose", commandSource, StringComparison.Ordinal);
        Assert.Contains("--iterations <n> --warmUp <ms> --duration <sec> --streams <n> --infStreams <n> --useCudaGraph", commandSource, StringComparison.Ordinal);
        Assert.Contains("--noDataTransfers --useSpinWait --threads <n> --avgRuns <n> --percentile <0..100> --sleepTime <ms> --idleTime <ms>", commandSource, StringComparison.Ordinal);
        Assert.Contains("--loadInputs input:file --dumpOutput --dumpRawBindingsToFile <path>", commandSource, StringComparison.Ordinal);
        Assert.Contains("--exportOutput <path> --exportTimes <path> --exportProfile <path> --saveProfile <path>", commandSource, StringComparison.Ordinal);
        Assert.Contains("--exportReport|--report <path.json|path.md>", commandSource, StringComparison.Ordinal);
        Assert.Contains("Input options:", commandSource, StringComparison.Ordinal);
        Assert.Contains("Build options:", commandSource, StringComparison.Ordinal);
        Assert.Contains("Runtime options:", commandSource, StringComparison.Ordinal);
        Assert.Contains("Deployment options:", commandSource, StringComparison.Ordinal);
        Assert.Contains("Report options:", commandSource, StringComparison.Ordinal);
        Assert.Contains("Evidence options:", commandSource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec ProofClassification=", commandSource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec NormalizedCommandSha256=", commandSource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec LoadEngineDiagnosticsState=", commandSource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec WorkspaceBytes=", commandSource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec BuilderConfigDeploymentSnapshot=", commandSource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec ReportPath=", commandSource, StringComparison.Ordinal);
        Assert.Contains("--evidenceSidecar <evidence.json>", commandSource, StringComparison.Ordinal);
        Assert.Contains("PreflightMetadata", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md")), StringComparison.Ordinal);
        Assert.Contains("LoadedEngineDiagnostics", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md")), StringComparison.Ordinal);
        Assert.Contains("ReadbackSha256", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md")), StringComparison.Ordinal);
        Assert.Contains("WorkspaceBytes", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md")), StringComparison.Ordinal);
        Assert.Contains("ArtifactProofBoundary", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md")), StringComparison.Ordinal);
        Assert.Contains("HasTensorOutputProof", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md")), StringComparison.Ordinal);
        Assert.Contains("HasRawBindingProof=false", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md")), StringComparison.Ordinal);
        Assert.Contains("第二层是 bounded runtime", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md")), StringComparison.Ordinal);
        Assert.Contains("runtime-output-captured-unverified", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md")), StringComparison.Ordinal);
        Assert.Contains("new TensorRtExecService().Execute(options)", formSource, StringComparison.Ordinal);
        Assert.Contains("FormatReportLog(report)", formSource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec ReportPath=", formSource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec ProofClassification=", formSource, StringComparison.Ordinal);
        Assert.Contains("BuildEvidenceOnly=", formSource, StringComparison.Ordinal);
        Assert.Contains("DryRun=", formSource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec NormalizedCommandSha256=", formSource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec LoadEngineDiagnosticsState=", formSource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec WorkspaceBytes=", formSource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec BuilderConfigDeploymentSnapshot=", formSource, StringComparison.Ordinal);
        Assert.Contains("_reportPath", formSource, StringComparison.Ordinal);
        Assert.Contains("_evidenceSidecarPath", formSource, StringComparison.Ordinal);
        Assert.Contains("OnBrowseEvidenceSidecar", formSource, StringComparison.Ordinal);
        Assert.Contains("_loadEnginePath", formSource, StringComparison.Ordinal);
        Assert.Contains("_plugins", formSource, StringComparison.Ordinal);
        Assert.Contains("_timingCachePath", formSource, StringComparison.Ordinal);
        Assert.Contains("_profilingVerbosity", formSource, StringComparison.Ordinal);
        Assert.Contains("_builderOptimizationLevel", formSource, StringComparison.Ordinal);
        Assert.Contains("_maxAuxStreams", formSource, StringComparison.Ordinal);
        Assert.Contains("_dryRun", formSource, StringComparison.Ordinal);
        Assert.Contains("_deviceOrdinal", formSource, StringComparison.Ordinal);
        Assert.Contains("_dlaCore", formSource, StringComparison.Ordinal);
        Assert.Contains("_tacticSources", formSource, StringComparison.Ordinal);
        Assert.Contains("_memoryPoolSizes", formSource, StringComparison.Ordinal);
        Assert.Contains("_inputIoFormats", formSource, StringComparison.Ordinal);
        Assert.Contains("_outputIoFormats", formSource, StringComparison.Ordinal);
        Assert.Contains("_calibrationCachePath", formSource, StringComparison.Ordinal);
        Assert.Contains("_allowGpuFallback", formSource, StringComparison.Ordinal);
        Assert.Contains("_directIo", formSource, StringComparison.Ordinal);
        Assert.Contains("_stronglyTyped", formSource, StringComparison.Ordinal);
        Assert.Contains("_minTiming", formSource, StringComparison.Ordinal);
        Assert.Contains("_avgTiming", formSource, StringComparison.Ordinal);
        Assert.Contains("_precisionConstraints", formSource, StringComparison.Ordinal);
        Assert.Contains("_layerPrecisions", formSource, StringComparison.Ordinal);
        Assert.Contains("_layerOutputTypes", formSource, StringComparison.Ordinal);
        Assert.Contains("_versionCompatible", formSource, StringComparison.Ordinal);
        Assert.Contains("_excludeLeanRuntime", formSource, StringComparison.Ordinal);
        Assert.Contains("_stripWeights", formSource, StringComparison.Ordinal);
        Assert.Contains("_refit", formSource, StringComparison.Ordinal);
        Assert.Contains("_weightStreamingBudget", formSource, StringComparison.Ordinal);
        Assert.Contains("_exportTimingCachePath", formSource, StringComparison.Ordinal);
        Assert.Contains("_safe", formSource, StringComparison.Ordinal);
        Assert.Contains("_consistency", formSource, StringComparison.Ordinal);
        Assert.Contains("_builderCache", formSource, StringComparison.Ordinal);
        Assert.Contains("_noBuilderCache", formSource, StringComparison.Ordinal);
        Assert.Contains("_dumpLayerInfo", formSource, StringComparison.Ordinal);
        Assert.Contains("_dumpProfile", formSource, StringComparison.Ordinal);
        Assert.Contains("_separateProfileRun", formSource, StringComparison.Ordinal);
        Assert.Contains("_infStreams", formSource, StringComparison.Ordinal);
        Assert.Contains("OnBrowseCalibrationCache", formSource, StringComparison.Ordinal);
        Assert.Contains("_layerInfoPath", formSource, StringComparison.Ordinal);
        Assert.Contains("_iterations", formSource, StringComparison.Ordinal);
        Assert.Contains("_warmUp", formSource, StringComparison.Ordinal);
        Assert.Contains("_duration", formSource, StringComparison.Ordinal);
        Assert.Contains("_streams", formSource, StringComparison.Ordinal);
        Assert.Contains("_useCudaGraph", formSource, StringComparison.Ordinal);
        Assert.Contains("_noDataTransfers", formSource, StringComparison.Ordinal);
        Assert.Contains("_useSpinWait", formSource, StringComparison.Ordinal);
        Assert.Contains("_loadInputs", formSource, StringComparison.Ordinal);
        Assert.Contains("_dumpRawBindingsPath", formSource, StringComparison.Ordinal);
        Assert.Contains("_exportOutputPath", formSource, StringComparison.Ordinal);
        Assert.Contains("_exportTimesPath", formSource, StringComparison.Ordinal);
        Assert.Contains("_exportProfilePath", formSource, StringComparison.Ordinal);
        Assert.Contains("_saveProfilePath", formSource, StringComparison.Ordinal);
        Assert.Contains("_commandPreview", formSource, StringComparison.Ordinal);
        Assert.Contains("CreateOptionsFromControls", formSource, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtExecHelpAndParityArtifactsExposeAdvancedTrtexecAliasesWithoutProofPromotion()
    {
        string commandSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Console", "TensorRtExecCommand.cs"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md"));
        string featureMatrix = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-feature-matrix.json"));
        string parityJson = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-trtexec-parity-matrix.json"));
        string parityMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-trtexec-parity-matrix.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "tensorrt-exec-trtexec-parity-matrix.md"));
        string combined = commandSource + readme + featureMatrix + parityJson + parityMarkdown + article;

        foreach (string marker in new[]
        {
            "--shapes",
            "--inputShapes",
            "--batch",
            "--fp8",
            "--best",
            "--dumpRefit",
            "--allowWeightStreaming",
            "--markDebug",
            "--dumpDebugTensors",
            "--plugin",
            "--dynamicPlugins",
            "--setPluginsToSerialize",
            "--timingCache",
            "--profilingVerbosity",
            "--verbose",
            "--sleepTime",
            "--idleTime",
            "shape-alias-batch",
            "wait-idle-controls",
            "Plugin alias compatibility",
            "parse-report-only",
            "not runtime proof",
            "\"isRuntimeProof\": false"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("\"isRuntimeProof\": true", parityJson, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canPromotePackageConsumerRuntime=true", combined, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TensorRtExecDocumentationIsLinkedFromRepositoryEntrypoints()
    {
        string rootReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string zhReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string samplesReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "README.md"));

        Assert.Contains("applications/TensorRtExec", rootReadme, StringComparison.Ordinal);
        Assert.Contains("applications/TensorRtExec", zhReadme, StringComparison.Ordinal);
        Assert.Contains("applications/TensorRtExec/README.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("Evidence Ladder For Asset-Dependent Samples", samplesReadme, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime belongs to release proof records", samplesReadme, StringComparison.Ordinal);
        string trtexecCoverage = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "user-acceptance", "trtexec-option-coverage.md"));
        Assert.Contains("--minTiming", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("--precisionConstraints", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("--weightStreamingBudget", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("--safe", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("--builderCache", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("TrtexecAlignmentStatus=parse-only", trtexecCoverage, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtExecOnnxToEngineAndYoloVisionDocsKeepProofChainSeparation()
    {
        string tensorRtExecReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md"));
        string onnxToEngineReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "OnnxToEngine", "README.md"));
        string yoloReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        string boundaryArticle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "onnxtoengine-and-tensorrtexec-boundary.md"));
        string coverage = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "user-acceptance", "trtexec-option-coverage.md"));
        string combined = tensorRtExecReadme + onnxToEngineReadme + yoloReadme + boundaryArticle + coverage;

        foreach (string marker in new[]
        {
            "ProofClassification",
            "BuildEvidenceOnly",
            "DryRun",
            "NormalizedCommandSha256",
            "InferenceRan",
            "OutputMatch",
            "IsRuntimeExecutionProof",
            "IsRealModelRuntimeProof",
            "IsPackageConsumerRuntimeProof",
            "PreflightMetadata",
            "OptionImplementationStatus",
            "Layer/profile diagnostic switch parity",
            "--exportReport",
            "--report",
            "build-only",
            "parse-only",
            "sidecar-only",
            "synthetic-input-runtime is not real-model-runtime",
            "package-consumer-runtime belongs to release proof records",
            "external-runtime-proof-record.json",
            "YoloVision Passed=True"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("samples/YoloDet", combined, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet.csproj", combined, StringComparison.Ordinal);
    }
}
