using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseCandidateSmokeScaffoldTests
{
    [Fact]
    public void YoloVisionSampleRunEvidenceTemplatesStayValidatorBackedAndNonPromotable()
    {
        string validator = ReadText("eng", "Test-SampleRunEvidenceRecord.ps1");
        string readme = ReadText("samples", "YoloVision", "README.md");

        foreach (string templateName in new[] { "yolovision", "yolox-s" })
        {
            using JsonDocument document = ReadJson("artifacts", "user-acceptance", $"sample-run-evidence-record.{templateName}.template.json");
            JsonElement root = document.RootElement;
            string markdown = ReadText("artifacts", "user-acceptance", $"sample-run-evidence-record.{templateName}.template.md");

            Assert.Equal("sample-run-evidence-record-template", root.GetProperty("recordKind").GetString());
            Assert.True(root.GetProperty("templateOnly").GetBoolean());
            Assert.Equal("YoloVision", root.GetProperty("sampleName").GetString());
            Assert.Equal("template-only", root.GetProperty("proofClassification").GetString());
            Assert.Equal("owner-action-required", root.GetProperty("validatorState").GetString());
            Assert.False(root.GetProperty("isSmokePassed").GetBoolean());
            Assert.False(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.Contains("--input-data", root.GetProperty("sampleRunCommand").GetString(), StringComparison.Ordinal);
            Assert.Contains(root.GetProperty("expectedEvidenceLines").EnumerateArray(), static item =>
                string.Equals(item.GetString(), "InputSource=external InputFile=...", StringComparison.Ordinal));
            Assert.Contains(root.GetProperty("expectedEvidenceLines").EnumerateArray(), static item =>
                string.Equals(item.GetString(), "YoloVision Passed=True", StringComparison.Ordinal));

            string rawJson = root.GetRawText();
            foreach (string marker in new[]
            {
                "modelSha256",
                "labelsSha256",
                "inputAssetSha256",
                "preprocessedInputTensorSha256",
                "sampleRunLogSha256",
                "stdoutSummary",
                "stderrSummary",
                "package-consumer-runtime is forbidden in sample run evidence records",
                "Sample run evidence can promote only to real-model-runtime, never package-consumer-runtime"
            })
            {
                Assert.Contains(marker, rawJson, StringComparison.OrdinalIgnoreCase);
                Assert.Contains(marker, markdown, StringComparison.OrdinalIgnoreCase);
            }
        }

        foreach (string marker in new[]
        {
            "package-consumer-runtime-forbidden",
            "yolovision-external-input-evidence-line",
            "declared-real-model-promotion",
            "canPromoteRealModelRuntime=true",
            "FailOnNotProof",
            "disallowedProofClassifications = @(\"package-consumer-runtime\")"
        })
        {
            Assert.Contains(marker, validator, StringComparison.Ordinal);
        }

        Assert.Contains("eng\\Test-SampleRunEvidenceRecord.ps1", readme, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("sample-run-evidence-record.yolovision.template.json", readme, StringComparison.Ordinal);
        Assert.Contains("sample-run-evidence-record.yolox-s.template.json", readme, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet", readme, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TrtexecOptionCoverageStaysBackedByMatrixParserAndTests()
    {
        string coverage = ReadText("artifacts", "user-acceptance", "trtexec-option-coverage.md");
        string parityJson = ReadText("samples", "OnnxToEngine", "trtexec-parity-matrix.json");
        string parityMarkdown = ReadText("samples", "OnnxToEngine", "trtexec-parity-matrix.md");
        string parser = ReadText("src", "JYPPX.TensorRtSharp.Tools", "TrtexecLikeParser.cs");
        string onnxTests = ReadText("tests", "JYPPX.ProjectQuality.Tests", "OnnxToEngineTrtexecLikeTests.cs");
        string appTests = ReadText("tests", "JYPPX.ProjectQuality.Tests", "TensorRtExecApplicationTests.cs");

        string combinedImplementation = parser + onnxTests + appTests;
        string combinedMatrix = parityJson + parityMarkdown + coverage;

        foreach (string option in new[]
        {
            "--onnx",
            "--saveEngine",
            "--loadEngine",
            "--minShapes",
            "--optShapes",
            "--maxShapes",
            "--shapes",
            "--fp16",
            "--bf16",
            "--int8",
            "--calib",
            "--workspace",
            "--memPoolSize",
            "--tacticSources",
            "--timingCacheFile",
            "--timingCache",
            "--profilingVerbosity",
            "--verbose",
            "--dumpLayerInfo",
            "--exportLayerInfo",
            "--minTiming",
            "--avgTiming",
            "--precisionConstraints",
            "--layerPrecisions",
            "--layerOutputTypes",
            "--versionCompatible",
            "--excludeLeanRuntime",
            "--stripWeights",
            "--refit",
            "--weightStreamingBudget",
            "--exportTimingCache",
            "--safe",
            "--consistency",
            "--builderCache",
            "--noBuilderCache",
            "--noDataTransfers",
            "--useSpinWait",
            "--threads",
            "--avgRuns",
            "--percentile",
            "--sleepTime",
            "--idleTime",
            "--loadInputs",
            "--dumpOutput",
            "--dumpRawBindingsToFile",
            "--exportOutput",
            "--exportTimes",
            "--exportProfile",
            "--saveProfile",
            "--exportReport",
            "--evidenceSidecar",
            "--dryRun",
            "--previewOnly",
            "--buildOnly",
            "--skipInference"
        })
        {
            Assert.Contains(option, combinedMatrix, StringComparison.Ordinal);
            Assert.Contains(option, combinedImplementation, StringComparison.Ordinal);
        }

        foreach (string boundary in new[]
        {
            "parse-only/build-only evidence 不能提升 real-model-runtime 或 package-consumer-runtime",
            "build-only artifact 必须保持 `HasTensorOutputProof=false`",
            "sample run evidence record 声明 `package-consumer-runtime` 必须被 validator 拒绝",
            "not runtime proof",
            "not real-model-runtime proof",
            "not package-consumer-runtime proof",
            "not full trtexec replacement proof",
            "ownerEvidenceRequiredForPromotion",
            "implementationClass",
            "requiresRuntimeProof",
            "requiresOwnerEvidence",
            "\"canPromoteRuntimeProof\": false"
        })
        {
            Assert.Contains(boundary, combinedMatrix, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument parityDocument = JsonDocument.Parse(parityJson);
        JsonElement[] entries = parityDocument.RootElement.GetProperty("entries").EnumerateArray().ToArray();
        Assert.True(entries.Length >= 26);
        Assert.All(entries, entry =>
        {
            Assert.True(entry.TryGetProperty("implementationClass", out _));
            Assert.True(entry.TryGetProperty("proofBoundary", out _));
            Assert.True(entry.GetProperty("requiresRuntimeProof").GetBoolean());
            Assert.True(entry.GetProperty("requiresOwnerEvidence").GetBoolean());
            Assert.False(entry.GetProperty("canPromoteRuntimeProof").GetBoolean());
        });
        Assert.Contains(entries, entry => entry.GetProperty("implementationClass").GetString() == "parse-only");
        Assert.Contains(entries, entry => entry.GetProperty("implementationClass").GetString() == "build-only" || entry.GetProperty("implementationClass").GetString() == "preflight-only");
        Assert.Contains(entries, entry => entry.GetProperty("option").GetString() == "--exportTimingCache" && entry.GetProperty("implementationClass").GetString() == "applied-build-cache-lifecycle");
        Assert.Contains(entries, entry => entry.GetProperty("option").GetString() == "--loadInputs" && entry.GetProperty("implementationClass").GetString() == "bounded-artifact");
        Assert.Contains(entries, entry => entry.GetProperty("option").GetString() == "--dumpOutput" && entry.GetProperty("implementationClass").GetString() == "bounded-artifact");
        Assert.Contains(entries, entry => entry.GetProperty("option").GetString() == "--dumpRawBindingsToFile" && entry.GetProperty("implementationClass").GetString() == "bounded-artifact");
    }

    [Fact]
    public void YoloVisionReadmeMatrixAndTemplatesAgreeOnTaskEvidenceRequirements()
    {
        string readme = ReadText("samples", "YoloVision", "README.md");
        string matrix = ReadText("samples", "YoloVision", "yolo-model-matrix.json");
        string assetTemplate = ReadText("samples", "assets", "yolovision-assets.template.json");
        string sampleTemplate = ReadText("artifacts", "user-acceptance", "sample-run-evidence-record.yolovision.template.json");
        string combined = readme + matrix + assetTemplate + sampleTemplate;

        foreach (string task in new[] { "det", "cls", "seg", "obb", "pose", "sem" })
        {
            Assert.Contains($"--task {task}", readme, StringComparison.OrdinalIgnoreCase);
            Assert.Contains($"\"{task}\"", matrix, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "YOLOv5",
            "YOLOv6",
            "YOLOv7",
            "YOLOv8",
            "YOLOv9",
            "YOLOv10",
            "YOLOv11",
            "YOLOv26",
            "custom",
            "YoloVision Passed=True",
            "sample-run-evidence",
            "real-model-runtime",
            "package-consumer-runtime"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("samples/YoloDet", combined, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet.csproj", combined, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canPromoteRealModelRuntime\": true", combined, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void YoloVisionCapabilityMatrixExposesMachineReadableJsonBoundary()
    {
        string program = ReadText("samples", "YoloVision", "Program.cs");
        string capabilitySource = ReadText("samples", "YoloVision", "YoloCapabilityMatrix.cs");

        Assert.Contains("--list-capabilities --json", program, StringComparison.Ordinal);
        Assert.Contains("YoloCapabilityMatrix.FormatJson()", program, StringComparison.Ordinal);
        Assert.Contains("matrixId = \"yolovision-capability-matrix\"", capabilitySource, StringComparison.Ordinal);
        Assert.Contains("entryCount = Entries.Count", capabilitySource, StringComparison.Ordinal);
        Assert.Contains("canPromoteRealModelRuntime = false", capabilitySource, StringComparison.Ordinal);
        Assert.Contains("canPromotePackageConsumerRuntime = false", capabilitySource, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec report", capabilitySource, StringComparison.Ordinal);
        Assert.Contains("OnnxToEngine report", capabilitySource, StringComparison.Ordinal);
        Assert.Contains("not package-consumer-runtime proof", capabilitySource, StringComparison.Ordinal);

        foreach (string family in new[] { "YoloV5", "YoloV6", "YoloV7", "YoloV8", "YoloV9", "YoloV10", "YoloV11", "YoloV26", "Custom" })
        {
            Assert.Contains(family, capabilitySource, StringComparison.Ordinal);
        }

        foreach (string task in new[] { "Detection", "Classification", "Segmentation", "OrientedBoundingBox", "Pose", "SemanticSegmentation" })
        {
            Assert.Contains(task, capabilitySource, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void TensorRtExecGuiCliFieldMapLocksCommandPreviewBoundary()
    {
        string json = ReadText("applications", "TensorRtExec", "tensor-rt-exec-gui-cli-field-map.json");
        string markdown = ReadText("applications", "TensorRtExec", "tensor-rt-exec-gui-cli-field-map.md");
        string appReadme = ReadText("applications", "README.md");
        string tensorRtExecReadme = ReadText("applications", "TensorRtExec", "README.md");
        string options = ReadText("applications", "TensorRtExec", "Core", "TensorRtExecOptions.cs");
        string form = ReadText("applications", "TensorRtExec", "WinForms", "MainForm.cs");

        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        JsonElement fields = root.GetProperty("fields");

        Assert.Equal("tensor-rt-exec-gui-cli-field-map", root.GetProperty("mapId").GetString());
        Assert.True(fields.GetArrayLength() >= 25);
        Assert.Contains("not runtime proof", root.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("command preview and screenshots are not proof", root.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        foreach (string option in new[]
        {
            "--onnx",
            "--saveEngine",
            "--loadEngine",
            "--minShapes",
            "--optShapes",
            "--maxShapes",
            "--plugins",
            "--fp16",
            "--int8",
            "--timingCacheFile",
            "--exportTimingCache",
            "--dumpLayerInfo",
            "--exportLayerInfo",
            "--dumpProfile",
            "--separateProfileRun",
            "--loadInputs",
            "--dumpOutput",
            "--dumpRawBindingsToFile",
            "--exportOutput",
            "--exportTimes",
            "--exportProfile",
            "--saveProfile",
            "--buildOnly",
            "--skipInference",
            "--dryRun",
            "--exportReport",
            "--evidenceSidecar"
        })
        {
            Assert.Contains(option, json, StringComparison.Ordinal);
            Assert.Contains(option, options + form + tensorRtExecReadme, StringComparison.Ordinal);
        }

        Assert.Contains("_commandPreview", json + form, StringComparison.Ordinal);
        Assert.Contains("TensorRtExecOptions.ToArgumentLine()", markdown, StringComparison.Ordinal);
        Assert.Contains("tensor-rt-exec-gui-cli-field-map.json", appReadme + tensorRtExecReadme, StringComparison.Ordinal);
        Assert.Contains("not package-consumer-runtime proof", json + markdown + appReadme + tensorRtExecReadme, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("precheck-only", json, StringComparison.Ordinal);
        Assert.Contains("parse-report-only", json, StringComparison.Ordinal);
        Assert.Contains("build-only-not-runtime-proof", json, StringComparison.Ordinal);
        Assert.Contains("ownerEvidenceRequiredForPromotion", json, StringComparison.Ordinal);
        Assert.Contains("sidecar-not-proof", json, StringComparison.Ordinal);

        Assert.All(fields.EnumerateArray(), field =>
        {
            Assert.True(field.TryGetProperty("implementationClass", out _));
            Assert.True(field.TryGetProperty("proofBoundary", out _));
            Assert.True(field.GetProperty("requiresRuntimeProof").GetBoolean());
            Assert.True(field.GetProperty("requiresOwnerEvidence").GetBoolean());
            Assert.False(field.GetProperty("canPromoteRuntimeProof").GetBoolean());
        });
    }

    private static JsonDocument ReadJson(params string[] pathParts)
    {
        return JsonDocument.Parse(ReadText(pathParts));
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}
