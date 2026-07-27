using System.Reflection;
using System.Text.Json;
using System.IO;
using JYPPX.TensorRtSharp;
using Xunit;
using YoloVisionSample;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionManagedPipelineTests
{
    [Fact]
    public void ProgramEmitsPointerFreeBindingMetadataSummary()
    {
        string program = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "Program.cs"));
        Assert.Contains("BindingReport Ready=", program, StringComparison.Ordinal);
        Assert.Contains("BindingMetadata Index=", program, StringComparison.Ordinal);
        Assert.Contains("TensorRtEngineBindingReport", program, StringComparison.Ordinal);
        Assert.DoesNotContain("IntPtr", program, StringComparison.Ordinal);
    }

    [Fact]
    public void OnnxSampleSupportLoadsExternalRawAndFloatTensorInputs()
    {
        string directory = Path.Combine(Path.GetTempPath(), "jyppx-yolovision-input-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string rawPath = Path.Combine(directory, "input.raw");
        string textPath = Path.Combine(directory, "input.txt");
        string binPath = Path.Combine(directory, "input.bin");
        try
        {
            File.WriteAllBytes(rawPath, new byte[] { 0, 127, 255 });
            File.WriteAllText(textPath, "0.1, 0.2 0.3");
            byte[] floatBytes = new byte[sizeof(float) * 3];
            Buffer.BlockCopy(new[] { 1.0f, 2.0f, 3.0f }, 0, floatBytes, 0, floatBytes.Length);
            File.WriteAllBytes(binPath, floatBytes);

            float[] raw = CreateInputValuesForTesting(3, "ramp", inputPath: rawPath);
            float[] text = CreateInputValuesForTesting(3, "ramp", inputDataPath: textPath);
            float[] bin = CreateInputValuesForTesting(3, "ramp", inputDataPath: binPath);

            Assert.Equal(new[] { 0.0f, 127.0f / 255.0f, 1.0f }, raw);
            Assert.Equal(new[] { 0.1f, 0.2f, 0.3f }, text);
            Assert.Equal(new[] { 1.0f, 2.0f, 3.0f }, bin);
            Assert.Throws<ArgumentException>(() => CreateInputValuesForTesting(2, "ramp", inputPath: rawPath));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void ImagePreprocessorWritesPpmFloatTensorAndEvidenceMetadata()
    {
        string directory = Path.Combine(Path.GetTempPath(), "jyppx-yolovision-preprocess-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string imagePath = Path.Combine(directory, "input.ppm");
        string tensorPath = Path.Combine(directory, "input.fp32.bin");
        try
        {
            File.WriteAllText(imagePath, "P3\n2 1\n255\n255 0 0 0 255 0\n");

            YoloImagePreprocessResult result = YoloImagePreprocessor.Preprocess(
                imagePath,
                tensorPath,
                new[] { 1, 3, 4, 4 },
                YoloPreprocessOptions.Default);

            Assert.True(File.Exists(tensorPath), tensorPath);
            Assert.Equal(Path.GetFullPath(imagePath), result.SourcePath);
            Assert.Equal(Path.GetFullPath(tensorPath), result.TensorPath);
            Assert.Equal(2, result.SourceWidth);
            Assert.Equal(1, result.SourceHeight);
            Assert.Equal(4, result.TargetWidth);
            Assert.Equal(4, result.TargetHeight);
            Assert.True(result.LetterboxEnabled);
            Assert.Equal(4, result.ResizedWidth);
            Assert.Equal(2, result.ResizedHeight);
            Assert.Equal(1, result.PadY);
            Assert.Equal(48, result.TensorElementCount);
            Assert.Equal(48 * sizeof(float), new FileInfo(tensorPath).Length);
            Assert.Equal(64, result.SourceSha256.Length);
            Assert.Equal(64, result.TensorSha256.Length);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void LayoutInferenceRecognizesCommonRank3Outputs()
    {
        Assert.Equal(YoloOutputLayout.ChannelsFirst, YoloOutputLayoutInference.InferRank3(new[] { 1, 84, 8400 }, YoloOutputLayout.Auto));
        Assert.Equal(YoloOutputLayout.BoxesFirst, YoloOutputLayoutInference.InferRank3(new[] { 1, 8400, 84 }, YoloOutputLayout.Auto));
        Assert.Equal(YoloOutputLayout.ChannelsFirst, YoloOutputLayoutInference.Parse("channels-first"));
        Assert.Equal(YoloOutputLayout.EndToEndNms, YoloOutputLayoutInference.Parse("end2end"));
    }

    [Fact]
    public void YoloXProfileUsesOfficialPreprocessDefaultsAndTopLeftLetterbox()
    {
        string directory = Path.Combine(Path.GetTempPath(), "jyppx-yolox-preprocess-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string imagePath = Path.Combine(directory, "input.ppm");
        string tensorPath = Path.Combine(directory, "input.fp32.bin");
        try
        {
            File.WriteAllText(imagePath, "P3\n2 1\n255\n255 0 0 0 255 0\n");
            YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
            {
                "--family", "yolox",
                "--task", "det",
                "--input-shape", "1x3x4x4"
            }, labelCount: 80);

            Assert.Equal("NCHW", profile.Preprocess.TensorLayout);
            Assert.Equal("BGR", profile.Preprocess.ColorOrder);
            Assert.False(profile.Preprocess.Normalize);
            Assert.Equal(1.0f, profile.Preprocess.Scale);
            Assert.Equal("top-left", profile.Preprocess.LetterboxAlignment);

            YoloImagePreprocessResult result = YoloImagePreprocessor.Preprocess(
                imagePath,
                tensorPath,
                profile.InputShape,
                profile.Preprocess);

            Assert.Equal("top-left", result.LetterboxAlignment);
            Assert.Equal(0, result.PadX);
            Assert.Equal(0, result.PadY);
            Assert.Equal(4, result.ResizedWidth);
            Assert.Equal(2, result.ResizedHeight);
            byte[] bytes = File.ReadAllBytes(tensorPath);
            float[] tensor = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, tensor, 0, bytes.Length);
            Assert.Equal(0.0f, tensor[0]);
            Assert.Equal(0.0f, tensor[16]);
            Assert.Equal(255.0f, tensor[32]);
            Assert.Equal(114.0f, tensor[12]);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void DetectionDecoderAppliesScoreFilteringAndClassAwareNms()
    {
        float[] values =
        {
            10, 10, 2, 2, 0.9f, 0.8f, 0.1f,
            10.2f, 10.1f, 2, 2, 0.85f, 0.7f, 0.2f,
            40, 40, 3, 3, 0.9f, 0.1f, 0.95f
        };
        YoloPostprocessOptions options = new YoloPostprocessOptions(
            YoloOutputLayout.BoxesFirst,
            hasObjectness: true,
            classCount: 2,
            confidenceThreshold: 0.5f,
            iouThreshold: 0.45f,
            topK: 10,
            applyNms: true);

        IReadOnlyList<YoloDetection> detections = YoloDetectionDecoder.Decode(values, new[] { 1, 3, 7 }, options);

        Assert.Equal(2, detections.Count);
        Assert.Contains(detections, detection => detection.ClassIndex == 0);
        Assert.Contains(detections, detection => detection.ClassIndex == 1);
        Assert.All(detections, detection => Assert.InRange(detection.SourceIndex, 0, 2));
    }

    [Fact]
    public void DetectionPostprocessRejectsNonFiniteThresholds()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new YoloPostprocessOptions(
            YoloOutputLayout.BoxesFirst,
            hasObjectness: true,
            classCount: 2,
            confidenceThreshold: float.NaN,
            iouThreshold: 0.45f,
            topK: 10,
            applyNms: true));
        Assert.Throws<ArgumentOutOfRangeException>(() => new YoloPostprocessOptions(
            YoloOutputLayout.BoxesFirst,
            hasObjectness: true,
            classCount: 2,
            confidenceThreshold: 0.25f,
            iouThreshold: float.PositiveInfinity,
            topK: 10,
            applyNms: true));
    }

    [Fact]
    public void DetectionDecoderRejectsMalformedRawHeadRows()
    {
        YoloPostprocessOptions options = new YoloPostprocessOptions(
            YoloOutputLayout.BoxesFirst,
            hasObjectness: true,
            classCount: 2,
            confidenceThreshold: 0.25f,
            iouThreshold: 0.45f,
            topK: 10,
            applyNms: true);

        Assert.Throws<InvalidOperationException>(() => YoloDetectionDecoder.Decode(
            new[] { float.NaN, 10.0f, 2.0f, 2.0f, 0.9f, 0.8f, 0.1f },
            new[] { 1, 1, 7 },
            options));
        Assert.Throws<InvalidOperationException>(() => YoloDetectionDecoder.Decode(
            new[] { 10.0f, 10.0f, -2.0f, 2.0f, 0.9f, 0.8f, 0.1f },
            new[] { 1, 1, 7 },
            options));
        Assert.Throws<InvalidOperationException>(() => YoloDetectionDecoder.Decode(
            new[] { 10.0f, 10.0f, 2.0f, 2.0f, float.PositiveInfinity, 0.8f, 0.1f },
            new[] { 1, 1, 7 },
            options));
        Assert.Throws<InvalidOperationException>(() => YoloDetectionDecoder.Decode(
            new[] { 10.0f, 10.0f, 2.0f, 2.0f, 0.9f, 0.8f, float.NegativeInfinity },
            new[] { 1, 1, 7 },
            options));
    }

    [Fact]
    public void YoloXDecoderTransformsRawGridAndStrideCoordinatesBeforeNms()
    {
        YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
        {
            "--family", "yolox",
            "--task", "det",
            "--input-shape", "1x3x32x32",
            "--class-count", "2",
            "--confidence", "0.5",
            "--no-nms"
        }, labelCount: 0);
        float[] values = new float[21 * 7];
        int offset = 6 * 7;
        values[offset] = 0.5f;
        values[offset + 1] = 0.25f;
        values[offset + 2] = MathF.Log(2.0f);
        values[offset + 3] = MathF.Log(0.5f);
        values[offset + 4] = 0.9f;
        values[offset + 5] = 0.8f;
        values[offset + 6] = 0.1f;

        IReadOnlyList<YoloDetection> detections = YoloSampleRunner.DecodeDetections(values, new[] { 1, 21, 7 }, profile);

        YoloDetection detection = Assert.Single(detections);
        Assert.Equal(0, detection.ClassIndex);
        Assert.Equal(0.72f, detection.Score, 5);
        Assert.Equal(20.0f, detection.CenterX, 5);
        Assert.Equal(10.0f, detection.CenterY, 5);
        Assert.Equal(16.0f, detection.Width, 5);
        Assert.Equal(4.0f, detection.Height, 5);
        Assert.Equal(6, detection.SourceIndex);
    }

    [Fact]
    public void YoloXProfileRejectsUnsupportedTasksAndOutputContracts()
    {
        Assert.Throws<NotSupportedException>(() => YoloModelProfile.FromArgs(
            new[] { "--family", "yolox", "--task", "cls" },
            labelCount: 0));

        YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
        {
            "--family", "x",
            "--task", "det",
            "--input-shape", "1x3x32x32",
            "--layout", "channels-first",
            "--class-count", "2"
        }, labelCount: 0);

        Assert.Throws<NotSupportedException>(() => YoloSampleRunner.DecodeDetections(new float[21 * 7], new[] { 1, 7, 21 }, profile));
    }

    [Fact]
    public void UnifiedVisionResultDecodesClassificationAndSemanticOutputs()
    {
        YoloModelProfile classificationProfile = YoloModelProfile.FromArgs(new[]
        {
            "--task", "cls",
            "--class-count", "4",
            "--confidence", "0.2",
            "--top-k", "2"
        }, labelCount: 0);

        YoloVisionResult classification = YoloSampleRunner.DecodeOutput(
            new[] { 0.1f, 0.9f, 0.4f, 0.05f },
            new[] { 1, 4 },
            classificationProfile);

        Assert.Equal(YoloTaskType.Classification, classification.TaskType);
        Assert.Equal(2, classification.Classifications.Count);
        Assert.Equal(1, classification.Classifications[0].ClassIndex);
        Assert.Equal(2, classification.Classifications[1].ClassIndex);
        Assert.Empty(classification.Detections);

        YoloModelProfile semanticProfile = YoloModelProfile.FromArgs(new[]
        {
            "--task", "sem",
            "--class-count", "2"
        }, labelCount: 0);

        YoloVisionResult semantic = YoloSampleRunner.DecodeOutput(
            new[] { 0.1f, 0.9f, 0.2f, 0.8f },
            new[] { 1, 1, 2, 2 },
            semanticProfile);

        Assert.Equal(YoloTaskType.SemanticSegmentation, semantic.TaskType);
        Assert.True(semantic.HasSemanticMap);
        Assert.NotNull(semantic.SemanticMap);
        Assert.Equal(2, semantic.SemanticMap!.ClassCount);
        Assert.Equal(2, semantic.SemanticMap.Width);
        Assert.Equal(1, semantic.SemanticMap.Height);
        Assert.Equal(new[] { 0.1f, 0.2f, 0.9f, 0.8f }, semantic.SemanticMap.Values);
    }

    [Fact]
    public void UnifiedVisionResultCarriesExplicitDiagnosticsForSingleOutputAuxiliaryTasks()
    {
        float[] values =
        {
            10, 10, 2, 2, 0.9f, 0.8f, 0.1f
        };
        YoloModelProfile segmentationProfile = YoloModelProfile.FromArgs(new[]
        {
            "--task", "seg",
            "--layout", "boxes-first",
            "--class-count", "2"
        }, labelCount: 0);
        YoloModelProfile obbProfile = YoloModelProfile.FromArgs(new[]
        {
            "--task", "obb",
            "--layout", "boxes-first",
            "--class-count", "2"
        }, labelCount: 0);
        YoloModelProfile poseProfile = YoloModelProfile.FromArgs(new[]
        {
            "--task", "pose",
            "--layout", "boxes-first",
            "--class-count", "2"
        }, labelCount: 0);

        YoloVisionResult segmentation = YoloSampleRunner.DecodeOutput(values, new[] { 1, 1, 7 }, segmentationProfile);
        YoloVisionResult obb = YoloSampleRunner.DecodeOutput(values, new[] { 1, 1, 7 }, obbProfile);
        YoloVisionResult pose = YoloSampleRunner.DecodeOutput(values, new[] { 1, 1, 7 }, poseProfile);

        Assert.Single(segmentation.Detections);
        Assert.Contains("mask prototypes", segmentation.Diagnostic, StringComparison.Ordinal);
        Assert.Single(obb.Detections);
        Assert.Contains("angle channels", obb.Diagnostic, StringComparison.Ordinal);
        Assert.Single(pose.Detections);
        Assert.Contains("keypoint channels", pose.Diagnostic, StringComparison.Ordinal);
    }

    [Fact]
    public void OutputReportCapturesLabelsAndCopiedTensorHashes()
    {
        string directory = Path.Combine(Path.GetTempPath(), "jyppx-yolovision-report-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string labelsPath = Path.Combine(directory, "labels.txt");
        try
        {
            File.WriteAllLines(labelsPath, new[] { "person", "car" });
            YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
            {
                "--family", "v8",
                "--task", "det",
                "--layout", "boxes-first",
                "--class-count", "2"
            }, labelCount: 2);
            YoloRuntimeOutputSet outputs = new YoloRuntimeOutputSet(new[]
            {
                new YoloRuntimeOutputTensor(
                    "boxes",
                    YoloOutputTensorRole.Detection,
                    new[] { 10.0f, 10.0f, 2.0f, 2.0f, 0.9f, 0.8f, 0.1f },
                    new[] { 1, 1, 7 })
            });
            YoloVisionResult result = YoloSampleRunner.DecodeRuntimeOutputs(outputs, profile);
            YoloVisionOutputReportContext context = new YoloVisionOutputReportContext(
                "model.onnx",
                string.Empty,
                string.Empty,
                "ramp",
                new[] { 1, 3, 640, 640 },
                tensorRtLine: 10,
                profileIndex: 0,
                engineDeviceMemoryBytes: 1024,
                elapsedMilliseconds: 1.25,
                labelsPath);

            string json = YoloVisionOutputReport.ToJson(context, outputs, profile, result, File.ReadAllLines(labelsPath));
            using JsonDocument document = JsonDocument.Parse(json);
            JsonElement root = document.RootElement;
            JsonElement labels = root.GetProperty("labels");
            JsonElement output = root.GetProperty("outputs").EnumerateArray().Single();

            Assert.Equal(labelsPath, labels.GetProperty("path").GetString());
            Assert.Equal(2, labels.GetProperty("classCount").GetInt32());
            Assert.Equal(64, labels.GetProperty("sha256").GetString()!.Length);
            Assert.Equal(64, output.GetProperty("valueSha256").GetString()!.Length);
            Assert.Equal(7, output.GetProperty("valuePreview").GetArrayLength());
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void OutputReportIncludesPointerFreeBindingMetadataAndSemanticRoleMapping()
    {
        YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
        {
            "--family", "v8",
            "--task", "det",
            "--layout", "boxes-first",
            "--class-count", "2"
        }, labelCount: 2);
        YoloRuntimeOutputSet outputs = new YoloRuntimeOutputSet(new[]
        {
            new YoloRuntimeOutputTensor(
                "boxes",
                YoloOutputTensorRole.Detection,
                new[] { 10.0f, 10.0f, 2.0f, 2.0f, 0.9f, 0.8f, 0.1f },
                new[] { 1, 1, 7 })
        });
        YoloVisionResult result = YoloSampleRunner.DecodeRuntimeOutputs(outputs, profile);
        TensorRtEngineBindingReport bindingReport = new TensorRtEngineBindingReport(
            "binding-fixture",
            profileIndex: 0,
            new[]
            {
                new TensorRtEngineTensorBinding(
                    index: 0,
                    name: "images",
                    TensorRtDataType.Float,
                    TensorRtIOMode.Input,
                    new TensorRtDims(new[] { -1, 3, 640, 640 }),
                    TensorRtTensorLocation.Device,
                    isShapeInferenceIO: false,
                    bytesPerComponent: 4,
                    componentsPerElement: 1,
                    TensorRtTensorFormat.Linear,
                    formatDescription: "Linear",
                    vectorizedDimension: -1,
                    profileIndex: 0,
                    profileMinShape: new TensorRtDims(new[] { 1, 3, 640, 640 }),
                    profileOptShape: new TensorRtDims(new[] { 1, 3, 640, 640 }),
                    profileMaxShape: new TensorRtDims(new[] { 4, 3, 640, 640 }),
                    diagnostics: Array.Empty<string>()),
                new TensorRtEngineTensorBinding(
                    index: 1,
                    name: "boxes",
                    TensorRtDataType.Float,
                    TensorRtIOMode.Output,
                    new TensorRtDims(new[] { 1, 1, 7 }),
                    TensorRtTensorLocation.Device,
                    isShapeInferenceIO: false,
                    bytesPerComponent: 4,
                    componentsPerElement: 1,
                    TensorRtTensorFormat.Linear,
                    formatDescription: "Linear",
                    vectorizedDimension: -1,
                    profileIndex: 0,
                    profileMinShape: null,
                    profileOptShape: null,
                    profileMaxShape: null,
                    diagnostics: new[] { "fixture-diagnostic" })
            },
            readiness: null);

        using JsonDocument document = JsonDocument.Parse(YoloVisionOutputReport.ToJson(
            new YoloVisionOutputReportContext(
                "model.onnx",
                string.Empty,
                string.Empty,
                "ramp",
                new[] { 1, 3, 640, 640 },
                tensorRtLine: 10,
                profileIndex: 0,
                engineDeviceMemoryBytes: 1024,
                elapsedMilliseconds: 1.0),
            outputs,
            profile,
            result,
            new[] { "person", "car" },
            bindingReport));

        JsonElement bindingMetadata = document.RootElement.GetProperty("bindingMetadata");
        Assert.Equal("binding-fixture", bindingMetadata.GetProperty("engineName").GetString());
        Assert.False(bindingMetadata.GetProperty("isRuntimeProof").GetBoolean());
        JsonElement[] tensors = bindingMetadata.GetProperty("tensors").EnumerateArray().ToArray();
        Assert.Equal(2, tensors.Length);
        Assert.Equal("input", tensors[0].GetProperty("semanticRole").GetString());
        Assert.False(tensors[0].GetProperty("valueCaptured").GetBoolean());
        Assert.Equal("boxes", tensors[1].GetProperty("semanticRole").GetString());
        Assert.True(tensors[1].GetProperty("valueCaptured").GetBoolean());
        Assert.Equal(3, tensors[1].GetProperty("runtimeShape").GetArrayLength());
        Assert.Contains("fixture-diagnostic", tensors[1].GetProperty("diagnostics").EnumerateArray().Select(static item => item.GetString()), StringComparer.Ordinal);
    }

    [Fact]
    public void VisualizationWriterCreatesTaskSpecificSvgEvidence()
    {
        YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
        {
            "--family", "v8",
            "--task", "seg",
            "--layout", "boxes-first",
            "--class-count", "2"
        }, labelCount: 2);
        YoloDetection detection = new YoloDetection(0, 0.93f, 0.5f, 0.5f, 0.25f, 0.2f, 0);
        YoloSegmentationMask mask = new YoloSegmentationMask(2, 2, new[] { 1.0f, 0.0f, 0.0f, 1.0f });
        YoloVisionResult result = YoloVisionResult.FromSegmentations(new[]
        {
            new YoloSegmentationPrediction(detection, mask)
        });

        string svg = YoloVisionVisualizationWriter.ToSvg(result, new[] { "person", "car" }, profile, new[] { 1, 3, 640, 640 });

        Assert.Contains("<svg", svg, StringComparison.Ordinal);
        Assert.Contains("YoloVision seg", svg, StringComparison.Ordinal);
        Assert.Contains("mask 2x2", svg, StringComparison.Ordinal);
        Assert.Contains("data-mask-cell=\"true\"", svg, StringComparison.Ordinal);
        Assert.Contains("person", svg, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet", svg, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SegmentationOutputReportDistinguishesActiveAndTotalPrototypePixels()
    {
        YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
        {
            "--family", "v8",
            "--task", "seg",
            "--layout", "boxes-first",
            "--class-count", "1"
        }, labelCount: 1);
        YoloSegmentationMask mask = new YoloSegmentationMask(
            2,
            2,
            new[] { 0.2f, 0.6f, 0.8f, 0.4f },
            YoloSegmentationMaskValueKind.Probability,
            threshold: 0.6f);
        YoloVisionResult result = YoloVisionResult.FromSegmentations(new[]
        {
            new YoloSegmentationPrediction(new YoloDetection(0, 0.9f, 0.5f, 0.5f, 0.4f, 0.4f), mask)
        });
        YoloRuntimeOutputSet outputs = new YoloRuntimeOutputSet(new[]
        {
            new YoloRuntimeOutputTensor("boxes", YoloOutputTensorRole.Detection, new[] { 0.5f }, new[] { 1 }),
            new YoloRuntimeOutputTensor("proto", YoloOutputTensorRole.MaskPrototypes, new[] { 0.2f, 0.6f, 0.8f, 0.4f }, new[] { 1, 2, 2 })
        });

        using JsonDocument document = JsonDocument.Parse(YoloVisionOutputReport.ToJson(
            new YoloVisionOutputReportContext(
                "model.onnx",
                string.Empty,
                string.Empty,
                "ramp",
                new[] { 1, 3, 640, 640 },
                tensorRtLine: 10,
                profileIndex: 0,
                engineDeviceMemoryBytes: 0,
                elapsedMilliseconds: 1.0),
            outputs,
            profile,
            result,
            new[] { "person" }));
        JsonElement prediction = document.RootElement.GetProperty("predictions")[0];

        Assert.Equal(2, prediction.GetProperty("maskPixelCount").GetInt32());
        Assert.Equal(4, prediction.GetProperty("maskTotalPixelCount").GetInt32());
        Assert.Equal(0.6f, prediction.GetProperty("maskThreshold").GetSingle());
        Assert.Equal("probability", prediction.GetProperty("maskValueKind").GetString());
        Assert.Equal("prototype-grid-before-crop-resize", prediction.GetProperty("maskPixelCountScope").GetString());
    }

    [Fact]
    public void ModelProfileParserSupportsYoloFamiliesTasksAndPreprocess()
    {
        YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
        {
            "--family", "v11",
            "--task", "obb",
            "--input-shape", "1x3x1024x1024",
            "--layout", "boxes-first",
            "--class-count", "80",
            "--iou-threshold", "0.5",
            "--nms-mode", "class-agnostic",
            "--color-order", "BGR"
        }, labelCount: 0);

        Assert.Equal(YoloModelFamily.YoloV11, profile.Family);
        Assert.Equal(YoloTaskType.OrientedBoundingBox, profile.TaskType);
        Assert.Equal(YoloOutputLayout.BoxesFirst, profile.Postprocess.Layout);
        Assert.True(profile.Postprocess.ApplyNms);
        Assert.Equal(YoloNmsMode.ClassAgnostic, profile.Postprocess.NmsMode);
        Assert.Equal("BGR", profile.Preprocess.ColorOrder);
        Assert.Equal(new[] { 1, 3, 1024, 1024 }, profile.InputShape);
    }

    [Fact]
    public void ModelProfileParserSupportsNmsAliasesAndDisableSwitch()
    {
        YoloModelProfile classAgnostic = YoloModelProfile.FromArgs(new[]
        {
            "--task", "det",
            "--nms-mode", "global"
        }, labelCount: 0);
        YoloModelProfile disabledByMode = YoloModelProfile.FromArgs(new[]
        {
            "--task", "det",
            "--nms-mode", "none"
        }, labelCount: 0);
        YoloModelProfile disabledBySwitch = YoloModelProfile.FromArgs(new[]
        {
            "--task", "det",
            "--nms-mode", "class-agnostic",
            "--no-nms"
        }, labelCount: 0);

        Assert.Equal(YoloNmsMode.ClassAgnostic, classAgnostic.Postprocess.NmsMode);
        Assert.True(classAgnostic.Postprocess.ApplyNms);
        Assert.Equal(YoloNmsMode.None, disabledByMode.Postprocess.NmsMode);
        Assert.False(disabledByMode.Postprocess.ApplyNms);
        Assert.Equal(YoloNmsMode.ClassAgnostic, disabledBySwitch.Postprocess.NmsMode);
        Assert.False(disabledBySwitch.Postprocess.ApplyNms);
    }

    [Fact]
    public void ModelProfileParserSupportsAllPromisedFamiliesAndTasks()
    {
        (string Alias, YoloModelFamily Family)[] families =
        {
            ("v5", YoloModelFamily.YoloV5),
            ("v6", YoloModelFamily.YoloV6),
            ("v7", YoloModelFamily.YoloV7),
            ("v8", YoloModelFamily.YoloV8),
            ("v9", YoloModelFamily.YoloV9),
            ("v10", YoloModelFamily.YoloV10),
            ("v11", YoloModelFamily.YoloV11),
            ("v26", YoloModelFamily.YoloV26),
            ("yolox", YoloModelFamily.YoloX),
            ("x", YoloModelFamily.YoloX)
        };
        foreach ((string alias, YoloModelFamily expectedFamily) in families)
        {
            YoloModelProfile profile = YoloModelProfile.FromArgs(new[] { "--family", alias, "--task", "det" }, labelCount: 0);

            Assert.Equal(expectedFamily, profile.Family);
        }

        (string Alias, YoloTaskType Task)[] tasks =
        {
            ("det", YoloTaskType.Detection),
            ("cls", YoloTaskType.Classification),
            ("seg", YoloTaskType.Segmentation),
            ("obb", YoloTaskType.OrientedBoundingBox),
            ("pose", YoloTaskType.Pose),
            ("sem", YoloTaskType.SemanticSegmentation)
        };
        foreach ((string alias, YoloTaskType expectedTask) in tasks)
        {
            YoloModelProfile profile = YoloModelProfile.FromArgs(new[] { "--family", "v8", "--task", alias }, labelCount: 0);

            Assert.Equal(expectedTask, profile.TaskType);
        }
    }

    [Fact]
    public void CapabilityMatrixCoversPromisedFamiliesTasksAndOfflineCliSurface()
    {
        Assert.Equal(60, YoloCapabilityMatrix.Entries.Count);

        foreach (YoloModelFamily family in Enum.GetValues<YoloModelFamily>())
        {
            foreach (YoloTaskType task in Enum.GetValues<YoloTaskType>())
            {
                Assert.Contains(
                    YoloCapabilityMatrix.Entries,
                    entry => entry.Family == family &&
                        entry.TaskType == task &&
                        !string.IsNullOrWhiteSpace(entry.DecodePath) &&
                        !string.IsNullOrWhiteSpace(entry.AuxiliaryMetadata) &&
                        !string.IsNullOrWhiteSpace(entry.EvidenceLevel));
            }
        }

        string table = YoloCapabilityMatrix.FormatConsoleTable();
        Assert.Contains("YoloVision Capability Matrix", table, StringComparison.Ordinal);
        Assert.Contains("v5 | det | Detection", table, StringComparison.Ordinal);
        Assert.Contains("v10 | det | Detection | supported | YOLOv10 end-to-end [1,N,6]", table, StringComparison.Ordinal);
        Assert.Contains("v26 | pose | Pose", table, StringComparison.Ordinal);
        Assert.Contains("yolox | det | Detection | supported", table, StringComparison.Ordinal);
        Assert.Contains("yolox | cls | Classification | unsupported-family-task", table, StringComparison.Ordinal);
        Assert.Contains("managed-metadata-ready", table, StringComparison.Ordinal);

        string program = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "Program.cs"));
        Assert.Contains("--list-capabilities", program, StringComparison.Ordinal);
        Assert.Contains("YoloCapabilityMatrix.FormatConsoleTable()", program, StringComparison.Ordinal);
        Assert.Contains("--preprocess-only", program, StringComparison.Ordinal);
        Assert.Contains("YoloVision PreprocessOnly=True", program, StringComparison.Ordinal);
        Assert.Contains("ImagePreprocessConfig", program, StringComparison.Ordinal);
        Assert.Contains("--preflight", program, StringComparison.Ordinal);
        Assert.Contains("YoloVisionPreflightReport.Create", program, StringComparison.Ordinal);
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        string schema = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "yolovision-preflight.schema.json"));
        Assert.Contains("yolovision-preflight.v1", readme, StringComparison.Ordinal);
        Assert.Contains("isRuntimeProof=false", readme, StringComparison.Ordinal);
        Assert.Contains("\"schemaVersion\": { \"const\": \"yolovision-preflight.v1\" }", schema, StringComparison.Ordinal);
        Assert.Contains("\"tensorRtRuntimeProbed\": { \"const\": false }", schema, StringComparison.Ordinal);
        string outputValidator = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionOutputReport.ps1"));
        Assert.Contains("\"yolox\"", outputValidator, StringComparison.Ordinal);
    }

    [Fact]
    public void OfficialYoloXAcquisitionPinsUpstreamAssetsToEDriveAndPreservesPublishBoundary()
    {
        string manifestPath = Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-yolox-official-assets.json");
        using JsonDocument manifest = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = manifest.RootElement;

        Assert.Equal("e1052df71842031413f6030723c3607b839c80ce", root.GetProperty("upstreamRevision").GetString());
        Assert.Equal("0.1.1rc0", root.GetProperty("upstreamTag").GetString());
        Assert.Equal(6, root.GetProperty("assets").GetArrayLength());
        JsonElement model = root.GetProperty("assets").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "yolox-s-onnx");
        Assert.Equal(35858002, model.GetProperty("expectedLength").GetInt64());
        Assert.Equal("c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063", model.GetProperty("expectedSha256").GetString());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("canPublishPublicly").GetBoolean());

        string acquisitionScript = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Acquire-YoloXOfficialAssets.ps1"));
        Assert.Contains("downloads\\yolox-apache", acquisitionScript, StringComparison.Ordinal);
        Assert.Contains("YOLOX assets must not be downloaded to the C drive", acquisitionScript, StringComparison.Ordinal);
        Assert.Contains("Write-P6Ppm", acquisitionScript, StringComparison.Ordinal);
        Assert.Contains("Write-CocoLabels", acquisitionScript, StringComparison.Ordinal);
        string manifestValidator = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-SampleAssetManifest.ps1"));
        Assert.Contains("-example.json", manifestValidator, StringComparison.Ordinal);

        using JsonDocument example = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-yolox-s-example.json")));
        Assert.Equal("real-model-runtime", example.RootElement.GetProperty("proofClassification").GetString());
        Assert.True(example.RootElement.GetProperty("isSmokePassed").GetBoolean());
        Assert.False(example.RootElement.GetProperty("boundary").GetProperty("isPackageConsumerRuntime").GetBoolean());
        Assert.False(example.RootElement.GetProperty("boundary").GetProperty("canPublishPublicly").GetBoolean());
        string proofPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "yolox-official-runtime-proof-closure.json");
        using JsonDocument proof = JsonDocument.Parse(File.ReadAllText(proofPath));
        Assert.True(proof.RootElement.GetProperty("boundary").GetProperty("isRealModelRuntimeProof").GetBoolean());
        Assert.False(proof.RootElement.GetProperty("boundary").GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(proof.RootElement.GetProperty("boundary").GetProperty("canPublishPublicly").GetBoolean());
    }

    [Fact]
    public void OfficialYoloV10AcquisitionPinsAgplReleaseAssetsToEDriveAndKeepsNonProofBoundary()
    {
        string manifestPath = Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-yolov10-official-assets.json");
        using JsonDocument manifest = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = manifest.RootElement;

        Assert.Equal("yolovision-yolov10-official-asset-acquisition-manifest", root.GetProperty("recordKind").GetString());
        Assert.Equal("799ff3be47d21173bcf29b351820d4b8e955e0fe", root.GetProperty("upstreamRevision").GetString());
        Assert.Equal("v1.1", root.GetProperty("upstreamTag").GetString());
        Assert.Equal("AGPL-3.0-only", root.GetProperty("license").GetProperty("spdxId").GetString());
        Assert.Equal(2, root.GetProperty("assets").GetArrayLength());
        JsonElement model = root.GetProperty("assets").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "yolov10n-onnx");
        Assert.Equal(9386466, model.GetProperty("expectedLength").GetInt64());
        Assert.Equal("7025ea1913f9a259cf8a8465ed608e10610d1bb376db2e0348b13e3bd286e0d3", model.GetProperty("expectedSha256").GetString());
        Assert.Equal("end2end", root.GetProperty("modelContract").GetProperty("recommendedLayout").GetString());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("performsPublish").GetBoolean());

        string acquisitionScript = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Acquire-YoloV10OfficialAssets.ps1"));
        Assert.Contains("downloads\\yolov10-agpl", acquisitionScript, StringComparison.Ordinal);
        Assert.Contains("YOLOv10 assets must not be downloaded to the C drive", acquisitionScript, StringComparison.Ordinal);
        Assert.Contains("compatibleRuntimeInputsReady", acquisitionScript, StringComparison.Ordinal);
        Assert.Contains("canPromotePackageConsumerRuntime = $false", acquisitionScript, StringComparison.Ordinal);

        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "README.md"));
        Assert.Contains("Acquire-YoloV10OfficialAssets.ps1", readme, StringComparison.Ordinal);
        Assert.Contains("AGPL-3.0-only", readme, StringComparison.Ordinal);

        string proofPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "yolov10-official-runtime-proof-closure.json");
        using JsonDocument proof = JsonDocument.Parse(File.ReadAllText(proofPath));
        Assert.Equal("source-tree-real-model-runtime", proof.RootElement.GetProperty("proofClassification").GetString());
        Assert.True(proof.RootElement.GetProperty("boundary").GetProperty("isRealModelRuntimeProof").GetBoolean());
        Assert.False(proof.RootElement.GetProperty("boundary").GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(proof.RootElement.GetProperty("boundary").GetProperty("canPublishPublicly").GetBoolean());
    }

    [Fact]
    public void PreflightReportCapturesAssetHashesAndNeverClaimsRuntimeProof()
    {
        string directory = Path.Combine(Path.GetTempPath(), "jyppx-yolovision-preflight", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string modelPath = Path.Combine(directory, "model.onnx");
        string labelsPath = Path.Combine(directory, "labels.txt");
        string inputPath = Path.Combine(directory, "input.fp32.bin");
        try
        {
            File.WriteAllBytes(modelPath, new byte[] { 1, 2, 3, 4 });
            File.WriteAllLines(labelsPath, new[] { "person", "car" });
            File.WriteAllBytes(inputPath, new byte[4 * sizeof(float)]);
            string[] args =
            {
                "--preflight",
                "--model", modelPath,
                "--labels", labelsPath,
                "--input-data", inputPath,
                "--family", "v8",
                "--task", "seg",
                "--input-shape", "1x3x640x640",
                "--output-role-map", "boxes:det,proto:mask-prototypes",
                "--mask-coefficient-count", "32"
            };
            IReadOnlyList<string> labels = File.ReadAllLines(labelsPath);
            YoloModelProfile profile = YoloModelProfile.FromArgs(args, labels.Count);
            YoloVisionPreflightResult result = YoloVisionPreflightReport.Create(
                args,
                profile,
                modelPath,
                labelsPath,
                string.Empty,
                inputPath,
                string.Empty,
                labels,
                YoloRuntimeOutputRoleResolver.CreateMetadata(args, profile.TaskType));

            using JsonDocument document = JsonDocument.Parse(result.Json);
            JsonElement root = document.RootElement;
            Assert.Equal("yolovision-preflight.v1", root.GetProperty("schemaVersion").GetString());
            Assert.Equal("ready-for-runtime-precheck", result.State);
            Assert.False(result.HasBlockers);
            Assert.Equal(64, result.NormalizedCommandSha256.Length);
            Assert.Equal(64, root.GetProperty("assets").GetProperty("model").GetProperty("sha256").GetString()!.Length);
            Assert.True(root.GetProperty("assets").GetProperty("model").GetProperty("exists").GetBoolean());
            Assert.True(root.GetProperty("output").GetProperty("metadataDeclared").GetBoolean());
            Assert.False(root.GetProperty("execution").GetProperty("tensorRtRuntimeProbed").GetBoolean());
            Assert.False(root.GetProperty("execution").GetProperty("onnxParserInvoked").GetBoolean());
            Assert.False(root.GetProperty("boundary").GetProperty("isRuntimeProof").GetBoolean());
            Assert.False(root.GetProperty("boundary").GetProperty("canPromoteRealModelRuntime").GetBoolean());
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void StrictPreflightTurnsMissingAssetsIntoBlockersWithoutExecutingAnything()
    {
        string modelPath = Path.Combine(Path.GetTempPath(), "missing-yolovision-model.onnx");
        string[] args =
        {
            "--preflight",
            "--strict-preflight",
            "--model", modelPath,
            "--family", "v11",
            "--task", "pose",
            "--input-shape", "1x3x640x640"
        };
        YoloModelProfile profile = YoloModelProfile.FromArgs(args, labelCount: 0);
        YoloVisionPreflightResult result = YoloVisionPreflightReport.Create(
            args,
            profile,
            modelPath,
            string.Empty,
            string.Empty,
            string.Empty,
            string.Empty,
            Array.Empty<string>(),
            metadata: null);

        Assert.Equal("invalid", result.State);
        Assert.True(result.HasBlockers);
        Assert.Contains("precheck", result.Json, StringComparison.Ordinal);
        Assert.Contains("tensorRtRuntimeProbed", result.Json, StringComparison.Ordinal);
        Assert.Contains("blocker", result.Json, StringComparison.Ordinal);
    }

    [Fact]
    public void PreflightRejectsConflictingInputSources()
    {
        string[] args =
        {
            "--preflight",
            "--model", "model.onnx",
            "--image", "image.ppm",
            "--input-data", "input.bin",
            "--family", "v8",
            "--task", "det"
        };
        YoloModelProfile profile = YoloModelProfile.FromArgs(args, labelCount: 1);
        YoloVisionPreflightResult result = YoloVisionPreflightReport.Create(
            args,
            profile,
            "model.onnx",
            string.Empty,
            string.Empty,
            "input.bin",
            "image.ppm",
            new[] { "person" },
            metadata: null);

        Assert.Equal("invalid", result.State);
        Assert.True(result.HasBlockers);
        Assert.Contains("Choose exactly one input source", result.Json, StringComparison.Ordinal);
        Assert.False(result.Json.Contains("engineBuildInvoked\": true", StringComparison.Ordinal));
    }

    [Fact]
    public void RuntimeOutputRoleResolverUsesExplicitMapsNamesHeuristicsAndMetadata()
    {
        string[] args =
        {
            "--output-role-map", "boxes:det,proto:mask-prototypes,kpts:pose-keypoints,theta:obb-angles",
            "--mask-coefficient-count", "32",
            "--mask-threshold", "0.65",
            "--aux-channel-start", "84",
            "--aux-layout", "boxes-first"
        };

        Assert.Equal(YoloOutputTensorRole.Detection, YoloRuntimeOutputRoleResolver.ResolveRole("boxes", YoloTaskType.Segmentation, args, isPrimary: true));
        Assert.Equal(YoloOutputTensorRole.MaskPrototypes, YoloRuntimeOutputRoleResolver.ResolveRole("proto", YoloTaskType.Segmentation, args, isPrimary: false));
        Assert.Equal(YoloOutputTensorRole.PoseKeypoints, YoloRuntimeOutputRoleResolver.ResolveRole("kpts", YoloTaskType.Pose, args, isPrimary: false));
        Assert.Equal(YoloOutputTensorRole.ObbAngles, YoloRuntimeOutputRoleResolver.ResolveRole("theta", YoloTaskType.OrientedBoundingBox, args, isPrimary: false));
        Assert.Equal(YoloOutputTensorRole.Classification, YoloRuntimeOutputRoleResolver.ResolveRole("logits", YoloTaskType.Detection, Array.Empty<string>(), isPrimary: false));
        Assert.Equal(YoloOutputTensorRole.SemanticMap, YoloRuntimeOutputRoleResolver.GetPrimaryOutputRole(YoloTaskType.SemanticSegmentation));

        YoloMultiOutputMetadata? metadata = YoloRuntimeOutputRoleResolver.CreateMetadata(args, YoloTaskType.Segmentation);

        Assert.NotNull(metadata);
        Assert.Equal(32, metadata!.MaskCoefficientCount);
        Assert.Equal(0.65f, metadata.MaskThreshold);
        Assert.Equal(84, metadata.AuxiliaryChannelStart);
        Assert.Equal(YoloOutputLayout.BoxesFirst, metadata.AuxiliaryLayout);
        Assert.Null(YoloRuntimeOutputRoleResolver.CreateMetadata(Array.Empty<string>(), YoloTaskType.Segmentation));
    }

    [Fact]
    public void RuntimeOutputRoleResolverCreatesPoseAndObbMetadataOnlyWhenDeclared()
    {
        string[] poseArgs =
        {
            "--pose-keypoints-output", "kpts",
            "--keypoint-count", "17",
            "--keypoint-stride", "3"
        };
        string[] obbArgs =
        {
            "--obb-angle-output", "angle",
            "--angle-degrees"
        };

        YoloMultiOutputMetadata? poseMetadata = YoloRuntimeOutputRoleResolver.CreateMetadata(poseArgs, YoloTaskType.Pose);
        YoloMultiOutputMetadata? obbMetadata = YoloRuntimeOutputRoleResolver.CreateMetadata(obbArgs, YoloTaskType.OrientedBoundingBox);

        Assert.NotNull(poseMetadata);
        Assert.Equal(17, poseMetadata!.PoseKeypointCount);
        Assert.Equal(3, poseMetadata.PoseKeypointStride);
        Assert.NotNull(obbMetadata);
        Assert.True(obbMetadata!.ObbAngleInDegrees);
        Assert.Null(YoloRuntimeOutputRoleResolver.CreateMetadata(Array.Empty<string>(), YoloTaskType.OrientedBoundingBox));
    }

    [Fact]
    public void ClassAgnosticNmsSuppressesOverlappingBoxesAcrossClasses()
    {
        YoloDetection first = new YoloDetection(0, 0.9f, 10, 10, 4, 4);
        YoloDetection second = new YoloDetection(1, 0.8f, 10.1f, 10.1f, 4, 4);

        Assert.Equal(2, YoloDetectionDecoder.ApplyClassAwareNms(new[] { first, second }, 0.45f).Count);
        Assert.Single(YoloDetectionDecoder.ApplyClassAgnosticNms(new[] { first, second }, 0.45f));
    }

    [Fact]
    public void EndToEndOutputValidatesRankAndChannels()
    {
        YoloEndToEndOutput output = YoloEndToEndOutput.FromShape(new[] { 1, 300, 6 });

        Assert.Equal(300, output.DetectionCount);
        Assert.Equal(6, output.ChannelCount);
        Assert.Equal(1800, output.ValueCount);
        Assert.Throws<NotSupportedException>(() => YoloEndToEndOutput.FromShape(new[] { 300, 6 }));
        Assert.Throws<NotSupportedException>(() => YoloEndToEndOutput.FromShape(new[] { 2, 300, 6 }));
        Assert.Throws<NotSupportedException>(() => YoloEndToEndOutput.FromShape(new[] { 1, 300, 7 }));
    }

    [Fact]
    public void YoloV10EndToEndDecoderUsesXyxyScoreClassColumnsWithoutSecondNms()
    {
        YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
        {
            "--family", "v10",
            "--task", "det",
            "--layout", "end2end",
            "--class-count", "3",
            "--confidence", "0.25",
            "--top-k", "10"
        }, labelCount: 0);
        float[] values =
        {
            10.0f, 20.0f, 30.0f, 40.0f, 0.90f, 2.0f,
            10.5f, 20.5f, 30.5f, 40.5f, 0.80f, 2.0f,
            0.0f, 0.0f, 5.0f, 5.0f, 0.20f, 1.0f
        };

        IReadOnlyList<YoloDetection> detections = YoloSampleRunner.DecodeDetections(values, new[] { 1, 3, 6 }, profile);

        Assert.Equal(YoloOutputLayout.EndToEndNms, profile.Postprocess.Layout);
        Assert.False(profile.Postprocess.HasObjectness);
        Assert.False(profile.Postprocess.ApplyNms);
        Assert.Equal(YoloNmsMode.None, profile.Postprocess.NmsMode);
        Assert.Equal(2, detections.Count);
        Assert.Equal(2, detections[0].ClassIndex);
        Assert.Equal(0.90f, detections[0].Score, 5);
        Assert.Equal(20.0f, detections[0].CenterX, 5);
        Assert.Equal(30.0f, detections[0].CenterY, 5);
        Assert.Equal(20.0f, detections[0].Width, 5);
        Assert.Equal(20.0f, detections[0].Height, 5);
        Assert.Equal(0, detections[0].SourceIndex);
        Assert.Equal(1, detections[1].SourceIndex);
    }

    [Fact]
    public void YoloV10EndToEndDecoderRejectsMalformedRows()
    {
        YoloPostprocessOptions options = new YoloPostprocessOptions(
            YoloOutputLayout.EndToEndNms,
            hasObjectness: true,
            classCount: 2,
            confidenceThreshold: 0.25f,
            iouThreshold: 0.45f,
            topK: 10,
            applyNms: true);

        Assert.Throws<ArgumentException>(() => YoloDetectionDecoder.DecodeEndToEnd(
            new[] { 0.0f, 0.0f, 2.0f, 2.0f, 0.9f },
            new[] { 1, 1, 6 },
            options));
        Assert.Throws<InvalidOperationException>(() => YoloDetectionDecoder.DecodeEndToEnd(
            new[] { 2.0f, 0.0f, 1.0f, 2.0f, 0.9f, 0.0f },
            new[] { 1, 1, 6 },
            options));
        Assert.Throws<InvalidOperationException>(() => YoloDetectionDecoder.DecodeEndToEnd(
            new[] { 0.0f, 0.0f, 2.0f, 2.0f, 0.9f, 0.5f },
            new[] { 1, 1, 6 },
            options));
        Assert.Throws<InvalidOperationException>(() => YoloDetectionDecoder.DecodeEndToEnd(
            new[] { 0.0f, 0.0f, 2.0f, 2.0f, 0.9f, 2.0f },
            new[] { 1, 1, 6 },
            options));
    }

    [Fact]
    public void YoloV10EndToEndManagedSmokeCommandRunsWithoutRuntimeAssets()
    {
        Assert.Equal(0, YoloVisionCommand.Run(new[] { "--self-test-end2end" }));
    }

    [Fact]
    public void MaskComposerValidatesPrototypeShapeAndCombinesPlanes()
    {
        YoloSegmentationMask mask = YoloMaskComposer.ComposeLinearMask(
            new[] { 0.5f, 2.0f },
            new[] { 1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f },
            prototypeCount: 2,
            width: 2,
            height: 2);

        Assert.Equal(new[] { 20.5f, 41.0f, 61.5f, 82.0f }, mask.Values);
        Assert.Throws<ArgumentException>(() => YoloMaskComposer.ComposeLinearMask(new[] { 1.0f }, new[] { 1.0f, 2.0f }, 2, 1, 1));
    }

    [Fact]
    public void MaskComposerProducesStableProbabilitiesAndThresholdedPixelCounts()
    {
        YoloSegmentationMask mask = YoloMaskComposer.ComposeProbabilityMask(
            new[] { 1.0f },
            new[] { -2.0f, 0.0f, 2.0f },
            prototypeCount: 1,
            width: 3,
            height: 1,
            threshold: 0.5f);

        Assert.Equal(YoloSegmentationMaskValueKind.Probability, mask.ValueKind);
        Assert.Equal(0.119203f, mask.Values[0], precision: 5);
        Assert.Equal(0.5f, mask.Values[1], precision: 5);
        Assert.Equal(0.880797f, mask.Values[2], precision: 5);
        Assert.Equal(2, mask.CountPixelsAtOrAboveThreshold());
        Assert.Equal(0.0f, YoloMaskComposer.Sigmoid(float.NegativeInfinity));
        Assert.Equal(1.0f, YoloMaskComposer.Sigmoid(float.PositiveInfinity));
        Assert.Throws<ArgumentOutOfRangeException>(() => new YoloSegmentationMask(1, 1, new[] { 0.5f }, YoloSegmentationMaskValueKind.Probability, float.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new YoloSegmentationMask(1, 1, new[] { 0.5f }, (YoloSegmentationMaskValueKind)99, 0.5f));
        YoloSegmentationMask nanMask = new YoloSegmentationMask(1, 1, new[] { float.NaN });
        Assert.Equal(0.0f, nanMask.GetProbability(0));
        Assert.Equal(0, nanMask.CountPixelsAtOrAboveThreshold());
    }

    [Fact]
    public void PoseAndObbHelpersValidateCommonLayouts()
    {
        YoloPoseKeypoint[] keypoints = YoloPoseDecoder.DecodeFlatKeypoints(new[] { 1.0f, 2.0f, 0.9f, 3.0f, 4.0f, 0.8f }, keypointCount: 2);
        YoloObbDetection obb = YoloObbDecoder.Decode(new YoloDetection(0, 0.7f, 10, 10, 4, 2), 90.0f, angleInDegrees: true);
        YoloSemanticMap semantic = new YoloSemanticMap(2, 1, 2, new[] { 0.1f, 0.2f, 0.3f, 0.4f });

        Assert.Equal(2, keypoints.Length);
        Assert.Equal(0.9f, keypoints[0].Score);
        Assert.Equal(MathF.PI / 2.0f, obb.AngleRadians, precision: 5);
        Assert.Equal(2, semantic.ClassCount);
        Assert.Throws<ArgumentException>(() => YoloPoseDecoder.DecodeFlatKeypoints(new[] { 1.0f, 2.0f }, keypointCount: 2));
    }

    [Fact]
    public void MultiOutputSegmentationComposesMasksForKeptDetections()
    {
        YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
        {
            "--task", "seg",
            "--layout", "boxes-first",
            "--class-count", "2",
            "--has-objectness", "false",
            "--confidence", "0.25",
            "--top-k", "4",
            "--no-nms"
        }, labelCount: 0);

        float[] boxAndCoefficientValues =
        {
            10, 10, 2, 2, 0.9f, 0.1f, 0.5f, 2.0f,
            20, 20, 2, 2, 0.1f, 0.95f, 1.0f, 0.25f
        };
        float[] prototypes =
        {
            1, 2, 3, 4,
            10, 20, 30, 40
        };

        YoloVisionResult result = YoloSampleRunner.DecodeSegmentationOutputs(
            boxAndCoefficientValues,
            new[] { 1, 2, 8 },
            prototypes,
            new[] { 2, 2, 2 },
            profile,
            YoloMultiOutputMetadata.ForSegmentation(maskCoefficientCount: 2));

        Assert.Equal(YoloTaskType.Segmentation, result.TaskType);
        Assert.True(result.HasSegmentationMasks);
        Assert.Equal(2, result.Segmentations.Count);
        Assert.Equal(1, result.Segmentations[0].Detection.ClassIndex);
        Assert.Equal(YoloSegmentationMaskValueKind.Probability, result.Segmentations[0].Mask.ValueKind);
        Assert.Equal(YoloMaskComposer.Sigmoid(3.5f), result.Segmentations[0].Mask.Values[0], precision: 5);
        Assert.Equal(YoloMaskComposer.Sigmoid(14.0f), result.Segmentations[0].Mask.Values[3], precision: 5);
        Assert.Equal(0, result.Segmentations[1].Detection.ClassIndex);
        Assert.Equal(YoloMaskComposer.Sigmoid(20.5f), result.Segmentations[1].Mask.Values[0], precision: 5);
        Assert.Equal(YoloMaskComposer.Sigmoid(82.0f), result.Segmentations[1].Mask.Values[3], precision: 5);
    }

    [Fact]
    public void RuntimeOutputSetRoutesSegmentationRolesToManagedDecoder()
    {
        YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
        {
            "--task", "seg",
            "--layout", "boxes-first",
            "--class-count", "2",
            "--has-objectness", "false",
            "--confidence", "0.25",
            "--top-k", "4",
            "--no-nms"
        }, labelCount: 0);

        YoloRuntimeOutputSet outputs = new YoloRuntimeOutputSet(new[]
        {
            new YoloRuntimeOutputTensor(
                "boxes",
                YoloOutputTensorRole.Detection,
                new[] { 10, 10, 2, 2, 0.9f, 0.1f, 0.5f, 2.0f },
                new[] { 1, 1, 8 }),
            new YoloRuntimeOutputTensor(
                "proto",
                YoloOutputTensorRole.MaskPrototypes,
                new[] { 1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f },
                new[] { 2, 2, 2 })
        });

        YoloVisionResult result = YoloSampleRunner.DecodeRuntimeOutputs(
            outputs,
            profile,
            YoloMultiOutputMetadata.ForSegmentation(maskCoefficientCount: 2));

        Assert.True(result.HasSegmentationMasks);
        Assert.Single(result.Segmentations);
        Assert.Equal(YoloSegmentationMaskValueKind.Probability, result.Segmentations[0].Mask.ValueKind);
        Assert.Equal(YoloMaskComposer.Sigmoid(20.5f), result.Segmentations[0].Mask.Values[0], precision: 5);
        Assert.Equal(YoloMaskComposer.Sigmoid(82.0f), result.Segmentations[0].Mask.Values[3], precision: 5);
        Assert.Contains("MaskPrototypes:proto", outputs.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void OutputReportSerializesPointerFreeJsonForOwnerReview()
    {
        string directory = Path.Combine(Path.GetTempPath(), "jyppx-yolovision-output-report", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string modelPath = Path.Combine(directory, "model.onnx");
        string inputPath = Path.Combine(directory, "input.bin");
        try
        {
            File.WriteAllBytes(modelPath, new byte[] { 1, 2, 3, 4 });
            File.WriteAllBytes(inputPath, new byte[] { 5, 6, 7, 8 });

            YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
            {
                "--family", "v8",
                "--task", "det",
                "--layout", "boxes-first",
                "--class-count", "2",
                "--has-objectness", "false",
                "--confidence", "0.25",
                "--top-k", "4",
                "--no-nms"
            }, labelCount: 0);
            YoloRuntimeOutputSet outputs = new YoloRuntimeOutputSet(new[]
            {
                new YoloRuntimeOutputTensor("boxes", YoloOutputTensorRole.Detection, new[] { 10, 10, 2, 2, 0.9f, 0.1f }, new[] { 1, 1, 6 })
            });
            YoloVisionResult result = YoloSampleRunner.DecodeRuntimeOutputs(outputs, profile);

            string json = YoloVisionOutputReport.ToJson(
                new YoloVisionOutputReportContext(modelPath, inputPath, string.Empty, "ramp", new[] { 1, 3, 640, 640 }, 11, 0, 1234, 1.5),
                outputs,
                profile,
                result,
                new[] { "person", "car" });

            using JsonDocument document = JsonDocument.Parse(json);
            JsonElement root = document.RootElement;

            Assert.Equal("yolovision-output.v1", root.GetProperty("schemaVersion").GetString());
            Assert.Equal("det", root.GetProperty("task").GetString());
            Assert.Equal("yolov8", root.GetProperty("modelFamily").GetString());
            Assert.Equal("external-tensor", root.GetProperty("input").GetProperty("sourceKind").GetString());
            Assert.Equal(64, root.GetProperty("input").GetProperty("sha256").GetString()!.Length);
            Assert.Equal("in-memory-from-onnx", root.GetProperty("engine").GetProperty("materialization").GetString());
            Assert.Equal(64, root.GetProperty("engine").GetProperty("modelSha256").GetString()!.Length);
            Assert.Equal("boxes", root.GetProperty("outputs")[0].GetProperty("role").GetString());
            Assert.Single(root.GetProperty("predictions").EnumerateArray());
            Assert.Equal("person", root.GetProperty("predictions")[0].GetProperty("className").GetString());
            Assert.False(root.GetProperty("boundary").GetProperty("isRuntimeProof").GetBoolean());
            Assert.Contains("not runtime proof", root.GetProperty("boundary").GetProperty("evidenceKind").GetString(), StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void MultiOutputPoseMapsKeypointsToKeptDetections()
    {
        YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
        {
            "--task", "pose",
            "--layout", "boxes-first",
            "--class-count", "1",
            "--has-objectness", "false",
            "--confidence", "0.25",
            "--top-k", "4",
            "--no-nms"
        }, labelCount: 0);

        float[] boxValues =
        {
            10, 10, 2, 2, 0.8f,
            20, 20, 2, 2, 0.9f
        };
        float[] keypointValues =
        {
            1, 2, 0.7f, 3, 4, 0.8f,
            5, 6, 0.9f, 7, 8, 0.6f
        };

        YoloVisionResult result = YoloSampleRunner.DecodePoseOutputs(
            boxValues,
            new[] { 1, 2, 5 },
            keypointValues,
            new[] { 1, 2, 6 },
            profile,
            YoloMultiOutputMetadata.ForPose(keypointCount: 2, auxiliaryLayout: YoloOutputLayout.BoxesFirst));

        Assert.Equal(YoloTaskType.Pose, result.TaskType);
        Assert.True(result.HasPoses);
        Assert.Equal(2, result.Poses.Count);
        Assert.Equal(20, result.Poses[0].Detection.CenterX);
        Assert.Equal(5, result.Poses[0].Keypoints[0].X);
        Assert.Equal(8, result.Poses[0].Keypoints[1].Y);
    }

    [Fact]
    public void RuntimeOutputSetRoutesPoseAndObbRolesToManagedDecoders()
    {
        YoloModelProfile poseProfile = YoloModelProfile.FromArgs(new[]
        {
            "--task", "pose",
            "--layout", "boxes-first",
            "--class-count", "1",
            "--has-objectness", "false",
            "--confidence", "0.25",
            "--top-k", "4",
            "--no-nms"
        }, labelCount: 0);
        YoloRuntimeOutputSet poseOutputs = new YoloRuntimeOutputSet(new[]
        {
            new YoloRuntimeOutputTensor("boxes", YoloOutputTensorRole.Detection, new[] { 10, 10, 2, 2, 0.8f }, new[] { 1, 1, 5 }),
            new YoloRuntimeOutputTensor("keypoints", YoloOutputTensorRole.PoseKeypoints, new[] { 1, 2, 0.7f, 3, 4, 0.8f }, new[] { 1, 1, 6 })
        });

        YoloVisionResult pose = YoloSampleRunner.DecodeRuntimeOutputs(
            poseOutputs,
            poseProfile,
            YoloMultiOutputMetadata.ForPose(keypointCount: 2, auxiliaryLayout: YoloOutputLayout.BoxesFirst));

        YoloModelProfile obbProfile = YoloModelProfile.FromArgs(new[]
        {
            "--task", "obb",
            "--layout", "boxes-first",
            "--class-count", "1",
            "--has-objectness", "false",
            "--confidence", "0.25",
            "--top-k", "4",
            "--no-nms"
        }, labelCount: 0);
        YoloRuntimeOutputSet obbOutputs = new YoloRuntimeOutputSet(new[]
        {
            new YoloRuntimeOutputTensor("boxes", YoloOutputTensorRole.Detection, new[] { 10, 10, 2, 2, 0.8f }, new[] { 1, 1, 5 }),
            new YoloRuntimeOutputTensor("angle", YoloOutputTensorRole.ObbAngles, new[] { 90.0f }, new[] { 1, 1, 1 })
        });

        YoloVisionResult obb = YoloSampleRunner.DecodeRuntimeOutputs(
            obbOutputs,
            obbProfile,
            YoloMultiOutputMetadata.ForObb(angleInDegrees: true, auxiliaryLayout: YoloOutputLayout.BoxesFirst));

        Assert.True(pose.HasPoses);
        Assert.Equal(2, pose.Poses[0].Keypoints.Length);
        Assert.True(obb.HasOrientedBoxes);
        Assert.Equal(MathF.PI / 2.0f, obb.OrientedBoxes[0].AngleRadians, precision: 5);
    }

    [Fact]
    public void MultiOutputObbConvertsAnglesForKeptDetections()
    {
        YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
        {
            "--task", "obb",
            "--layout", "boxes-first",
            "--class-count", "1",
            "--has-objectness", "false",
            "--confidence", "0.25",
            "--top-k", "4",
            "--no-nms"
        }, labelCount: 0);

        YoloVisionResult result = YoloSampleRunner.DecodeObbOutputs(
            new[] { 10, 10, 2, 2, 0.6f, 20, 20, 2, 2, 0.95f },
            new[] { 1, 2, 5 },
            new[] { 45.0f, 90.0f },
            new[] { 1, 2, 1 },
            profile,
            YoloMultiOutputMetadata.ForObb(angleInDegrees: true, auxiliaryLayout: YoloOutputLayout.BoxesFirst));

        Assert.Equal(YoloTaskType.OrientedBoundingBox, result.TaskType);
        Assert.True(result.HasOrientedBoxes);
        Assert.Equal(2, result.OrientedBoxes.Count);
        Assert.Equal(MathF.PI / 2.0f, result.OrientedBoxes[0].AngleRadians, precision: 5);
        Assert.Equal(MathF.PI / 4.0f, result.OrientedBoxes[1].AngleRadians, precision: 5);
    }

    [Fact]
    public void SegmentationSpatialTransformUsesExplicitCoordinatesAndOptionalBoxCrop()
    {
        YoloImagePreprocessResult preprocess = CreatePreprocess(
            sourceWidth: 4,
            sourceHeight: 4,
            targetWidth: 4,
            targetHeight: 4,
            resizedWidth: 4,
            resizedHeight: 4,
            padX: 0,
            padY: 0,
            scaleX: 1.0f,
            scaleY: 1.0f);
        YoloSegmentationPrediction prediction = new YoloSegmentationPrediction(
            new YoloDetection(0, 0.9f, 2.0f, 2.0f, 2.0f, 2.0f, sourceIndex: 3),
            new YoloSegmentationMask(2, 2, new[] { 1.0f, 1.0f, 1.0f, 1.0f }));

        YoloSegmentationSpatialTransformResult cropped = YoloSegmentationSpatialTransform.Apply(
            prediction,
            preprocess,
            new YoloSegmentationSpatialTransformOptions(YoloSegmentationCoordinateSpace.ModelInputPixels, cropToDetection: true));
        YoloSegmentationSpatialTransformResult uncropped = YoloSegmentationSpatialTransform.Apply(
            prediction,
            preprocess,
            new YoloSegmentationSpatialTransformOptions(YoloSegmentationCoordinateSpace.ModelInputPixels, cropToDetection: false));
        YoloSegmentationSpatialTransformResult normalized = YoloSegmentationSpatialTransform.Apply(
            new YoloSegmentationPrediction(
                new YoloDetection(0, 0.9f, 0.5f, 0.5f, 0.5f, 0.5f, sourceIndex: 3),
                prediction.Mask),
            preprocess,
            new YoloSegmentationSpatialTransformOptions(YoloSegmentationCoordinateSpace.Normalized, cropToDetection: true));

        Assert.Equal(4, cropped.Mask.CountPixelsAtOrAboveThreshold());
        Assert.Equal(16, uncropped.Mask.CountPixelsAtOrAboveThreshold());
        Assert.Equal(cropped.Mask.Values, normalized.Mask.Values);
        Assert.Equal(2.0f, cropped.Detection.CenterX);
        Assert.Equal(2.0f, cropped.Detection.CenterY);
        Assert.Equal(2.0f, cropped.Detection.Width);
        Assert.Equal(2.0f, cropped.Detection.Height);
        Assert.Equal(3, cropped.Detection.SourceIndex);
        Assert.Equal("bilinear", cropped.Interpolation);
        Assert.Contains("explicit-preprocess-metadata-transform", cropped.Boundary, StringComparison.Ordinal);
    }

    [Fact]
    public void SegmentationSpatialTransformRemovesLetterboxPaddingBeforeSourceResize()
    {
        YoloImagePreprocessResult preprocess = CreatePreprocess(
            sourceWidth: 4,
            sourceHeight: 2,
            targetWidth: 4,
            targetHeight: 4,
            resizedWidth: 4,
            resizedHeight: 2,
            padX: 0,
            padY: 1,
            scaleX: 1.0f,
            scaleY: 1.0f);
        YoloSegmentationMask prototype = new YoloSegmentationMask(
            4,
            4,
            new[]
            {
                0.0f, 0.0f, 0.0f, 0.0f,
                1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f,
                0.0f, 0.0f, 0.0f, 0.0f
            });
        YoloSegmentationPrediction prediction = new YoloSegmentationPrediction(
            new YoloDetection(0, 0.9f, 2.0f, 2.0f, 4.0f, 4.0f),
            prototype);

        YoloSegmentationSpatialTransformResult result = YoloSegmentationSpatialTransform.Apply(
            prediction,
            preprocess,
            new YoloSegmentationSpatialTransformOptions(YoloSegmentationCoordinateSpace.ModelInputPixels, cropToDetection: false));

        Assert.Equal(4, result.Mask.Width);
        Assert.Equal(2, result.Mask.Height);
        Assert.Equal(8, result.Mask.CountPixelsAtOrAboveThreshold());
        Assert.All(result.Mask.Values, value => Assert.Equal(1.0f, value, precision: 5));
        Assert.Equal(1.0f, result.Detection.CenterY);
        Assert.Equal(2.0f, result.Detection.Height);
    }

    [Fact]
    public void SegmentationSpatialTransformOptionsAndMetadataFailClosed()
    {
        Assert.Null(YoloSegmentationSpatialTransformOptions.FromArgs(Array.Empty<string>(), YoloTaskType.Segmentation));
        Assert.Throws<ArgumentException>(() => YoloSegmentationSpatialTransformOptions.FromArgs(
            new[] { "--mask-spatial-transform" },
            YoloTaskType.Segmentation));
        Assert.Throws<ArgumentException>(() => YoloSegmentationSpatialTransformOptions.FromArgs(
            new[] { "--mask-spatial-transform", "--mask-coordinate-space", "model-input" },
            YoloTaskType.Detection));
        Assert.Throws<ArgumentException>(() => YoloSegmentationSpatialTransformOptions.FromArgs(
            new[]
            {
                "--mask-spatial-transform",
                "--mask-coordinate-space", "model-input",
                "--mask-crop-to-box"
            },
            YoloTaskType.Segmentation));
        YoloSegmentationSpatialTransformOptions options = YoloSegmentationSpatialTransformOptions.FromArgs(
            new[]
            {
                "--mask-spatial-transform",
                "--mask-coordinate-space", "normalized",
                "--mask-crop-to-box", "false"
            },
            YoloTaskType.Segmentation)!;
        Assert.Equal(YoloSegmentationCoordinateSpace.Normalized, options.CoordinateSpace);
        Assert.False(options.CropToDetection);

        YoloImagePreprocessResult invalid = CreatePreprocess(
            sourceWidth: 4,
            sourceHeight: 4,
            targetWidth: 4,
            targetHeight: 4,
            resizedWidth: 4,
            resizedHeight: 4,
            padX: 1,
            padY: 0,
            scaleX: 1.0f,
            scaleY: 1.0f);
        Assert.Throws<ArgumentException>(() => YoloSegmentationSpatialTransform.Apply(
            new YoloSegmentationPrediction(
                new YoloDetection(0, 0.9f, 2.0f, 2.0f, 2.0f, 2.0f),
                new YoloSegmentationMask(1, 1, new[] { 1.0f })),
            invalid,
            options));

        Assert.Throws<ArgumentException>(() => new YoloVisionOutputReportContext(
            "model.onnx",
            string.Empty,
            string.Empty,
            "ramp",
            new[] { 1, 3, 4, 4 },
            tensorRtLine: 10,
            profileIndex: 0,
            engineDeviceMemoryBytes: 0,
            elapsedMilliseconds: 1.0,
            labelsPath: string.Empty,
            imagePreprocess: null,
            segmentationSpatialTransform: options));

        YoloVisionResult segmentation = YoloVisionResult.FromSegmentations(new[]
        {
            new YoloSegmentationPrediction(
                new YoloDetection(0, 0.9f, 2.0f, 2.0f, 2.0f, 2.0f),
                new YoloSegmentationMask(1, 1, new[] { 1.0f }))
        });
        YoloModelProfile profile = YoloModelProfile.FromArgs(
            new[] { "--family", "v8", "--task", "seg", "--class-count", "1" },
            labelCount: 1);
        Assert.Throws<ArgumentException>(() => YoloVisionVisualizationWriter.ToSvg(
            segmentation,
            new[] { "person" },
            profile,
            new[] { 1, 3, 4, 4 },
            imagePreprocess: null,
            segmentationSpatialTransform: options));
    }

    [Fact]
    public void SegmentationSpatialTransformUsesRoundedResizeDimensionsAsEffectiveScale()
    {
        YoloImagePreprocessResult preprocess = CreatePreprocess(
            sourceWidth: 7,
            sourceHeight: 5,
            targetWidth: 4,
            targetHeight: 4,
            resizedWidth: 4,
            resizedHeight: 3,
            padX: 0,
            padY: 0,
            scaleX: 4.0f / 7.0f,
            scaleY: 4.0f / 7.0f);
        YoloSegmentationPrediction prediction = new YoloSegmentationPrediction(
            new YoloDetection(0, 0.9f, 1.0f, 1.0f, 2.0f, 2.0f),
            new YoloSegmentationMask(1, 1, new[] { 1.0f }));

        YoloSegmentationSpatialTransformResult result = YoloSegmentationSpatialTransform.Apply(
            prediction,
            preprocess,
            new YoloSegmentationSpatialTransformOptions(
                YoloSegmentationCoordinateSpace.ModelInputPixels,
                cropToDetection: false));

        Assert.Equal(4.0f / 7.0f, result.EffectiveScaleX, precision: 5);
        Assert.Equal(3.0f / 5.0f, result.EffectiveScaleY, precision: 5);
        Assert.Equal(1.75f, result.Detection.CenterX, precision: 5);
        Assert.Equal(5.0f / 3.0f, result.Detection.CenterY, precision: 5);
        Assert.Equal(3.5f, result.Detection.Width, precision: 5);
        Assert.Equal(10.0f / 3.0f, result.Detection.Height, precision: 5);
    }

    [Fact]
    public void SegmentationSpatialTransformPreflightRequiresImageAndRecordsExplicitIntent()
    {
        string[] missingImageArgs =
        {
            "--preflight",
            "--model", "model.onnx",
            "--family", "v8",
            "--task", "seg",
            "--class-count", "1",
            "--mask-coefficient-count", "32",
            "--mask-spatial-transform",
            "--mask-coordinate-space", "normalized"
        };
        YoloModelProfile missingImageProfile = YoloModelProfile.FromArgs(missingImageArgs, labelCount: 1);
        YoloVisionPreflightResult missingImage = YoloVisionPreflightReport.Create(
            missingImageArgs,
            missingImageProfile,
            "model.onnx",
            string.Empty,
            string.Empty,
            string.Empty,
            string.Empty,
            new[] { "person" },
            YoloRuntimeOutputRoleResolver.CreateMetadata(missingImageArgs, missingImageProfile.TaskType));

        Assert.True(missingImage.HasBlockers);
        using (JsonDocument missingImageDocument = JsonDocument.Parse(missingImage.Json))
        {
            Assert.Contains(missingImageDocument.RootElement.GetProperty("checks").EnumerateArray(), static check =>
                check.GetProperty("id").GetString() == "segmentation-spatial-transform-image" &&
                !check.GetProperty("passed").GetBoolean());
        }

        string[] imageArgs = missingImageArgs.Concat(new[] { "--image", "image.ppm" }).ToArray();
        YoloModelProfile imageProfile = YoloModelProfile.FromArgs(imageArgs, labelCount: 1);
        YoloVisionPreflightResult image = YoloVisionPreflightReport.Create(
            imageArgs,
            imageProfile,
            "model.onnx",
            string.Empty,
            string.Empty,
            string.Empty,
            "image.ppm",
            new[] { "person" },
            YoloRuntimeOutputRoleResolver.CreateMetadata(imageArgs, imageProfile.TaskType));

        Assert.False(image.HasBlockers);
        using JsonDocument imageDocument = JsonDocument.Parse(image.Json);
        JsonElement spatial = imageDocument.RootElement.GetProperty("output").GetProperty("spatialTransform");
        Assert.True(spatial.GetProperty("requested").GetBoolean());
        Assert.Equal("normalized", spatial.GetProperty("coordinateSpace").GetString());
        Assert.True(spatial.GetProperty("cropToDetection").GetBoolean());
        Assert.True(spatial.GetProperty("requiresImagePreprocessMetadata").GetBoolean());
        Assert.Contains(imageDocument.RootElement.GetProperty("checks").EnumerateArray(), static check =>
            check.GetProperty("id").GetString() == "segmentation-spatial-transform-image" &&
            check.GetProperty("passed").GetBoolean());
    }

    [Fact]
    public void SegmentationSpatialTransformIsWrittenToJsonAndSourceImageSvg()
    {
        YoloImagePreprocessResult preprocess = CreatePreprocess(
            sourceWidth: 4,
            sourceHeight: 4,
            targetWidth: 4,
            targetHeight: 4,
            resizedWidth: 4,
            resizedHeight: 4,
            padX: 0,
            padY: 0,
            scaleX: 1.0f,
            scaleY: 1.0f);
        YoloSegmentationSpatialTransformOptions spatialOptions = new YoloSegmentationSpatialTransformOptions(
            YoloSegmentationCoordinateSpace.ModelInputPixels,
            cropToDetection: true);
        YoloSegmentationPrediction prediction = new YoloSegmentationPrediction(
            new YoloDetection(0, 0.9f, 2.0f, 2.0f, 2.0f, 2.0f),
            new YoloSegmentationMask(2, 2, new[] { 1.0f, 1.0f, 1.0f, 1.0f }));
        YoloVisionResult result = YoloVisionResult.FromSegmentations(new[] { prediction });
        YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
        {
            "--family", "v8",
            "--task", "seg",
            "--layout", "boxes-first",
            "--class-count", "1"
        }, labelCount: 1);
        YoloRuntimeOutputSet outputs = new YoloRuntimeOutputSet(new[]
        {
            new YoloRuntimeOutputTensor("boxes", YoloOutputTensorRole.Detection, new[] { 0.5f }, new[] { 1 }),
            new YoloRuntimeOutputTensor("proto", YoloOutputTensorRole.MaskPrototypes, new[] { 1.0f, 1.0f, 1.0f, 1.0f }, new[] { 1, 2, 2 })
        });

        using JsonDocument document = JsonDocument.Parse(YoloVisionOutputReport.ToJson(
            new YoloVisionOutputReportContext(
                "model.onnx",
                string.Empty,
                string.Empty,
                "ramp",
                new[] { 1, 3, 4, 4 },
                tensorRtLine: 10,
                profileIndex: 0,
                engineDeviceMemoryBytes: 0,
                elapsedMilliseconds: 1.0,
                labelsPath: string.Empty,
                imagePreprocess: preprocess,
                segmentationSpatialTransform: spatialOptions),
            outputs,
            profile,
            result,
            new[] { "person" }));
        JsonElement spatial = document.RootElement.GetProperty("predictions")[0].GetProperty("spatialTransform");
        string svg = YoloVisionVisualizationWriter.ToSvg(
            result,
            new[] { "person" },
            profile,
            new[] { 1, 3, 4, 4 },
            preprocess,
            spatialOptions);

        Assert.True(spatial.GetProperty("applied").GetBoolean());
        Assert.Equal("model-input-pixels", spatial.GetProperty("coordinateSpace").GetString());
        Assert.Equal(16, spatial.GetProperty("finalMaskTotalPixelCount").GetInt32());
        Assert.Equal(4, spatial.GetProperty("finalMaskPixelCount").GetInt32());
        Assert.Equal("source-image-after-explicit-preprocess-inverse-and-optional-box-crop", spatial.GetProperty("finalMaskScope").GetString());
        Assert.Contains("data-spatial-mask-cell=\"true\"", svg, StringComparison.Ordinal);
        Assert.Contains("spatial mask: explicit preprocess inverse", svg, StringComparison.Ordinal);
    }

    private static YoloImagePreprocessResult CreatePreprocess(
        int sourceWidth,
        int sourceHeight,
        int targetWidth,
        int targetHeight,
        int resizedWidth,
        int resizedHeight,
        int padX,
        int padY,
        float scaleX,
        float scaleY)
    {
        return new YoloImagePreprocessResult(
            sourcePath: "source.ppm",
            sourceSha256: string.Empty,
            sourceWidth,
            sourceHeight,
            tensorPath: "tensor.bin",
            tensorSha256: string.Empty,
            tensorElementCount: checked(targetWidth * targetHeight * 3),
            targetWidth,
            targetHeight,
            tensorLayout: "NCHW",
            colorOrder: "RGB",
            resizeMode: padX == 0 && padY == 0 && resizedWidth == targetWidth && resizedHeight == targetHeight ? "stretch" : "letterbox",
            normalized: true,
            scale: 1.0f / 255.0f,
            letterboxEnabled: padX != 0 || padY != 0 || resizedWidth != targetWidth || resizedHeight != targetHeight,
            letterboxAlignment: "center",
            resizedWidth,
            resizedHeight,
            padX,
            padY,
            resizeScaleX: scaleX,
            resizeScaleY: scaleY,
            fillValue: 114);
    }

    private static float[] CreateInputValuesForTesting(
        int count,
        string inputPattern,
        string inputPath = "",
        string inputDataPath = "")
    {
        Type type = typeof(YoloModelProfile).Assembly.GetType("JYPPX.SampleSupport.TensorRtOnnxSample")
            ?? throw new InvalidOperationException("TensorRtOnnxSample helper type was not found.");
        System.Reflection.MethodInfo method = type.GetMethod(
            "CreateInputValuesForTesting",
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)
            ?? throw new InvalidOperationException("CreateInputValuesForTesting helper method was not found.");
        try
        {
            return (float[])method.Invoke(null, new object[] { count, inputPattern, inputPath, inputDataPath })!;
        }
        catch (TargetInvocationException exception) when (exception.InnerException != null)
        {
            throw exception.InnerException;
        }
    }
}
