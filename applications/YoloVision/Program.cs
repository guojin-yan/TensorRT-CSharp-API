using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using JYPPX.SampleSupport;
using JYPPX.TensorRtSharp;

namespace YoloVisionSample;

#if !JYPPX_PROJECT_QUALITY_SOURCE_HOST
internal static class Program
{
    public static int Main(string[] args)
    {
        return YoloVisionCommand.Run(args);
    }
}
#endif

/// <summary>
/// Runs the reusable, pointer-free YoloVision command-line pipeline.
/// 运行可复用、无指针暴露的 YoloVision 命令行流水线。
/// </summary>
public static class YoloVisionCommand
{
    /// <summary>
    /// Parses YoloVision arguments and executes capability, preflight, preprocessing, or TensorRT inference mode.
    /// 解析 YoloVision 参数，并执行能力查询、预检、预处理或 TensorRT 推理模式。
    /// </summary>
    /// <param name="args">Command-line arguments. 命令行参数。</param>
    /// <returns>Zero on success, or a nonzero argument/output validation code. 成功返回零，参数或输出校验失败时返回非零值。</returns>
    public static int Run(string[] args)
    {
        if (args == null)
        {
            throw new ArgumentNullException(nameof(args));
        }

        if (SampleCommandLine.HasSwitch(args, "--help"))
        {
            PrintUsage();
            return 0;
        }

        if (SampleCommandLine.HasSwitch(args, "--list-capabilities") || SampleCommandLine.HasSwitch(args, "--capabilities"))
        {
            if (SampleCommandLine.HasSwitch(args, "--json"))
            {
                Console.WriteLine(YoloCapabilityMatrix.FormatJson());
                return 0;
            }

            Console.WriteLine(YoloCapabilityMatrix.FormatConsoleTable());
            return 0;
        }

        if (SampleCommandLine.HasSwitch(args, "--self-test-end2end"))
        {
            return RunEndToEndManagedSmoke();
        }

        if (SampleCommandLine.HasSwitch(args, "--self-test-capabilities"))
        {
            return RunCapabilitySelfTest();
        }

        try
        {
            string labelsPath = ResolveOptionalFullPath(SampleCommandLine.GetStringArgument(args, "--labels", string.Empty));
            bool preflight = SampleCommandLine.HasSwitch(args, "--preflight") ||
                             SampleCommandLine.HasSwitch(args, "--dryRun") ||
                             SampleCommandLine.HasSwitch(args, "--previewOnly");
            if (preflight && SampleCommandLine.HasSwitch(args, "--preprocess-only"))
            {
                throw new ArgumentException("--preflight cannot be combined with --preprocess-only. Use --preprocess-only when you want to write a tensor.");
            }

            IReadOnlyList<string> labels = preflight
                ? ReadLabelsForPreflight(labelsPath)
                : TensorRtOnnxSample.ReadLabels(labelsPath);
            YoloModelProfile profile = YoloModelProfile.FromArgs(args, labels.Count);
            YoloSegmentationSpatialTransformOptions? segmentationSpatialTransform =
                YoloSegmentationSpatialTransformOptions.FromArgs(args, profile.TaskType);
            if (preflight)
            {
                string modelPath = ResolveOptionalFullPath(GetFirstStringArgument(args, string.Empty, "--model", "--onnx", "--onnxFile"));
                string inputPath = ResolveOptionalFullPath(SampleCommandLine.GetStringArgument(args, "--input", string.Empty));
                string inputDataPath = ResolveOptionalFullPath(SampleCommandLine.GetStringArgument(args, "--input-data", string.Empty));
                string imagePath = ResolveOptionalFullPath(SampleCommandLine.GetStringArgument(args, "--image", string.Empty));
                YoloMultiOutputMetadata? preflightMetadata = YoloRuntimeOutputRoleResolver.CreateMetadata(args, profile.TaskType);
                YoloVisionPreflightResult preflightResult = YoloVisionPreflightReport.Create(
                    args,
                    profile,
                    modelPath,
                    labelsPath,
                    inputPath,
                    inputDataPath,
                    imagePath,
                    labels,
                    preflightMetadata);
                string reportPath = ResolveOptionalFullPath(GetFirstStringArgument(args, string.Empty, "--preflight-report", "--preflight-output"));
                if (!string.IsNullOrWhiteSpace(reportPath))
                {
                    YoloVisionPreflightReport.Write(reportPath, preflightResult);
                    Console.WriteLine($"YoloVision PreflightReport={reportPath}");
                }
                else
                {
                    Console.WriteLine(preflightResult.Json);
                }

                Console.WriteLine($"YoloVision PreflightOnly=True State={preflightResult.State} OwnerAction={preflightResult.HasOwnerAction} NormalizedCommandSha256={preflightResult.NormalizedCommandSha256}");
                return preflightResult.HasBlockers ? 2 : 0;
            }

            if (SampleCommandLine.HasSwitch(args, "--preprocess-only"))
            {
                YoloImagePreprocessResult? preprocessOnlyResult = TryPreprocessImageInput(args, profile);
                if (preprocessOnlyResult == null)
                {
                    throw new ArgumentException("--preprocess-only requires --image <path>.");
                }

                PrintImagePreprocess(preprocessOnlyResult);
                Console.WriteLine("YoloVision PreprocessOnly=True");
                return 0;
            }

            YoloImagePreprocessResult? imagePreprocess = TryPreprocessImageInput(args, profile);
            if (segmentationSpatialTransform != null && imagePreprocess == null)
            {
                throw new ArgumentException("--mask-spatial-transform requires --image so the exact preprocessing metadata is available.");
            }

            string[] effectiveArgs = imagePreprocess == null ? args : AddOrReplaceArgument(args, "--input-data", imagePreprocess.TensorPath);
            OnnxSampleOptions options = OnnxSampleOptions.FromArgs(effectiveArgs, "1x3x640x640");

            OnnxSampleMultiOutputResult result = TensorRtOnnxSample.RunSingleFloatInputOutputs(options);
            OnnxSampleOutputTensor primaryOutput = result.PrimaryOutput;
            Console.WriteLine($"YoloVision TensorRtLine={(int)result.Line} Model={options.ModelPath}");
            Console.WriteLine($"Profile Family={profile.Family} Task={profile.TaskType} Layout={profile.Postprocess.Layout} Nms={profile.Postprocess.ApplyNms} NmsMode={profile.Postprocess.NmsMode} ClassificationScoreMode={profile.Postprocess.ClassificationScoreMode}");
            Console.WriteLine($"InputSource={(options.UsesExternalInput ? "external" : "synthetic")} InputFile={GetInputFileSummary(options)}");
            if (imagePreprocess != null)
            {
                PrintImagePreprocess(imagePreprocess);
            }

            Console.WriteLine($"Input={result.InputName}:{result.InputShape} Inputs={result.Inputs.Count} Output={primaryOutput.Name}:{primaryOutput.Shape} Outputs={result.Outputs.Count}");
            foreach (OnnxSampleInputTensor input in result.Inputs)
            {
                Console.WriteLine($"RuntimeInput Tensor={input.Name} Shape={input.Shape} Elements={input.ElementCount} Bytes={input.ByteLength} Source={input.SourceClassification} SourcePath={input.SourcePath} Sha256={input.Sha256}");
            }
            Console.WriteLine($"ProfileIndex={result.ProfileIndex} EngineDeviceMemory={result.EngineDeviceMemoryBytes}");
            PrintBindingReport(result.Report);
            Console.WriteLine($"Execution {result.ExecutionSummary} ElapsedMs={result.ElapsedMilliseconds:0.###}");
            PrintReferenceValidation(result.ReferenceValidation);

            YoloRuntimeOutputSet runtimeOutputs = new YoloRuntimeOutputSet(result.Outputs.Select(output =>
                new YoloRuntimeOutputTensor(
                    output.Name,
                    YoloRuntimeOutputRoleResolver.ResolveRole(output.Name, profile.TaskType, args, string.Equals(output.Name, result.PrimaryOutputName, StringComparison.Ordinal)),
                    output.Values,
                    output.Shape.Values)));
            YoloMultiOutputMetadata? metadata = YoloRuntimeOutputRoleResolver.CreateMetadata(args, profile.TaskType);
            YoloVisionResult visionResult = result.Outputs.Count == 1 && metadata == null
                ? YoloSampleRunner.DecodeOutput(primaryOutput.Values, primaryOutput.Shape.Values, profile)
                : YoloSampleRunner.DecodeRuntimeOutputs(runtimeOutputs, profile, metadata);
            Console.WriteLine($"Postprocess {visionResult}");
            PrintVisionResult(visionResult, labels, profile);

            string outputJsonPath = SampleCommandLine.GetStringArgument(
                args,
                "--output",
                SampleCommandLine.GetStringArgument(args, "--output-json", string.Empty));
            if (!string.IsNullOrWhiteSpace(outputJsonPath))
            {
                YoloVisionOutputReport.Write(
                    outputJsonPath,
                    options,
                    result,
                    runtimeOutputs,
                    profile,
                    visionResult,
                    labels,
                    labelsPath,
                    imagePreprocess,
                    segmentationSpatialTransform);
                Console.WriteLine($"OutputJson={Path.GetFullPath(outputJsonPath)}");
            }

            string segmentationMaskOutputDirectory = SampleCommandLine.GetStringArgument(
                args,
                "--segmentation-mask-output-directory",
                string.Empty);
            if (!string.IsNullOrWhiteSpace(segmentationMaskOutputDirectory))
            {
                string manifestPath = YoloSegmentationMaskArtifactWriter.Write(
                    segmentationMaskOutputDirectory,
                    visionResult,
                    labels,
                    imagePreprocess,
                    segmentationSpatialTransform);
                Console.WriteLine($"SegmentationMaskArtifacts={manifestPath}");
            }

            string semanticArtifactOutputDirectory = SampleCommandLine.GetStringArgument(
                args,
                "--semantic-artifact-output-directory",
                string.Empty);
            if (!string.IsNullOrWhiteSpace(semanticArtifactOutputDirectory))
            {
                string manifestPath = YoloSemanticMapArtifactWriter.Write(
                    semanticArtifactOutputDirectory,
                    visionResult,
                    labels);
                Console.WriteLine($"SemanticMapArtifacts={manifestPath}");
            }

            string visualizationPath = SampleCommandLine.GetStringArgument(
                args,
                "--visualization",
                SampleCommandLine.GetStringArgument(args, "--visualization-svg", string.Empty));
            string visualizationBackgroundPath = ResolveOptionalFullPath(
                SampleCommandLine.GetStringArgument(args, "--visualization-background", string.Empty));
            if (!string.IsNullOrWhiteSpace(visualizationBackgroundPath) && string.IsNullOrWhiteSpace(visualizationPath))
            {
                throw new ArgumentException("--visualization-background requires --visualization <path>.");
            }

            if (!string.IsNullOrWhiteSpace(visualizationPath))
            {
                if (string.IsNullOrWhiteSpace(visualizationBackgroundPath))
                {
                    YoloVisionVisualizationWriter.Write(
                        visualizationPath,
                        visionResult,
                        labels,
                        profile,
                        options.InputShape.Values,
                        imagePreprocess,
                        segmentationSpatialTransform);
                }
                else
                {
                    if (imagePreprocess == null)
                    {
                        throw new ArgumentException("--visualization-background requires --image so source-space coordinates are available.");
                    }

                    YoloVisionVisualizationWriter.Write(
                        visualizationPath,
                        visionResult,
                        labels,
                        profile,
                        options.InputShape.Values,
                        imagePreprocess,
                        segmentationSpatialTransform,
                        visualizationBackgroundPath);
                }

                Console.WriteLine($"Visualization={Path.GetFullPath(visualizationPath)}");
            }

            bool success = !result.ReferenceValidation.Requested ||
                (result.ReferenceValidation.Completed && result.ReferenceValidation.Passed);
            Console.WriteLine("OutputValidated=" + (result.ReferenceValidation.Requested && result.ReferenceValidation.Completed && result.ReferenceValidation.Passed));
            Console.WriteLine("YoloVision Passed=" + success);
            return success ? 0 : 1;
        }
        catch (SampleSkippedException exception)
        {
            Console.WriteLine($"YoloVision=Skipped Reason={exception.Message}");
            return 0;
        }
        catch (Exception exception) when (TensorRtSampleSupport.IsDeploymentException(exception))
        {
            Console.WriteLine($"YoloVision=Skipped Reason={exception.Message}");
            return 0;
        }
        catch (ArgumentException exception)
        {
            Console.WriteLine($"YoloVision=InvalidArguments Reason={exception.Message}");
            PrintUsage();
            return 2;
        }
        catch (FileNotFoundException exception)
        {
            Console.WriteLine($"YoloVision=InvalidArguments Reason={exception.Message}");
            PrintUsage();
            return 2;
        }
        catch (NotSupportedException exception)
        {
            Console.WriteLine($"YoloVision=UnsupportedOutput Reason={exception.Message}");
            return 2;
        }
        catch (Exception exception) when (exception is InvalidDataException || exception is JsonException)
        {
            Console.WriteLine($"YoloVision=InvalidArguments Reason={exception.Message}");
            return 2;
        }
    }

    private static void PrintReferenceValidation(OnnxSampleReferenceValidationResult validation)
    {
        if (!validation.Requested)
        {
            return;
        }

        Console.WriteLine($"ReferenceOutputValidation Requested=True Completed={validation.Completed} Passed={validation.Passed} Tensors={validation.TensorComparisons.Count} AbsTolerance={validation.AbsoluteTolerance:R} RelTolerance={validation.RelativeTolerance:R} NaNPolicy={validation.NaNPolicy} InfinityPolicy={validation.InfinityPolicy}");
        foreach (OnnxSampleReferenceTensorComparison comparison in validation.TensorComparisons)
        {
            Console.WriteLine($"ReferenceOutputTensor Tensor={comparison.TensorName} Passed={comparison.Passed} Compared={comparison.ComparedElementCount} Mismatches={comparison.MismatchCount} FirstMismatch={comparison.FirstMismatchIndex} MaxAbs={comparison.MaximumAbsoluteError:R} MaxRel={comparison.MaximumRelativeError:R} ReferenceSha256={comparison.ReferenceSha256} Source={comparison.SourceClassification} Diagnostic={comparison.Diagnostic}");
        }
    }

    private static void PrintVisionResult(YoloVisionResult result, IReadOnlyList<string> labels, YoloModelProfile profile)
    {
        if (result.HasDiagnostic)
        {
            Console.WriteLine($"PostprocessDiagnostic={result.Diagnostic}");
        }

        if (result.TaskType == YoloTaskType.Classification)
        {
            if (result.Classifications.Count == 0)
            {
                Console.WriteLine($"Classifications=0 Confidence={profile.Postprocess.ConfidenceThreshold:0.###}");
                return;
            }

            foreach (YoloClassificationPrediction prediction in result.Classifications)
            {
                Console.WriteLine($"Classification Class={TensorRtOnnxSample.LabelOrIndex(labels, prediction.ClassIndex)} Score={prediction.Score:0.######}");
            }

            return;
        }

        if (result.TaskType == YoloTaskType.SemanticSegmentation && result.SemanticMap != null)
        {
            Console.WriteLine($"SemanticMap Classes={result.SemanticMap.ClassCount} Width={result.SemanticMap.Width} Height={result.SemanticMap.Height} Values={result.SemanticMap.Values.Length}");
            return;
        }

        if (result.TaskType == YoloTaskType.Segmentation)
        {
            Console.WriteLine($"Segmentations={result.Segmentations.Count}");
            foreach (YoloSegmentationPrediction segmentation in result.Segmentations)
            {
                Console.WriteLine(
                    $"Segmentation Class={TensorRtOnnxSample.LabelOrIndex(labels, segmentation.Detection.ClassIndex)} " +
                    $"Score={segmentation.Detection.Score:0.######} Mask={segmentation.Mask.Width}x{segmentation.Mask.Height}");
            }

            return;
        }

        if (result.TaskType == YoloTaskType.OrientedBoundingBox)
        {
            Console.WriteLine($"OrientedBoxes={result.OrientedBoxes.Count}");
            foreach (YoloObbDetection orientedBox in result.OrientedBoxes)
            {
                Console.WriteLine(
                    $"Obb Class={TensorRtOnnxSample.LabelOrIndex(labels, orientedBox.Box.ClassIndex)} " +
                    $"Score={orientedBox.Box.Score:0.######} AngleRadians={orientedBox.AngleRadians:0.######}");
            }

            return;
        }

        if (result.TaskType == YoloTaskType.Pose)
        {
            Console.WriteLine($"Poses={result.Poses.Count}");
            foreach (YoloPosePrediction pose in result.Poses)
            {
                Console.WriteLine(
                    $"Pose Class={TensorRtOnnxSample.LabelOrIndex(labels, pose.Detection.ClassIndex)} " +
                    $"Score={pose.Detection.Score:0.######} Keypoints={pose.Keypoints.Length}");
            }

            return;
        }

        if (result.Detections.Count == 0)
        {
            Console.WriteLine($"Detections=0 Confidence={profile.Postprocess.ConfidenceThreshold:0.###} Note=If InputSource=synthetic, this is pipeline evidence only. Use --input-data with a preprocessed tensor for real image evidence.");
            return;
        }

        foreach (YoloDetection detection in result.Detections)
        {
            Console.WriteLine(
                $"Detection Class={TensorRtOnnxSample.LabelOrIndex(labels, detection.ClassIndex)} Score={detection.Score:0.######} " +
                $"BoxCxCyWh={detection.CenterX:0.###},{detection.CenterY:0.###},{detection.Width:0.###},{detection.Height:0.###}");
        }
    }

    private static int RunEndToEndManagedSmoke()
    {
        YoloPostprocessOptions options = new YoloPostprocessOptions(
            YoloOutputLayout.EndToEndNms,
            hasObjectness: true,
            classCount: 3,
            confidenceThreshold: 0.25f,
            iouThreshold: 0.45f,
            topK: 10,
            applyNms: true);
        float[] values =
        {
            10.0f, 20.0f, 30.0f, 40.0f, 0.90f, 2.0f,
            10.5f, 20.5f, 30.5f, 40.5f, 0.80f, 2.0f,
            0.0f, 0.0f, 5.0f, 5.0f, 0.20f, 1.0f
        };

        IReadOnlyList<YoloDetection> detections = YoloDetectionDecoder.Decode(values, new[] { 1, 3, 6 }, options);
        bool passed = detections.Count == 2 &&
                      detections[0].ClassIndex == 2 &&
                      Math.Abs(detections[0].Score - 0.90f) < 0.00001f &&
                      detections[0].SourceIndex == 0 &&
                      detections[1].SourceIndex == 1 &&
                      !options.ApplyNms &&
                      options.NmsMode == YoloNmsMode.None;
        Console.WriteLine(
            $"YoloVision ManagedSmoke=YOLOv10EndToEnd Passed={passed} Detections={detections.Count} " +
            $"Layout={options.Layout} ApplyNms={options.ApplyNms} NmsMode={options.NmsMode}");
        Console.WriteLine(
            "YoloVision ManagedSmokeBoundary=managed-array-decode-only IsRuntimeProof=False " +
            "IsRealModelRuntimeProof=False IsPackageConsumerRuntimeProof=False");
        return passed ? 0 : 2;
    }

    private static int RunCapabilitySelfTest()
    {
        string json = YoloCapabilityMatrix.FormatJson();
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        JsonElement entries = root.GetProperty("entries");
        bool passed =
            root.GetProperty("matrixId").GetString() == "yolovision-capability-matrix" &&
            root.GetProperty("entryCount").GetInt32() == YoloCapabilityMatrix.Entries.Count &&
            root.GetProperty("proofBoundary").GetString()!.Contains("not runtime proof", StringComparison.Ordinal) &&
            entries.GetArrayLength() == 60 &&
            YoloCapabilityMatrix.Entries.Count(static entry => entry.Supported) == 55 &&
            YoloCapabilityMatrix.Entries.Any(static entry =>
                string.Equals(entry.FamilyAlias, "yolox", StringComparison.Ordinal) &&
                string.Equals(entry.TaskAlias, "det", StringComparison.Ordinal) &&
                entry.Supported) &&
            YoloCapabilityMatrix.Entries.Count(static entry =>
                string.Equals(entry.FamilyAlias, "yolox", StringComparison.Ordinal) &&
                !entry.Supported) == 5 &&
            YoloCapabilityMatrix.FormatConsoleTable().Contains("YoloVision Capability Matrix", StringComparison.Ordinal);

        Console.WriteLine(
            $"YoloVision CapabilitySelfTest Passed={passed} Entries={YoloCapabilityMatrix.Entries.Count} " +
            $"Supported={YoloCapabilityMatrix.Entries.Count(static entry => entry.Supported)} Unsupported={YoloCapabilityMatrix.Entries.Count(static entry => !entry.Supported)}");
        Console.WriteLine(
            "YoloVision CapabilitySelfTestBoundary=offline-matrix-contract-only IsRuntimeProof=False " +
            "IsRealModelRuntimeProof=False IsPackageConsumerRuntimeProof=False");
        return passed ? 0 : 2;
    }

    private static void PrintBindingReport(TensorRtEngineBindingReport report)
    {
        Console.WriteLine(
            $"BindingReport Ready={report.IsReadyForEnqueue} Profile={report.ProfileIndex} " +
            $"Tensors={report.Tensors.Count} Inputs={report.GetInputs().Count} Outputs={report.GetOutputs().Count}");
        foreach (TensorRtEngineTensorBinding binding in report.Tensors)
        {
            Console.WriteLine(
                $"BindingMetadata Index={binding.Index} Name={binding.Name} Mode={binding.IOMode} " +
                $"DataType={binding.DataType} Shape={binding.EngineShape} Location={binding.Location} " +
                $"Format={binding.Format} VectorizedDimension={binding.VectorizedDimension} " +
                $"Profile={binding.ProfileIndex}");
        }
    }

    private static string GetInputFileSummary(OnnxSampleOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.InputDataPath))
        {
            return options.InputDataPath;
        }

        if (!string.IsNullOrWhiteSpace(options.InputPath))
        {
            return options.InputPath;
        }

        return options.InputPattern;
    }

    private static string ResolveOptionalFullPath(string path)
    {
        return string.IsNullOrWhiteSpace(path) ? string.Empty : Path.GetFullPath(path);
    }

    private static IReadOnlyList<string> ReadLabelsForPreflight(string labelsPath)
    {
        return string.IsNullOrWhiteSpace(labelsPath) || !File.Exists(labelsPath)
            ? Array.Empty<string>()
            : TensorRtOnnxSample.ReadLabels(labelsPath);
    }

    private static string GetFirstStringArgument(string[] args, string defaultValue, params string[] names)
    {
        foreach (string name in names)
        {
            string value = SampleCommandLine.GetStringArgument(args, name, string.Empty);
            if (!string.IsNullOrWhiteSpace(value))
            {
                return value;
            }
        }

        return defaultValue;
    }

    private static void PrintImagePreprocess(YoloImagePreprocessResult result)
    {
        Console.WriteLine(
            $"ImagePreprocess Source={result.SourcePath} SourceSha256={result.SourceSha256} " +
            $"SourceSize={result.SourceWidth}x{result.SourceHeight} Tensor={result.TensorPath} " +
            $"TensorSha256={result.TensorSha256} TensorElements={result.TensorElementCount}");
        Console.WriteLine(
            $"ImagePreprocessConfig Mode={result.ResizeMode} Layout={result.TensorLayout} Color={result.ColorOrder} " +
            $"Target={result.TargetWidth}x{result.TargetHeight} Resized={result.ResizedWidth}x{result.ResizedHeight} " +
            $"Pad={result.PadX},{result.PadY} Crop={result.CropX},{result.CropY} ShorterSide={result.ResizeShorterSide} " +
            $"Alignment={result.LetterboxAlignment} Scale={result.ResizeScaleX:0.######},{result.ResizeScaleY:0.######} " +
            $"Normalize={result.Normalized} ValueScale={result.Scale:0.########} Fill={result.FillValue}");
        Console.WriteLine(
            $"ImagePreprocessNormalization Mean={FormatFloatTriplet(result.Mean)} Std={FormatFloatTriplet(result.StandardDeviation)} " +
            $"ContractSha256={result.PreprocessContractSha256}");
    }

    private static YoloImagePreprocessResult? TryPreprocessImageInput(string[] args, YoloModelProfile profile)
    {
        string imagePath = SampleCommandLine.GetStringArgument(
            args,
            "--image",
            SampleCommandLine.GetStringArgument(args, "--input-image", string.Empty));
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            return null;
        }

        if (!string.IsNullOrWhiteSpace(SampleCommandLine.GetStringArgument(args, "--input", string.Empty)) ||
            !string.IsNullOrWhiteSpace(SampleCommandLine.GetStringArgument(args, "--input-data", string.Empty)))
        {
            throw new ArgumentException("--image cannot be combined with --input or --input-data. Use one input source per run.");
        }

        string outputPath = SampleCommandLine.GetStringArgument(
            args,
            "--preprocessed-output",
            SampleCommandLine.GetStringArgument(args, "--preprocessed-tensor-output", string.Empty));
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            outputPath = CreateDefaultPreprocessedTensorPath(imagePath, profile);
        }

        return YoloImagePreprocessor.Preprocess(imagePath, outputPath, profile.InputShape, profile.Preprocess);
    }

    private static string CreateDefaultPreprocessedTensorPath(string imagePath, YoloModelProfile profile)
    {
        string imageName = Path.GetFileNameWithoutExtension(imagePath);
        foreach (char invalid in Path.GetInvalidFileNameChars())
        {
            imageName = imageName.Replace(invalid, '_');
        }

        if (string.IsNullOrWhiteSpace(imageName))
        {
            imageName = "image";
        }

        ResolveProfileInputSize(profile, out int width, out int height);
        string fileName = $"{imageName}-{profile.TaskType}-{height}x{width}-{profile.Preprocess.TensorLayout}-{profile.Preprocess.ColorOrder}.fp32.bin";
        return Path.Combine("artifacts", "yolovision", "preprocessed", fileName);
    }

    private static void ResolveProfileInputSize(YoloModelProfile profile, out int width, out int height)
    {
        if (profile.InputShape.Length < 4)
        {
            width = 1;
            height = 1;
            return;
        }

        string normalizedLayout = (profile.Preprocess.TensorLayout ?? string.Empty)
            .Replace("-", string.Empty, StringComparison.Ordinal)
            .Replace("_", string.Empty, StringComparison.Ordinal)
            .ToUpperInvariant();
        if (string.Equals(normalizedLayout, "NHWC", StringComparison.Ordinal) ||
            string.Equals(normalizedLayout, "CHANNELSLAST", StringComparison.Ordinal))
        {
            width = profile.InputShape[2];
            height = profile.InputShape[1];
            return;
        }

        width = profile.InputShape[3];
        height = profile.InputShape[2];
    }

    private static string[] AddOrReplaceArgument(string[] args, string name, string value)
    {
        string[] copy = (string[])args.Clone();
        for (int index = 0; index < copy.Length - 1; index++)
        {
            if (string.Equals(copy[index], name, StringComparison.OrdinalIgnoreCase))
            {
                copy[index + 1] = value;
                return copy;
            }
        }

        string[] expanded = new string[copy.Length + 2];
        Array.Copy(copy, expanded, copy.Length);
        expanded[^2] = name;
        expanded[^1] = value;
        return expanded;
    }

    private static string FormatFloatTriplet(float[] values)
    {
        return string.Join(",", values.Select(value => value.ToString("R", CultureInfo.InvariantCulture)));
    }

    private static void PrintUsage()
    {
        Console.WriteLine("YoloVision sample");
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project applications/YoloVision -- --model yolo.onnx --labels labels.txt --input-shape 1x3x640x640 --tensor-rt-line 10");
        Console.WriteLine("Options:");
        Console.WriteLine("  --list-capabilities      Print the offline YOLO family/task capability matrix without TensorRT runtime or model assets.");
        Console.WriteLine("  --list-capabilities --json  Print the same capability matrix as machine-readable JSON.");
        Console.WriteLine("  --self-test-end2end     Run the deterministic YOLOv10 six-column managed decoder smoke without CUDA/TensorRT.");
        Console.WriteLine("  --self-test-capabilities  Validate the offline capability matrix JSON/table contract without CUDA/TensorRT.");
        Console.WriteLine("  --family custom|v5|v6|v7|v8|v9|v10|v11|v26|yolox");
        Console.WriteLine("  --task det|cls|seg|obb|pose|sem");
        Console.WriteLine("  --layout auto|channels-first|boxes-first|end2end");
        Console.WriteLine("  --has-objectness auto|true|false");
        Console.WriteLine("  --class-count <count>     Defaults to labels count when labels are provided.");
        Console.WriteLine("  --confidence <value>      Default: 0 for classification; 0.25 for other tasks.");
        Console.WriteLine("  --iou-threshold <value>   Default: 0.45");
        Console.WriteLine("  --top-k <count>           Default: 10");
        Console.WriteLine("  --classification-score-mode raw|logits|probabilities  Raw preserves legacy scores; logits applies stable softmax; probabilities validates [0,1] and sum=1.");
        Console.WriteLine("  --nms-mode class-aware|class-agnostic|none");
        Console.WriteLine("  --no-nms                  Keep score filtering only.");
        Console.WriteLine("  --tensor-layout NCHW|NHWC --color-order RGB|BGR --resize letterbox|stretch|shorter-side-center-crop");
        Console.WriteLine("  --resize-shorter-side <n> Classification defaults to the shorter input side (224 for the default profile).");
        Console.WriteLine("  --letterbox-alignment center|top-left  YOLOX defaults to top-left; other families default to center.");
        Console.WriteLine("  --normalize|--no-normalize  YOLOX defaults to raw 0..255 values; other families default to 1/255 normalization.");
        Console.WriteLine("  --output-role-map <map>   Example: boxes:det,proto:mask-prototypes,kpts:pose-keypoints,angle:obb-angles.");
        Console.WriteLine("  --mask-prototypes-output <name>  Segmentation prototype tensor name.");
        Console.WriteLine("  --mask-coefficient-count <count> Segmentation mask coefficient count.");
        Console.WriteLine("  --mask-threshold <value>         Segmentation probability threshold in [0,1]. Default: 0.5.");
        Console.WriteLine("  --mask-spatial-transform        Opt in to explicit prototype-to-source-image mask mapping; requires --image.");
        Console.WriteLine("  --mask-coordinate-space model-input|normalized  Required with --mask-spatial-transform.");
        Console.WriteLine("  --mask-crop-to-box true|false   Crop the transformed mask to its detection box. Default: true.");
        Console.WriteLine("  --segmentation-mask-output-directory <path>  Write hashed prototype/source probability masks, thresholded u8 masks, and a manifest.");
        Console.WriteLine("  --semantic-artifact-output-directory <path>  Write the full-resolution int32 class-index map, SHA256, histogram, and manifest.");
        Console.WriteLine("  --pose-keypoints-output <name>   Pose keypoint tensor name.");
        Console.WriteLine("  --keypoint-count <count>         Pose keypoint count; --keypoint-stride defaults to 3.");
        Console.WriteLine("  --obb-angle-output <name>        Optional separate OBB angle tensor; embedded angle uses --aux-channel-start.");
        Console.WriteLine("  --angle-degrees|--angle-radians  OBB angle unit; official YOLOv8 OBB exports use radians.");
        Console.WriteLine("  --aux-channel-start <index>      Optional channel start for auxiliary data embedded in detection rows.");
        Console.WriteLine("  --aux-layout auto|channels-first|boxes-first");
        Console.WriteLine("  --output <path>        Write a YoloVision output JSON report for owner/golden-output review.");
        Console.WriteLine("  --output-json <path>   Alias for --output.");
        Console.WriteLine("  --visualization <path> Write an SVG visualization for detections, classification, segmentation, OBB, pose, or semantic maps.");
        Console.WriteLine("  --visualization-svg <path> Alias for --visualization.");
        Console.WriteLine("  --visualization-background <path> Embed a same-size JPEG/PNG/BMP source image and map predictions back to source coordinates; requires --image and --visualization.");
        Console.WriteLine("  --input-pattern <pattern> zeros, ones, or ramp. Default: ramp.");
        Console.WriteLine("  --input <path>           Raw byte tensor normalized to [0,1]; byte count must match input element count.");
        Console.WriteLine("  --input-data <path>      Float tensor data from .bin/.raw float32 or comma/space/newline text.");
        Console.WriteLine("  --input-shapes <map>     Named multi-input shapes, for example left:1x4,right:1x4.");
        Console.WriteLine("  --min-shapes/--opt-shapes/--max-shapes <map>  Complete named dynamic profile; all three maps are required together.");
        Console.WriteLine("  --load-inputs <map>      Named float32/text input files; every model input needs exactly one source.");
        Console.WriteLine("  --load-byte-inputs <map> Named raw-byte input files normalized to [0,1].");
        Console.WriteLine("  --input-patterns <map>   Named synthetic sources using zeros, ones, or ramp.");
        Console.WriteLine("  --reference-outputs <map>  Structured schemaVersion=1 output references using tensor:path mappings.");
        Console.WriteLine("  --reference-abs-tolerance/--reference-rel-tolerance <n>  Finite non-negative comparison tolerances.");
        Console.WriteLine("  --reference-nan-policy <reject|equal> --reference-infinity-policy <exact|reject>");
        Console.WriteLine("  --noTF32                  Disable TensorRT TF32 tactics when strict FP32 parity is required.");
        Console.WriteLine("  --image <path>           Decode JPEG/PNG/BMP/PPM, preprocess to fp32, and feed it as --input-data.");
        Console.WriteLine("  --input-image <path>     Alias for --image.");
        Console.WriteLine("  --preprocessed-output <path>  Optional fp32 tensor path written by --image preprocessing.");
        Console.WriteLine("  --mean <r,g,b> --std <r,g,b>  Per-channel normalization after --scale; defaults to 0,0,0 and 1,1,1.");
        Console.WriteLine("  --preprocess-only        Write the preprocessed tensor and hashes without requiring an ONNX model or TensorRT runtime.");
        Console.WriteLine("  --preflight|--dryRun|--previewOnly  Validate profile, asset paths/hashes, and output metadata without TensorRT or ONNX execution.");
        Console.WriteLine("  --preflight-report <path>  Write a yolovision-preflight.v1 JSON report; --strict-preflight returns exit code 2 for missing owner assets.");
    }
}
