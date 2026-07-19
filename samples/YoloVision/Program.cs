using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using JYPPX.SampleSupport;
using JYPPX.TensorRtSharp;

namespace YoloVisionSample;

internal static class Program
{
    public static int Main(string[] args)
    {
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
            string[] effectiveArgs = imagePreprocess == null ? args : AddOrReplaceArgument(args, "--input-data", imagePreprocess.TensorPath);
            OnnxSampleOptions options = OnnxSampleOptions.FromArgs(effectiveArgs, "1x3x640x640");

            OnnxSampleMultiOutputResult result = TensorRtOnnxSample.RunSingleFloatInputOutputs(options);
            OnnxSampleOutputTensor primaryOutput = result.PrimaryOutput;
            Console.WriteLine($"YoloVision TensorRtLine={(int)result.Line} Model={options.ModelPath}");
            Console.WriteLine($"Profile Family={profile.Family} Task={profile.TaskType} Layout={profile.Postprocess.Layout} Nms={profile.Postprocess.ApplyNms} NmsMode={profile.Postprocess.NmsMode}");
            Console.WriteLine($"InputSource={(options.UsesExternalInput ? "external" : "synthetic")} InputFile={GetInputFileSummary(options)}");
            if (imagePreprocess != null)
            {
                PrintImagePreprocess(imagePreprocess);
            }

            Console.WriteLine($"Input={result.InputName}:{result.InputShape} Output={primaryOutput.Name}:{primaryOutput.Shape} Outputs={result.Outputs.Count}");
            Console.WriteLine($"ProfileIndex={result.ProfileIndex} EngineDeviceMemory={result.EngineDeviceMemoryBytes}");
            PrintBindingReport(result.Report);
            Console.WriteLine($"Execution {result.ExecutionSummary} ElapsedMs={result.ElapsedMilliseconds:0.###}");

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
                YoloVisionOutputReport.Write(outputJsonPath, options, result, runtimeOutputs, profile, visionResult, labels, labelsPath, imagePreprocess);
                Console.WriteLine($"OutputJson={Path.GetFullPath(outputJsonPath)}");
            }

            string visualizationPath = SampleCommandLine.GetStringArgument(
                args,
                "--visualization",
                SampleCommandLine.GetStringArgument(args, "--visualization-svg", string.Empty));
            if (!string.IsNullOrWhiteSpace(visualizationPath))
            {
                YoloVisionVisualizationWriter.Write(visualizationPath, visionResult, labels, profile, options.InputShape.Values);
                Console.WriteLine($"Visualization={Path.GetFullPath(visualizationPath)}");
            }

            Console.WriteLine("YoloVision Passed=True");
            return 0;
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
            $"Pad={result.PadX},{result.PadY} Scale={result.ResizeScaleX:0.######},{result.ResizeScaleY:0.######} " +
            $"Normalize={result.Normalized} ValueScale={result.Scale:0.########} Fill={result.FillValue}");
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

    private static void PrintUsage()
    {
        Console.WriteLine("YoloVision sample");
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project samples/YoloVision -- --model yolo.onnx --labels labels.txt --input-shape 1x3x640x640 --tensor-rt-line 10");
        Console.WriteLine("Options:");
        Console.WriteLine("  --list-capabilities      Print the offline YOLO family/task capability matrix without TensorRT runtime or model assets.");
        Console.WriteLine("  --list-capabilities --json  Print the same capability matrix as machine-readable JSON.");
        Console.WriteLine("  --family custom|v5|v6|v7|v8|v9|v10|v11|v26");
        Console.WriteLine("  --task det|cls|seg|obb|pose|sem");
        Console.WriteLine("  --layout auto|channels-first|boxes-first");
        Console.WriteLine("  --has-objectness auto|true|false");
        Console.WriteLine("  --class-count <count>     Defaults to labels count when labels are provided.");
        Console.WriteLine("  --confidence <value>      Default: 0.25");
        Console.WriteLine("  --iou-threshold <value>   Default: 0.45");
        Console.WriteLine("  --top-k <count>           Default: 10");
        Console.WriteLine("  --nms-mode class-aware|class-agnostic|none");
        Console.WriteLine("  --no-nms                  Keep score filtering only.");
        Console.WriteLine("  --output-role-map <map>   Example: boxes:det,proto:mask-prototypes,kpts:pose-keypoints,angle:obb-angles.");
        Console.WriteLine("  --mask-prototypes-output <name>  Segmentation prototype tensor name.");
        Console.WriteLine("  --mask-coefficient-count <count> Segmentation mask coefficient count.");
        Console.WriteLine("  --pose-keypoints-output <name>   Pose keypoint tensor name.");
        Console.WriteLine("  --keypoint-count <count>         Pose keypoint count; --keypoint-stride defaults to 3.");
        Console.WriteLine("  --obb-angle-output <name>        OBB angle tensor name; --angle-degrees or --angle-radians controls units.");
        Console.WriteLine("  --aux-channel-start <index>      Optional channel start for auxiliary data embedded in detection rows.");
        Console.WriteLine("  --aux-layout auto|channels-first|boxes-first");
        Console.WriteLine("  --output <path>        Write a YoloVision output JSON report for owner/golden-output review.");
        Console.WriteLine("  --output-json <path>   Alias for --output.");
        Console.WriteLine("  --visualization <path> Write an SVG visualization for detections, classification, segmentation, OBB, pose, or semantic maps.");
        Console.WriteLine("  --visualization-svg <path> Alias for --visualization.");
        Console.WriteLine("  --input-pattern <pattern> zeros, ones, or ramp. Default: ramp.");
        Console.WriteLine("  --input <path>           Raw byte tensor normalized to [0,1]; byte count must match input element count.");
        Console.WriteLine("  --input-data <path>      Float tensor data from .bin/.raw float32 or comma/space/newline text.");
        Console.WriteLine("  --image <path>           Decode .bmp/.ppm image, preprocess to fp32, and feed it as --input-data.");
        Console.WriteLine("  --input-image <path>     Alias for --image.");
        Console.WriteLine("  --preprocessed-output <path>  Optional fp32 tensor path written by --image preprocessing.");
        Console.WriteLine("  --preprocess-only        Write the preprocessed tensor and hashes without requiring an ONNX model or TensorRT runtime.");
        Console.WriteLine("  --preflight|--dryRun|--previewOnly  Validate profile, asset paths/hashes, and output metadata without TensorRT or ONNX execution.");
        Console.WriteLine("  --preflight-report <path>  Write a yolovision-preflight.v1 JSON report; --strict-preflight returns exit code 2 for missing owner assets.");
    }
}
