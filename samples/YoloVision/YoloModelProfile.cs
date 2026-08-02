using System;
using System.Globalization;

namespace YoloVisionSample;

public sealed class YoloModelProfile
{
    public YoloModelProfile(
        YoloModelFamily family,
        YoloTaskType taskType,
        string inputName,
        string outputName,
        int[] inputShape,
        YoloPreprocessOptions preprocess,
        YoloPostprocessOptions postprocess)
    {
        Family = family;
        TaskType = taskType;
        InputName = inputName ?? string.Empty;
        OutputName = outputName ?? string.Empty;
        InputShape = inputShape ?? throw new ArgumentNullException(nameof(inputShape));
        Preprocess = preprocess ?? throw new ArgumentNullException(nameof(preprocess));
        Postprocess = postprocess ?? throw new ArgumentNullException(nameof(postprocess));
    }

    public YoloModelFamily Family { get; }

    public YoloTaskType TaskType { get; }

    public string InputName { get; }

    public string OutputName { get; }

    public int[] InputShape { get; }

    public YoloPreprocessOptions Preprocess { get; }

    public YoloPostprocessOptions Postprocess { get; }

    public static YoloModelProfile FromArgs(string[] args, int labelCount)
    {
        if (args == null)
        {
            throw new ArgumentNullException(nameof(args));
        }

        YoloModelFamily family = ParseFamily(GetStringArgument(args, "--family", "custom"));
        YoloTaskType taskType = ParseTask(GetStringArgument(args, "--task", "det"));
        if (family == YoloModelFamily.YoloX && taskType != YoloTaskType.Detection)
        {
            throw new NotSupportedException("The built-in YOLOX profile supports detection models only.");
        }

        string tensorLayout = GetStringArgument(args, "--tensor-layout", "NCHW");
        string defaultInputShape = taskType == YoloTaskType.Classification ? "1x3x224x224" : "1x3x640x640";
        int[] inputShape = ParseShape(GetStringArgument(args, "--input-shape", defaultInputShape), "--input-shape");
        YoloOutputLayout layout = YoloOutputLayoutInference.Parse(GetStringArgument(args, "--layout", "auto"));
        bool? hasObjectness = ParseOptionalBoolean(GetStringArgument(args, "--has-objectness", "auto"));
        int classCount = GetPositiveIntArgument(args, "--class-count", Math.Max(0, labelCount));
        float confidence = GetFloatArgument(
            args,
            "--confidence",
            taskType == YoloTaskType.Classification ? 0.0f : YoloPostprocessOptions.Default.ConfidenceThreshold);
        float iouThreshold = GetFloatArgument(args, "--iou-threshold", YoloPostprocessOptions.Default.IouThreshold);
        int topK = GetPositiveIntArgument(args, "--top-k", 10);
        YoloNmsMode nmsMode = ParseNmsMode(GetStringArgument(args, "--nms-mode", "class-aware"));
        YoloClassificationScoreMode classificationScoreMode = ParseClassificationScoreMode(
            GetStringArgument(args, "--classification-score-mode", "raw"));
        bool applyNms = !HasSwitch(args, "--no-nms") && nmsMode != YoloNmsMode.None;
        if (taskType is YoloTaskType.Classification or YoloTaskType.SemanticSegmentation)
        {
            applyNms = false;
            nmsMode = YoloNmsMode.None;
        }
        bool yoloX = family == YoloModelFamily.YoloX;
        if (HasSwitch(args, "--normalize") && HasSwitch(args, "--no-normalize"))
        {
            throw new ArgumentException("--normalize and --no-normalize cannot be used together.");
        }

        bool normalize = yoloX
            ? HasSwitch(args, "--normalize")
            : !HasSwitch(args, "--no-normalize");
        string defaultResizeMode = taskType == YoloTaskType.Classification
            ? "shorter-side-center-crop"
            : "letterbox";
        int resizeShorterSide = GetPositiveIntArgument(
            args,
            "--resize-shorter-side",
            GetInputShorterSide(inputShape, tensorLayout));

        YoloPreprocessOptions preprocess = new YoloPreprocessOptions(
            tensorLayout,
            GetStringArgument(args, "--color-order", yoloX ? "BGR" : "RGB"),
            GetStringArgument(args, "--resize", defaultResizeMode),
            GetFloatArgument(args, "--scale", yoloX ? 1.0f : 1.0f / 255.0f),
            normalize,
            preserveAspectRatio: !HasSwitch(args, "--stretch"),
            GetStringArgument(args, "--letterbox-alignment", yoloX ? "top-left" : "center"),
            resizeShorterSide);

        YoloPostprocessOptions postprocess = new YoloPostprocessOptions(
            layout,
            hasObjectness,
            classCount,
            confidence,
            iouThreshold,
            topK,
            applyNms,
            nmsMode,
            classificationScoreMode);

        return new YoloModelProfile(
            family,
            taskType,
            GetStringArgument(args, "--input-name", string.Empty),
            GetStringArgument(args, "--output-name", string.Empty),
            inputShape,
            preprocess,
            postprocess);
    }

    private static int GetInputShorterSide(int[] inputShape, string tensorLayout)
    {
        if (inputShape.Length != 4)
        {
            return 224;
        }

        string normalizedLayout = Normalize(tensorLayout);
        int height = normalizedLayout is "nhwc" or "channelslast" ? inputShape[1] : inputShape[2];
        int width = normalizedLayout is "nhwc" or "channelslast" ? inputShape[2] : inputShape[3];
        return Math.Min(height, width);
    }

    private static YoloModelFamily ParseFamily(string value)
    {
        string normalized = Normalize(value);
        return normalized switch
        {
            "yolov5" or "v5" or "5" => YoloModelFamily.YoloV5,
            "yolov6" or "v6" or "6" => YoloModelFamily.YoloV6,
            "yolov7" or "v7" or "7" => YoloModelFamily.YoloV7,
            "yolov8" or "v8" or "8" => YoloModelFamily.YoloV8,
            "yolov9" or "v9" or "9" => YoloModelFamily.YoloV9,
            "yolov10" or "v10" or "10" => YoloModelFamily.YoloV10,
            "yolov11" or "v11" or "11" => YoloModelFamily.YoloV11,
            "yolov26" or "v26" or "26" => YoloModelFamily.YoloV26,
            "yolox" or "x" => YoloModelFamily.YoloX,
            _ => YoloModelFamily.Custom
        };
    }

    private static YoloTaskType ParseTask(string value)
    {
        string normalized = Normalize(value);
        return normalized switch
        {
            "det" or "detect" or "detection" => YoloTaskType.Detection,
            "cls" or "classify" or "classification" => YoloTaskType.Classification,
            "seg" or "segment" or "segmentation" => YoloTaskType.Segmentation,
            "obb" or "orientedbbox" or "orientedboundingbox" => YoloTaskType.OrientedBoundingBox,
            "pose" or "keypoint" or "keypoints" => YoloTaskType.Pose,
            "sem" or "semantic" or "semanticsegmentation" => YoloTaskType.SemanticSegmentation,
            _ => throw new ArgumentException($"Unsupported YOLO task '{value}'.")
        };
    }

    private static YoloNmsMode ParseNmsMode(string value)
    {
        string normalized = Normalize(value);
        return normalized switch
        {
            "" or "auto" or "classaware" or "aware" => YoloNmsMode.ClassAware,
            "classagnostic" or "agnostic" or "global" => YoloNmsMode.ClassAgnostic,
            "none" or "off" or "disabled" or "disable" => YoloNmsMode.None,
            _ => throw new ArgumentException($"Unsupported YOLO NMS mode '{value}'.")
        };
    }

    private static YoloClassificationScoreMode ParseClassificationScoreMode(string value)
    {
        string normalized = Normalize(value);
        return normalized switch
        {
            "" or "raw" => YoloClassificationScoreMode.Raw,
            "logit" or "logits" or "softmax" => YoloClassificationScoreMode.Logits,
            "probability" or "probabilities" or "prob" => YoloClassificationScoreMode.Probabilities,
            _ => throw new ArgumentException($"Unsupported classification score mode '{value}'. Use raw, logits, or probabilities.")
        };
    }

    private static bool? ParseOptionalBoolean(string value)
    {
        if (string.Equals(value, "auto", StringComparison.OrdinalIgnoreCase) || string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        if (bool.TryParse(value, out bool parsed))
        {
            return parsed;
        }

        throw new ArgumentException($"Expected true, false, or auto, got '{value}'.");
    }

    private static int[] ParseShape(string value, string argumentName)
    {
        string[] tokens = value.Split(new[] { 'x', 'X', ',', ';' }, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length == 0)
        {
            throw new ArgumentException($"{argumentName} must contain at least one dimension.");
        }

        int[] values = new int[tokens.Length];
        for (int index = 0; index < tokens.Length; index++)
        {
            if (!int.TryParse(tokens[index].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int dimension) || dimension <= 0)
            {
                throw new ArgumentException($"{argumentName} must contain positive integer dimensions.");
            }

            values[index] = dimension;
        }

        return values;
    }

    private static string Normalize(string value)
    {
        return (value ?? string.Empty).Trim().Replace("-", string.Empty, StringComparison.Ordinal).Replace("_", string.Empty, StringComparison.Ordinal).ToLowerInvariant();
    }

    private static bool HasSwitch(string[] args, string name)
    {
        for (int index = 0; index < args.Length; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static string GetStringArgument(string[] args, string name, string defaultValue)
    {
        for (int index = 0; index < args.Length - 1; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }
        }

        return defaultValue;
    }

    private static int GetPositiveIntArgument(string[] args, string name, int defaultValue)
    {
        string value = GetStringArgument(args, name, defaultValue.ToString(CultureInfo.InvariantCulture));
        return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) && parsed > 0 ? parsed : defaultValue;
    }

    private static float GetFloatArgument(string[] args, string name, float defaultValue)
    {
        string value = GetStringArgument(args, name, defaultValue.ToString(CultureInfo.InvariantCulture));
        return float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed) ? parsed : defaultValue;
    }
}
