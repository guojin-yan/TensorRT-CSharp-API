using System;

namespace YoloVisionSample;

public enum YoloSegmentationCoordinateSpace
{
    ModelInputPixels,
    Normalized
}

public sealed class YoloSegmentationSpatialTransformOptions
{
    public YoloSegmentationSpatialTransformOptions(
        YoloSegmentationCoordinateSpace coordinateSpace,
        bool cropToDetection = true)
    {
        if (coordinateSpace is not YoloSegmentationCoordinateSpace.ModelInputPixels and not YoloSegmentationCoordinateSpace.Normalized)
        {
            throw new ArgumentOutOfRangeException(nameof(coordinateSpace));
        }

        CoordinateSpace = coordinateSpace;
        CropToDetection = cropToDetection;
    }

    public YoloSegmentationCoordinateSpace CoordinateSpace { get; }

    public bool CropToDetection { get; }

    public static YoloSegmentationSpatialTransformOptions? FromArgs(string[] args, YoloTaskType taskType)
    {
        if (args == null)
        {
            throw new ArgumentNullException(nameof(args));
        }

        if (!HasSwitch(args, "--mask-spatial-transform"))
        {
            return null;
        }

        if (taskType != YoloTaskType.Segmentation)
        {
            throw new ArgumentException("--mask-spatial-transform is only valid with --task seg.");
        }

        string coordinateSpace = GetStringArgument(args, "--mask-coordinate-space", string.Empty);
        if (string.IsNullOrWhiteSpace(coordinateSpace))
        {
            throw new ArgumentException("--mask-spatial-transform requires --mask-coordinate-space model-input|normalized.");
        }

        YoloSegmentationCoordinateSpace parsedCoordinateSpace = coordinateSpace.Trim().ToLowerInvariant() switch
        {
            "model-input" or "modelinput" or "input-pixels" or "pixels" => YoloSegmentationCoordinateSpace.ModelInputPixels,
            "normalized" or "normalised" => YoloSegmentationCoordinateSpace.Normalized,
            _ => throw new ArgumentException("--mask-coordinate-space must be model-input or normalized.")
        };
        bool cropToDetection = GetBooleanArgument(args, "--mask-crop-to-box", defaultValue: true);
        return new YoloSegmentationSpatialTransformOptions(parsedCoordinateSpace, cropToDetection);
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

    private static bool GetBooleanArgument(string[] args, string name, bool defaultValue)
    {
        string text = GetStringArgument(args, name, string.Empty);
        if (string.IsNullOrWhiteSpace(text))
        {
            if (HasSwitch(args, name))
            {
                throw new ArgumentException($"{name} requires true or false.");
            }

            return defaultValue;
        }

        if (bool.TryParse(text, out bool value))
        {
            return value;
        }

        throw new ArgumentException($"{name} must be true or false.");
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
}

public sealed class YoloSegmentationSpatialTransformResult
{
    internal YoloSegmentationSpatialTransformResult(
        YoloSegmentationMask mask,
        YoloDetection detection,
        YoloImagePreprocessResult preprocess,
        YoloSegmentationSpatialTransformOptions options)
    {
        Mask = mask ?? throw new ArgumentNullException(nameof(mask));
        Detection = detection;
        Preprocess = preprocess ?? throw new ArgumentNullException(nameof(preprocess));
        Options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public YoloSegmentationMask Mask { get; }

    public YoloDetection Detection { get; }

    public YoloImagePreprocessResult Preprocess { get; }

    public YoloSegmentationSpatialTransformOptions Options { get; }

    public string Interpolation => "bilinear";

    public float EffectiveScaleX => Preprocess.ResizedWidth / (float)Preprocess.SourceWidth;

    public float EffectiveScaleY => Preprocess.ResizedHeight / (float)Preprocess.SourceHeight;

    public string Scope => "source-image-after-explicit-preprocess-inverse-and-optional-box-crop";

    public string Boundary => "explicit-preprocess-metadata-transform; owner must validate exporter-specific mask alignment";
}

public static class YoloSegmentationSpatialTransform
{
    public static YoloSegmentationSpatialTransformResult Apply(
        YoloSegmentationPrediction prediction,
        YoloImagePreprocessResult preprocess,
        YoloSegmentationSpatialTransformOptions options)
    {
        if (prediction == null)
        {
            throw new ArgumentNullException(nameof(prediction));
        }

        if (preprocess == null)
        {
            throw new ArgumentNullException(nameof(preprocess));
        }

        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        ValidatePreprocess(preprocess);
        ResolveModelDetection(prediction.Detection, preprocess, options.CoordinateSpace, out float left, out float top, out float right, out float bottom);
        float effectiveScaleX = preprocess.ResizedWidth / (float)preprocess.SourceWidth;
        float effectiveScaleY = preprocess.ResizedHeight / (float)preprocess.SourceHeight;
        float[] values = new float[checked(preprocess.SourceWidth * preprocess.SourceHeight)];
        for (int sourceY = 0; sourceY < preprocess.SourceHeight; sourceY++)
        {
            float modelY = preprocess.PadY + ((sourceY + 0.5f) * effectiveScaleY) - 0.5f;
            float prototypeY = ((modelY + 0.5f) * prediction.Mask.Height / preprocess.TargetHeight) - 0.5f;
            for (int sourceX = 0; sourceX < preprocess.SourceWidth; sourceX++)
            {
                float modelX = preprocess.PadX + ((sourceX + 0.5f) * effectiveScaleX) - 0.5f;
                if (options.CropToDetection && (modelX < left || modelX >= right || modelY < top || modelY >= bottom))
                {
                    continue;
                }

                float prototypeX = ((modelX + 0.5f) * prediction.Mask.Width / preprocess.TargetWidth) - 0.5f;
                values[sourceY * preprocess.SourceWidth + sourceX] = SampleBilinear(prediction.Mask, prototypeX, prototypeY);
            }
        }

        YoloSegmentationMask transformedMask = new YoloSegmentationMask(
            preprocess.SourceWidth,
            preprocess.SourceHeight,
            values,
            YoloSegmentationMaskValueKind.Probability,
            prediction.Mask.Threshold);
        YoloDetection transformedDetection = TransformDetectionToSource(
            prediction.Detection,
            preprocess,
            options.CoordinateSpace,
            effectiveScaleX,
            effectiveScaleY);
        return new YoloSegmentationSpatialTransformResult(transformedMask, transformedDetection, preprocess, options);
    }

    private static float SampleBilinear(YoloSegmentationMask mask, float x, float y)
    {
        float clampedX = Math.Clamp(x, 0.0f, mask.Width - 1.0f);
        float clampedY = Math.Clamp(y, 0.0f, mask.Height - 1.0f);
        int x0 = (int)MathF.Floor(clampedX);
        int y0 = (int)MathF.Floor(clampedY);
        int x1 = Math.Min(mask.Width - 1, x0 + 1);
        int y1 = Math.Min(mask.Height - 1, y0 + 1);
        float xWeight = clampedX - x0;
        float yWeight = clampedY - y0;
        float top = Lerp(mask.GetProbability(y0 * mask.Width + x0), mask.GetProbability(y0 * mask.Width + x1), xWeight);
        float bottom = Lerp(mask.GetProbability(y1 * mask.Width + x0), mask.GetProbability(y1 * mask.Width + x1), xWeight);
        return Lerp(top, bottom, yWeight);
    }

    private static float Lerp(float from, float to, float weight)
    {
        return from + (to - from) * weight;
    }

    private static YoloDetection TransformDetectionToSource(
        YoloDetection detection,
        YoloImagePreprocessResult preprocess,
        YoloSegmentationCoordinateSpace coordinateSpace,
        float effectiveScaleX,
        float effectiveScaleY)
    {
        ResolveModelDetection(detection, preprocess, coordinateSpace, out float left, out float top, out float right, out float bottom);
        float sourceLeft = Math.Clamp((left - preprocess.PadX) / effectiveScaleX, 0.0f, preprocess.SourceWidth);
        float sourceTop = Math.Clamp((top - preprocess.PadY) / effectiveScaleY, 0.0f, preprocess.SourceHeight);
        float sourceRight = Math.Clamp((right - preprocess.PadX) / effectiveScaleX, 0.0f, preprocess.SourceWidth);
        float sourceBottom = Math.Clamp((bottom - preprocess.PadY) / effectiveScaleY, 0.0f, preprocess.SourceHeight);
        return new YoloDetection(
            detection.ClassIndex,
            detection.Score,
            (sourceLeft + sourceRight) / 2.0f,
            (sourceTop + sourceBottom) / 2.0f,
            Math.Max(0.0f, sourceRight - sourceLeft),
            Math.Max(0.0f, sourceBottom - sourceTop),
            detection.SourceIndex);
    }

    private static void ResolveModelDetection(
        YoloDetection detection,
        YoloImagePreprocessResult preprocess,
        YoloSegmentationCoordinateSpace coordinateSpace,
        out float left,
        out float top,
        out float right,
        out float bottom)
    {
        float centerX = detection.CenterX;
        float centerY = detection.CenterY;
        float width = detection.Width;
        float height = detection.Height;
        if (!float.IsFinite(centerX) || !float.IsFinite(centerY) || !float.IsFinite(width) || !float.IsFinite(height) || width < 0.0f || height < 0.0f)
        {
            throw new ArgumentException("Detection coordinates must be finite and sizes must be non-negative.", nameof(detection));
        }

        if (coordinateSpace == YoloSegmentationCoordinateSpace.Normalized)
        {
            centerX *= preprocess.TargetWidth;
            centerY *= preprocess.TargetHeight;
            width *= preprocess.TargetWidth;
            height *= preprocess.TargetHeight;
        }

        left = centerX - width / 2.0f;
        top = centerY - height / 2.0f;
        right = centerX + width / 2.0f;
        bottom = centerY + height / 2.0f;
    }

    private static void ValidatePreprocess(YoloImagePreprocessResult preprocess)
    {
        if (preprocess.SourceWidth <= 0 || preprocess.SourceHeight <= 0 ||
            preprocess.TargetWidth <= 0 || preprocess.TargetHeight <= 0 ||
            preprocess.ResizedWidth <= 0 || preprocess.ResizedHeight <= 0)
        {
            throw new ArgumentException("Preprocess source, target, and resized dimensions must be positive.", nameof(preprocess));
        }

        if (preprocess.PadX < 0 || preprocess.PadY < 0 ||
            preprocess.PadX + preprocess.ResizedWidth > preprocess.TargetWidth ||
            preprocess.PadY + preprocess.ResizedHeight > preprocess.TargetHeight)
        {
            throw new ArgumentException("Preprocess padding and resized dimensions must fit inside the model input.", nameof(preprocess));
        }

        if (!float.IsFinite(preprocess.ResizeScaleX) || !float.IsFinite(preprocess.ResizeScaleY) ||
            preprocess.ResizeScaleX <= 0.0f || preprocess.ResizeScaleY <= 0.0f)
        {
            throw new ArgumentException("Preprocess resize scales must be finite and positive.", nameof(preprocess));
        }
    }
}
