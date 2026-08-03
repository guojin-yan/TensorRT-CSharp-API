using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using JYPPX.SampleSupport;

namespace YoloVisionSample;

public static class YoloVisionVisualizationWriter
{
    private static readonly string[] Palette =
    {
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4f46e5"
    };

    public static void Write(
        string outputPath,
        YoloVisionResult result,
        IReadOnlyList<string> labels,
        YoloModelProfile profile,
        IReadOnlyList<int> inputShape)
    {
        Write(outputPath, result, labels, profile, inputShape, imagePreprocess: null, segmentationSpatialTransform: null);
    }

    public static void Write(
        string outputPath,
        YoloVisionResult result,
        IReadOnlyList<string> labels,
        YoloModelProfile profile,
        IReadOnlyList<int> inputShape,
        YoloImagePreprocessResult? imagePreprocess,
        YoloSegmentationSpatialTransformOptions? segmentationSpatialTransform)
    {
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Visualization path must not be empty.", nameof(outputPath));
        }

        string fullPath = Path.GetFullPath(outputPath);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(
            fullPath,
            ToSvg(result, labels, profile, inputShape, imagePreprocess, segmentationSpatialTransform),
            Encoding.UTF8);
    }

    public static void Write(
        string outputPath,
        YoloVisionResult result,
        IReadOnlyList<string> labels,
        YoloModelProfile profile,
        IReadOnlyList<int> inputShape,
        YoloImagePreprocessResult imagePreprocess,
        YoloSegmentationSpatialTransformOptions? segmentationSpatialTransform,
        string backgroundImagePath)
    {
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Visualization path must not be empty.", nameof(outputPath));
        }

        string fullPath = Path.GetFullPath(outputPath);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(
            fullPath,
            ToSvg(result, labels, profile, inputShape, imagePreprocess, segmentationSpatialTransform, backgroundImagePath),
            Encoding.UTF8);
    }

    public static string ToSvg(
        YoloVisionResult result,
        IReadOnlyList<string> labels,
        YoloModelProfile profile,
        IReadOnlyList<int> inputShape)
    {
        return ToSvg(result, labels, profile, inputShape, imagePreprocess: null, segmentationSpatialTransform: null);
    }

    public static string ToSvg(
        YoloVisionResult result,
        IReadOnlyList<string> labels,
        YoloModelProfile profile,
        IReadOnlyList<int> inputShape,
        YoloImagePreprocessResult? imagePreprocess,
        YoloSegmentationSpatialTransformOptions? segmentationSpatialTransform)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        if (segmentationSpatialTransform != null && imagePreprocess == null)
        {
            throw new ArgumentException(
                "Segmentation spatial transform requires image preprocessing metadata.",
                nameof(segmentationSpatialTransform));
        }

        if (segmentationSpatialTransform != null && result.TaskType != YoloTaskType.Segmentation)
        {
            throw new ArgumentException(
                "Segmentation spatial transform can only visualize a segmentation result.",
                nameof(result));
        }

        labels ??= Array.Empty<string>();
        bool useSpatialSegmentation = result.TaskType == YoloTaskType.Segmentation &&
                                      imagePreprocess != null &&
                                      segmentationSpatialTransform != null;
        int modelWidth = useSpatialSegmentation ? imagePreprocess!.SourceWidth : GetShapeDimension(inputShape, 3, 640);
        int modelHeight = useSpatialSegmentation ? imagePreprocess!.SourceHeight : GetShapeDimension(inputShape, 2, 640);
        int canvasWidth = Math.Max(modelWidth, 360);
        int canvasHeight = Math.Max(modelHeight + (useSpatialSegmentation ? 66 : 0), 260);

        StringBuilder builder = new StringBuilder();
        builder.AppendLine($"""<svg xmlns="http://www.w3.org/2000/svg" width="{canvasWidth}" height="{canvasHeight}" viewBox="0 0 {canvasWidth} {canvasHeight}" role="img" aria-label="YoloVision visualization">""");
        builder.AppendLine("""  <rect width="100%" height="100%" fill="#f8fafc"/>""");
        builder.AppendLine($"""  <text x="16" y="28" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700" fill="#111827">YoloVision {Escape(ToTaskAlias(result.TaskType))}</text>""");
        builder.AppendLine($"""  <text x="16" y="50" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569">Family={Escape(profile.Family.ToString())} Detections={result.Detections.Count} Classifications={result.Classifications.Count} Segmentations={result.Segmentations.Count} Poses={result.Poses.Count}</text>""");
        builder.AppendLine($"""  <rect x="1" y="65" width="{canvasWidth - 2}" height="{canvasHeight - 66}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>""");

        switch (result.TaskType)
        {
            case YoloTaskType.Classification:
                AppendClassification(builder, result.Classifications, labels, canvasWidth);
                break;
            case YoloTaskType.SemanticSegmentation:
                AppendSemantic(builder, result.SemanticMap, canvasWidth, canvasHeight);
                break;
            case YoloTaskType.Segmentation:
                if (useSpatialSegmentation)
                {
                    AppendSpatialSegmentation(
                        builder,
                        result.Segmentations,
                        labels,
                        imagePreprocess!,
                        segmentationSpatialTransform!,
                        contentTop: 66);
                }
                else
                {
                    AppendSegmentation(builder, result.Segmentations, labels, modelWidth, modelHeight);
                }
                break;
            case YoloTaskType.OrientedBoundingBox:
                AppendObb(builder, result.OrientedBoxes, labels, modelWidth, modelHeight);
                break;
            case YoloTaskType.Pose:
                AppendPose(builder, result.Poses, labels, modelWidth, modelHeight);
                break;
            default:
                AppendDetections(builder, result.Detections, labels, modelWidth, modelHeight);
                break;
        }

        if (result.HasDiagnostic)
        {
            builder.AppendLine($"""  <text x="16" y="{canvasHeight - 16}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#64748b">{Escape(result.Diagnostic)}</text>""");
        }

        builder.AppendLine("</svg>");
        return builder.ToString();
    }

    public static string ToSvg(
        YoloVisionResult result,
        IReadOnlyList<string> labels,
        YoloModelProfile profile,
        IReadOnlyList<int> inputShape,
        YoloImagePreprocessResult imagePreprocess,
        YoloSegmentationSpatialTransformOptions? segmentationSpatialTransform,
        string backgroundImagePath)
    {
        if (inputShape == null)
        {
            throw new ArgumentNullException(nameof(inputShape));
        }

        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        if (imagePreprocess == null)
        {
            throw new ArgumentNullException(nameof(imagePreprocess));
        }

        if (string.IsNullOrWhiteSpace(backgroundImagePath))
        {
            throw new ArgumentException("Visualization background path must not be empty.", nameof(backgroundImagePath));
        }

        string fullBackgroundPath = Path.GetFullPath(backgroundImagePath);
        if (!File.Exists(fullBackgroundPath))
        {
            throw new FileNotFoundException("Visualization background image was not found.", fullBackgroundPath);
        }

        if (segmentationSpatialTransform != null && result.TaskType != YoloTaskType.Segmentation)
        {
            throw new ArgumentException(
                "Segmentation spatial transform can only visualize a segmentation result.",
                nameof(result));
        }

        labels ??= Array.Empty<string>();
        const int contentTop = 66;
        int sourceWidth = imagePreprocess.SourceWidth;
        int sourceHeight = imagePreprocess.SourceHeight;
        if (sourceWidth <= 0 || sourceHeight <= 0)
        {
            throw new ArgumentException("Source image dimensions must be positive.", nameof(imagePreprocess));
        }

        (int backgroundWidth, int backgroundHeight) = ReadImageDimensions(fullBackgroundPath);
        if (backgroundWidth != sourceWidth || backgroundHeight != sourceHeight)
        {
            throw new ArgumentException(
                $"Visualization background dimensions {backgroundWidth}x{backgroundHeight} do not match the preprocessed source image {sourceWidth}x{sourceHeight}.",
                nameof(backgroundImagePath));
        }

        int canvasWidth = Math.Max(sourceWidth, 360);
        int canvasHeight = Math.Max(sourceHeight + contentTop, 260);
        string mimeType = ResolveImageMimeType(fullBackgroundPath);
        string imageData = Convert.ToBase64String(File.ReadAllBytes(fullBackgroundPath));

        StringBuilder builder = new StringBuilder();
        builder.AppendLine($"""<svg xmlns="http://www.w3.org/2000/svg" width="{canvasWidth}" height="{canvasHeight}" viewBox="0 0 {canvasWidth} {canvasHeight}" role="img" aria-label="YoloVision source image visualization">""");
        builder.AppendLine("""  <rect width="100%" height="100%" fill="#f8fafc"/>""");
        builder.AppendLine($"""  <text x="16" y="28" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700" fill="#111827">YoloVision {Escape(ToTaskAlias(result.TaskType))}</text>""");
        builder.AppendLine($"""  <text x="16" y="50" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569">Family={Escape(profile.Family.ToString())} Detections={result.Detections.Count} Classifications={result.Classifications.Count} Segmentations={result.Segmentations.Count} Poses={result.Poses.Count}</text>""");
        builder.AppendLine($"""  <image data-source-image="true" x="0" y="{contentTop}" width="{sourceWidth}" height="{sourceHeight}" preserveAspectRatio="none" href="data:{mimeType};base64,{imageData}"/>""");

        switch (result.TaskType)
        {
            case YoloTaskType.Classification:
                AppendSourceClassification(builder, result.Classifications, labels, sourceWidth, contentTop);
                break;
            case YoloTaskType.SemanticSegmentation:
                AppendSourceSemantic(builder, result.SemanticMap, labels, sourceWidth, sourceHeight, contentTop);
                break;
            case YoloTaskType.Segmentation:
                if (segmentationSpatialTransform == null)
                {
                    throw new ArgumentException(
                        "Source-image segmentation visualization requires an explicit spatial transform.",
                        nameof(segmentationSpatialTransform));
                }

                AppendSpatialSegmentation(
                    builder,
                    result.Segmentations,
                    labels,
                    imagePreprocess,
                    segmentationSpatialTransform,
                    contentTop);
                break;
            case YoloTaskType.OrientedBoundingBox:
                AppendSourceObb(builder, result.OrientedBoxes, labels, imagePreprocess, contentTop);
                break;
            case YoloTaskType.Pose:
                AppendSourcePose(builder, result.Poses, labels, imagePreprocess, contentTop);
                break;
            default:
                AppendSourceDetections(builder, result.Detections, labels, imagePreprocess, contentTop);
                break;
        }

        builder.AppendLine("</svg>");
        return builder.ToString();
    }

    private static void AppendSourceDetections(
        StringBuilder builder,
        IReadOnlyList<YoloDetection> detections,
        IReadOnlyList<string> labels,
        YoloImagePreprocessResult preprocess,
        int contentTop)
    {
        if (detections.Count == 0)
        {
            AppendEmpty(builder, "No detections above threshold.");
            return;
        }

        int index = 0;
        foreach (YoloDetection detection in detections.Take(100))
        {
            YoloDetection sourceDetection = TransformDetectionToSource(detection, preprocess);
            string color = Palette[index % Palette.Length];
            AppendSourceIndexedBox(
                builder,
                sourceDetection,
                preprocess.SourceWidth,
                preprocess.SourceHeight,
                contentTop,
                color,
                index + 1);
            index++;
        }

        AppendSourceDetectionLegend(builder, detections, labels, contentTop);
    }

    private static void AppendSourceIndexedBox(
        StringBuilder builder,
        YoloDetection detection,
        int sourceWidth,
        int sourceHeight,
        int contentTop,
        string color,
        int number)
    {
        float left = Math.Clamp(detection.Left, 0.0f, sourceWidth);
        float top = Math.Clamp(detection.Top, 0.0f, sourceHeight);
        float right = Math.Clamp(detection.Right, 0.0f, sourceWidth);
        float bottom = Math.Clamp(detection.Bottom, 0.0f, sourceHeight);
        float width = Math.Max(1.0f, right - left);
        float height = Math.Max(1.0f, bottom - top);
        float tagY = Math.Max(contentTop, contentTop + top - 22.0f);
        builder.AppendLine($"""  <rect x="{Format(left)}" y="{Format(contentTop + top)}" width="{Format(width)}" height="{Format(height)}" fill="none" stroke="{color}" stroke-width="3"/>""");
        builder.AppendLine($"""  <rect x="{Format(left)}" y="{Format(tagY)}" width="24" height="22" fill="{color}"/>""");
        builder.AppendLine($"""  <text x="{Format(left + 7)}" y="{Format(tagY + 16)}" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="700" fill="#ffffff">{number}</text>""");
    }

    private static void AppendSourceDetectionLegend(
        StringBuilder builder,
        IReadOnlyList<YoloDetection> detections,
        IReadOnlyList<string> labels,
        int contentTop)
    {
        int count = Math.Min(detections.Count, 12);
        int panelHeight = 40 + count * 25;
        builder.AppendLine($"""  <rect x="16" y="{contentTop + 16}" width="238" height="{panelHeight}" fill="#111827" opacity="0.84"/>""");
        builder.AppendLine($"""  <text x="32" y="{contentTop + 43}" font-family="Segoe UI, Arial, sans-serif" font-size="15" font-weight="700" fill="#ffffff">Detections</text>""");
        for (int index = 0; index < count; index++)
        {
            YoloDetection detection = detections[index];
            string color = Palette[index % Palette.Length];
            int y = contentTop + 60 + index * 25;
            builder.AppendLine($"""  <rect x="32" y="{y}" width="18" height="18" fill="{color}"/>""");
            builder.AppendLine($"""  <text x="38" y="{y + 14}" font-family="Segoe UI, Arial, sans-serif" font-size="11" font-weight="700" fill="#ffffff">{index + 1}</text>""");
            builder.AppendLine($"""  <text x="60" y="{y + 14}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#ffffff">{Escape(LabelOrIndex(labels, detection.ClassIndex))} {detection.Score:0.000}</text>""");
        }
    }

    private static void AppendSourcePose(
        StringBuilder builder,
        IReadOnlyList<YoloPosePrediction> poses,
        IReadOnlyList<string> labels,
        YoloImagePreprocessResult preprocess,
        int contentTop)
    {
        if (poses.Count == 0)
        {
            AppendEmpty(builder, "No poses above threshold.");
            return;
        }

        int index = 0;
        foreach (YoloPosePrediction pose in poses.Take(50))
        {
            bool normalized = IsNormalized(pose.Detection);
            YoloDetection sourceDetection = TransformDetectionToSource(pose.Detection, preprocess);
            string color = Palette[index % Palette.Length];
            AppendSourceBox(
                builder,
                sourceDetection,
                labels,
                preprocess.SourceWidth,
                preprocess.SourceHeight,
                contentTop,
                color,
                "pose");
            foreach (YoloPoseKeypoint keypoint in pose.Keypoints)
            {
                float modelX = normalized ? keypoint.X * preprocess.TargetWidth : keypoint.X;
                float modelY = normalized ? keypoint.Y * preprocess.TargetHeight : keypoint.Y;
                float x = TransformCoordinateToSource(modelX, preprocess, horizontal: true);
                float y = TransformCoordinateToSource(modelY, preprocess, horizontal: false);
                builder.AppendLine($"""  <circle cx="{Format(x)}" cy="{Format(contentTop + y)}" r="4" fill="{color}" stroke="#ffffff" stroke-width="1" opacity="{Format(Math.Clamp(keypoint.Score, 0.25f, 1.0f))}"/>""");
            }

            index++;
        }
    }

    private static void AppendSourceObb(
        StringBuilder builder,
        IReadOnlyList<YoloObbDetection> boxes,
        IReadOnlyList<string> labels,
        YoloImagePreprocessResult preprocess,
        int contentTop)
    {
        if (boxes.Count == 0)
        {
            AppendEmpty(builder, "No oriented boxes above threshold.");
            return;
        }

        int index = 0;
        foreach (YoloObbDetection oriented in boxes.Take(100))
        {
            YoloDetection sourceDetection = TransformDetectionToSource(oriented.Box, preprocess);
            BoxRect rect = ResolveBox(sourceDetection, preprocess.SourceWidth, preprocess.SourceHeight);
            string color = Palette[index % Palette.Length];
            double angleDegrees = oriented.AngleRadians * 180.0 / Math.PI;
            string label = $"{LabelOrIndex(labels, oriented.Box.ClassIndex)} {oriented.Box.Score:0.###} angle={angleDegrees:0.#}";
            builder.AppendLine($"""  <g transform="rotate({Format(angleDegrees)} {Format(rect.CenterX)} {Format(contentTop + rect.CenterY)})">""");
            builder.AppendLine($"""    <rect x="{Format(rect.X)}" y="{Format(contentTop + rect.Y)}" width="{Format(rect.Width)}" height="{Format(rect.Height)}" fill="none" stroke="{color}" stroke-width="3"/>""");
            builder.AppendLine("  </g>");
            AppendLabel(builder, rect.X, contentTop + rect.Y, color, label);
            index++;
        }
    }

    private static void AppendSourceClassification(
        StringBuilder builder,
        IReadOnlyList<YoloClassificationPrediction> predictions,
        IReadOnlyList<string> labels,
        int sourceWidth,
        int contentTop)
    {
        if (predictions.Count == 0)
        {
            AppendEmpty(builder, "No classification scores above threshold.");
            return;
        }

        int count = Math.Min(predictions.Count, 5);
        int panelWidth = Math.Min(440, Math.Max(280, sourceWidth - 32));
        int panelHeight = 42 + count * 31;
        builder.AppendLine($"""  <rect x="16" y="{contentTop + 16}" width="{panelWidth}" height="{panelHeight}" fill="#111827" opacity="0.82"/>""");
        builder.AppendLine($"""  <text x="32" y="{contentTop + 43}" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="700" fill="#ffffff">Top predictions</text>""");
        for (int index = 0; index < count; index++)
        {
            YoloClassificationPrediction prediction = predictions[index];
            int y = contentTop + 68 + index * 31;
            string label = $"{index + 1}. {LabelOrIndex(labels, prediction.ClassIndex)}";
            builder.AppendLine($"""  <text x="32" y="{y + 14}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#ffffff">{Escape(label)}</text>""");
            builder.AppendLine($"""  <text x="{panelWidth - 70}" y="{y + 14}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#ffffff">{prediction.Score:0.000}</text>""");
        }
    }

    private static void AppendSourceSemantic(
        StringBuilder builder,
        YoloSemanticMap? map,
        IReadOnlyList<string> labels,
        int sourceWidth,
        int sourceHeight,
        int contentTop)
    {
        if (map == null)
        {
            AppendEmpty(builder, "No semantic map available.");
            return;
        }

        int columns = Math.Max(1, Math.Min(sourceWidth, 96));
        int rows = Math.Max(1, Math.Min(sourceHeight, 72));
        float cellWidth = sourceWidth / (float)columns;
        float cellHeight = sourceHeight / (float)rows;
        for (int row = 0; row < rows; row++)
        {
            int mapY = Math.Min(map.Height - 1, row * map.Height / rows);
            for (int column = 0; column < columns; column++)
            {
                int mapX = Math.Min(map.Width - 1, column * map.Width / columns);
                int classIndex = InferSemanticClassAt(map, mapX, mapY);
                string color = Palette[classIndex % Palette.Length];
                string opacity = IsSemanticBackgroundClass(labels, classIndex) ? "0.10" : "0.52";
                builder.AppendLine($"""  <rect data-semantic-cell="true" x="{Format(column * cellWidth)}" y="{Format(contentTop + row * cellHeight)}" width="{Format(cellWidth + 0.5f)}" height="{Format(cellHeight + 0.5f)}" fill="{color}" opacity="{opacity}"/>""");
            }
        }

        AppendSourceSemanticLegend(builder, map, labels, sourceWidth, contentTop);
    }

    private static void AppendSourceSemanticLegend(
        StringBuilder builder,
        YoloSemanticMap map,
        IReadOnlyList<string> labels,
        int sourceWidth,
        int contentTop)
    {
        int[] histogram = map.GetClassHistogram();
        int[] activeClasses = Enumerable.Range(0, histogram.Length)
            .Where(classIndex => histogram[classIndex] > 0)
            .OrderByDescending(classIndex => histogram[classIndex])
            .ThenBy(classIndex => classIndex)
            .Take(8)
            .ToArray();
        int panelWidth = Math.Min(310, Math.Max(250, sourceWidth - 32));
        int panelHeight = 48 + activeClasses.Length * 27;
        int panelX = Math.Max(16, sourceWidth - panelWidth - 16);
        int pixelCount = checked(map.Width * map.Height);
        builder.AppendLine($"""  <rect data-semantic-legend="true" x="{panelX}" y="{contentTop + 16}" width="{panelWidth}" height="{panelHeight}" fill="#111827" opacity="0.86"/>""");
        builder.AppendLine($"""  <text x="{panelX + 16}" y="{contentTop + 43}" font-family="Segoe UI, Arial, sans-serif" font-size="15" font-weight="700" fill="#ffffff">Semantic classes</text>""");

        for (int index = 0; index < activeClasses.Length; index++)
        {
            int classIndex = activeClasses[index];
            int y = contentTop + 58 + index * 27;
            string color = Palette[classIndex % Palette.Length];
            string label = LabelOrIndex(labels, classIndex);
            string percentage = (100.0 * histogram[classIndex] / pixelCount).ToString("0.0", CultureInfo.InvariantCulture);
            string count = histogram[classIndex].ToString("N0", CultureInfo.InvariantCulture);
            builder.AppendLine($"""  <rect x="{panelX + 16}" y="{y}" width="18" height="18" fill="{color}"/>""");
            builder.AppendLine($"""  <text x="{panelX + 44}" y="{y + 14}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#ffffff">{Escape(label)}  {count} px ({percentage}%)</text>""");
        }
    }

    private static bool IsSemanticBackgroundClass(IReadOnlyList<string> labels, int classIndex)
    {
        if ((uint)classIndex >= (uint)labels.Count)
        {
            return false;
        }

        string label = labels[classIndex].Trim();
        return label.Equals("background", StringComparison.OrdinalIgnoreCase) ||
               label.Equals("__background__", StringComparison.OrdinalIgnoreCase);
    }

    private static void AppendDetections(StringBuilder builder, IReadOnlyList<YoloDetection> detections, IReadOnlyList<string> labels, int width, int height)
    {
        if (detections.Count == 0)
        {
            AppendEmpty(builder, "No detections above threshold.");
            return;
        }

        int index = 0;
        foreach (YoloDetection detection in detections.Take(100))
        {
            AppendBox(builder, detection, labels, width, height, index, "det");
            index++;
        }
    }

    private static void AppendSegmentation(StringBuilder builder, IReadOnlyList<YoloSegmentationPrediction> segmentations, IReadOnlyList<string> labels, int width, int height)
    {
        if (segmentations.Count == 0)
        {
            AppendEmpty(builder, "No segmentation masks above threshold.");
            return;
        }

        int index = 0;
        foreach (YoloSegmentationPrediction segmentation in segmentations.Take(100))
        {
            BoxRect rect = ResolveBox(segmentation.Detection, width, height);
            string color = Palette[index % Palette.Length];
            AppendSegmentationMaskPreview(builder, segmentation.Mask, rect, color);
            AppendBox(builder, segmentation.Detection, labels, width, height, index, $"mask {segmentation.Mask.Width}x{segmentation.Mask.Height}");
            index++;
        }
    }

    private static void AppendSegmentationMaskPreview(StringBuilder builder, YoloSegmentationMask mask, BoxRect rect, string color)
    {
        int columns = Math.Max(1, Math.Min(mask.Width, 24));
        int rows = Math.Max(1, Math.Min(mask.Height, 24));
        float cellWidth = rect.Width / columns;
        float cellHeight = rect.Height / rows;
        for (int row = 0; row < rows; row++)
        {
            int sourceY = Math.Min(mask.Height - 1, row * mask.Height / rows);
            for (int column = 0; column < columns; column++)
            {
                int sourceX = Math.Min(mask.Width - 1, column * mask.Width / columns);
                float probability = mask.GetProbability(sourceY * mask.Width + sourceX);
                if (probability < mask.Threshold)
                {
                    continue;
                }

                float opacity = 0.12f + probability * 0.46f;
                builder.AppendLine($"""  <rect data-mask-cell="true" x="{Format(rect.X + column * cellWidth)}" y="{Format(rect.Y + row * cellHeight)}" width="{Format(cellWidth + 0.25f)}" height="{Format(cellHeight + 0.25f)}" fill="{color}" opacity="{Format(opacity)}"/>""");
            }
        }
    }

    private static void AppendSpatialSegmentation(
        StringBuilder builder,
        IReadOnlyList<YoloSegmentationPrediction> segmentations,
        IReadOnlyList<string> labels,
        YoloImagePreprocessResult preprocess,
        YoloSegmentationSpatialTransformOptions options,
        int contentTop)
    {
        if (segmentations.Count == 0)
        {
            AppendEmpty(builder, "No segmentation masks above threshold.");
            return;
        }

        int index = 0;
        foreach (YoloSegmentationPrediction segmentation in segmentations.Take(100))
        {
            YoloSegmentationSpatialTransformResult transform =
                YoloSegmentationSpatialTransform.Apply(segmentation, preprocess, options);
            string color = Palette[index % Palette.Length];
            AppendSourceMaskPreview(builder, transform.Mask, preprocess.SourceWidth, preprocess.SourceHeight, contentTop, color);
            AppendSourceBox(builder, transform.Detection, labels, preprocess.SourceWidth, preprocess.SourceHeight, contentTop, color);
            index++;
        }

        builder.AppendLine($"""  <text x="16" y="{contentTop + preprocess.SourceHeight - 8}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#334155">spatial mask: explicit preprocess inverse, crop={options.CropToDetection.ToString().ToLowerInvariant()}</text>""");
    }

    private static void AppendSourceMaskPreview(
        StringBuilder builder,
        YoloSegmentationMask mask,
        int sourceWidth,
        int sourceHeight,
        int contentTop,
        string color)
    {
        int columns = Math.Max(1, Math.Min(mask.Width, 48));
        int rows = Math.Max(1, Math.Min(mask.Height, 48));
        float cellWidth = sourceWidth / (float)columns;
        float cellHeight = sourceHeight / (float)rows;
        for (int row = 0; row < rows; row++)
        {
            int sourceY = Math.Min(mask.Height - 1, row * mask.Height / rows);
            for (int column = 0; column < columns; column++)
            {
                int sourceX = Math.Min(mask.Width - 1, column * mask.Width / columns);
                float probability = mask.GetProbability(sourceY * mask.Width + sourceX);
                if (probability < mask.Threshold)
                {
                    continue;
                }

                float opacity = 0.12f + probability * 0.46f;
                builder.AppendLine($"""  <rect data-spatial-mask-cell="true" x="{Format(column * cellWidth)}" y="{Format(contentTop + row * cellHeight)}" width="{Format(cellWidth + 0.25f)}" height="{Format(cellHeight + 0.25f)}" fill="{color}" opacity="{Format(opacity)}"/>""");
            }
        }
    }

    private static void AppendSourceBox(
        StringBuilder builder,
        YoloDetection detection,
        IReadOnlyList<string> labels,
        int sourceWidth,
        int sourceHeight,
        int contentTop,
        string color,
        string prefix = "mask")
    {
        float left = Math.Clamp(detection.Left, 0.0f, sourceWidth);
        float top = Math.Clamp(detection.Top, 0.0f, sourceHeight);
        float right = Math.Clamp(detection.Right, 0.0f, sourceWidth);
        float bottom = Math.Clamp(detection.Bottom, 0.0f, sourceHeight);
        float width = Math.Max(1.0f, right - left);
        float height = Math.Max(1.0f, bottom - top);
        builder.AppendLine($"""  <rect x="{Format(left)}" y="{Format(contentTop + top)}" width="{Format(width)}" height="{Format(height)}" fill="none" stroke="{color}" stroke-width="2"/>""");
        AppendLabel(
            builder,
            left,
            contentTop + top,
            color,
            $"{prefix} {LabelOrIndex(labels, detection.ClassIndex)} {detection.Score:0.###}");
    }

    private static void AppendObb(StringBuilder builder, IReadOnlyList<YoloObbDetection> boxes, IReadOnlyList<string> labels, int width, int height)
    {
        if (boxes.Count == 0)
        {
            AppendEmpty(builder, "No oriented boxes above threshold.");
            return;
        }

        int index = 0;
        foreach (YoloObbDetection oriented in boxes.Take(100))
        {
            BoxRect rect = ResolveBox(oriented.Box, width, height);
            string color = Palette[index % Palette.Length];
            double angleDegrees = oriented.AngleRadians * 180.0 / Math.PI;
            string label = $"{LabelOrIndex(labels, oriented.Box.ClassIndex)} {oriented.Box.Score:0.###} angle={angleDegrees:0.#}";
            builder.AppendLine($"""  <g transform="rotate({Format(angleDegrees)} {Format(rect.CenterX)} {Format(rect.CenterY)})">""");
            builder.AppendLine($"""    <rect x="{Format(rect.X)}" y="{Format(rect.Y)}" width="{Format(rect.Width)}" height="{Format(rect.Height)}" fill="none" stroke="{color}" stroke-width="2"/>""");
            builder.AppendLine("  </g>");
            AppendLabel(builder, rect.X, rect.Y, color, label);
            index++;
        }
    }

    private static void AppendPose(StringBuilder builder, IReadOnlyList<YoloPosePrediction> poses, IReadOnlyList<string> labels, int width, int height)
    {
        if (poses.Count == 0)
        {
            AppendEmpty(builder, "No poses above threshold.");
            return;
        }

        int index = 0;
        foreach (YoloPosePrediction pose in poses.Take(50))
        {
            AppendBox(builder, pose.Detection, labels, width, height, index, "pose");
            string color = Palette[index % Palette.Length];
            foreach (YoloPoseKeypoint keypoint in pose.Keypoints)
            {
                float x = ScaleCoordinate(keypoint.X, width);
                float y = ScaleCoordinate(keypoint.Y, height);
                builder.AppendLine($"""  <circle cx="{Format(x)}" cy="{Format(y)}" r="3" fill="{color}" opacity="{Format(Math.Clamp(keypoint.Score, 0.25f, 1.0f))}"/>""");
            }

            index++;
        }
    }

    private static void AppendClassification(StringBuilder builder, IReadOnlyList<YoloClassificationPrediction> predictions, IReadOnlyList<string> labels, int canvasWidth)
    {
        if (predictions.Count == 0)
        {
            AppendEmpty(builder, "No classification scores above threshold.");
            return;
        }

        int row = 0;
        int barWidth = Math.Max(80, canvasWidth - 180);
        foreach (YoloClassificationPrediction prediction in predictions.Take(12))
        {
            float clampedScore = Math.Clamp(prediction.Score, 0.0f, 1.0f);
            int y = 88 + row * 28;
            string color = Palette[row % Palette.Length];
            string label = $"{LabelOrIndex(labels, prediction.ClassIndex)} {prediction.Score:0.###}";
            builder.AppendLine($"""  <text x="18" y="{y + 14}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#334155">{Escape(label)}</text>""");
            builder.AppendLine($"""  <rect x="140" y="{y}" width="{barWidth}" height="18" fill="#e2e8f0"/>""");
            builder.AppendLine($"""  <rect x="140" y="{y}" width="{Format(barWidth * clampedScore)}" height="18" fill="{color}"/>""");
            row++;
        }
    }

    private static void AppendSemantic(StringBuilder builder, YoloSemanticMap? map, int canvasWidth, int canvasHeight)
    {
        if (map == null)
        {
            AppendEmpty(builder, "No semantic map available.");
            return;
        }

        int cols = Math.Max(1, Math.Min(map.Width, 32));
        int rows = Math.Max(1, Math.Min(map.Height, 24));
        float cellWidth = (canvasWidth - 32.0f) / cols;
        float cellHeight = (canvasHeight - 96.0f) / rows;
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                int classIndex = InferSemanticClass(map, col, row);
                string color = Palette[classIndex % Palette.Length];
                builder.AppendLine($"""  <rect x="{Format(16 + col * cellWidth)}" y="{Format(80 + row * cellHeight)}" width="{Format(cellWidth + 0.5f)}" height="{Format(cellHeight + 0.5f)}" fill="{color}" opacity="0.58"/>""");
            }
        }

        builder.AppendLine($"""  <text x="16" y="{canvasHeight - 18}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#334155">semantic classes={map.ClassCount} map={map.Width}x{map.Height}</text>""");
    }

    private static void AppendBox(StringBuilder builder, YoloDetection detection, IReadOnlyList<string> labels, int width, int height, int index, string prefix)
    {
        BoxRect rect = ResolveBox(detection, width, height);
        string color = Palette[index % Palette.Length];
        string label = $"{prefix} {LabelOrIndex(labels, detection.ClassIndex)} {detection.Score:0.###}";
        builder.AppendLine($"""  <rect x="{Format(rect.X)}" y="{Format(rect.Y)}" width="{Format(rect.Width)}" height="{Format(rect.Height)}" fill="none" stroke="{color}" stroke-width="2"/>""");
        AppendLabel(builder, rect.X, rect.Y, color, label);
    }

    private static void AppendLabel(StringBuilder builder, float x, float y, string color, string label)
    {
        float labelY = Math.Max(70, y - 18);
        int labelWidth = Math.Max(80, label.Length * 7 + 12);
        builder.AppendLine($"""  <rect x="{Format(Math.Max(4, x))}" y="{Format(labelY)}" width="{labelWidth}" height="18" fill="{color}" opacity="0.92"/>""");
        builder.AppendLine($"""  <text x="{Format(Math.Max(8, x + 4))}" y="{Format(labelY + 13)}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#ffffff">{Escape(label)}</text>""");
    }

    private static void AppendEmpty(StringBuilder builder, string message)
    {
        builder.AppendLine($"""  <text x="18" y="96" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#64748b">{Escape(message)}</text>""");
    }

    private static BoxRect ResolveBox(YoloDetection detection, int width, int height)
    {
        float centerX = ScaleCoordinate(detection.CenterX, width);
        float centerY = ScaleCoordinate(detection.CenterY, height);
        float boxWidth = Math.Max(1.0f, ScaleSize(detection.Width, width));
        float boxHeight = Math.Max(1.0f, ScaleSize(detection.Height, height));
        float x = Math.Clamp(centerX - boxWidth / 2.0f, 2.0f, Math.Max(2.0f, width - 2.0f));
        float y = Math.Clamp(centerY - boxHeight / 2.0f, 68.0f, Math.Max(68.0f, height - 2.0f));
        boxWidth = Math.Min(boxWidth, Math.Max(1.0f, width - x - 2.0f));
        boxHeight = Math.Min(boxHeight, Math.Max(1.0f, height - y - 2.0f));
        return new BoxRect(x, y, boxWidth, boxHeight);
    }

    private static YoloDetection TransformDetectionToSource(
        YoloDetection detection,
        YoloImagePreprocessResult preprocess)
    {
        bool normalized = IsNormalized(detection);
        float centerX = normalized ? detection.CenterX * preprocess.TargetWidth : detection.CenterX;
        float centerY = normalized ? detection.CenterY * preprocess.TargetHeight : detection.CenterY;
        float width = normalized ? detection.Width * preprocess.TargetWidth : detection.Width;
        float height = normalized ? detection.Height * preprocess.TargetHeight : detection.Height;
        if (!float.IsFinite(centerX) || !float.IsFinite(centerY) ||
            !float.IsFinite(width) || !float.IsFinite(height) ||
            width < 0.0f || height < 0.0f)
        {
            throw new ArgumentException("Detection coordinates must be finite and sizes must be non-negative.", nameof(detection));
        }

        float left = TransformCoordinateToSource(centerX - width / 2.0f, preprocess, horizontal: true);
        float top = TransformCoordinateToSource(centerY - height / 2.0f, preprocess, horizontal: false);
        float right = TransformCoordinateToSource(centerX + width / 2.0f, preprocess, horizontal: true);
        float bottom = TransformCoordinateToSource(centerY + height / 2.0f, preprocess, horizontal: false);
        return new YoloDetection(
            detection.ClassIndex,
            detection.Score,
            (left + right) / 2.0f,
            (top + bottom) / 2.0f,
            Math.Max(0.0f, right - left),
            Math.Max(0.0f, bottom - top),
            detection.SourceIndex);
    }

    private static float TransformCoordinateToSource(
        float modelCoordinate,
        YoloImagePreprocessResult preprocess,
        bool horizontal)
    {
        float sourceSize = horizontal ? preprocess.SourceWidth : preprocess.SourceHeight;
        float resizedSize = horizontal ? preprocess.ResizedWidth : preprocess.ResizedHeight;
        float offset = preprocess.CenterCropEnabled
            ? (horizontal ? preprocess.CropX : preprocess.CropY)
            : -(horizontal ? preprocess.PadX : preprocess.PadY);
        float sourceCoordinate = (modelCoordinate + offset) / (resizedSize / sourceSize);
        return Math.Clamp(sourceCoordinate, 0.0f, sourceSize);
    }

    private static bool IsNormalized(YoloDetection detection)
    {
        return MathF.Abs(detection.CenterX) <= 1.5f &&
               MathF.Abs(detection.CenterY) <= 1.5f &&
               MathF.Abs(detection.Width) <= 1.5f &&
               MathF.Abs(detection.Height) <= 1.5f;
    }

    private static int InferSemanticClass(YoloSemanticMap map, int x, int y)
    {
        int sourceX = Math.Clamp((int)MathF.Round(x * (map.Width - 1.0f) / Math.Max(1, Math.Min(map.Width, 32) - 1)), 0, map.Width - 1);
        int sourceY = Math.Clamp((int)MathF.Round(y * (map.Height - 1.0f) / Math.Max(1, Math.Min(map.Height, 24) - 1)), 0, map.Height - 1);
        int spatialIndex = sourceY * map.Width + sourceX;
        int bestClass = 0;
        float bestScore = float.NegativeInfinity;
        for (int classIndex = 0; classIndex < map.ClassCount; classIndex++)
        {
            int index = classIndex * map.Width * map.Height + spatialIndex;
            if (index >= 0 && index < map.Values.Length && map.Values[index] > bestScore)
            {
                bestScore = map.Values[index];
                bestClass = classIndex;
            }
        }

        return bestClass;
    }

    private static int InferSemanticClassAt(YoloSemanticMap map, int x, int y)
    {
        int spatialIndex = y * map.Width + x;
        int bestClass = 0;
        float bestScore = float.NegativeInfinity;
        for (int classIndex = 0; classIndex < map.ClassCount; classIndex++)
        {
            float score = map.Values[classIndex * map.Width * map.Height + spatialIndex];
            if (score > bestScore)
            {
                bestScore = score;
                bestClass = classIndex;
            }
        }

        return bestClass;
    }

    private static string ResolveImageMimeType(string path)
    {
        return Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".jpg" or ".jpeg" => "image/jpeg",
            ".png" => "image/png",
            ".bmp" => "image/bmp",
            _ => throw new NotSupportedException("Visualization background must be a JPEG, PNG, or BMP image.")
        };
    }

    private static (int Width, int Height) ReadImageDimensions(string path)
    {
        string extension = Path.GetExtension(path).ToLowerInvariant();
        using FileStream stream = File.OpenRead(path);
        using BinaryReader reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        try
        {
            return extension switch
            {
                ".jpg" or ".jpeg" => ReadJpegDimensions(reader),
                ".png" => ReadPngDimensions(reader),
                ".bmp" => ReadBmpDimensions(reader),
                _ => throw new NotSupportedException("Visualization background must be a JPEG, PNG, or BMP image.")
            };
        }
        catch (EndOfStreamException exception)
        {
            throw new InvalidDataException("Visualization background image header is truncated.", exception);
        }
    }

    private static (int Width, int Height) ReadPngDimensions(BinaryReader reader)
    {
        byte[] header = reader.ReadBytes(24);
        byte[] signature = { 137, 80, 78, 71, 13, 10, 26, 10 };
        if (header.Length != 24 || !header.Take(8).SequenceEqual(signature) ||
            ReadBigEndianInt32(header, 8) != 13 ||
            header[12] != (byte)'I' || header[13] != (byte)'H' || header[14] != (byte)'D' || header[15] != (byte)'R')
        {
            throw new InvalidDataException("Visualization background is not a valid PNG header.");
        }

        int width = ReadBigEndianInt32(header, 16);
        int height = ReadBigEndianInt32(header, 20);
        return ValidateImageDimensions(width, height);
    }

    private static (int Width, int Height) ReadBmpDimensions(BinaryReader reader)
    {
        byte[] header = reader.ReadBytes(26);
        if (header.Length != 26 || header[0] != (byte)'B' || header[1] != (byte)'M')
        {
            throw new InvalidDataException("Visualization background is not a valid BMP header.");
        }

        int width = BitConverter.ToInt32(header, 18);
        int rawHeight = BitConverter.ToInt32(header, 22);
        if (rawHeight == int.MinValue)
        {
            throw new InvalidDataException("Visualization background contains an invalid BMP height.");
        }

        int height = Math.Abs(rawHeight);
        return ValidateImageDimensions(width, height);
    }

    private static (int Width, int Height) ReadJpegDimensions(BinaryReader reader)
    {
        if (reader.ReadByte() != 0xff || reader.ReadByte() != 0xd8)
        {
            throw new InvalidDataException("Visualization background is not a valid JPEG header.");
        }

        while (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            byte prefix;
            do
            {
                prefix = reader.ReadByte();
            }
            while (prefix != 0xff && reader.BaseStream.Position < reader.BaseStream.Length);

            byte marker;
            do
            {
                marker = reader.ReadByte();
            }
            while (marker == 0xff);

            if (marker == 0xd9 || marker == 0xda)
            {
                break;
            }

            if (marker is 0x01 or >= 0xd0 and <= 0xd7)
            {
                continue;
            }

            int segmentLength = ReadBigEndianUInt16(reader);
            if (segmentLength < 2)
            {
                throw new InvalidDataException("Visualization background contains an invalid JPEG segment.");
            }

            if (marker is 0xc0 or 0xc1 or 0xc2 or 0xc3 or 0xc5 or 0xc6 or 0xc7 or 0xc9 or 0xca or 0xcb or 0xcd or 0xce or 0xcf)
            {
                if (segmentLength < 7)
                {
                    throw new InvalidDataException("Visualization background contains an invalid JPEG frame header.");
                }

                reader.ReadByte();
                int height = ReadBigEndianUInt16(reader);
                int width = ReadBigEndianUInt16(reader);
                return ValidateImageDimensions(width, height);
            }

            reader.BaseStream.Seek(segmentLength - 2, SeekOrigin.Current);
        }

        throw new InvalidDataException("Visualization background JPEG dimensions could not be resolved.");
    }

    private static int ReadBigEndianUInt16(BinaryReader reader)
    {
        int high = reader.ReadByte();
        int low = reader.ReadByte();
        return (high << 8) | low;
    }

    private static int ReadBigEndianInt32(byte[] bytes, int offset)
    {
        return (bytes[offset] << 24) |
               (bytes[offset + 1] << 16) |
               (bytes[offset + 2] << 8) |
               bytes[offset + 3];
    }

    private static (int Width, int Height) ValidateImageDimensions(int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            throw new InvalidDataException("Visualization background dimensions must be positive.");
        }

        return (width, height);
    }

    private static float ScaleCoordinate(float value, int limit)
    {
        return MathF.Abs(value) <= 1.5f ? value * limit : value;
    }

    private static float ScaleSize(float value, int limit)
    {
        return MathF.Abs(value) <= 1.5f ? value * limit : value;
    }

    private static int GetShapeDimension(IReadOnlyList<int> shape, int index, int fallback)
    {
        if (shape == null || shape.Count <= index || shape[index] <= 0)
        {
            return fallback;
        }

        return shape[index];
    }

    private static string LabelOrIndex(IReadOnlyList<string> labels, int index)
    {
        return index >= 0 && index < labels.Count && !string.IsNullOrWhiteSpace(labels[index])
            ? labels[index]
            : index.ToString(CultureInfo.InvariantCulture);
    }

    private static string ToTaskAlias(YoloTaskType taskType)
    {
        return taskType switch
        {
            YoloTaskType.Detection => "det",
            YoloTaskType.Classification => "cls",
            YoloTaskType.Segmentation => "seg",
            YoloTaskType.OrientedBoundingBox => "obb",
            YoloTaskType.Pose => "pose",
            YoloTaskType.SemanticSegmentation => "sem",
            _ => "det"
        };
    }

    private static string Format(double value)
    {
        return value.ToString("0.###", CultureInfo.InvariantCulture);
    }

    private static string Escape(string value)
    {
        return (value ?? string.Empty)
            .Replace("&", "&amp;", StringComparison.Ordinal)
            .Replace("<", "&lt;", StringComparison.Ordinal)
            .Replace(">", "&gt;", StringComparison.Ordinal)
            .Replace("\"", "&quot;", StringComparison.Ordinal);
    }

    private readonly struct BoxRect
    {
        public BoxRect(float x, float y, float width, float height)
        {
            X = x;
            Y = y;
            Width = width;
            Height = height;
        }

        public float X { get; }

        public float Y { get; }

        public float Width { get; }

        public float Height { get; }

        public float CenterX => X + Width / 2.0f;

        public float CenterY => Y + Height / 2.0f;
    }
}
