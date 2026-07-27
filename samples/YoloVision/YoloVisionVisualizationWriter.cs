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

        File.WriteAllText(fullPath, ToSvg(result, labels, profile, inputShape), Encoding.UTF8);
    }

    public static string ToSvg(
        YoloVisionResult result,
        IReadOnlyList<string> labels,
        YoloModelProfile profile,
        IReadOnlyList<int> inputShape)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        labels ??= Array.Empty<string>();
        int modelWidth = GetShapeDimension(inputShape, 3, 640);
        int modelHeight = GetShapeDimension(inputShape, 2, 640);
        int canvasWidth = Math.Max(modelWidth, 360);
        int canvasHeight = Math.Max(modelHeight, 260);

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
                AppendSegmentation(builder, result.Segmentations, labels, modelWidth, modelHeight);
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
