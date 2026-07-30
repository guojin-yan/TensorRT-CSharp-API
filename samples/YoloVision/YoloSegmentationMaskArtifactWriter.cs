using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace YoloVisionSample;

public static class YoloSegmentationMaskArtifactWriter
{
    public const string SchemaVersion = "yolovision-segmentation-mask-artifacts.v1";

    public static string Write(
        string outputDirectory,
        YoloVisionResult result,
        IReadOnlyList<string> labels,
        YoloImagePreprocessResult? imagePreprocess = null,
        YoloSegmentationSpatialTransformOptions? spatialTransform = null)
    {
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            throw new ArgumentException("Segmentation mask output directory is required.", nameof(outputDirectory));
        }
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }
        if (labels == null)
        {
            throw new ArgumentNullException(nameof(labels));
        }
        if (result.TaskType != YoloTaskType.Segmentation)
        {
            throw new ArgumentException("Segmentation mask artifacts require a segmentation result.", nameof(result));
        }
        if ((imagePreprocess == null) != (spatialTransform == null))
        {
            throw new ArgumentException("Image preprocess metadata and spatial transform options must be provided together.");
        }

        string directory = Path.GetFullPath(outputDirectory);
        Directory.CreateDirectory(directory);
        List<PredictionArtifacts> predictions = new List<PredictionArtifacts>(result.Segmentations.Count);
        for (int index = 0; index < result.Segmentations.Count; index++)
        {
            YoloSegmentationPrediction prediction = result.Segmentations[index];
            string stem = string.Create(
                CultureInfo.InvariantCulture,
                $"prediction-{index:D2}-class-{prediction.Detection.ClassIndex}-source-{prediction.Detection.SourceIndex}");
            MaskArtifact prototypeProbability = WriteProbabilityArtifact(
                directory,
                stem + "-prototype-probability.f32.bin",
                "prototype-grid-probability",
                prediction.Mask);

            YoloSegmentationSpatialTransformResult? transformed = null;
            MaskArtifact? sourceProbability = null;
            MaskArtifact? sourceThresholded = null;
            if (imagePreprocess != null && spatialTransform != null)
            {
                transformed = YoloSegmentationSpatialTransform.Apply(prediction, imagePreprocess, spatialTransform);
                sourceProbability = WriteProbabilityArtifact(
                    directory,
                    stem + "-source-probability.f32.bin",
                    "source-image-probability",
                    transformed.Mask);
                sourceThresholded = WriteThresholdedArtifact(
                    directory,
                    stem + "-source-thresholded.u8.bin",
                    transformed.Mask);
            }

            predictions.Add(new PredictionArtifacts(
                index,
                prediction,
                LabelOrIndex(labels, prediction.Detection.ClassIndex),
                prototypeProbability,
                transformed,
                sourceProbability,
                sourceThresholded));
        }

        string manifestPath = Path.Combine(directory, "segmentation-mask-artifacts.manifest.json");
        using (FileStream stream = File.Create(manifestPath))
        using (Utf8JsonWriter writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteString("schemaVersion", SchemaVersion);
            writer.WriteString("task", "seg");
            writer.WriteNumber("predictionCount", predictions.Count);
            writer.WriteBoolean("spatialTransformApplied", imagePreprocess != null && spatialTransform != null);
            writer.WriteString("probabilityDataType", "float32-little-endian");
            writer.WriteString("thresholdedDataType", "uint8-0-or-1");
            writer.WritePropertyName("predictions");
            writer.WriteStartArray();
            foreach (PredictionArtifacts prediction in predictions)
            {
                WritePrediction(writer, prediction);
            }
            writer.WriteEndArray();
            writer.WritePropertyName("boundary");
            writer.WriteStartObject();
            writer.WriteBoolean("isRuntimeProof", false);
            writer.WriteBoolean("isPackageConsumerProof", false);
            writer.WriteBoolean("isPostPublishProof", false);
            writer.WriteString(
                "evidenceKind",
                "deterministic source-tree mask artifacts for independent comparison; requires model/input/runtime/provenance records and strict validation before proof promotion");
            writer.WriteEndObject();
            writer.WriteEndObject();
        }

        return manifestPath;
    }

    private static void WritePrediction(Utf8JsonWriter writer, PredictionArtifacts prediction)
    {
        writer.WriteStartObject();
        writer.WriteNumber("index", prediction.Index);
        writer.WriteNumber("classId", prediction.Prediction.Detection.ClassIndex);
        writer.WriteString("className", prediction.ClassName);
        writer.WriteNumber("score", prediction.Prediction.Detection.Score);
        writer.WriteNumber("sourceIndex", prediction.Prediction.Detection.SourceIndex);
        writer.WritePropertyName("modelInputBox");
        WriteBox(writer, prediction.Prediction.Detection);
        writer.WritePropertyName("prototypeProbability");
        WriteArtifact(writer, prediction.PrototypeProbability);
        if (prediction.Transformed != null && prediction.SourceProbability != null && prediction.SourceThresholded != null)
        {
            writer.WritePropertyName("sourceBox");
            WriteBox(writer, prediction.Transformed.Detection);
            writer.WritePropertyName("sourceProbability");
            WriteArtifact(writer, prediction.SourceProbability);
            writer.WritePropertyName("sourceThresholded");
            WriteArtifact(writer, prediction.SourceThresholded);
            writer.WritePropertyName("spatialTransform");
            writer.WriteStartObject();
            writer.WriteString("coordinateSpace", prediction.Transformed.Options.CoordinateSpace == YoloSegmentationCoordinateSpace.Normalized ? "normalized" : "model-input-pixels");
            writer.WriteBoolean("cropToDetection", prediction.Transformed.Options.CropToDetection);
            writer.WriteString("interpolation", prediction.Transformed.Interpolation);
            writer.WriteString("scope", prediction.Transformed.Scope);
            writer.WriteString("boundary", prediction.Transformed.Boundary);
            writer.WriteEndObject();
        }
        writer.WriteEndObject();
    }

    private static void WriteBox(Utf8JsonWriter writer, YoloDetection detection)
    {
        writer.WriteStartObject();
        writer.WriteNumber("x", detection.CenterX);
        writer.WriteNumber("y", detection.CenterY);
        writer.WriteNumber("width", detection.Width);
        writer.WriteNumber("height", detection.Height);
        writer.WriteEndObject();
    }

    private static void WriteArtifact(Utf8JsonWriter writer, MaskArtifact artifact)
    {
        writer.WriteStartObject();
        writer.WriteString("role", artifact.Role);
        writer.WriteString("fileName", Path.GetFileName(artifact.Path));
        writer.WriteString("path", artifact.Path);
        writer.WritePropertyName("shape");
        writer.WriteStartArray();
        writer.WriteNumberValue(artifact.Height);
        writer.WriteNumberValue(artifact.Width);
        writer.WriteEndArray();
        writer.WriteNumber("elementCount", artifact.ElementCount);
        writer.WriteNumber("byteLength", artifact.ByteLength);
        writer.WriteString("sha256", artifact.Sha256);
        writer.WriteString("dataType", artifact.DataType);
        writer.WriteNumber("threshold", artifact.Threshold);
        writer.WriteNumber("activePixelCount", artifact.ActivePixelCount);
        writer.WriteEndObject();
    }

    private static MaskArtifact WriteProbabilityArtifact(string directory, string fileName, string role, YoloSegmentationMask mask)
    {
        string path = Path.Combine(directory, fileName);
        using (FileStream stream = File.Create(path))
        using (BinaryWriter writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false))
        {
            for (int index = 0; index < mask.Values.Length; index++)
            {
                writer.Write(mask.GetProbability(index));
            }
        }

        return CreateArtifact(path, role, "float32-little-endian", mask, mask.CountPixelsAtOrAboveThreshold());
    }

    private static MaskArtifact WriteThresholdedArtifact(string directory, string fileName, YoloSegmentationMask mask)
    {
        string path = Path.Combine(directory, fileName);
        byte[] values = new byte[mask.Values.Length];
        int activePixelCount = 0;
        for (int index = 0; index < values.Length; index++)
        {
            if (mask.GetProbability(index) >= mask.Threshold)
            {
                values[index] = 1;
                activePixelCount++;
            }
        }
        File.WriteAllBytes(path, values);
        return CreateArtifact(path, "source-image-thresholded", "uint8-0-or-1", mask, activePixelCount);
    }

    private static MaskArtifact CreateArtifact(
        string path,
        string role,
        string dataType,
        YoloSegmentationMask mask,
        int activePixelCount)
    {
        FileInfo file = new FileInfo(path);
        using FileStream stream = File.OpenRead(path);
        string sha256 = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        return new MaskArtifact(
            path,
            role,
            dataType,
            mask.Width,
            mask.Height,
            mask.Values.Length,
            file.Length,
            sha256,
            mask.Threshold,
            activePixelCount);
    }

    private static string LabelOrIndex(IReadOnlyList<string> labels, int classIndex)
    {
        return classIndex >= 0 && classIndex < labels.Count ? labels[classIndex] : classIndex.ToString(CultureInfo.InvariantCulture);
    }

    private sealed record MaskArtifact(
        string Path,
        string Role,
        string DataType,
        int Width,
        int Height,
        int ElementCount,
        long ByteLength,
        string Sha256,
        float Threshold,
        int ActivePixelCount);

    private sealed record PredictionArtifacts(
        int Index,
        YoloSegmentationPrediction Prediction,
        string ClassName,
        MaskArtifact PrototypeProbability,
        YoloSegmentationSpatialTransformResult? Transformed,
        MaskArtifact? SourceProbability,
        MaskArtifact? SourceThresholded);
}
