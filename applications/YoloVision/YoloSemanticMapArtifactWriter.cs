using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace YoloVisionSample;

public static class YoloSemanticMapArtifactWriter
{
    public const string SchemaVersion = "yolovision-semantic-map-artifacts.v1";

    public static string Write(
        string outputDirectory,
        YoloVisionResult result,
        IReadOnlyList<string> labels)
    {
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            throw new ArgumentException("Semantic map output directory is required.", nameof(outputDirectory));
        }
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }
        if (labels == null)
        {
            throw new ArgumentNullException(nameof(labels));
        }
        if (result.TaskType != YoloTaskType.SemanticSegmentation || result.SemanticMap == null)
        {
            throw new ArgumentException("Semantic map artifacts require a semantic segmentation result.", nameof(result));
        }

        string directory = Path.GetFullPath(outputDirectory);
        Directory.CreateDirectory(directory);
        YoloSemanticMap map = result.SemanticMap;
        int[] classIndices = map.GetClassIndexMap();
        int[] histogram = map.GetClassHistogram();
        string classIndexPath = Path.Combine(directory, "semantic-class-index.i32.bin");
        WriteInt32LittleEndian(classIndexPath, classIndices);
        FileInfo artifact = new FileInfo(classIndexPath);
        string artifactSha256 = ComputeSha256(classIndexPath);
        int dominantClassId = GetDominantClassId(histogram);

        string manifestPath = Path.Combine(directory, "semantic-map-artifacts.manifest.json");
        using (FileStream stream = File.Create(manifestPath))
        using (Utf8JsonWriter writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteString("schemaVersion", SchemaVersion);
            writer.WriteString("task", "sem");
            writer.WriteNumber("classCount", map.ClassCount);
            writer.WriteNumber("width", map.Width);
            writer.WriteNumber("height", map.Height);
            writer.WriteNumber("pixelCount", classIndices.Length);
            writer.WriteNumber("dominantClassId", dominantClassId);
            writer.WriteString("dominantClassName", LabelOrIndex(labels, dominantClassId));
            writer.WritePropertyName("classIndexArtifact");
            writer.WriteStartObject();
            writer.WriteString("role", "semantic-class-index-map");
            writer.WriteString("fileName", Path.GetFileName(classIndexPath));
            writer.WriteString("path", classIndexPath);
            writer.WritePropertyName("shape");
            writer.WriteStartArray();
            writer.WriteNumberValue(map.Height);
            writer.WriteNumberValue(map.Width);
            writer.WriteEndArray();
            writer.WriteNumber("elementCount", classIndices.Length);
            writer.WriteNumber("byteLength", artifact.Length);
            writer.WriteString("sha256", artifactSha256);
            writer.WriteString("dataType", "int32-little-endian");
            writer.WriteString("layout", "row-major-hw");
            writer.WriteEndObject();
            writer.WritePropertyName("classHistogram");
            WriteHistogram(writer, histogram, labels);
            writer.WritePropertyName("boundary");
            writer.WriteStartObject();
            writer.WriteBoolean("isRuntimeProof", false);
            writer.WriteBoolean("isPackageConsumerProof", false);
            writer.WriteBoolean("isPostPublishProof", false);
            writer.WriteString(
                "evidenceKind",
                "deterministic full-resolution semantic argmax artifact; runtime promotion also requires model, input, raw output, runtime, and reference-comparison provenance");
            writer.WriteEndObject();
            writer.WriteEndObject();
        }

        return manifestPath;
    }

    internal static void WriteHistogram(
        Utf8JsonWriter writer,
        int[] histogram,
        IReadOnlyList<string> labels)
    {
        writer.WriteStartArray();
        for (int classId = 0; classId < histogram.Length; classId++)
        {
            writer.WriteStartObject();
            writer.WriteNumber("classId", classId);
            writer.WriteString("className", LabelOrIndex(labels, classId));
            writer.WriteNumber("pixelCount", histogram[classId]);
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
    }

    private static void WriteInt32LittleEndian(string path, int[] values)
    {
        using FileStream stream = File.Create(path);
        using BinaryWriter writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false);
        foreach (int value in values)
        {
            writer.Write(value);
        }
    }

    private static int GetDominantClassId(int[] histogram)
    {
        int dominantClassId = 0;
        for (int classId = 1; classId < histogram.Length; classId++)
        {
            if (histogram[classId] > histogram[dominantClassId])
            {
                dominantClassId = classId;
            }
        }
        return dominantClassId;
    }

    private static string LabelOrIndex(IReadOnlyList<string> labels, int classIndex)
    {
        return classIndex >= 0 && classIndex < labels.Count
            ? labels[classIndex]
            : classIndex.ToString(CultureInfo.InvariantCulture);
    }

    private static string ComputeSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }
}
