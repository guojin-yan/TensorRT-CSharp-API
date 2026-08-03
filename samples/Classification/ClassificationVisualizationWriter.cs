using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace ClassificationSample;

public static class ClassificationVisualizationWriter
{
    public static void Write(
        string outputPath,
        string backgroundImagePath,
        ClassificationImagePreprocessResult preprocess,
        IReadOnlyList<ClassificationPrediction> predictions)
    {
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Visualization path must not be empty.", nameof(outputPath));
        }
        if (string.IsNullOrWhiteSpace(backgroundImagePath))
        {
            throw new ArgumentException("Visualization background path must not be empty.", nameof(backgroundImagePath));
        }
        if (preprocess == null)
        {
            throw new ArgumentNullException(nameof(preprocess));
        }
        if (predictions == null || predictions.Count == 0)
        {
            throw new ArgumentException("Visualization requires at least one classification prediction.", nameof(predictions));
        }

        string fullBackgroundPath = Path.GetFullPath(backgroundImagePath);
        if (!File.Exists(fullBackgroundPath))
        {
            throw new FileNotFoundException("Visualization background image was not found.", fullBackgroundPath);
        }

        (int width, int height) = ReadImageDimensions(fullBackgroundPath);
        if (width != preprocess.SourceWidth || height != preprocess.SourceHeight)
        {
            throw new ArgumentException(
                $"Visualization background dimensions {width}x{height} do not match the preprocessed source image {preprocess.SourceWidth}x{preprocess.SourceHeight}.",
                nameof(backgroundImagePath));
        }

        string fullOutputPath = Path.GetFullPath(outputPath);
        string? outputDirectory = Path.GetDirectoryName(fullOutputPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        File.WriteAllText(
            fullOutputPath,
            ToSvg(fullBackgroundPath, width, height, predictions),
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
    }

    private static string ToSvg(
        string backgroundImagePath,
        int sourceWidth,
        int sourceHeight,
        IReadOnlyList<ClassificationPrediction> predictions)
    {
        const int contentTop = 72;
        int canvasWidth = Math.Max(sourceWidth, 360);
        int canvasHeight = sourceHeight + contentTop;
        int count = Math.Min(predictions.Count, 5);
        int panelWidth = Math.Min(520, Math.Max(320, sourceWidth - 32));
        int panelHeight = 62 + count * 42;
        string mimeType = ResolveImageMimeType(backgroundImagePath);
        string imageData = Convert.ToBase64String(File.ReadAllBytes(backgroundImagePath));

        var builder = new StringBuilder();
        builder.AppendLine($"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{canvasWidth}\" height=\"{canvasHeight}\" viewBox=\"0 0 {canvasWidth} {canvasHeight}\" role=\"img\" aria-label=\"Classification result\">");
        builder.AppendLine("  <rect width=\"100%\" height=\"100%\" fill=\"#f8fafc\"/>");
        builder.AppendLine("  <text x=\"16\" y=\"30\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"20\" font-weight=\"700\" fill=\"#111827\">TensorRtSharp Classification</text>");
        builder.AppendLine($"  <text x=\"16\" y=\"54\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"13\" fill=\"#475569\">Top-{count} predictions from the real TensorRT execution</text>");
        builder.AppendLine($"  <image data-source-image=\"true\" x=\"0\" y=\"{contentTop}\" width=\"{sourceWidth}\" height=\"{sourceHeight}\" preserveAspectRatio=\"none\" href=\"data:{mimeType};base64,{imageData}\"/>");
        builder.AppendLine($"  <rect x=\"16\" y=\"{contentTop + 16}\" width=\"{panelWidth}\" height=\"{panelHeight}\" rx=\"6\" fill=\"#111827\" opacity=\"0.88\"/>");
        builder.AppendLine($"  <text x=\"36\" y=\"{contentTop + 48}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"18\" font-weight=\"700\" fill=\"#ffffff\">Top predictions</text>");

        for (int index = 0; index < count; index++)
        {
            ClassificationPrediction prediction = predictions[index];
            int rowY = contentTop + 72 + index * 42;
            int barWidth = (int)Math.Round(Math.Clamp(prediction.Score, 0.0f, 1.0f) * (panelWidth - 44));
            string rankAndLabel = $"{index + 1}. {prediction.Label}";
            string score = prediction.Score.ToString("0.0000", CultureInfo.InvariantCulture);
            builder.AppendLine($"  <rect x=\"36\" y=\"{rowY}\" width=\"{panelWidth - 40}\" height=\"28\" rx=\"3\" fill=\"#374151\"/>");
            if (barWidth > 0)
            {
                builder.AppendLine($"  <rect x=\"36\" y=\"{rowY}\" width=\"{barWidth}\" height=\"28\" rx=\"3\" fill=\"#0ea5e9\" opacity=\"0.72\"/>");
            }
            builder.AppendLine($"  <text x=\"46\" y=\"{rowY + 19}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"13\" fill=\"#ffffff\">{Escape(rankAndLabel)}</text>");
            builder.AppendLine($"  <text x=\"{panelWidth - 58}\" y=\"{rowY + 19}\" text-anchor=\"end\" font-family=\"Consolas, monospace\" font-size=\"13\" fill=\"#ffffff\">{score}</text>");
        }

        builder.AppendLine("</svg>");
        return builder.ToString();
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

        return ValidateImageDimensions(ReadBigEndianInt32(header, 16), ReadBigEndianInt32(header, 20));
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

        return ValidateImageDimensions(width, Math.Abs(rawHeight));
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
        return (reader.ReadByte() << 8) | reader.ReadByte();
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

    private static string Escape(string value)
    {
        return (value ?? string.Empty)
            .Replace("&", "&amp;", StringComparison.Ordinal)
            .Replace("<", "&lt;", StringComparison.Ordinal)
            .Replace(">", "&gt;", StringComparison.Ordinal)
            .Replace("\"", "&quot;", StringComparison.Ordinal)
            .Replace("'", "&apos;", StringComparison.Ordinal);
    }
}
