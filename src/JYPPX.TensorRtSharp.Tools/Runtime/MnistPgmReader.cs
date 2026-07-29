using System;
using System.Globalization;
using System.IO;
using System.Text;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Reads and preprocesses the P5 PGM assets shipped with TensorRT MNIST samples.
/// 读取并预处理 TensorRT MNIST 样例附带的 P5 PGM 资产。
/// </summary>
public static class MnistPgmReader
{
    public static MnistPgmImage Read(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("PGM path must not be empty.", nameof(path));
        }

        byte[] bytes = File.ReadAllBytes(path);
        int offset = 0;
        string magic = ReadToken(bytes, ref offset);
        if (!string.Equals(magic, "P5", StringComparison.Ordinal))
        {
            throw new InvalidDataException("MNIST input must be a binary P5 PGM file.");
        }

        int width = ParsePositiveInt(ReadToken(bytes, ref offset), "width");
        int height = ParsePositiveInt(ReadToken(bytes, ref offset), "height");
        int maxValue = ParsePositiveInt(ReadToken(bytes, ref offset), "max value");
        if (maxValue != 255)
        {
            throw new InvalidDataException("MNIST PGM max value must be 255.");
        }

        ConsumePixelSeparator(bytes, ref offset);
        int pixelCount = checked(width * height);
        if (bytes.Length - offset != pixelCount)
        {
            throw new InvalidDataException(
                $"PGM pixel payload length mismatch. Expected={pixelCount} Actual={bytes.Length - offset}.");
        }

        byte[] pixels = new byte[pixelCount];
        Buffer.BlockCopy(bytes, offset, pixels, 0, pixelCount);
        return new MnistPgmImage(width, height, maxValue, pixels);
    }

    public static float[] ToTensorInput(MnistPgmImage image)
    {
        if (image == null)
        {
            throw new ArgumentNullException(nameof(image));
        }

        float[] values = new float[image.Pixels.Length];
        for (int index = 0; index < image.Pixels.Length; index++)
        {
            values[index] = 1.0f - (image.Pixels[index] / 255.0f);
        }

        return values;
    }

    private static string ReadToken(byte[] bytes, ref int offset)
    {
        SkipWhitespaceAndComments(bytes, ref offset);
        if (offset >= bytes.Length)
        {
            throw new InvalidDataException("Unexpected end of PGM header.");
        }

        int start = offset;
        while (offset < bytes.Length && !IsWhitespace(bytes[offset]) && bytes[offset] != (byte)'#')
        {
            offset++;
        }

        if (offset == start)
        {
            throw new InvalidDataException("PGM header token is empty.");
        }

        return Encoding.ASCII.GetString(bytes, start, offset - start);
    }

    private static void SkipWhitespaceAndComments(byte[] bytes, ref int offset)
    {
        while (offset < bytes.Length)
        {
            while (offset < bytes.Length && IsWhitespace(bytes[offset]))
            {
                offset++;
            }

            if (offset >= bytes.Length || bytes[offset] != (byte)'#')
            {
                return;
            }

            while (offset < bytes.Length && bytes[offset] != (byte)'\n')
            {
                offset++;
            }
        }
    }

    private static void ConsumePixelSeparator(byte[] bytes, ref int offset)
    {
        if (offset >= bytes.Length || !IsWhitespace(bytes[offset]))
        {
            throw new InvalidDataException("PGM header must be followed by a whitespace separator.");
        }

        byte first = bytes[offset++];
        if (first == (byte)'\r' && offset < bytes.Length && bytes[offset] == (byte)'\n')
        {
            offset++;
        }
    }

    private static bool IsWhitespace(byte value)
    {
        return value == (byte)' ' ||
            value == (byte)'\t' ||
            value == (byte)'\r' ||
            value == (byte)'\n' ||
            value == (byte)'\f';
    }

    private static int ParsePositiveInt(string value, string fieldName)
    {
        if (!int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int parsed) || parsed <= 0)
        {
            throw new InvalidDataException($"PGM {fieldName} is invalid.");
        }

        return parsed;
    }
}
