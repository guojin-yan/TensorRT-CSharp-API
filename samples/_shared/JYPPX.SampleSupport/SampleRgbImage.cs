using System;
using System.Globalization;
using System.IO;

namespace JYPPX.SampleSupport;

internal sealed class SampleRgbImage
{
    public SampleRgbImage(int width, int height, byte[] pixels)
    {
        if (width <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width));
        }
        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }
        if (pixels == null || pixels.Length != checked(width * height * 3))
        {
            throw new ArgumentException("RGB pixels must contain exactly width * height * 3 bytes.", nameof(pixels));
        }

        Width = width;
        Height = height;
        Pixels = (byte[])pixels.Clone();
    }

    public int Width { get; }

    public int Height { get; }

    public byte[] Pixels { get; }
}

internal static class SampleRgbImageDecoder
{
    public static SampleRgbImage Decode(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Image path must not be empty.", nameof(path));
        }

        string fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Input image file was not found.", fullPath);
        }

        string extension = Path.GetExtension(fullPath);
        if (string.Equals(extension, ".ppm", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(extension, ".pnm", StringComparison.OrdinalIgnoreCase))
        {
            return DecodePpm(fullPath);
        }
        if (string.Equals(extension, ".bmp", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(extension, ".dib", StringComparison.OrdinalIgnoreCase))
        {
            return DecodeBmp(fullPath);
        }

        throw new NotSupportedException("Image decoding supports uncompressed .bmp and .ppm/.pnm files. Use --input-data for externally preprocessed JPG/PNG assets.");
    }

    private static SampleRgbImage DecodeBmp(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length < 54 || bytes[0] != (byte)'B' || bytes[1] != (byte)'M')
        {
            throw new ArgumentException("BMP image is not a valid Windows bitmap file.");
        }

        int pixelOffset = ReadInt32LittleEndian(bytes, 10);
        int dibSize = ReadInt32LittleEndian(bytes, 14);
        if (dibSize < 40 || bytes.Length < 14 + dibSize)
        {
            throw new ArgumentException("BMP image uses an unsupported DIB header.");
        }

        int width = ReadInt32LittleEndian(bytes, 18);
        int signedHeight = ReadInt32LittleEndian(bytes, 22);
        short planes = ReadInt16LittleEndian(bytes, 26);
        short bitsPerPixel = ReadInt16LittleEndian(bytes, 28);
        int compression = ReadInt32LittleEndian(bytes, 30);
        if (width <= 0 || signedHeight == 0 || planes != 1 || compression != 0 || (bitsPerPixel != 24 && bitsPerPixel != 32))
        {
            throw new ArgumentException("BMP decoder supports only uncompressed 24-bit or 32-bit RGB bitmaps.");
        }

        int height = Math.Abs(signedHeight);
        bool topDown = signedHeight < 0;
        int rowStride = checked(((width * bitsPerPixel + 31) / 32) * 4);
        if (pixelOffset < 0 || (long)pixelOffset + (long)rowStride * height > bytes.Length)
        {
            throw new ArgumentException("BMP pixel data is truncated.");
        }

        byte[] rgb = new byte[checked(width * height * 3)];
        int bytesPerPixel = bitsPerPixel / 8;
        for (int y = 0; y < height; y++)
        {
            int sourceY = topDown ? y : height - 1 - y;
            int sourceRow = pixelOffset + sourceY * rowStride;
            for (int x = 0; x < width; x++)
            {
                int sourceIndex = sourceRow + x * bytesPerPixel;
                int targetIndex = (y * width + x) * 3;
                rgb[targetIndex] = bytes[sourceIndex + 2];
                rgb[targetIndex + 1] = bytes[sourceIndex + 1];
                rgb[targetIndex + 2] = bytes[sourceIndex];
            }
        }

        return new SampleRgbImage(width, height, rgb);
    }

    private static SampleRgbImage DecodePpm(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        PpmReader reader = new PpmReader(bytes);
        string magic = reader.ReadToken();
        bool binary = string.Equals(magic, "P6", StringComparison.Ordinal);
        bool ascii = string.Equals(magic, "P3", StringComparison.Ordinal);
        if (!binary && !ascii)
        {
            throw new ArgumentException("PPM decoder supports P6 binary and P3 ASCII images only.");
        }

        int width = reader.ReadPositiveInt("PPM width");
        int height = reader.ReadPositiveInt("PPM height");
        int maxValue = reader.ReadPositiveInt("PPM max value");
        if (maxValue > 255)
        {
            throw new ArgumentException("PPM decoder supports max value in the range 1..255.");
        }

        byte[] rgb = new byte[checked(width * height * 3)];
        if (binary)
        {
            reader.SkipRasterSeparator();
            if (reader.Position + rgb.Length > bytes.Length)
            {
                throw new ArgumentException("PPM pixel data is truncated.");
            }

            Array.Copy(bytes, reader.Position, rgb, 0, rgb.Length);
            if (maxValue != 255)
            {
                ScalePpmValues(rgb, maxValue);
            }
        }
        else
        {
            for (int index = 0; index < rgb.Length; index++)
            {
                int value = reader.ReadNonNegativeInt("PPM channel");
                if (value > maxValue)
                {
                    throw new ArgumentException("PPM channel value exceeds max value.");
                }

                rgb[index] = (byte)MathF.Round(value * 255.0f / maxValue);
            }
        }

        return new SampleRgbImage(width, height, rgb);
    }

    private static void ScalePpmValues(byte[] rgb, int maxValue)
    {
        for (int index = 0; index < rgb.Length; index++)
        {
            if (rgb[index] > maxValue)
            {
                throw new ArgumentException("PPM channel value exceeds max value.");
            }
            rgb[index] = (byte)MathF.Round(rgb[index] * 255.0f / maxValue);
        }
    }

    private static int ReadInt32LittleEndian(byte[] bytes, int offset)
    {
        return bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24);
    }

    private static short ReadInt16LittleEndian(byte[] bytes, int offset)
    {
        return (short)(bytes[offset] | (bytes[offset + 1] << 8));
    }

    private sealed class PpmReader
    {
        private readonly byte[] _bytes;

        public PpmReader(byte[] bytes)
        {
            _bytes = bytes ?? throw new ArgumentNullException(nameof(bytes));
        }

        public int Position { get; private set; }

        public string ReadToken()
        {
            SkipWhitespaceAndComments();
            int start = Position;
            while (Position < _bytes.Length && !char.IsWhiteSpace((char)_bytes[Position]))
            {
                Position++;
            }
            if (Position == start)
            {
                throw new ArgumentException("Unexpected end of PPM header.");
            }

            return System.Text.Encoding.ASCII.GetString(_bytes, start, Position - start);
        }

        public int ReadPositiveInt(string name)
        {
            int value = ReadNonNegativeInt(name);
            if (value <= 0)
            {
                throw new ArgumentException($"{name} must be positive.");
            }

            return value;
        }

        public int ReadNonNegativeInt(string name)
        {
            string token = ReadToken();
            if (!int.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) || value < 0)
            {
                throw new ArgumentException($"{name} must be a non-negative integer.");
            }

            return value;
        }

        public void SkipRasterSeparator()
        {
            if (Position >= _bytes.Length || !char.IsWhiteSpace((char)_bytes[Position]))
            {
                throw new ArgumentException("PPM binary header must end with whitespace before pixel data.");
            }

            bool isCrLf = _bytes[Position] == (byte)'\r' && Position + 1 < _bytes.Length && _bytes[Position + 1] == (byte)'\n';
            Position += isCrLf ? 2 : 1;
        }

        private void SkipWhitespaceAndComments()
        {
            while (Position < _bytes.Length)
            {
                byte current = _bytes[Position];
                if (char.IsWhiteSpace((char)current))
                {
                    Position++;
                    continue;
                }
                if (current == (byte)'#')
                {
                    while (Position < _bytes.Length && _bytes[Position] != (byte)'\n')
                    {
                        Position++;
                    }
                    continue;
                }

                break;
            }
        }
    }
}
