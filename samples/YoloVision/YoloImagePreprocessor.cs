using System;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;

namespace YoloVisionSample;

public sealed class YoloImagePreprocessResult
{
    public YoloImagePreprocessResult(
        string sourcePath,
        string sourceSha256,
        int sourceWidth,
        int sourceHeight,
        string tensorPath,
        string tensorSha256,
        int tensorElementCount,
        int targetWidth,
        int targetHeight,
        string tensorLayout,
        string colorOrder,
        string resizeMode,
        bool normalized,
        float scale,
        bool letterboxEnabled,
        int resizedWidth,
        int resizedHeight,
        int padX,
        int padY,
        float resizeScaleX,
        float resizeScaleY,
        byte fillValue)
    {
        SourcePath = sourcePath ?? string.Empty;
        SourceSha256 = sourceSha256 ?? string.Empty;
        SourceWidth = sourceWidth;
        SourceHeight = sourceHeight;
        TensorPath = tensorPath ?? string.Empty;
        TensorSha256 = tensorSha256 ?? string.Empty;
        TensorElementCount = tensorElementCount;
        TargetWidth = targetWidth;
        TargetHeight = targetHeight;
        TensorLayout = tensorLayout ?? string.Empty;
        ColorOrder = colorOrder ?? string.Empty;
        ResizeMode = resizeMode ?? string.Empty;
        Normalized = normalized;
        Scale = scale;
        LetterboxEnabled = letterboxEnabled;
        ResizedWidth = resizedWidth;
        ResizedHeight = resizedHeight;
        PadX = padX;
        PadY = padY;
        ResizeScaleX = resizeScaleX;
        ResizeScaleY = resizeScaleY;
        FillValue = fillValue;
    }

    public string SourcePath { get; }

    public string SourceSha256 { get; }

    public int SourceWidth { get; }

    public int SourceHeight { get; }

    public string TensorPath { get; }

    public string TensorSha256 { get; }

    public int TensorElementCount { get; }

    public int TargetWidth { get; }

    public int TargetHeight { get; }

    public string TensorLayout { get; }

    public string ColorOrder { get; }

    public string ResizeMode { get; }

    public bool Normalized { get; }

    public float Scale { get; }

    public bool LetterboxEnabled { get; }

    public int ResizedWidth { get; }

    public int ResizedHeight { get; }

    public int PadX { get; }

    public int PadY { get; }

    public float ResizeScaleX { get; }

    public float ResizeScaleY { get; }

    public byte FillValue { get; }
}

public static class YoloImagePreprocessor
{
    private const byte DefaultLetterboxFill = 114;

    public static YoloImagePreprocessResult Preprocess(
        string imagePath,
        string tensorPath,
        int[] inputShape,
        YoloPreprocessOptions options)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            throw new ArgumentException("Image path must not be empty.", nameof(imagePath));
        }

        if (string.IsNullOrWhiteSpace(tensorPath))
        {
            throw new ArgumentException("Preprocessed tensor output path must not be empty.", nameof(tensorPath));
        }

        if (inputShape == null)
        {
            throw new ArgumentNullException(nameof(inputShape));
        }

        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        string fullImagePath = Path.GetFullPath(imagePath);
        if (!File.Exists(fullImagePath))
        {
            throw new FileNotFoundException("Input image file was not found.", fullImagePath);
        }

        string layout = NormalizeLayout(options.TensorLayout);
        string colorOrder = NormalizeColorOrder(options.ColorOrder);
        ResolveInputShape(inputShape, layout, out int channelCount, out int targetHeight, out int targetWidth);
        if (channelCount != 3)
        {
            throw new ArgumentException("YoloVision image preprocessing currently supports 3-channel tensors only.");
        }

        RgbImage image = DecodeRgbImage(fullImagePath);
        bool letterbox = options.PreserveAspectRatio && !string.Equals(options.ResizeMode, "stretch", StringComparison.OrdinalIgnoreCase);
        ResizePlan plan = CreateResizePlan(image.Width, image.Height, targetWidth, targetHeight, letterbox);
        byte[] targetPixels = ResizeToTarget(image, plan, DefaultLetterboxFill);
        float[] tensor = ToTensor(targetPixels, targetWidth, targetHeight, layout, colorOrder, options.Normalize, options.Scale);

        string fullTensorPath = Path.GetFullPath(tensorPath);
        string? directory = Path.GetDirectoryName(fullTensorPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        WriteFloat32Tensor(fullTensorPath, tensor);
        return new YoloImagePreprocessResult(
            fullImagePath,
            ComputeFileSha256(fullImagePath),
            image.Width,
            image.Height,
            fullTensorPath,
            ComputeFileSha256(fullTensorPath),
            tensor.Length,
            targetWidth,
            targetHeight,
            layout,
            colorOrder,
            letterbox ? "letterbox" : "stretch",
            options.Normalize,
            options.Scale,
            letterbox,
            plan.ResizedWidth,
            plan.ResizedHeight,
            plan.PadX,
            plan.PadY,
            plan.ResizeScaleX,
            plan.ResizeScaleY,
            DefaultLetterboxFill);
    }

    private static RgbImage DecodeRgbImage(string path)
    {
        string extension = Path.GetExtension(path);
        if (string.Equals(extension, ".ppm", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(extension, ".pnm", StringComparison.OrdinalIgnoreCase))
        {
            return DecodePpm(path);
        }

        if (string.Equals(extension, ".bmp", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(extension, ".dib", StringComparison.OrdinalIgnoreCase))
        {
            return DecodeBmp(path);
        }

        throw new NotSupportedException("YoloVision --image currently decodes uncompressed .bmp and .ppm/.pnm files. Use --input-data for externally preprocessed JPG/PNG assets.");
    }

    private static RgbImage DecodeBmp(string path)
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
        int rowStride = ((width * bitsPerPixel + 31) / 32) * 4;
        if (pixelOffset < 0 || pixelOffset + rowStride * height > bytes.Length)
        {
            throw new ArgumentException("BMP pixel data is truncated.");
        }

        byte[] rgb = new byte[width * height * 3];
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

        return new RgbImage(width, height, rgb);
    }

    private static RgbImage DecodePpm(string path)
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
        if (maxValue <= 0 || maxValue > 255)
        {
            throw new ArgumentException("PPM decoder supports max value in the range 1..255.");
        }

        byte[] rgb = new byte[width * height * 3];
        if (binary)
        {
            reader.SkipSingleWhitespace();
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

        return new RgbImage(width, height, rgb);
    }

    private static void ScalePpmValues(byte[] rgb, int maxValue)
    {
        for (int index = 0; index < rgb.Length; index++)
        {
            rgb[index] = (byte)MathF.Round(rgb[index] * 255.0f / maxValue);
        }
    }

    private static ResizePlan CreateResizePlan(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight, bool letterbox)
    {
        if (!letterbox)
        {
            return new ResizePlan(targetWidth, targetHeight, targetWidth, targetHeight, 0, 0, targetWidth / (float)sourceWidth, targetHeight / (float)sourceHeight);
        }

        float scale = MathF.Min(targetWidth / (float)sourceWidth, targetHeight / (float)sourceHeight);
        int resizedWidth = Math.Max(1, Math.Min(targetWidth, (int)MathF.Round(sourceWidth * scale)));
        int resizedHeight = Math.Max(1, Math.Min(targetHeight, (int)MathF.Round(sourceHeight * scale)));
        int padX = (targetWidth - resizedWidth) / 2;
        int padY = (targetHeight - resizedHeight) / 2;
        return new ResizePlan(targetWidth, targetHeight, resizedWidth, resizedHeight, padX, padY, scale, scale);
    }

    private static byte[] ResizeToTarget(RgbImage image, ResizePlan plan, byte fillValue)
    {
        int targetWidth = plan.TargetWidth;
        int targetHeight = plan.TargetHeight;
        byte[] target = new byte[targetWidth * targetHeight * 3];
        Array.Fill(target, fillValue);

        for (int y = 0; y < plan.ResizedHeight; y++)
        {
            float sourceY = MapCoordinate(y, plan.ResizedHeight, image.Height);
            int y0 = Math.Clamp((int)MathF.Floor(sourceY), 0, image.Height - 1);
            int y1 = Math.Clamp(y0 + 1, 0, image.Height - 1);
            float yWeight = sourceY - y0;

            for (int x = 0; x < plan.ResizedWidth; x++)
            {
                float sourceX = MapCoordinate(x, plan.ResizedWidth, image.Width);
                int x0 = Math.Clamp((int)MathF.Floor(sourceX), 0, image.Width - 1);
                int x1 = Math.Clamp(x0 + 1, 0, image.Width - 1);
                float xWeight = sourceX - x0;
                int targetIndex = ((y + plan.PadY) * targetWidth + (x + plan.PadX)) * 3;

                for (int channel = 0; channel < 3; channel++)
                {
                    float c00 = image.Pixels[(y0 * image.Width + x0) * 3 + channel];
                    float c01 = image.Pixels[(y0 * image.Width + x1) * 3 + channel];
                    float c10 = image.Pixels[(y1 * image.Width + x0) * 3 + channel];
                    float c11 = image.Pixels[(y1 * image.Width + x1) * 3 + channel];
                    float top = c00 + (c01 - c00) * xWeight;
                    float bottom = c10 + (c11 - c10) * xWeight;
                    target[targetIndex + channel] = (byte)Math.Clamp((int)MathF.Round(top + (bottom - top) * yWeight), 0, 255);
                }
            }
        }

        return target;
    }

    private static float MapCoordinate(int targetIndex, int targetLength, int sourceLength)
    {
        if (targetLength == sourceLength)
        {
            return targetIndex;
        }

        return Math.Clamp((targetIndex + 0.5f) * sourceLength / targetLength - 0.5f, 0.0f, sourceLength - 1.0f);
    }

    private static float[] ToTensor(byte[] rgbPixels, int width, int height, string layout, string colorOrder, bool normalize, float scale)
    {
        float[] tensor = new float[width * height * 3];
        bool bgr = string.Equals(colorOrder, "BGR", StringComparison.Ordinal);
        bool nchw = string.Equals(layout, "NCHW", StringComparison.Ordinal);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int pixelIndex = (y * width + x) * 3;
                byte first = rgbPixels[pixelIndex + (bgr ? 2 : 0)];
                byte second = rgbPixels[pixelIndex + 1];
                byte third = rgbPixels[pixelIndex + (bgr ? 0 : 2)];
                if (nchw)
                {
                    int planeOffset = y * width + x;
                    tensor[planeOffset] = ToFloat(first, normalize, scale);
                    tensor[width * height + planeOffset] = ToFloat(second, normalize, scale);
                    tensor[width * height * 2 + planeOffset] = ToFloat(third, normalize, scale);
                }
                else
                {
                    int targetIndex = (y * width + x) * 3;
                    tensor[targetIndex] = ToFloat(first, normalize, scale);
                    tensor[targetIndex + 1] = ToFloat(second, normalize, scale);
                    tensor[targetIndex + 2] = ToFloat(third, normalize, scale);
                }
            }
        }

        return tensor;
    }

    private static float ToFloat(byte value, bool normalize, float scale)
    {
        return normalize ? value * scale : value;
    }

    private static void WriteFloat32Tensor(string path, float[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    private static void ResolveInputShape(int[] shape, string layout, out int channels, out int height, out int width)
    {
        if (shape.Length != 4 || shape[0] != 1)
        {
            throw new ArgumentException("YoloVision --image requires a 4D batch-1 input shape.");
        }

        if (string.Equals(layout, "NCHW", StringComparison.Ordinal))
        {
            channels = shape[1];
            height = shape[2];
            width = shape[3];
            return;
        }

        channels = shape[3];
        height = shape[1];
        width = shape[2];
    }

    private static string NormalizeLayout(string value)
    {
        string normalized = (value ?? string.Empty).Trim().Replace("-", string.Empty, StringComparison.Ordinal).Replace("_", string.Empty, StringComparison.Ordinal).ToUpperInvariant();
        return normalized switch
        {
            "" or "NCHW" or "CHANNELSFIRST" => "NCHW",
            "NHWC" or "CHANNELSLAST" => "NHWC",
            _ => throw new ArgumentException($"Unsupported tensor layout '{value}'. Use NCHW or NHWC.")
        };
    }

    private static string NormalizeColorOrder(string value)
    {
        string normalized = (value ?? string.Empty).Trim().ToUpperInvariant();
        return normalized switch
        {
            "" or "RGB" => "RGB",
            "BGR" => "BGR",
            _ => throw new ArgumentException($"Unsupported color order '{value}'. Use RGB or BGR.")
        };
    }

    private static string ComputeFileSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static int ReadInt32LittleEndian(byte[] bytes, int offset)
    {
        return bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24);
    }

    private static short ReadInt16LittleEndian(byte[] bytes, int offset)
    {
        return (short)(bytes[offset] | (bytes[offset + 1] << 8));
    }

    private readonly struct RgbImage
    {
        public RgbImage(int width, int height, byte[] pixels)
        {
            Width = width;
            Height = height;
            Pixels = pixels;
        }

        public int Width { get; }

        public int Height { get; }

        public byte[] Pixels { get; }
    }

    private readonly struct ResizePlan
    {
        public ResizePlan(int targetWidth, int targetHeight, int resizedWidth, int resizedHeight, int padX, int padY, float resizeScaleX, float resizeScaleY)
        {
            TargetWidth = targetWidth;
            TargetHeight = targetHeight;
            ResizedWidth = resizedWidth;
            ResizedHeight = resizedHeight;
            PadX = padX;
            PadY = padY;
            ResizeScaleX = resizeScaleX;
            ResizeScaleY = resizeScaleY;
        }

        public int TargetWidth { get; }

        public int TargetHeight { get; }

        public int ResizedWidth { get; }

        public int ResizedHeight { get; }

        public int PadX { get; }

        public int PadY { get; }

        public float ResizeScaleX { get; }

        public float ResizeScaleY { get; }
    }

    private sealed class PpmReader
    {
        private readonly byte[] _bytes;

        public PpmReader(byte[] bytes)
        {
            _bytes = bytes;
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

        public void SkipSingleWhitespace()
        {
            if (Position < _bytes.Length && char.IsWhiteSpace((char)_bytes[Position]))
            {
                Position++;
            }
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
