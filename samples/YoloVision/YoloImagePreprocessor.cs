using System;
using System.IO;
using System.Security.Cryptography;
using JYPPX.SampleSupport;

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
        string letterboxAlignment,
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
        LetterboxAlignment = letterboxAlignment ?? string.Empty;
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

    public string LetterboxAlignment { get; }

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
        string letterboxAlignment = NormalizeLetterboxAlignment(options.LetterboxAlignment);
        ResolveInputShape(inputShape, layout, out int channelCount, out int targetHeight, out int targetWidth);
        if (channelCount != 3)
        {
            throw new ArgumentException("YoloVision image preprocessing currently supports 3-channel tensors only.");
        }

        SampleRgbImage image = SampleRgbImageDecoder.Decode(fullImagePath);
        bool letterbox = options.PreserveAspectRatio && !string.Equals(options.ResizeMode, "stretch", StringComparison.OrdinalIgnoreCase);
        ResizePlan plan = CreateResizePlan(image.Width, image.Height, targetWidth, targetHeight, letterbox, letterboxAlignment);
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
            letterboxAlignment,
            plan.ResizedWidth,
            plan.ResizedHeight,
            plan.PadX,
            plan.PadY,
            plan.ResizeScaleX,
            plan.ResizeScaleY,
            DefaultLetterboxFill);
    }

    private static ResizePlan CreateResizePlan(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight, bool letterbox, string letterboxAlignment)
    {
        if (!letterbox)
        {
            return new ResizePlan(targetWidth, targetHeight, targetWidth, targetHeight, 0, 0, targetWidth / (float)sourceWidth, targetHeight / (float)sourceHeight);
        }

        float scale = MathF.Min(targetWidth / (float)sourceWidth, targetHeight / (float)sourceHeight);
        bool topLeft = string.Equals(letterboxAlignment, "top-left", StringComparison.Ordinal);
        int resizedWidth = Math.Max(1, Math.Min(targetWidth, topLeft ? (int)(sourceWidth * scale) : (int)MathF.Round(sourceWidth * scale)));
        int resizedHeight = Math.Max(1, Math.Min(targetHeight, topLeft ? (int)(sourceHeight * scale) : (int)MathF.Round(sourceHeight * scale)));
        int padX = topLeft ? 0 : (targetWidth - resizedWidth) / 2;
        int padY = topLeft ? 0 : (targetHeight - resizedHeight) / 2;
        return new ResizePlan(targetWidth, targetHeight, resizedWidth, resizedHeight, padX, padY, scale, scale);
    }

    private static byte[] ResizeToTarget(SampleRgbImage image, ResizePlan plan, byte fillValue)
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

    private static string NormalizeLetterboxAlignment(string value)
    {
        string normalized = (value ?? string.Empty).Trim().Replace("_", "-", StringComparison.Ordinal).ToLowerInvariant();
        return normalized switch
        {
            "" or "center" or "centered" => "center",
            "topleft" or "top-left" => "top-left",
            _ => throw new ArgumentException($"Unsupported letterbox alignment '{value}'. Use center or top-left.")
        };
    }

    private static string ComputeFileSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
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

}
