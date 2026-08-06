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
        : this(
            sourcePath,
            sourceSha256,
            sourceWidth,
            sourceHeight,
            tensorPath,
            tensorSha256,
            tensorElementCount,
            targetWidth,
            targetHeight,
            tensorLayout,
            colorOrder,
            resizeMode,
            normalized,
            scale,
            letterboxEnabled,
            letterboxAlignment,
            resizedWidth,
            resizedHeight,
            padX,
            padY,
            resizeScaleX,
            resizeScaleY,
            fillValue,
            centerCropEnabled: false,
            resizeShorterSide: 0,
            cropX: 0,
            cropY: 0,
            mean: new float[3],
            standardDeviation: new[] { 1.0f, 1.0f, 1.0f },
            preprocessContractSha256: string.Empty)
    {
    }

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
        byte fillValue,
        bool centerCropEnabled,
        int resizeShorterSide,
        int cropX,
        int cropY)
        : this(
            sourcePath,
            sourceSha256,
            sourceWidth,
            sourceHeight,
            tensorPath,
            tensorSha256,
            tensorElementCount,
            targetWidth,
            targetHeight,
            tensorLayout,
            colorOrder,
            resizeMode,
            normalized,
            scale,
            letterboxEnabled,
            letterboxAlignment,
            resizedWidth,
            resizedHeight,
            padX,
            padY,
            resizeScaleX,
            resizeScaleY,
            fillValue,
            centerCropEnabled,
            resizeShorterSide,
            cropX,
            cropY,
            mean: new float[3],
            standardDeviation: new[] { 1.0f, 1.0f, 1.0f },
            preprocessContractSha256: string.Empty)
    {
    }

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
        byte fillValue,
        bool centerCropEnabled,
        int resizeShorterSide,
        int cropX,
        int cropY,
        float[] mean,
        float[] standardDeviation,
        string preprocessContractSha256)
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
        CenterCropEnabled = centerCropEnabled;
        ResizeShorterSide = resizeShorterSide;
        CropX = cropX;
        CropY = cropY;
        Mean = mean == null ? throw new ArgumentNullException(nameof(mean)) : (float[])mean.Clone();
        StandardDeviation = standardDeviation == null
            ? throw new ArgumentNullException(nameof(standardDeviation))
            : (float[])standardDeviation.Clone();
        PreprocessContractSha256 = preprocessContractSha256 ?? string.Empty;
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

    public bool CenterCropEnabled { get; }

    public int ResizeShorterSide { get; }

    public int CropX { get; }

    public int CropY { get; }

    public float[] Mean { get; }

    public float[] StandardDeviation { get; }

    public string PreprocessContractSha256 { get; }
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
        string resizeMode = NormalizeResizeMode(options.ResizeMode);
        ResolveInputShape(inputShape, layout, out int channelCount, out int targetHeight, out int targetWidth);
        if (channelCount != 3)
        {
            throw new ArgumentException("YoloVision image preprocessing currently supports 3-channel tensors only.");
        }

        SampleRgbImage image = OpenCvSampleRgbImageDecoder.Decode(fullImagePath);
        bool centerCrop = string.Equals(resizeMode, "shorter-side-center-crop", StringComparison.Ordinal);
        bool letterbox = !centerCrop && options.PreserveAspectRatio && !string.Equals(resizeMode, "stretch", StringComparison.Ordinal);
        ResizePlan plan;
        byte[] targetPixels;
        int resizeShorterSide = 0;
        int cropX = 0;
        int cropY = 0;
        byte fillValue = centerCrop ? (byte)0 : DefaultLetterboxFill;
        if (centerCrop)
        {
            resizeShorterSide = options.ResizeShorterSide > 0
                ? options.ResizeShorterSide
                : Math.Min(targetWidth, targetHeight);
            plan = CreateCenterCropResizePlan(
                image.Width,
                image.Height,
                targetWidth,
                targetHeight,
                resizeShorterSide,
                out cropX,
                out cropY);
            targetPixels = ResizeAndCenterCrop(image, plan.ResizedWidth, plan.ResizedHeight, targetWidth, targetHeight, cropX, cropY);
        }
        else
        {
            plan = CreateResizePlan(image.Width, image.Height, targetWidth, targetHeight, letterbox, letterboxAlignment);
            targetPixels = ResizeToTarget(image, plan, fillValue);
        }
        float[] tensor = ToTensor(targetPixels, targetWidth, targetHeight, layout, colorOrder, options);

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
            centerCrop ? resizeMode : letterbox ? "letterbox" : "stretch",
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
            fillValue,
            centerCrop,
            resizeShorterSide,
            cropX,
            cropY,
            options.Mean,
            options.StandardDeviation,
            options.ContractSha256);
    }

    private static ResizePlan CreateCenterCropResizePlan(
        int sourceWidth,
        int sourceHeight,
        int targetWidth,
        int targetHeight,
        int resizeShorterSide,
        out int cropX,
        out int cropY)
    {
        int resizedWidth;
        int resizedHeight;
        if (sourceWidth <= sourceHeight)
        {
            resizedWidth = resizeShorterSide;
            resizedHeight = Math.Max(1, (int)(resizeShorterSide * sourceHeight / (double)sourceWidth));
        }
        else
        {
            resizedHeight = resizeShorterSide;
            resizedWidth = Math.Max(1, (int)(resizeShorterSide * sourceWidth / (double)sourceHeight));
        }
        if (resizedWidth < targetWidth || resizedHeight < targetHeight)
        {
            throw new ArgumentException(
                "Resize shorter side does not produce an image large enough for the requested center crop.");
        }

        cropX = (resizedWidth - targetWidth) / 2;
        cropY = (resizedHeight - targetHeight) / 2;
        return new ResizePlan(
            targetWidth,
            targetHeight,
            resizedWidth,
            resizedHeight,
            0,
            0,
            resizedWidth / (float)sourceWidth,
            resizedHeight / (float)sourceHeight);
    }

    private static byte[] ResizeAndCenterCrop(
        SampleRgbImage image,
        int resizedWidth,
        int resizedHeight,
        int targetWidth,
        int targetHeight,
        int cropX,
        int cropY)
    {
        byte[] resized = ResizeBilinearAntialiased(image, resizedWidth, resizedHeight);
        byte[] target = new byte[checked(targetWidth * targetHeight * 3)];
        for (int y = 0; y < targetHeight; y++)
        {
            int sourceOffset = ((y + cropY) * resizedWidth + cropX) * 3;
            int targetOffset = y * targetWidth * 3;
            Buffer.BlockCopy(resized, sourceOffset, target, targetOffset, targetWidth * 3);
        }

        return target;
    }

    private static byte[] ResizeBilinearAntialiased(SampleRgbImage image, int targetWidth, int targetHeight)
    {
        double scaleX = image.Width / (double)targetWidth;
        double scaleY = image.Height / (double)targetHeight;
        double filterScaleX = Math.Max(1.0, scaleX);
        double filterScaleY = Math.Max(1.0, scaleY);
        byte[] target = new byte[checked(targetWidth * targetHeight * 3)];
        for (int y = 0; y < targetHeight; y++)
        {
            double sourceCenterY = (y + 0.5) * scaleY;
            int sourceYStart = Math.Max(0, (int)Math.Ceiling(sourceCenterY - filterScaleY - 0.5));
            int sourceYEnd = Math.Min(image.Height - 1, (int)Math.Floor(sourceCenterY + filterScaleY - 0.5));
            for (int x = 0; x < targetWidth; x++)
            {
                double sourceCenterX = (x + 0.5) * scaleX;
                int sourceXStart = Math.Max(0, (int)Math.Ceiling(sourceCenterX - filterScaleX - 0.5));
                int sourceXEnd = Math.Min(image.Width - 1, (int)Math.Floor(sourceCenterX + filterScaleX - 0.5));
                double weightSum = 0.0;
                double red = 0.0;
                double green = 0.0;
                double blue = 0.0;
                for (int sourceY = sourceYStart; sourceY <= sourceYEnd; sourceY++)
                {
                    double yWeight = Math.Max(
                        0.0,
                        1.0 - Math.Abs((sourceY + 0.5 - sourceCenterY) / filterScaleY));
                    for (int sourceX = sourceXStart; sourceX <= sourceXEnd; sourceX++)
                    {
                        double xWeight = Math.Max(
                            0.0,
                            1.0 - Math.Abs((sourceX + 0.5 - sourceCenterX) / filterScaleX));
                        double weight = xWeight * yWeight;
                        int sourceIndex = (sourceY * image.Width + sourceX) * 3;
                        weightSum += weight;
                        red += image.Pixels[sourceIndex] * weight;
                        green += image.Pixels[sourceIndex + 1] * weight;
                        blue += image.Pixels[sourceIndex + 2] * weight;
                    }
                }

                int targetIndex = (y * targetWidth + x) * 3;
                target[targetIndex] = ToByte(red / weightSum);
                target[targetIndex + 1] = ToByte(green / weightSum);
                target[targetIndex + 2] = ToByte(blue / weightSum);
            }
        }

        return target;
    }

    private static byte ToByte(double value)
    {
        return (byte)Math.Clamp((int)Math.Round(value, MidpointRounding.AwayFromZero), 0, 255);
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

    private static float[] ToTensor(
        byte[] rgbPixels,
        int width,
        int height,
        string layout,
        string colorOrder,
        YoloPreprocessOptions options)
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
                    tensor[planeOffset] = ToFloat(first, 0, options);
                    tensor[width * height + planeOffset] = ToFloat(second, 1, options);
                    tensor[width * height * 2 + planeOffset] = ToFloat(third, 2, options);
                }
                else
                {
                    int targetIndex = (y * width + x) * 3;
                    tensor[targetIndex] = ToFloat(first, 0, options);
                    tensor[targetIndex + 1] = ToFloat(second, 1, options);
                    tensor[targetIndex + 2] = ToFloat(third, 2, options);
                }
            }
        }

        return tensor;
    }

    private static float ToFloat(byte value, int channel, YoloPreprocessOptions options)
    {
        return options.Normalize
            ? (value * options.Scale - options.Mean[channel]) / options.StandardDeviation[channel]
            : value;
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

    private static string NormalizeResizeMode(string value)
    {
        string normalized = (value ?? string.Empty).Trim().Replace('_', '-').ToLowerInvariant();
        return normalized switch
        {
            "" or "letterbox" => "letterbox",
            "stretch" => "stretch",
            "shorter-side-center-crop" or "center-crop" => "shorter-side-center-crop",
            _ => throw new ArgumentException(
                $"Unsupported resize mode '{value}'. Use letterbox, stretch, or shorter-side-center-crop.")
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
