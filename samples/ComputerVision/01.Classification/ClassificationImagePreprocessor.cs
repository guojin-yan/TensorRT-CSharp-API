using System;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using JYPPX.SampleSupport;

namespace ClassificationSample;

public sealed class ClassificationPreprocessOptions
{
    public ClassificationPreprocessOptions(
        string resizeMode,
        int resizeShorterSide,
        string tensorLayout,
        string colorOrder,
        float scale,
        float[] mean,
        float[] standardDeviation)
    {
        ResizeMode = NormalizeResizeMode(resizeMode);
        if (resizeShorterSide <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(resizeShorterSide));
        }

        ResizeShorterSide = resizeShorterSide;
        TensorLayout = NormalizeLayout(tensorLayout);
        ColorOrder = NormalizeColorOrder(colorOrder);
        if (!float.IsFinite(scale) || scale <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(scale), "Scale must be finite and positive.");
        }

        Scale = scale;
        Mean = CopyTriplet(mean, nameof(mean), requirePositive: false);
        StandardDeviation = CopyTriplet(standardDeviation, nameof(standardDeviation), requirePositive: true);
    }

    public string ResizeMode { get; }

    public int ResizeShorterSide { get; }

    public string TensorLayout { get; }

    public string ColorOrder { get; }

    public float Scale { get; }

    public float[] Mean { get; }

    public float[] StandardDeviation { get; }

    public string ContractSha256 => ComputeSha256(Encoding.UTF8.GetBytes(ToCanonicalString()));

    public string ToCanonicalString()
    {
        return string.Join(
            ";",
            "resizeMode=" + ResizeMode,
            "resizeShorterSide=" + ResizeShorterSide.ToString(CultureInfo.InvariantCulture),
            "tensorLayout=" + TensorLayout,
            "colorOrder=" + ColorOrder,
            "scale=" + Scale.ToString("R", CultureInfo.InvariantCulture),
            "mean=" + FormatTriplet(Mean),
            "std=" + FormatTriplet(StandardDeviation),
            "interpolation=bilinear-half-pixel",
            "crop=center");
    }

    private static string NormalizeResizeMode(string value)
    {
        string normalized = (value ?? string.Empty).Trim().Replace('_', '-').ToLowerInvariant();
        return normalized switch
        {
            "" or "shorter-side-center-crop" or "center-crop" => "shorter-side-center-crop",
            "stretch" => "stretch",
            _ => throw new ArgumentException("Resize mode must be shorter-side-center-crop or stretch.", nameof(value))
        };
    }

    private static string NormalizeLayout(string value)
    {
        string normalized = (value ?? string.Empty).Trim().Replace("-", string.Empty, StringComparison.Ordinal).Replace("_", string.Empty, StringComparison.Ordinal).ToUpperInvariant();
        return normalized switch
        {
            "" or "NCHW" or "CHANNELSFIRST" => "NCHW",
            "NHWC" or "CHANNELSLAST" => "NHWC",
            _ => throw new ArgumentException("Tensor layout must be NCHW or NHWC.", nameof(value))
        };
    }

    private static string NormalizeColorOrder(string value)
    {
        string normalized = (value ?? string.Empty).Trim().ToUpperInvariant();
        return normalized switch
        {
            "" or "RGB" => "RGB",
            "BGR" => "BGR",
            _ => throw new ArgumentException("Color order must be RGB or BGR.", nameof(value))
        };
    }

    private static float[] CopyTriplet(float[] values, string name, bool requirePositive)
    {
        if (values == null || values.Length != 3)
        {
            throw new ArgumentException("Value must contain exactly three channels.", name);
        }

        float[] copy = (float[])values.Clone();
        foreach (float value in copy)
        {
            if (!float.IsFinite(value) || (requirePositive && value <= 0.0f))
            {
                throw new ArgumentOutOfRangeException(name, requirePositive
                    ? "Every channel must be finite and positive."
                    : "Every channel must be finite.");
            }
        }

        return copy;
    }

    private static string FormatTriplet(float[] values)
    {
        return string.Join(",", Array.ConvertAll(values, static value => value.ToString("R", CultureInfo.InvariantCulture)));
    }

    private static string ComputeSha256(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }
}

public sealed class ClassificationImagePreprocessResult
{
    public ClassificationImagePreprocessResult(
        string sourcePath,
        string sourceSha256,
        int sourceWidth,
        int sourceHeight,
        string tensorPath,
        string tensorSha256,
        int tensorElementCount,
        int targetWidth,
        int targetHeight,
        int resizedWidth,
        int resizedHeight,
        int cropX,
        int cropY,
        ClassificationPreprocessOptions options)
    {
        SourcePath = sourcePath;
        SourceSha256 = sourceSha256;
        SourceWidth = sourceWidth;
        SourceHeight = sourceHeight;
        TensorPath = tensorPath;
        TensorSha256 = tensorSha256;
        TensorElementCount = tensorElementCount;
        TargetWidth = targetWidth;
        TargetHeight = targetHeight;
        ResizedWidth = resizedWidth;
        ResizedHeight = resizedHeight;
        CropX = cropX;
        CropY = cropY;
        Options = options;
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

    public int ResizedWidth { get; }

    public int ResizedHeight { get; }

    public int CropX { get; }

    public int CropY { get; }

    public ClassificationPreprocessOptions Options { get; }
}

public static class ClassificationImagePreprocessor
{
    public static ClassificationImagePreprocessResult Preprocess(
        string imagePath,
        string tensorPath,
        int[] inputShape,
        ClassificationPreprocessOptions options)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            throw new ArgumentException("Image path must not be empty.", nameof(imagePath));
        }
        if (string.IsNullOrWhiteSpace(tensorPath))
        {
            throw new ArgumentException("Tensor path must not be empty.", nameof(tensorPath));
        }
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        ResolveInputShape(inputShape, options.TensorLayout, out int channels, out int targetHeight, out int targetWidth);
        if (channels != 3)
        {
            throw new ArgumentException("Classification image preprocessing requires a three-channel input tensor.", nameof(inputShape));
        }

        string fullImagePath = Path.GetFullPath(imagePath);
        SampleRgbImage image = OpenCvSampleRgbImageDecoder.Decode(fullImagePath);
        CreateResizePlan(image.Width, image.Height, targetWidth, targetHeight, options, out int resizedWidth, out int resizedHeight, out int cropX, out int cropY);
        byte[] resized = ResizeBilinear(image, resizedWidth, resizedHeight);
        float[] tensor = CropAndNormalize(resized, resizedWidth, resizedHeight, targetWidth, targetHeight, cropX, cropY, options);

        string fullTensorPath = Path.GetFullPath(tensorPath);
        string? directory = Path.GetDirectoryName(fullTensorPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        byte[] tensorBytes = new byte[checked(tensor.Length * sizeof(float))];
        Buffer.BlockCopy(tensor, 0, tensorBytes, 0, tensorBytes.Length);
        File.WriteAllBytes(fullTensorPath, tensorBytes);
        return new ClassificationImagePreprocessResult(
            fullImagePath,
            ComputeFileSha256(fullImagePath),
            image.Width,
            image.Height,
            fullTensorPath,
            ComputeSha256(tensorBytes),
            tensor.Length,
            targetWidth,
            targetHeight,
            resizedWidth,
            resizedHeight,
            cropX,
            cropY,
            options);
    }

    private static void CreateResizePlan(
        int sourceWidth,
        int sourceHeight,
        int targetWidth,
        int targetHeight,
        ClassificationPreprocessOptions options,
        out int resizedWidth,
        out int resizedHeight,
        out int cropX,
        out int cropY)
    {
        if (string.Equals(options.ResizeMode, "stretch", StringComparison.Ordinal))
        {
            resizedWidth = targetWidth;
            resizedHeight = targetHeight;
            cropX = 0;
            cropY = 0;
            return;
        }

        int shorterSide = options.ResizeShorterSide;
        float resizeScale = shorterSide / (float)Math.Min(sourceWidth, sourceHeight);
        resizedWidth = Math.Max(1, (int)MathF.Round(sourceWidth * resizeScale));
        resizedHeight = Math.Max(1, (int)MathF.Round(sourceHeight * resizeScale));
        if (resizedWidth < targetWidth || resizedHeight < targetHeight)
        {
            throw new ArgumentException("Resize shorter side does not produce an image large enough for the requested center crop.");
        }

        cropX = (resizedWidth - targetWidth) / 2;
        cropY = (resizedHeight - targetHeight) / 2;
    }

    private static byte[] ResizeBilinear(SampleRgbImage image, int targetWidth, int targetHeight)
    {
        byte[] target = new byte[checked(targetWidth * targetHeight * 3)];
        for (int y = 0; y < targetHeight; y++)
        {
            float sourceY = MapCoordinate(y, targetHeight, image.Height);
            int y0 = Math.Clamp((int)MathF.Floor(sourceY), 0, image.Height - 1);
            int y1 = Math.Clamp(y0 + 1, 0, image.Height - 1);
            float yWeight = sourceY - y0;
            for (int x = 0; x < targetWidth; x++)
            {
                float sourceX = MapCoordinate(x, targetWidth, image.Width);
                int x0 = Math.Clamp((int)MathF.Floor(sourceX), 0, image.Width - 1);
                int x1 = Math.Clamp(x0 + 1, 0, image.Width - 1);
                float xWeight = sourceX - x0;
                int targetIndex = (y * targetWidth + x) * 3;
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

    private static float[] CropAndNormalize(
        byte[] resized,
        int resizedWidth,
        int resizedHeight,
        int targetWidth,
        int targetHeight,
        int cropX,
        int cropY,
        ClassificationPreprocessOptions options)
    {
        if (cropX < 0 || cropY < 0 || cropX + targetWidth > resizedWidth || cropY + targetHeight > resizedHeight)
        {
            throw new ArgumentOutOfRangeException(nameof(cropX), "Center crop exceeds the resized image.");
        }

        float[] tensor = new float[checked(targetWidth * targetHeight * 3)];
        bool bgr = string.Equals(options.ColorOrder, "BGR", StringComparison.Ordinal);
        bool nchw = string.Equals(options.TensorLayout, "NCHW", StringComparison.Ordinal);
        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                int sourceIndex = (((y + cropY) * resizedWidth) + x + cropX) * 3;
                for (int outputChannel = 0; outputChannel < 3; outputChannel++)
                {
                    int sourceChannel = bgr ? 2 - outputChannel : outputChannel;
                    float value = (resized[sourceIndex + sourceChannel] * options.Scale - options.Mean[outputChannel]) /
                        options.StandardDeviation[outputChannel];
                    int targetIndex = nchw
                        ? outputChannel * targetWidth * targetHeight + y * targetWidth + x
                        : (y * targetWidth + x) * 3 + outputChannel;
                    tensor[targetIndex] = value;
                }
            }
        }

        return tensor;
    }

    private static float MapCoordinate(int targetIndex, int targetLength, int sourceLength)
    {
        if (targetLength == sourceLength)
        {
            return targetIndex;
        }

        return Math.Clamp((targetIndex + 0.5f) * sourceLength / targetLength - 0.5f, 0.0f, sourceLength - 1.0f);
    }

    private static void ResolveInputShape(int[] shape, string layout, out int channels, out int height, out int width)
    {
        if (shape == null || shape.Length != 4 || shape[0] != 1)
        {
            throw new ArgumentException("Classification --image requires a 4D batch-1 input shape.", nameof(shape));
        }
        if (string.Equals(layout, "NCHW", StringComparison.Ordinal))
        {
            channels = shape[1];
            height = shape[2];
            width = shape[3];
        }
        else
        {
            height = shape[1];
            width = shape[2];
            channels = shape[3];
        }
    }

    private static string ComputeFileSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static string ComputeSha256(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }
}
