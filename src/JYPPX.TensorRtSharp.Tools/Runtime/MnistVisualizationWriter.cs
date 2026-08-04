using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Writes an SVG visualization for one completed MNIST inference.
/// </summary>
public static class MnistVisualizationWriter
{
    private const int CanvasWidth = 1120;
    private const int CanvasHeight = 640;
    private const int PixelSize = 16;

    /// <summary>
    /// Writes the source pixels, prediction, confidence, and class probabilities to an SVG file.
    /// </summary>
    public static void Write(string path, MnistOnnxRuntimeResult result)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Visualization path must not be empty.", nameof(path));
        }

        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (!result.InferenceRan || result.PredictedDigit < 0 || result.PredictedDigit > 9 || result.Probabilities.Length != 10)
        {
            throw new InvalidOperationException("MNIST visualization requires a completed ten-class inference.");
        }

        if (!float.IsFinite(result.Confidence) ||
            result.Probabilities.Any(static probability => !float.IsFinite(probability) || probability < 0.0f || probability > 1.0f))
        {
            throw new InvalidDataException("MNIST visualization requires finite probabilities in the range [0, 1].");
        }

        MnistPgmImage image = MnistPgmReader.Read(result.InputPath);
        if (image.Width != 28 || image.Height != 28)
        {
            throw new InvalidDataException($"MNIST visualization requires a 28x28 PGM image. Actual={image.Width}x{image.Height}.");
        }

        string? directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(path, BuildSvg(image, result), new UTF8Encoding(false));
    }

    private static string BuildSvg(MnistPgmImage image, MnistOnnxRuntimeResult result)
    {
        StringBuilder svg = new StringBuilder(96 * 1024);
        svg.AppendLine($"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{CanvasWidth}\" height=\"{CanvasHeight}\" viewBox=\"0 0 {CanvasWidth} {CanvasHeight}\">");
        svg.AppendLine("  <rect width=\"1120\" height=\"640\" fill=\"#f4f5f7\"/>");
        svg.AppendLine("  <rect x=\"40\" y=\"40\" width=\"520\" height=\"560\" rx=\"6\" fill=\"#15191d\"/>");
        svg.AppendLine("  <text x=\"72\" y=\"82\" fill=\"#d7dde3\" font-family=\"Segoe UI,Arial,sans-serif\" font-size=\"18\">INPUT · 28 × 28 PGM</text>");

        const int imageX = 76;
        const int imageY = 112;
        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                byte source = image.Pixels[(y * image.Width) + x];
                int intensity = 255 - source;
                if (intensity == 0)
                {
                    continue;
                }

                string color = $"rgb({intensity},{intensity},{intensity})";
                svg.Append("  <rect x=\"").Append(imageX + (x * PixelSize)).Append("\" y=\"")
                    .Append(imageY + (y * PixelSize)).Append("\" width=\"").Append(PixelSize)
                    .Append("\" height=\"").Append(PixelSize).Append("\" fill=\"").Append(color).AppendLine("\"/>");
            }
        }

        svg.AppendLine("  <text x=\"608\" y=\"78\" fill=\"#40474f\" font-family=\"Segoe UI,Arial,sans-serif\" font-size=\"18\">TENSORRT CLASSIFICATION</text>");
        svg.AppendLine($"  <text x=\"608\" y=\"178\" fill=\"#111417\" font-family=\"Segoe UI,Arial,sans-serif\" font-size=\"104\" font-weight=\"700\">{result.PredictedDigit}</text>");
        svg.AppendLine($"  <text x=\"736\" y=\"140\" fill=\"#167a68\" font-family=\"Segoe UI,Arial,sans-serif\" font-size=\"28\" font-weight=\"600\">{FormatPercent(result.Confidence)}</text>");
        svg.AppendLine($"  <text x=\"736\" y=\"174\" fill=\"#68717a\" font-family=\"Segoe UI,Arial,sans-serif\" font-size=\"17\">expected {result.ExpectedDigit} · match {result.OutputMatch.ToString().ToLowerInvariant()}</text>");
        svg.AppendLine("  <line x1=\"608\" y1=\"216\" x2=\"1070\" y2=\"216\" stroke=\"#ccd2d8\"/>");

        int[] ranked = Enumerable.Range(0, result.Probabilities.Length)
            .OrderByDescending(index => result.Probabilities[index])
            .ThenBy(index => index)
            .ToArray();
        for (int rank = 0; rank < ranked.Length; rank++)
        {
            int digit = ranked[rank];
            float probability = result.Probabilities[digit];
            int y = 250 + (rank * 32);
            int width = Math.Max(1, (int)Math.Round(340 * Math.Clamp(probability, 0.0f, 1.0f)));
            string fill = digit == result.PredictedDigit ? "#167a68" : "#aeb7bf";
            svg.AppendLine($"  <text x=\"608\" y=\"{y + 16}\" fill=\"#30363c\" font-family=\"Consolas,monospace\" font-size=\"16\">{digit}</text>");
            svg.AppendLine($"  <rect x=\"638\" y=\"{y}\" width=\"340\" height=\"20\" rx=\"4\" fill=\"#e1e5e9\"/>");
            svg.AppendLine($"  <rect x=\"638\" y=\"{y}\" width=\"{width}\" height=\"20\" rx=\"4\" fill=\"{fill}\"/>");
            svg.AppendLine($"  <text x=\"994\" y=\"{y + 16}\" fill=\"#4e565e\" font-family=\"Consolas,monospace\" font-size=\"14\">{FormatPercent(probability)}</text>");
        }

        svg.AppendLine("  <text x=\"608\" y=\"602\" fill=\"#68717a\" font-family=\"Segoe UI,Arial,sans-serif\" font-size=\"15\">RGB-free · NCHW [1,1,28,28] · preprocess 1 - pixel / 255</text>");
        svg.AppendLine("</svg>");
        return svg.ToString();
    }

    private static string FormatPercent(float value) =>
        (value * 100.0f).ToString("0.000", CultureInfo.InvariantCulture) + "%";
}
