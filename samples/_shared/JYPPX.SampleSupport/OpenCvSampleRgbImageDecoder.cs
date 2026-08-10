using System;
using System.IO;
using OpenCvSharp;
using OpenCvSharp.Core;
using OpenCvSharp.ImgCodecs;
using ImgCodecsCv2 = OpenCvSharp.ImgCodecs.Cv2;

namespace JYPPX.SampleSupport;

internal static class OpenCvSampleRgbImageDecoder
{
    public static SampleRgbImage Decode(string path)
    {
        string extension = Path.GetExtension(path);
        if (string.Equals(extension, ".bmp", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(extension, ".dib", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(extension, ".ppm", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(extension, ".pnm", StringComparison.OrdinalIgnoreCase))
        {
            return SampleRgbImageDecoder.Decode(path);
        }

        if (!string.Equals(extension, ".jpg", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(extension, ".jpeg", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(extension, ".png", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException(
                "Image decoding supports JPEG/PNG through JYPPX.OpenCV.CSharp.API and BMP/PPM through the managed fallback.");
        }

        using Mat image = ImgCodecsCv2.ImRead(path, ImreadModes.Color);
        if (image.Empty || image.Rows <= 0 || image.Cols <= 0 || image.Channels != 3)
        {
            throw new InvalidDataException("JYPPX.OpenCV.CSharp.API could not decode the input image as three-channel BGR data.");
        }

        byte[] bgr = image.ToArray<byte>();
        int expectedLength = checked(image.Rows * image.Cols * 3);
        if (bgr.Length != expectedLength)
        {
            throw new InvalidDataException(
                $"JYPPX.OpenCV.CSharp.API returned {bgr.Length} image bytes; expected {expectedLength}.");
        }

        byte[] rgb = new byte[expectedLength];
        for (int index = 0; index < expectedLength; index += 3)
        {
            rgb[index] = bgr[index + 2];
            rgb[index + 1] = bgr[index + 1];
            rgb[index + 2] = bgr[index];
        }

        return new SampleRgbImage(image.Cols, image.Rows, rgb);
    }
}
