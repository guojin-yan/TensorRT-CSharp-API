using System;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class MnistOnnxRuntimeService
{
    internal static string ComputeSha256(byte[] bytes)
    {
        using SHA256 sha256 = SHA256.Create();
        byte[] hash = sha256.ComputeHash(bytes);
        StringBuilder builder = new StringBuilder(hash.Length * 2);
        foreach (byte item in hash)
        {
            builder.Append(item.ToString("x2", CultureInfo.InvariantCulture));
        }

        return builder.ToString();
    }

    private static string ComputeFileSha256(string path)
    {
        return ComputeSha256(File.ReadAllBytes(path));
    }

    private static byte[] ToBytes(float[] values)
    {
        byte[] bytes = new byte[checked(values.Length * sizeof(float))];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
