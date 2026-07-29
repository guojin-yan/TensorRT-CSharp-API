using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineRuntimeOutputArtifact
{
    public OnnxEngineRuntimeOutputArtifact(
        string tensorName,
        IReadOnlyList<int> shape,
        IReadOnlyList<float> values)
        : this(
            tensorName,
            shape,
            values?.Count ?? 0,
            CreatePreview(values),
            ToBytes(values))
    {
    }

    internal OnnxEngineRuntimeOutputArtifact(
        string tensorName,
        IReadOnlyList<int> shape,
        int elementCount,
        IReadOnlyList<float> preview,
        byte[] rawBytes)
    {
        TensorName = tensorName ?? string.Empty;
        Shape = shape?.ToArray() ?? Array.Empty<int>();
        ElementCount = Math.Max(0, elementCount);
        Preview = preview?.ToArray() ?? Array.Empty<float>();
        RawBytes = rawBytes?.ToArray() ?? Array.Empty<byte>();
        ByteLength = RawBytes.LongLength;
        Sha256 = ComputeSha256(RawBytes);
    }

    public string TensorName { get; }

    public IReadOnlyList<int> Shape { get; }

    public int ElementCount { get; }

    public IReadOnlyList<float> Preview { get; }

    public long ByteLength { get; }

    public string Sha256 { get; }

    internal byte[] RawBytes { get; }

    private static IReadOnlyList<float> CreatePreview(IReadOnlyList<float>? values)
    {
        return values == null ? Array.Empty<float>() : values.Take(8).ToArray();
    }

    private static byte[] ToBytes(IReadOnlyList<float>? values)
    {
        if (values == null || values.Count == 0)
        {
            return Array.Empty<byte>();
        }

        float[] floats = values.ToArray();
        byte[] bytes = new byte[checked(floats.Length * sizeof(float))];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static string ComputeSha256(byte[] bytes)
    {
        if (bytes.Length == 0)
        {
            return string.Empty;
        }

        using SHA256 sha256 = SHA256.Create();
        return Convert.ToHexString(sha256.ComputeHash(bytes)).ToLowerInvariant();
    }
}
