using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEnginePreflightMetadata
{
    public OnnxEnginePreflightMetadata(
        string kind,
        string path,
        bool exists,
        long lengthBytes,
        string sha256,
        string preflightState,
        string proofClassification,
        string evidenceBoundary)
    {
        Kind = kind ?? string.Empty;
        Path = path ?? string.Empty;
        Exists = exists;
        LengthBytes = lengthBytes;
        Sha256 = sha256 ?? string.Empty;
        PreflightState = preflightState ?? string.Empty;
        ProofClassification = proofClassification ?? string.Empty;
        EvidenceBoundary = evidenceBoundary ?? string.Empty;
    }

    public static OnnxEnginePreflightMetadata Empty { get; } = new OnnxEnginePreflightMetadata(
        string.Empty,
        string.Empty,
        false,
        0,
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty);

    public string Kind { get; }

    public string Path { get; }

    public bool Exists { get; }

    public long LengthBytes { get; }

    public string Sha256 { get; }

    public string PreflightState { get; }

    public string ProofClassification { get; }

    public string EvidenceBoundary { get; }

    public static OnnxEnginePreflightMetadata FromExistingEngine(string enginePath)
    {
        string fullPath = string.IsNullOrWhiteSpace(enginePath) ? string.Empty : System.IO.Path.GetFullPath(enginePath);
        bool exists = !string.IsNullOrWhiteSpace(fullPath) && File.Exists(fullPath);
        long lengthBytes = exists ? new FileInfo(fullPath).Length : 0;
        string sha256 = exists ? ComputeFileSha256(fullPath) : string.Empty;
        return new OnnxEnginePreflightMetadata(
            "load-engine-preflight",
            fullPath,
            exists,
            lengthBytes,
            sha256,
            "dependency-probe-only",
            "dependency-probe-only",
            "load-engine preflight records file metadata only; it does not deserialize, bind tensors, enqueue inference, validate outputs, or prove package-consumer-runtime.");
    }

    private static string ComputeFileSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        using SHA256 sha256 = SHA256.Create();
        byte[] hash = sha256.ComputeHash(stream);
        StringBuilder builder = new StringBuilder(hash.Length * 2);
        foreach (byte item in hash)
        {
            builder.Append(item.ToString("x2"));
        }

        return builder.ToString();
    }
}
