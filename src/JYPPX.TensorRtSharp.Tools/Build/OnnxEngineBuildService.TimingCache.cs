using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class OnnxEngineBuildService
{
    private const long MaxTimingCacheBytes = 512L * 1024L * 1024L;

    private static TimingCacheLease CreateTimingCacheLease(
        TensorRtBuilderConfig config,
        OnnxEngineBuildOptions options,
        List<string> log)
    {
        string inputPath = options.TimingCacheFile;
        string outputPath = options.DeploymentOptions.ExportTimingCachePath;
        bool inputRequested = !string.IsNullOrWhiteSpace(inputPath);
        bool outputRequested = !string.IsNullOrWhiteSpace(outputPath);
        if (!inputRequested && !outputRequested)
        {
            return new TimingCacheLease(null, OnnxEngineTimingCacheArtifact.Empty);
        }

        byte[]? inputBytes = null;
        long inputLengthBytes = 0;
        string inputSha256 = string.Empty;
        if (inputRequested)
        {
            FileInfo inputFile = new FileInfo(inputPath);
            if (!inputFile.Exists)
            {
                throw new FileNotFoundException("Timing cache file was not found.", inputPath);
            }

            if (inputFile.Length > MaxTimingCacheBytes)
            {
                throw new InvalidDataException($"Timing cache file exceeds the {MaxTimingCacheBytes} byte safety limit.");
            }

            inputBytes = File.ReadAllBytes(inputFile.FullName);
            inputLengthBytes = inputBytes.LongLength;
            inputSha256 = ComputeSha256(inputBytes);
        }

        TensorRtTimingCache cache = config.CreateTimingCache(inputBytes);
        try
        {
            config.SetTimingCache(cache, ignoreMismatch: false);
        }
        catch
        {
            cache.Dispose();
            throw;
        }

        log.Add($"TimingCache ImportRequested={inputRequested} Applied=True Path={inputPath} LengthBytes={inputLengthBytes} Sha256={inputSha256}");
        return new TimingCacheLease(
            cache,
            new OnnxEngineTimingCacheArtifact(
                inputRequested,
                inputApplied: true,
                inputPath,
                inputLengthBytes,
                inputSha256,
                outputRequested,
                outputWritten: false,
                outputPath,
                outputLengthBytes: 0,
                outputSha256: string.Empty,
                state: outputRequested ? "imported-export-pending" : "imported",
                evidenceBoundary: TimingCacheEvidenceBoundary));
    }

    private static OnnxEngineTimingCacheArtifact ExportTimingCache(
        TimingCacheLease lease,
        OnnxEngineBuildOptions options,
        List<string> log)
    {
        string outputPath = options.DeploymentOptions.ExportTimingCachePath;
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            return lease.Artifact;
        }

        if (lease.Cache == null)
        {
            log.Add("TimingCache ExportRequested=True Written=False Reason=timing cache owner was not created.");
            return CreateTimingCacheBoundaryArtifact(options, "export-not-applied");
        }

        using TensorRtHostMemory hostMemory = lease.Cache.Serialize();
        byte[] bytes = hostMemory.ToArray();
        string fullPath = Path.GetFullPath(outputPath);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllBytes(fullPath, bytes);
        string outputSha256 = ComputeSha256(bytes);
        log.Add($"TimingCache ExportRequested=True Written=True Path={fullPath} LengthBytes={bytes.LongLength} Sha256={outputSha256}");
        return new OnnxEngineTimingCacheArtifact(
            lease.Artifact.InputRequested,
            lease.Artifact.InputApplied,
            lease.Artifact.InputPath,
            lease.Artifact.InputLengthBytes,
            lease.Artifact.InputSha256,
            outputRequested: true,
            outputWritten: true,
            fullPath,
            bytes.LongLength,
            outputSha256,
            state: lease.Artifact.InputApplied ? "imported-and-exported" : "exported",
            evidenceBoundary: TimingCacheEvidenceBoundary);
    }

    private static OnnxEngineTimingCacheArtifact CreateTimingCacheBoundaryArtifact(
        OnnxEngineBuildOptions options,
        string state)
    {
        bool inputRequested = !string.IsNullOrWhiteSpace(options.TimingCacheFile);
        bool outputRequested = !string.IsNullOrWhiteSpace(options.DeploymentOptions.ExportTimingCachePath);
        if (!inputRequested && !outputRequested)
        {
            return OnnxEngineTimingCacheArtifact.Empty;
        }

        return new OnnxEngineTimingCacheArtifact(
            inputRequested,
            inputApplied: false,
            options.TimingCacheFile,
            inputLengthBytes: 0,
            inputSha256: string.Empty,
            outputRequested,
            outputWritten: false,
            options.DeploymentOptions.ExportTimingCachePath,
            outputLengthBytes: 0,
            outputSha256: string.Empty,
            state,
            TimingCacheEvidenceBoundary);
    }

    private const string TimingCacheEvidenceBoundary = "timing-cache import/export evidence is build-cache lifecycle metadata only; it is not model accuracy, runtime execution, real-model-runtime, or package-consumer-runtime proof.";

    private static string ComputeSha256(byte[] bytes)
    {
        using SHA256 sha256 = SHA256.Create();
        byte[] hash = sha256.ComputeHash(bytes ?? Array.Empty<byte>());
        StringBuilder builder = new StringBuilder(hash.Length * 2);
        foreach (byte item in hash)
        {
            builder.Append(item.ToString("x2"));
        }

        return builder.ToString();
    }

    private sealed class TimingCacheLease : IDisposable
    {
        public TimingCacheLease(TensorRtTimingCache? cache, OnnxEngineTimingCacheArtifact artifact)
        {
            Cache = cache;
            Artifact = artifact ?? OnnxEngineTimingCacheArtifact.Empty;
        }

        public TensorRtTimingCache? Cache { get; }

        public OnnxEngineTimingCacheArtifact Artifact { get; set; }

        public void Dispose()
        {
            Cache?.Dispose();
        }
    }

}
