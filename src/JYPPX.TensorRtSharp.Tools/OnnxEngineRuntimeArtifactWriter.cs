using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineRuntimeArtifactData
{
    public OnnxEngineRuntimeArtifactData(
        string tensorName,
        IReadOnlyList<int> shape,
        int inputElementCount,
        int outputElementCount,
        IReadOnlyList<float> inputPreview,
        IReadOnlyList<float> outputPreview,
        string executionSummary,
        byte[] rawOutputBytes,
        IReadOnlyList<float>? timingSamplesMilliseconds = null)
    {
        TensorName = tensorName ?? string.Empty;
        Shape = shape ?? Array.Empty<int>();
        InputElementCount = inputElementCount;
        OutputElementCount = outputElementCount;
        InputPreview = inputPreview ?? Array.Empty<float>();
        OutputPreview = outputPreview ?? Array.Empty<float>();
        ExecutionSummary = executionSummary ?? string.Empty;
        RawOutputBytes = rawOutputBytes ?? Array.Empty<byte>();
        TimingSamplesMilliseconds = timingSamplesMilliseconds ?? Array.Empty<float>();
    }

    public static OnnxEngineRuntimeArtifactData Empty { get; } = new OnnxEngineRuntimeArtifactData(
        string.Empty,
        Array.Empty<int>(),
        0,
        0,
        Array.Empty<float>(),
        Array.Empty<float>(),
        string.Empty,
        Array.Empty<byte>(),
        Array.Empty<float>());

    public static OnnxEngineRuntimeArtifactData CreateIdentityOutput(
        string tensorName,
        IReadOnlyList<int> shape,
        IReadOnlyList<float> inputValues,
        IReadOnlyList<float> outputValues,
        string executionSummary,
        IReadOnlyList<float>? timingSamplesMilliseconds = null)
    {
        return new OnnxEngineRuntimeArtifactData(
            tensorName,
            shape,
            inputValues?.Count ?? 0,
            outputValues?.Count ?? 0,
            Preview(inputValues),
            Preview(outputValues),
            executionSummary,
            ToBytes(outputValues),
            timingSamplesMilliseconds ?? Array.Empty<float>());
    }

    public static OnnxEngineRuntimeArtifactData CreateOutputSummary(
        string tensorName,
        IReadOnlyList<int> shape,
        int inputElementCount,
        IReadOnlyList<float> outputValues,
        string executionSummary,
        IReadOnlyList<float>? timingSamplesMilliseconds = null)
    {
        return new OnnxEngineRuntimeArtifactData(
            tensorName,
            shape,
            inputElementCount,
            outputValues?.Count ?? 0,
            Array.Empty<float>(),
            Preview(outputValues),
            executionSummary,
            ToBytes(outputValues),
            timingSamplesMilliseconds ?? Array.Empty<float>());
    }

    internal static OnnxEngineRuntimeArtifactData CreateBenchmarkOnly(
        int inputElementCount,
        string executionSummary,
        IReadOnlyList<float>? timingSamplesMilliseconds = null)
    {
        return new OnnxEngineRuntimeArtifactData(
            string.Empty,
            Array.Empty<int>(),
            inputElementCount,
            0,
            Array.Empty<float>(),
            Array.Empty<float>(),
            executionSummary,
            Array.Empty<byte>(),
            timingSamplesMilliseconds ?? Array.Empty<float>());
    }

    public string TensorName { get; }

    public IReadOnlyList<int> Shape { get; }

    public int InputElementCount { get; }

    public int OutputElementCount { get; }

    public IReadOnlyList<float> InputPreview { get; }

    public IReadOnlyList<float> OutputPreview { get; }

    public string ExecutionSummary { get; }

    public byte[] RawOutputBytes { get; }

    public IReadOnlyList<float> TimingSamplesMilliseconds { get; }

    public bool HasOutput => OutputElementCount > 0 && OutputPreview.Count > 0;

    public bool HasRawOutput => RawOutputBytes.Length > 0;

    private static IReadOnlyList<float> Preview(IReadOnlyList<float>? values)
    {
        if (values == null || values.Count == 0)
        {
            return Array.Empty<float>();
        }

        return values.Take(8).ToArray();
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
}

public static class OnnxEngineRuntimeArtifactWriter
{
    private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions { WriteIndented = true };

    public static void WriteArtifacts(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData? data = null)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        TrtexecLikeRuntimeOptions options = result.RuntimeOptions;
        if (!options.HasRuntimeDiagnostics)
        {
            return;
        }

        OnnxEngineRuntimeArtifactData artifactData = data ?? OnnxEngineRuntimeArtifactData.Empty;
        if (!string.IsNullOrWhiteSpace(options.ExportTimesPath))
        {
            WriteJson(options.ExportTimesPath, CreateTimesArtifact(result, artifactData));
        }

        if (!string.IsNullOrWhiteSpace(options.ExportOutputPath))
        {
            WriteJson(options.ExportOutputPath, CreateOutputArtifact(result, artifactData));
        }

        if (!string.IsNullOrWhiteSpace(options.ExportProfilePath))
        {
            WriteJson(options.ExportProfilePath, CreateProfileArtifact(result, artifactData));
        }

        string engineReadbackPath = GetEngineReadbackArtifactPath(options);
        if (!string.IsNullOrWhiteSpace(engineReadbackPath))
        {
            WriteJson(engineReadbackPath, CreateEngineReadbackArtifact(result, artifactData));
        }

        if (!string.IsNullOrWhiteSpace(options.SaveProfilePath))
        {
            WriteText(options.SaveProfilePath, CreateProfileText(result, artifactData));
        }

        if (!string.IsNullOrWhiteSpace(options.DumpRawBindingsToFile))
        {
            WriteRawBindingsOrBoundary(options.DumpRawBindingsToFile, result, artifactData);
        }
    }

    private static object CreateTimesArtifact(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        RuntimeArtifactProofBoundary proofBoundary = CreateProofBoundary(result, data, "timing");
        return new
        {
            ArtifactKind = "trtexec-like-times",
            ArtifactBoundary = proofBoundary.ArtifactProofBoundary,
            proofBoundary.ArtifactProofBoundary,
            proofBoundary.RuntimeProofClass,
            proofBoundary.HasTensorOutputProof,
            proofBoundary.HasRawBindingProof,
            proofBoundary.IsBuildOnlyEvidence,
            proofBoundary.IsDependencyProbeOnly,
            proofBoundary.IsSyntheticRuntime,
            proofBoundary.HasBenchmarkExecutionEvidence,
            proofBoundary.ModelSource,
            proofBoundary.EnginePath,
            proofBoundary.PreflightMetadata,
            result.State,
            result.ProofClassification,
            result.BuildEvidenceOnly,
            result.IsRuntimeExecutionProof,
            result.IsRealModelRuntimeProof,
            result.IsPackageConsumerRuntimeProof,
            result.NormalizedCommandSha256,
            result.ProfileIndex,
            result.ElapsedMilliseconds,
            TimingSamplesMilliseconds = result.BenchmarkSummary.TimingSamplesMilliseconds,
            TimingSampleCount = result.BenchmarkSummary.TimingSampleCount,
            AveragedTimingSamplesMilliseconds = result.BenchmarkSummary.AveragedTimingSamplesMilliseconds,
            AveragedTimingSampleCount = result.BenchmarkSummary.AveragedTimingSampleCount,
            AverageElapsedMilliseconds = result.BenchmarkSummary.AverageElapsedMilliseconds,
            MinElapsedMilliseconds = result.BenchmarkSummary.MinElapsedMilliseconds,
            MaxElapsedMilliseconds = result.BenchmarkSummary.MaxElapsedMilliseconds,
            PercentileRequested = result.BenchmarkSummary.PercentileRequested,
            PercentileElapsedMilliseconds = result.BenchmarkSummary.PercentileElapsedMilliseconds,
            AvgRunsRequested = result.BenchmarkSummary.AvgRunsRequested,
            AvgRunsExecuted = result.BenchmarkSummary.AvgRunsExecuted,
            ThreadsRequested = result.BenchmarkSummary.ThreadsRequested,
            ThreadsExecuted = result.BenchmarkSummary.ThreadsExecuted,
            NoDataTransfersRequested = result.BenchmarkSummary.NoDataTransfersRequested,
            NoDataTransfersApplied = result.BenchmarkSummary.NoDataTransfersApplied,
            UseSpinWaitRequested = result.BenchmarkSummary.UseSpinWaitRequested,
            UseSpinWaitApplied = result.BenchmarkSummary.UseSpinWaitApplied,
            UseCudaGraphRequested = result.BenchmarkSummary.UseCudaGraphRequested,
            UseCudaGraphApplied = result.BenchmarkSummary.UseCudaGraphApplied,
            UseCudaGraphFallbackReason = result.BenchmarkSummary.UseCudaGraphFallbackReason,
            SleepTimeMillisecondsRequested = result.BenchmarkSummary.SleepTimeMillisecondsRequested,
            SleepTimeMillisecondsApplied = result.BenchmarkSummary.SleepTimeMillisecondsApplied,
            IdleTimeMillisecondsRequested = result.BenchmarkSummary.IdleTimeMillisecondsRequested,
            IdleTimeMillisecondsApplied = result.BenchmarkSummary.IdleTimeMillisecondsApplied,
            IterationsRequested = result.BenchmarkSummary.IterationsRequested,
            MeasurementRoundsExecuted = result.BenchmarkSummary.MeasurementRoundsExecuted,
            MeasurementRoundsPerContext = result.BenchmarkSummary.MeasurementRoundsPerContext,
            InferenceIterationsExecuted = result.BenchmarkSummary.InferenceIterationsExecuted,
            WarmUpMillisecondsRequested = result.BenchmarkSummary.WarmUpMillisecondsRequested,
            WarmUpElapsedMilliseconds = result.BenchmarkSummary.WarmUpElapsedMilliseconds,
            WarmUpIterationsExecuted = result.BenchmarkSummary.WarmUpIterationsExecuted,
            DurationSecondsRequested = result.BenchmarkSummary.DurationSecondsRequested,
            MeasurementElapsedMilliseconds = result.BenchmarkSummary.MeasurementElapsedMilliseconds,
            StreamsRequested = result.BenchmarkSummary.StreamsRequested,
            InfStreamsRequested = result.BenchmarkSummary.InfStreamsRequested,
            ExecutionContextsCreated = result.BenchmarkSummary.ExecutionContextsCreated,
            ConcurrentStreamsExecuted = result.BenchmarkSummary.ConcurrentStreamsExecuted,
            BenchmarkBoundary = result.BenchmarkSummary.BenchmarkBoundary,
            RuntimeOptions = result.RuntimeOptions,
            data.ExecutionSummary
        };
    }

    private static object CreateOutputArtifact(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        RuntimeArtifactProofBoundary proofBoundary = CreateProofBoundary(result, data, "output");
        return new
        {
            ArtifactKind = "trtexec-like-output-summary",
            ArtifactBoundary = proofBoundary.ArtifactProofBoundary,
            proofBoundary.ArtifactProofBoundary,
            proofBoundary.RuntimeProofClass,
            proofBoundary.HasTensorOutputProof,
            proofBoundary.HasRawBindingProof,
            proofBoundary.IsBuildOnlyEvidence,
            proofBoundary.IsDependencyProbeOnly,
            proofBoundary.IsSyntheticRuntime,
            proofBoundary.HasBenchmarkExecutionEvidence,
            proofBoundary.ModelSource,
            proofBoundary.EnginePath,
            proofBoundary.PreflightMetadata,
            result.State,
            result.ProofClassification,
            result.BuildEvidenceOnly,
            result.IsRuntimeExecutionProof,
            result.IsRealModelRuntimeProof,
            result.IsPackageConsumerRuntimeProof,
            result.NormalizedCommandSha256,
            result.InferenceRan,
            result.OutputMatch,
            TensorName = data.TensorName,
            Shape = data.Shape,
            data.InputElementCount,
            data.OutputElementCount,
            data.InputPreview,
            data.OutputPreview,
            OutputComparisonSample = ReadFloatSample(data.RawOutputBytes, 64),
            OutputByteLength = data.RawOutputBytes.LongLength,
            OutputSha256 = ComputeSha256(data.RawOutputBytes),
            Note = "Output artifact is a bounded summary; it intentionally avoids large tensor dumps. The output hash is comparison evidence only and does not promote the proof classification."
        };
    }

    private static string ComputeSha256(byte[] bytes)
    {
        if (bytes == null || bytes.Length == 0)
        {
            return string.Empty;
        }

        using SHA256 sha256 = SHA256.Create();
        byte[] hash = sha256.ComputeHash(bytes);
        StringBuilder builder = new StringBuilder(hash.Length * 2);
        foreach (byte item in hash)
        {
            builder.Append(item.ToString("x2", CultureInfo.InvariantCulture));
        }

        return builder.ToString();
    }

    private static IReadOnlyList<float> ReadFloatSample(byte[] bytes, int maximumValues)
    {
        if (bytes == null || bytes.Length == 0 || maximumValues <= 0)
        {
            return Array.Empty<float>();
        }

        int count = Math.Min(bytes.Length / sizeof(float), maximumValues);
        float[] values = new float[count];
        Buffer.BlockCopy(bytes, 0, values, 0, count * sizeof(float));
        return values;
    }

    private static object CreateProfileArtifact(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        RuntimeArtifactProofBoundary proofBoundary = CreateProofBoundary(result, data, "profile");
        return new
        {
            ArtifactKind = "trtexec-like-profile-boundary",
            ArtifactBoundary = proofBoundary.ArtifactProofBoundary,
            proofBoundary.ArtifactProofBoundary,
            proofBoundary.RuntimeProofClass,
            proofBoundary.HasTensorOutputProof,
            proofBoundary.HasRawBindingProof,
            proofBoundary.IsBuildOnlyEvidence,
            proofBoundary.IsDependencyProbeOnly,
            proofBoundary.IsSyntheticRuntime,
            proofBoundary.HasBenchmarkExecutionEvidence,
            proofBoundary.ModelSource,
            proofBoundary.EnginePath,
            proofBoundary.PreflightMetadata,
            result.State,
            result.ProofClassification,
            result.BuildEvidenceOnly,
            result.IsRuntimeExecutionProof,
            result.IsRealModelRuntimeProof,
            result.IsPackageConsumerRuntimeProof,
            result.NormalizedCommandSha256,
            result.ProfileIndex,
            result.ElapsedMilliseconds,
            TimingSampleCount = result.BenchmarkSummary.TimingSampleCount,
            AverageElapsedMilliseconds = result.BenchmarkSummary.AverageElapsedMilliseconds,
            LayerProfileAvailable = false,
            LayerProfileBoundary = "TensorRT per-layer profiler data is not collected by this generic sample path; no layer timings are fabricated.",
            data.ExecutionSummary
        };
    }

    private static object CreateEngineReadbackArtifact(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        RuntimeArtifactProofBoundary proofBoundary = CreateProofBoundary(result, data, "engine-readback");
        OnnxLoadedEngineDiagnostics loaded = result.LoadedEngineDiagnostics;
        bool hasReadback = loaded.Attempted &&
            loaded.Succeeded &&
            !string.IsNullOrWhiteSpace(loaded.ReadbackFingerprint) &&
            !string.IsNullOrWhiteSpace(loaded.ReadbackSha256);

        return new
        {
            ArtifactKind = hasReadback ? "trtexec-like-engine-readback" : "trtexec-like-engine-readback-skipped",
            ArtifactBoundary = proofBoundary.ArtifactProofBoundary,
            proofBoundary.ArtifactProofBoundary,
            proofBoundary.RuntimeProofClass,
            proofBoundary.HasTensorOutputProof,
            proofBoundary.HasRawBindingProof,
            proofBoundary.IsBuildOnlyEvidence,
            proofBoundary.IsDependencyProbeOnly,
            proofBoundary.IsSyntheticRuntime,
            proofBoundary.HasBenchmarkExecutionEvidence,
            proofBoundary.ModelSource,
            proofBoundary.EnginePath,
            proofBoundary.PreflightMetadata,
            result.State,
            result.ProofClassification,
            result.BuildEvidenceOnly,
            IsRuntimeExecutionProof = false,
            IsRealModelRuntimeProof = false,
            IsPackageConsumerRuntimeProof = false,
            result.NormalizedCommandSha256,
            ReadbackAvailable = hasReadback,
            ReadbackFingerprint = hasReadback ? loaded.ReadbackFingerprint : string.Empty,
            ReadbackSha256 = hasReadback ? loaded.ReadbackSha256 : string.Empty,
            SkippedReason = hasReadback ? string.Empty : EngineReadbackSkippedReason(loaded),
            LoadedEngineDiagnostics = new
            {
                loaded.Attempted,
                loaded.Succeeded,
                loaded.DiagnosticsState,
                loaded.FailureReason,
                loaded.EngineName,
                loaded.IOTensorCount,
                loaded.LayerCount,
                loaded.OptimizationProfileCount,
                loaded.DeviceMemorySizeInBytes,
                loaded.AuxiliaryStreamCount,
                loaded.Capability,
                loaded.ProfilingVerbosity,
                loaded.InspectorInformationLength,
                loaded.IOTensorSummaries,
                loaded.EvidenceBoundary
            },
            Note = "Engine readback artifact is readonly metadata evidence only; it does not bind tensors, enqueue inference, validate outputs, or prove real-model/package-consumer runtime."
        };
    }

    private static string CreateProfileText(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        RuntimeArtifactProofBoundary proofBoundary = CreateProofBoundary(result, data, "profile");
        return string.Join(Environment.NewLine, new[]
        {
            "TensorRtSharp trtexec-like profile boundary",
            $"ArtifactKind: trtexec-like-profile-boundary",
            $"ArtifactBoundary: {proofBoundary.ArtifactProofBoundary}",
            $"ArtifactProofBoundary: {proofBoundary.ArtifactProofBoundary}",
            $"RuntimeProofClass: {proofBoundary.RuntimeProofClass}",
            $"HasTensorOutputProof: {proofBoundary.HasTensorOutputProof}",
            $"HasRawBindingProof: {proofBoundary.HasRawBindingProof}",
            $"IsBuildOnlyEvidence: {proofBoundary.IsBuildOnlyEvidence}",
            $"IsDependencyProbeOnly: {proofBoundary.IsDependencyProbeOnly}",
            $"IsSyntheticRuntime: {proofBoundary.IsSyntheticRuntime}",
            $"HasBenchmarkExecutionEvidence: {proofBoundary.HasBenchmarkExecutionEvidence}",
            $"ModelSource: {proofBoundary.ModelSource}",
            $"EnginePath: {proofBoundary.EnginePath}",
            $"PreflightKind: {proofBoundary.PreflightMetadata.Kind}",
            $"PreflightPath: {proofBoundary.PreflightMetadata.Path}",
            $"PreflightEvidenceBoundary: {proofBoundary.PreflightMetadata.EvidenceBoundary}",
            $"State: {result.State}",
            $"ProofClassification: {result.ProofClassification}",
            $"BuildEvidenceOnly: {result.BuildEvidenceOnly}",
            $"IsRuntimeExecutionProof: {result.IsRuntimeExecutionProof}",
            $"IsRealModelRuntimeProof: {result.IsRealModelRuntimeProof}",
            $"IsPackageConsumerRuntimeProof: {result.IsPackageConsumerRuntimeProof}",
            $"NormalizedCommandSha256: {result.NormalizedCommandSha256}",
            $"ElapsedMilliseconds: {result.ElapsedMilliseconds?.ToString("0.###") ?? string.Empty}",
            $"TimingSampleCount: {result.BenchmarkSummary.TimingSampleCount}",
            $"AveragedTimingSampleCount: {result.BenchmarkSummary.AveragedTimingSampleCount}",
            $"AverageElapsedMilliseconds: {result.BenchmarkSummary.AverageElapsedMilliseconds?.ToString("0.###", CultureInfo.InvariantCulture) ?? string.Empty}",
            $"PercentileRequested: {result.BenchmarkSummary.PercentileRequested?.ToString(CultureInfo.InvariantCulture) ?? string.Empty}",
            $"PercentileElapsedMilliseconds: {result.BenchmarkSummary.PercentileElapsedMilliseconds?.ToString("0.###", CultureInfo.InvariantCulture) ?? string.Empty}",
            $"IterationsRequested: {result.BenchmarkSummary.IterationsRequested}",
            $"MeasurementRoundsExecuted: {result.BenchmarkSummary.MeasurementRoundsExecuted}",
            $"MeasurementRoundsPerContext: {string.Join(",", result.BenchmarkSummary.MeasurementRoundsPerContext)}",
            $"InferenceIterationsExecuted: {result.BenchmarkSummary.InferenceIterationsExecuted}",
            $"WarmUpMillisecondsRequested: {result.BenchmarkSummary.WarmUpMillisecondsRequested}",
            $"WarmUpElapsedMilliseconds: {result.BenchmarkSummary.WarmUpElapsedMilliseconds.ToString("0.###", CultureInfo.InvariantCulture)}",
            $"DurationSecondsRequested: {result.BenchmarkSummary.DurationSecondsRequested}",
            $"MeasurementElapsedMilliseconds: {result.BenchmarkSummary.MeasurementElapsedMilliseconds.ToString("0.###", CultureInfo.InvariantCulture)}",
            $"ExecutionContextsCreated: {result.BenchmarkSummary.ExecutionContextsCreated}",
            $"ThreadsExecuted: {result.BenchmarkSummary.ThreadsExecuted}",
            $"UseSpinWaitApplied: {result.BenchmarkSummary.UseSpinWaitApplied}",
            $"UseCudaGraphRequested: {result.BenchmarkSummary.UseCudaGraphRequested}",
            $"UseCudaGraphApplied: {result.BenchmarkSummary.UseCudaGraphApplied}",
            $"UseCudaGraphFallbackReason: {result.BenchmarkSummary.UseCudaGraphFallbackReason}",
            $"BenchmarkBoundary: {result.BenchmarkSummary.BenchmarkBoundary}",
            $"ExecutionSummary: {data.ExecutionSummary}",
            "LayerProfileAvailable: False",
            "LayerProfileBoundary: TensorRT per-layer profiler data is not collected by this generic sample path; no layer timings are fabricated."
        });
    }

    private static void WriteRawBindingsOrBoundary(string path, OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        if (CanWriteRawBindings(result, data))
        {
            WriteBytes(path, data.RawOutputBytes);
            return;
        }

        RuntimeArtifactProofBoundary proofBoundary = CreateProofBoundary(result, data, "raw-bindings");
        WriteJson(path, new
        {
            ArtifactKind = "trtexec-like-raw-bindings-skipped",
            ArtifactBoundary = proofBoundary.ArtifactProofBoundary,
            proofBoundary.ArtifactProofBoundary,
            proofBoundary.RuntimeProofClass,
            proofBoundary.HasTensorOutputProof,
            proofBoundary.HasRawBindingProof,
            proofBoundary.IsBuildOnlyEvidence,
            proofBoundary.IsDependencyProbeOnly,
            proofBoundary.IsSyntheticRuntime,
            proofBoundary.HasBenchmarkExecutionEvidence,
            proofBoundary.ModelSource,
            proofBoundary.EnginePath,
            proofBoundary.PreflightMetadata,
            result.State,
            result.ProofClassification,
            result.BuildEvidenceOnly,
            result.IsRuntimeExecutionProof,
            result.IsRealModelRuntimeProof,
            result.IsPackageConsumerRuntimeProof,
            result.NormalizedCommandSha256,
            RawBindingBytesWritten = 0
        });
    }

    private static string GetEngineReadbackArtifactPath(TrtexecLikeRuntimeOptions options)
    {
        string sourcePath = FirstNonEmpty(
            options.ExportProfilePath,
            options.ExportTimesPath,
            options.ExportOutputPath,
            options.SaveProfilePath,
            options.DumpRawBindingsToFile);

        if (string.IsNullOrWhiteSpace(sourcePath))
        {
            return string.Empty;
        }

        string extension = Path.GetExtension(sourcePath);
        if (string.Equals(extension, ".json", StringComparison.OrdinalIgnoreCase))
        {
            return Path.ChangeExtension(sourcePath, ".engine-readback.json");
        }

        return sourcePath + ".engine-readback.json";
    }

    private static string FirstNonEmpty(params string[] values)
    {
        foreach (string value in values)
        {
            if (!string.IsNullOrWhiteSpace(value))
            {
                return value;
            }
        }

        return string.Empty;
    }

    private static string EngineReadbackSkippedReason(OnnxLoadedEngineDiagnostics loaded)
    {
        if (!loaded.Attempted)
        {
            return string.IsNullOrWhiteSpace(loaded.DiagnosticsState)
                ? "readonly engine diagnostics were not attempted"
                : loaded.DiagnosticsState;
        }

        if (!loaded.Succeeded)
        {
            return string.IsNullOrWhiteSpace(loaded.FailureReason)
                ? loaded.DiagnosticsState
                : loaded.FailureReason;
        }

        return "readonly engine diagnostics did not provide a readback fingerprint and SHA256";
    }

    private static bool CanWriteRawBindings(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        return result.InferenceRan &&
            result.OutputMatch &&
            data.HasRawOutput &&
            string.Equals(result.ModelSource, "embedded-dynamic-identity", StringComparison.Ordinal);
    }

    private static RuntimeArtifactProofBoundary CreateProofBoundary(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data, string artifactKind)
    {
        return new RuntimeArtifactProofBoundary(
            Boundary(result, data, artifactKind),
            result.ProofClassification,
            result.InferenceRan && result.OutputMatch && data.HasOutput,
            CanWriteRawBindings(result, data),
            result.BuildEvidenceOnly,
            string.Equals(result.ProofClassification, "dependency-probe-only", StringComparison.Ordinal),
            string.Equals(result.ProofClassification, "synthetic-input-runtime", StringComparison.Ordinal),
            result.InferenceRan && result.BenchmarkSummary.TimingSampleCount > 0,
            result.ModelSource,
            result.EnginePath,
            result.PreflightMetadata);
    }

    private static string Boundary(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data, string artifactKind)
    {
        if (result.InferenceRan && result.OutputMatch && data.HasOutput)
        {
            if (!string.Equals(result.ModelSource, "embedded-dynamic-identity", StringComparison.Ordinal))
            {
                return "runtime-executed-bounded-input; this is sample runtime evidence only, not real-model or package-consumer proof.";
            }

            return "runtime-executed-synthetic-input; this is sample runtime evidence only, not real-model or package-consumer proof.";
        }

        if (result.InferenceRan && data.HasOutput)
        {
            return "runtime-output-captured-unverified; TensorRT enqueue and output readback completed, but no reference output matched, so this is not runtime proof, real-model proof, or package-consumer proof.";
        }

        if (result.InferenceRan && result.BenchmarkSummary.NoDataTransfersApplied && result.BenchmarkSummary.TimingSampleCount > 0)
        {
            return "runtime-benchmark-executed-no-data-transfers; TensorRT enqueue timing is recorded, but input H2D and output D2H/readback were suppressed, so this is not tensor-correctness, real-model, or package-consumer proof.";
        }

        if (string.Equals(result.ProofClassification, "precheck", StringComparison.Ordinal))
        {
            return artifactKind + "-skipped-precheck; dry-run did not probe TensorRT runtime, parse ONNX, build engine, or run inference.";
        }

        if (string.Equals(result.ProofClassification, "build-only", StringComparison.Ordinal))
        {
            return artifactKind + "-skipped-build-only; engine build evidence does not include inference output, timing, or package-consumer runtime proof.";
        }

        if (string.Equals(result.ProofClassification, "dependency-probe-only", StringComparison.Ordinal))
        {
            return artifactKind + "-skipped-dependency-probe-only; runtime dependencies were probed but inference did not execute.";
        }

        return artifactKind + "-skipped; generic external model binding/output semantics are not inferred by this sample path.";
    }

    private static void WriteJson(string path, object value)
    {
        WriteText(path, JsonSerializer.Serialize(value, JsonOptions));
    }

    private static void WriteText(string path, string content)
    {
        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(fullPath, content, Encoding.UTF8);
    }

    private static void WriteBytes(string path, byte[] bytes)
    {
        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllBytes(fullPath, bytes);
    }

    private sealed class RuntimeArtifactProofBoundary
    {
        public RuntimeArtifactProofBoundary(
            string artifactProofBoundary,
            string runtimeProofClass,
            bool hasTensorOutputProof,
            bool hasRawBindingProof,
            bool isBuildOnlyEvidence,
            bool isDependencyProbeOnly,
            bool isSyntheticRuntime,
            bool hasBenchmarkExecutionEvidence,
            string modelSource,
            string enginePath,
            OnnxEnginePreflightMetadata preflightMetadata)
        {
            ArtifactProofBoundary = artifactProofBoundary ?? string.Empty;
            RuntimeProofClass = runtimeProofClass ?? string.Empty;
            HasTensorOutputProof = hasTensorOutputProof;
            HasRawBindingProof = hasRawBindingProof;
            IsBuildOnlyEvidence = isBuildOnlyEvidence;
            IsDependencyProbeOnly = isDependencyProbeOnly;
            IsSyntheticRuntime = isSyntheticRuntime;
            HasBenchmarkExecutionEvidence = hasBenchmarkExecutionEvidence;
            ModelSource = modelSource ?? string.Empty;
            EnginePath = enginePath ?? string.Empty;
            PreflightMetadata = preflightMetadata ?? OnnxEnginePreflightMetadata.Empty;
        }

        public string ArtifactProofBoundary { get; }

        public string RuntimeProofClass { get; }

        public bool HasTensorOutputProof { get; }

        public bool HasRawBindingProof { get; }

        public bool IsBuildOnlyEvidence { get; }

        public bool IsDependencyProbeOnly { get; }

        public bool IsSyntheticRuntime { get; }

        public bool HasBenchmarkExecutionEvidence { get; }

        public string ModelSource { get; }

        public string EnginePath { get; }

        public OnnxEnginePreflightMetadata PreflightMetadata { get; }
    }
}
