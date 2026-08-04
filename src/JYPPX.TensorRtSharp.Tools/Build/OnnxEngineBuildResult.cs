using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineBuildResult
{
    public OnnxEngineBuildResult(
        bool success,
        bool skipped,
        string state,
        TensorRtApiLine tensorRtLine,
        string modelSource,
        string enginePath,
        bool parsed,
        bool engineSaved,
        bool engineFileRoundTrip,
        bool inferenceRan,
        bool outputMatch,
        int profileIndex,
        float? elapsedMilliseconds,
        string skipReason,
        string normalizedCommandLine,
        IReadOnlyList<string> diagnostics,
        IReadOnlyList<string> logLines,
        OnnxEngineBuildEvidenceSidecar? evidenceSidecar = null,
        TrtexecLikeRuntimeOptions? runtimeOptions = null,
        OnnxEngineBenchmarkSummary? benchmarkSummary = null,
        OnnxEnginePreflightMetadata? preflightMetadata = null,
        OnnxLoadedEngineDiagnostics? loadedEngineDiagnostics = null,
        OnnxEngineTimingCacheArtifact? timingCacheArtifact = null,
        OnnxEngineCapabilityProbe? capabilityProbe = null,
        ulong workspaceBytes = 0,
        TensorRtBuilderConfigDeploymentSnapshot? builderConfigDeploymentSnapshot = null,
        OnnxEngineParserPreflightSnapshot? parserPreflightSnapshot = null,
        OnnxEngineRefitSnapshot? refitSnapshot = null,
        OnnxEngineRefitPersistenceSnapshot? refitPersistenceSnapshot = null,
        bool outputValidated = false,
        bool identityOutputMatch = false)
        : this(
            success,
            skipped,
            state,
            tensorRtLine,
            modelSource,
            enginePath,
            parsed,
            engineSaved,
            engineFileRoundTrip,
            inferenceRan,
            outputMatch,
            profileIndex,
            elapsedMilliseconds,
            skipReason,
            normalizedCommandLine,
            TrtexecLikeDeploymentOptions.Default,
            diagnostics,
            logLines,
            evidenceSidecar,
            runtimeOptions,
            benchmarkSummary,
            preflightMetadata,
            loadedEngineDiagnostics,
            timingCacheArtifact,
            capabilityProbe,
            workspaceBytes,
            builderConfigDeploymentSnapshot,
            parserPreflightSnapshot,
            refitSnapshot,
            refitPersistenceSnapshot,
            outputValidated,
            identityOutputMatch)
    {
    }

    public OnnxEngineBuildResult(
        bool success,
        bool skipped,
        string state,
        TensorRtApiLine tensorRtLine,
        string modelSource,
        string enginePath,
        bool parsed,
        bool engineSaved,
        bool engineFileRoundTrip,
        bool inferenceRan,
        bool outputMatch,
        int profileIndex,
        float? elapsedMilliseconds,
        string skipReason,
        string normalizedCommandLine,
        TrtexecLikeDeploymentOptions deploymentOptions,
        IReadOnlyList<string> diagnostics,
        IReadOnlyList<string> logLines,
        OnnxEngineBuildEvidenceSidecar? evidenceSidecar = null,
        TrtexecLikeRuntimeOptions? runtimeOptions = null,
        OnnxEngineBenchmarkSummary? benchmarkSummary = null,
        OnnxEnginePreflightMetadata? preflightMetadata = null,
        OnnxLoadedEngineDiagnostics? loadedEngineDiagnostics = null,
        OnnxEngineTimingCacheArtifact? timingCacheArtifact = null,
        OnnxEngineCapabilityProbe? capabilityProbe = null,
        ulong workspaceBytes = 0,
        TensorRtBuilderConfigDeploymentSnapshot? builderConfigDeploymentSnapshot = null,
        OnnxEngineParserPreflightSnapshot? parserPreflightSnapshot = null,
        OnnxEngineRefitSnapshot? refitSnapshot = null,
        OnnxEngineRefitPersistenceSnapshot? refitPersistenceSnapshot = null,
        bool outputValidated = false,
        bool identityOutputMatch = false)
    {
        Success = success;
        Skipped = skipped;
        State = state ?? string.Empty;
        TensorRtLine = tensorRtLine;
        ModelSource = modelSource ?? string.Empty;
        EnginePath = enginePath ?? string.Empty;
        Parsed = parsed;
        EngineSaved = engineSaved;
        EngineFileRoundTrip = engineFileRoundTrip;
        InferenceRan = inferenceRan;
        OutputMatch = outputMatch;
        OutputValidated = outputValidated;
        IdentityOutputMatch = identityOutputMatch;
        ProfileIndex = profileIndex;
        ElapsedMilliseconds = elapsedMilliseconds;
        SkipReason = skipReason ?? string.Empty;
        NormalizedCommandLine = normalizedCommandLine ?? string.Empty;
        DeploymentOptions = deploymentOptions ?? TrtexecLikeDeploymentOptions.Default;
        RuntimeOptions = runtimeOptions ?? TrtexecLikeRuntimeOptions.Default;
        BenchmarkSummary = benchmarkSummary ?? OnnxEngineBenchmarkSummary.Empty;
        PreflightMetadata = preflightMetadata ?? OnnxEnginePreflightMetadata.Empty;
        LoadedEngineDiagnostics = loadedEngineDiagnostics ?? OnnxLoadedEngineDiagnostics.Empty;
        TimingCacheArtifact = timingCacheArtifact ?? OnnxEngineTimingCacheArtifact.Empty;
        CapabilityProbe = capabilityProbe ?? OnnxEngineCapabilityProbe.Empty;
        WorkspaceBytes = workspaceBytes;
        BuilderConfigDeploymentSnapshot = builderConfigDeploymentSnapshot;
        ParserPreflightSnapshot = parserPreflightSnapshot ?? OnnxEngineParserPreflightSnapshot.Empty;
        RefitSnapshot = refitSnapshot ?? OnnxEngineRefitSnapshot.Empty;
        RefitPersistenceSnapshot = refitPersistenceSnapshot ?? OnnxEngineRefitPersistenceSnapshot.Empty;
        Diagnostics = diagnostics ?? Array.Empty<string>();
        LogLines = logLines ?? Array.Empty<string>();
        EvidenceSidecar = evidenceSidecar ?? OnnxEngineBuildEvidenceSidecarReader.Empty;
    }

    public bool Success { get; }

    public bool Skipped { get; }

    public string State { get; }

    public TensorRtApiLine TensorRtLine { get; }

    public string ModelSource { get; }

    public string EnginePath { get; }

    public bool Parsed { get; }

    public bool EngineSaved { get; }

    public bool EngineFileRoundTrip { get; }

    public bool InferenceRan { get; }

    public bool OutputMatch { get; }

    public bool OutputValidated { get; }

    public bool IdentityOutputMatch { get; }

    public int ProfileIndex { get; }

    public float? ElapsedMilliseconds { get; }

    public string SkipReason { get; }

    public string NormalizedCommandLine { get; }

    public string NormalizedCommandSha256 => ComputeSha256(NormalizedCommandLine);

    public TrtexecLikeDeploymentOptions DeploymentOptions { get; }

    public TrtexecLikeRuntimeOptions RuntimeOptions { get; }

    public OnnxEngineBenchmarkSummary BenchmarkSummary { get; }

    public OnnxEnginePreflightMetadata PreflightMetadata { get; }

    public OnnxLoadedEngineDiagnostics LoadedEngineDiagnostics { get; }

    public OnnxEngineTimingCacheArtifact TimingCacheArtifact { get; }

    public OnnxEngineCapabilityProbe CapabilityProbe { get; }

    public ulong WorkspaceBytes { get; }

    /// <summary>
    /// Gets builder-config values copied after deployment options were applied.
    /// 获取应用部署选项后复制读回的 builder-config 实际值。
    /// </summary>
    /// <remarks>
    /// This is build/deployment diagnostics only. It is not model runtime or package-consumer proof.
    /// 该数据只属于 build/deployment 诊断，不是模型 runtime 或 package-consumer proof。
    /// </remarks>
    public TensorRtBuilderConfigDeploymentSnapshot? BuilderConfigDeploymentSnapshot { get; }

    public OnnxEngineParserPreflightSnapshot ParserPreflightSnapshot { get; }

    /// <summary>
    /// Gets copied diagnostics for the optional ONNX stripped-plan refit lifecycle.
    /// 获取可选 ONNX stripped-plan 重整生命周期的复制诊断。
    /// </summary>
    public OnnxEngineRefitSnapshot RefitSnapshot { get; }

    /// <summary>
    /// Gets copied evidence for the optional persisted-plan independent reload lifecycle.
    /// 获取可选持久化 plan 独立重新加载生命周期的复制证据。
    /// </summary>
    public OnnxEngineRefitPersistenceSnapshot RefitPersistenceSnapshot { get; }

    public bool IsRuntimeExecutionProof => InferenceRan && OutputMatch;

    public bool BuildEvidenceOnly => string.Equals(ProofClassification, "build-only", StringComparison.Ordinal) ||
        string.Equals(ProofClassification, "dependency-probe-only", StringComparison.Ordinal) ||
        string.Equals(ProofClassification, "precheck", StringComparison.Ordinal);

    public string ProofClassification
    {
        get
        {
            if (IsRuntimeExecutionProof)
            {
                return "synthetic-input-runtime";
            }

            if (Skipped || State.Contains("preflight", StringComparison.OrdinalIgnoreCase))
            {
                return "dependency-probe-only";
            }

            if (State.Contains("dry-run", StringComparison.OrdinalIgnoreCase) ||
                State.Contains("preview", StringComparison.OrdinalIgnoreCase) ||
                State.Contains("precheck", StringComparison.OrdinalIgnoreCase))
            {
                return "precheck";
            }

            return "build-only";
        }
    }

    public IReadOnlyList<string> EvidenceClassifications { get; } = new[]
    {
        "build-only",
        "dependency-probe-only",
        "precheck",
        "synthetic-input-runtime",
        "real-model-runtime",
        "package-consumer-runtime"
    };

    public bool IsRealModelRuntimeProof => string.Equals(ProofClassification, "real-model-runtime", StringComparison.Ordinal);

    public bool IsPackageConsumerRuntimeProof => string.Equals(ProofClassification, "package-consumer-runtime", StringComparison.Ordinal);

    public string StdoutSummary => string.IsNullOrWhiteSpace(EvidenceSidecar.StdoutSummary) ? CreateSummary(LogLines) : EvidenceSidecar.StdoutSummary;

    public string StderrSummary => EvidenceSidecar.StderrSummary;

    public OnnxEngineBuildModelEvidence ModelEvidence => EvidenceSidecar.HasModelEvidence
        ? EvidenceSidecar.ToModelEvidence(ModelSource)
        : new OnnxEngineBuildModelEvidence(ModelSource, string.Empty, string.Empty, string.Empty, string.Empty);

    public OnnxEngineBuildEvidenceSidecar EvidenceSidecar { get; }

    public IReadOnlyList<string> Diagnostics { get; }

    public IReadOnlyList<string> LogLines { get; }

    private static string CreateSummary(IReadOnlyList<string> lines)
    {
        if (lines == null || lines.Count == 0)
        {
            return string.Empty;
        }

        return string.Join(" | ", lines).Trim();
    }

    private static string ComputeSha256(string value)
    {
        using SHA256 sha256 = SHA256.Create();
        byte[] hash = sha256.ComputeHash(Encoding.UTF8.GetBytes(value ?? string.Empty));
        StringBuilder builder = new StringBuilder(hash.Length * 2);
        foreach (byte item in hash)
        {
            builder.Append(item.ToString("x2"));
        }

        return builder.ToString();
    }

}
