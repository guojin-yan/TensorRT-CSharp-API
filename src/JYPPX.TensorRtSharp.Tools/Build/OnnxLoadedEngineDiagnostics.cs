using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxLoadedEngineDiagnostics
{
    public OnnxLoadedEngineDiagnostics(
        bool attempted,
        bool succeeded,
        string diagnosticsState,
        string failureReason,
        string engineName,
        int ioTensorCount,
        int layerCount,
        int optimizationProfileCount,
        ulong deviceMemorySizeInBytes,
        int auxiliaryStreamCount,
        string capability,
        string profilingVerbosity,
        int inspectorInformationLength,
        IReadOnlyList<string> ioTensorSummaries,
        string readbackFingerprint,
        string readbackSha256,
        string evidenceBoundary,
        OnnxEngineBindingMetadata? bindingMetadata = null,
        OnnxEngineLayerInfoArtifact? layerInfoArtifact = null)
    {
        Attempted = attempted;
        Succeeded = succeeded;
        DiagnosticsState = diagnosticsState ?? string.Empty;
        FailureReason = failureReason ?? string.Empty;
        EngineName = engineName ?? string.Empty;
        IOTensorCount = ioTensorCount;
        LayerCount = layerCount;
        OptimizationProfileCount = optimizationProfileCount;
        DeviceMemorySizeInBytes = deviceMemorySizeInBytes;
        AuxiliaryStreamCount = auxiliaryStreamCount;
        Capability = capability ?? string.Empty;
        ProfilingVerbosity = profilingVerbosity ?? string.Empty;
        InspectorInformationLength = inspectorInformationLength;
        IOTensorSummaries = ioTensorSummaries ?? Array.Empty<string>();
        ReadbackFingerprint = readbackFingerprint ?? string.Empty;
        ReadbackSha256 = readbackSha256 ?? string.Empty;
        EvidenceBoundary = evidenceBoundary ?? string.Empty;
        BindingMetadata = bindingMetadata ?? OnnxEngineBindingMetadata.Empty;
        LayerInfoArtifact = layerInfoArtifact ?? OnnxEngineLayerInfoArtifact.Empty;
    }

    public static OnnxLoadedEngineDiagnostics Empty { get; } = new OnnxLoadedEngineDiagnostics(
        attempted: false,
        succeeded: false,
        diagnosticsState: string.Empty,
        failureReason: string.Empty,
        engineName: string.Empty,
        ioTensorCount: 0,
        layerCount: 0,
        optimizationProfileCount: 0,
        deviceMemorySizeInBytes: 0,
        auxiliaryStreamCount: 0,
        capability: string.Empty,
        profilingVerbosity: string.Empty,
        inspectorInformationLength: 0,
        ioTensorSummaries: Array.Empty<string>(),
        readbackFingerprint: string.Empty,
        readbackSha256: string.Empty,
        evidenceBoundary: string.Empty);

    public bool Attempted { get; }

    public bool Succeeded { get; }

    public string DiagnosticsState { get; }

    public string FailureReason { get; }

    public string EngineName { get; }

    public int IOTensorCount { get; }

    public int LayerCount { get; }

    public int OptimizationProfileCount { get; }

    public ulong DeviceMemorySizeInBytes { get; }

    public int AuxiliaryStreamCount { get; }

    public string Capability { get; }

    public string ProfilingVerbosity { get; }

    public int InspectorInformationLength { get; }

    public IReadOnlyList<string> IOTensorSummaries { get; }

    public string ReadbackFingerprint { get; }

    public string ReadbackSha256 { get; }

    public string EvidenceBoundary { get; }

    public OnnxEngineBindingMetadata BindingMetadata { get; }

    public OnnxEngineLayerInfoArtifact LayerInfoArtifact { get; }
}
