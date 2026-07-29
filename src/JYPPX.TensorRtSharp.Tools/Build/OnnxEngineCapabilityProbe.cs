using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineCapabilityProbe
{
    public OnnxEngineCapabilityProbe(
        bool attempted,
        string probeState,
        TensorRtApiLine tensorRtLine,
        string tensorRtVersion,
        string cudaToolkitVersion,
        bool runtimeAvailable,
        bool builderAvailable,
        bool builderConfigAvailable,
        bool engineInspectorApiAvailable,
        bool fp8FlagRequested,
        bool fp8FlagKnown,
        bool debugTensorOptionsRequested,
        bool debugTensorApiKnown,
        bool weightStreamingRequested,
        bool weightStreamingApiKnown,
        IReadOnlyList<string> probeItems,
        string evidenceBoundary)
    {
        Attempted = attempted;
        ProbeState = probeState ?? string.Empty;
        TensorRtLine = tensorRtLine;
        TensorRtVersion = tensorRtVersion ?? string.Empty;
        CudaToolkitVersion = cudaToolkitVersion ?? string.Empty;
        RuntimeAvailable = runtimeAvailable;
        BuilderAvailable = builderAvailable;
        BuilderConfigAvailable = builderConfigAvailable;
        EngineInspectorApiAvailable = engineInspectorApiAvailable;
        Fp8FlagRequested = fp8FlagRequested;
        Fp8FlagKnown = fp8FlagKnown;
        DebugTensorOptionsRequested = debugTensorOptionsRequested;
        DebugTensorApiKnown = debugTensorApiKnown;
        WeightStreamingRequested = weightStreamingRequested;
        WeightStreamingApiKnown = weightStreamingApiKnown;
        ProbeItems = probeItems ?? Array.Empty<string>();
        EvidenceBoundary = evidenceBoundary ?? string.Empty;
    }

    public static OnnxEngineCapabilityProbe Empty { get; } = new OnnxEngineCapabilityProbe(
        attempted: false,
        probeState: string.Empty,
        tensorRtLine: TensorRtApiLine.TensorRt10,
        tensorRtVersion: string.Empty,
        cudaToolkitVersion: string.Empty,
        runtimeAvailable: false,
        builderAvailable: false,
        builderConfigAvailable: false,
        engineInspectorApiAvailable: false,
        fp8FlagRequested: false,
        fp8FlagKnown: false,
        debugTensorOptionsRequested: false,
        debugTensorApiKnown: false,
        weightStreamingRequested: false,
        weightStreamingApiKnown: false,
        probeItems: Array.Empty<string>(),
        evidenceBoundary: string.Empty);

    public bool Attempted { get; }

    public string ProbeState { get; }

    public TensorRtApiLine TensorRtLine { get; }

    public string TensorRtVersion { get; }

    public string CudaToolkitVersion { get; }

    public bool RuntimeAvailable { get; }

    public bool BuilderAvailable { get; }

    public bool BuilderConfigAvailable { get; }

    public bool EngineInspectorApiAvailable { get; }

    public bool Fp8FlagRequested { get; }

    public bool Fp8FlagKnown { get; }

    public bool DebugTensorOptionsRequested { get; }

    public bool DebugTensorApiKnown { get; }

    public bool WeightStreamingRequested { get; }

    public bool WeightStreamingApiKnown { get; }

    public IReadOnlyList<string> ProbeItems { get; }

    public string EvidenceBoundary { get; }
}
