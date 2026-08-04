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
    private static OnnxEngineParserPreflightSnapshot CaptureParserPreflightSnapshot(
        TensorRtOnnxParser parser,
        byte[] model,
        string modelPath,
        bool parsed,
        List<string> log)
    {
        TensorRtOnnxParserDiagnosticSnapshot? diagnostics = null;
        try
        {
            diagnostics = parser.GetDiagnosticSnapshot();
        }
        catch (Exception exception) when (exception is TensorRtException || exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            log.Add($"ParserDiagnostics State=unavailable Reason={exception.GetType().Name}:{exception.Message}");
        }
        bool modelSupportAttempted = false;
        string modelSupportState = "not-attempted";
        bool modelSupported = false;
        long supportedSubgraphCount = 0;
        long unsupportedSubgraphCount = 0;
        int copiedSubgraphCount = 0;
        long copiedSupportedSubgraphCount = 0;
        long copiedUnsupportedSubgraphCount = 0;
        long copiedNodeCount = 0;

        try
        {
            TensorRtOnnxModelSupportSummary summary = parser.CheckModelSupport(model, modelPath).ToSummary();
            modelSupportAttempted = true;
            modelSupportState = "copied-readback";
            modelSupported = summary.IsSupported;
            supportedSubgraphCount = summary.ReportedSupportedSubgraphCount;
            unsupportedSubgraphCount = summary.ReportedUnsupportedSubgraphCount;
            copiedSubgraphCount = summary.CopiedSubgraphCount;
            copiedSupportedSubgraphCount = summary.CopiedSupportedSubgraphCount;
            copiedUnsupportedSubgraphCount = summary.CopiedUnsupportedSubgraphCount;
            copiedNodeCount = summary.CopiedNodeCount;
        }
        catch (Exception exception) when (exception is TensorRtException || exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            modelSupportState = "unavailable";
            log.Add($"ParserModelSupport State=unavailable Reason={exception.GetType().Name}:{exception.Message}");
        }

        OnnxEngineParserPreflightSnapshot snapshot = new OnnxEngineParserPreflightSnapshot(
            parser.Line, true, parsed, diagnostics == null ? "unavailable" : "copied-readback", diagnostics?.ErrorCount ?? 0, diagnostics?.Diagnostics.Count ?? 0,
            diagnostics?.DiagnosticSummary ?? string.Empty, diagnostics?.IdentityOperatorSupported ?? false, modelSupportAttempted,
            modelSupportState, modelSupported, supportedSubgraphCount, unsupportedSubgraphCount,
            copiedSubgraphCount, copiedSupportedSubgraphCount, copiedUnsupportedSubgraphCount, copiedNodeCount);
        log.Add($"ParserPreflightSnapshot State={snapshot.DiagnosticsState} Errors={snapshot.ErrorCount} Diagnostics={snapshot.CopiedDiagnosticCount} Identity={snapshot.IdentityOperatorSupported} ModelSupport={snapshot.ModelSupportState}:{snapshot.ModelSupported} Subgraphs={snapshot.CopiedSubgraphCount}");
        return snapshot;
    }

    private static TensorRtBuilderConfigDeploymentSnapshot? TryGetBuilderConfigDeploymentSnapshot(
        TensorRtBuilderConfig config,
        List<string> log)
    {
        try
        {
            TensorRtBuilderConfigDeploymentSnapshot snapshot = config.GetDeploymentSnapshot();
            log.Add($"BuilderConfigDeploymentSnapshot State=copied-readback Diagnostics={snapshot.Diagnostics.Count} Summary={snapshot}");
            return snapshot;
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            log.Add($"BuilderConfigDeploymentSnapshot State=unavailable Reason={exception.GetType().Name}:{exception.Message}");
            return null;
        }
    }

    private static IReadOnlyList<string> AppendCapabilityProbeLog(IReadOnlyList<string> logLines, OnnxEngineCapabilityProbe capabilityProbe)
    {
        List<string> merged = new List<string>(logLines ?? Array.Empty<string>());
        if (capabilityProbe.Attempted)
        {
            merged.Add($"CapabilityProbe State={capabilityProbe.ProbeState} Runtime={capabilityProbe.RuntimeAvailable} Builder={capabilityProbe.BuilderAvailable} BuilderConfig={capabilityProbe.BuilderConfigAvailable} EngineInspectorApi={capabilityProbe.EngineInspectorApiAvailable}");
            merged.Add("CapabilityProbe Items=" + string.Join("; ", capabilityProbe.ProbeItems));
            merged.Add("CapabilityProbe Boundary=" + capabilityProbe.EvidenceBoundary);
        }

        return merged;
    }

    private static OnnxEngineCapabilityProbe ProbeCapabilities(OnnxEngineBuildOptions options)
    {
        const string boundary = "capability-probe-only records host/tool API availability and requested advanced options; it does not build a proof model, enqueue inference, validate outputs, or promote parse-only options to runtime proof.";
        TrtexecLikeDeploymentOptions deployment = options.DeploymentOptions ?? TrtexecLikeDeploymentOptions.Default;
        bool fp8Requested = deployment.Fp8 || deployment.Best;
        bool debugRequested = !string.IsNullOrWhiteSpace(deployment.MarkDebug) || deployment.DumpDebugTensors;
        bool weightStreamingRequested = deployment.AllowWeightStreaming || deployment.WeightStreamingBudget.IsSpecified;

        try
        {
            TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
            TensorRtAdapterInfo adapter = TensorRtToolSupport.SelectAdapter(snapshot, options.TensorRtLine);
            bool modernTensorRtLine = options.TensorRtLine == TensorRtApiLine.TensorRt10 || options.TensorRtLine == TensorRtApiLine.TensorRt11;
            List<string> probeItems = new List<string>
            {
                "runtime:create:" + adapter.RuntimeCreationSupported,
                "builder:create:" + adapter.BuilderCreationSupported,
                "builder-config:create:" + adapter.BuilderCreationSupported,
                "engine-inspector:managed-api:" + modernTensorRtLine,
                "fp8-builder-flag:requested:" + fp8Requested + ":known:" + modernTensorRtLine,
                "debug-tensor-options:requested:" + debugRequested + ":known:" + modernTensorRtLine,
                "weight-streaming-options:requested:" + weightStreamingRequested + ":known:" + modernTensorRtLine
            };

            return new OnnxEngineCapabilityProbe(
                attempted: true,
                probeState: adapter.RuntimeCreationSupported || adapter.BuilderCreationSupported ? "capability-probe-only" : "dependency-unavailable",
                tensorRtLine: options.TensorRtLine,
                tensorRtVersion: snapshot.BuildInfo.TensorRtVersion,
                cudaToolkitVersion: snapshot.BuildInfo.CudaToolkitVersion,
                runtimeAvailable: adapter.RuntimeCreationSupported,
                builderAvailable: adapter.BuilderCreationSupported,
                builderConfigAvailable: adapter.BuilderCreationSupported,
                engineInspectorApiAvailable: modernTensorRtLine,
                fp8FlagRequested: fp8Requested,
                fp8FlagKnown: modernTensorRtLine,
                debugTensorOptionsRequested: debugRequested,
                debugTensorApiKnown: modernTensorRtLine,
                weightStreamingRequested: weightStreamingRequested,
                weightStreamingApiKnown: modernTensorRtLine,
                probeItems: probeItems,
                evidenceBoundary: boundary);
        }
        catch (Exception exception) when (exception is TensorRtException || exception is BridgeProbeException || exception is InvalidOperationException || exception is DllNotFoundException || exception is BadImageFormatException)
        {
            return new OnnxEngineCapabilityProbe(
                attempted: true,
                probeState: "capability-probe-failed",
                tensorRtLine: options.TensorRtLine,
                tensorRtVersion: string.Empty,
                cudaToolkitVersion: string.Empty,
                runtimeAvailable: false,
                builderAvailable: false,
                builderConfigAvailable: false,
                engineInspectorApiAvailable: options.TensorRtLine == TensorRtApiLine.TensorRt10 || options.TensorRtLine == TensorRtApiLine.TensorRt11,
                fp8FlagRequested: fp8Requested,
                fp8FlagKnown: false,
                debugTensorOptionsRequested: debugRequested,
                debugTensorApiKnown: false,
                weightStreamingRequested: weightStreamingRequested,
                weightStreamingApiKnown: false,
                probeItems: new[]
                {
                    "probe-error:" + exception.GetType().Name,
                    "runtime:create:false",
                    "builder:create:false"
                },
                evidenceBoundary: boundary);
        }
    }

    private static OnnxLoadedEngineDiagnostics ProbeLoadedEngineDiagnostics(
        OnnxEngineBuildOptions options,
        OnnxEnginePreflightMetadata preflightMetadata,
        List<string> log)
    {
        const string boundary = "load-engine readonly diagnostics may deserialize the engine and copy metadata, but it does not create execution bindings, enqueue inference, validate outputs, or prove package-consumer-runtime.";
        if (!preflightMetadata.Exists)
        {
            log.Add("LoadEngineReadonlyDiagnostics Attempted=False Succeeded=False Reason=engine file does not exist.");
            return new OnnxLoadedEngineDiagnostics(
                attempted: false,
                succeeded: false,
                diagnosticsState: "missing-engine-file",
                failureReason: "Serialized TensorRT engine file was not found.",
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
                evidenceBoundary: boundary);
        }

        try
        {
            TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
            TensorRtAdapterInfo adapter = TensorRtToolSupport.SelectAdapter(snapshot, options.TensorRtLine);
            log.Add($"LoadEngineReadonlyDiagnostics Preflight TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion} Runtime={adapter.RuntimeCreationSupported}");
            if (!adapter.RuntimeCreationSupported)
            {
                log.Add($"LoadEngineReadonlyDiagnostics Attempted=False Succeeded=False Reason={adapter.StatusMessage}");
                return new OnnxLoadedEngineDiagnostics(
                    attempted: false,
                    succeeded: false,
                    diagnosticsState: "runtime-unavailable",
                    failureReason: adapter.StatusMessage,
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
                    evidenceBoundary: boundary);
            }

            using TensorRtLogger logger = new TensorRtLogger(options.TensorRtLine);
            using TensorRtRuntime runtime = new TensorRtRuntime(logger);
            ConfigureRuntimeForEnginePolicies(runtime, options, log, "LoadEngineDiagnostics");
            using TensorRtEngine engine = runtime.DeserializeFromFile(preflightMetadata.Path);
            using TensorRtEngineInspector inspector = engine.CreateInspector();
            IReadOnlyList<TensorRtTensorInfo> tensors = engine.GetIOTensors();
            string inspectorInformation = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
            TryCollectLayerInformation(inspector, engine.LayerCount, options, log, "LoadEngine");
            string[] tensorSummaries = tensors
                .Select(static tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.DataType}:{tensor.Shape}")
                .ToArray();
            string readbackFingerprint = CreateLoadedEngineReadbackFingerprint(engine, inspectorInformation, tensorSummaries);
            string readbackSha256 = ComputeSha256(readbackFingerprint);
            log.Add($"LoadEngineReadonlyDiagnostics Attempted=True Succeeded=True IOTensors={engine.IOTensorCount} Layers={engine.LayerCount} Profiles={engine.OptimizationProfileCount} InspectorBytes={inspectorInformation.Length} ReadbackSha256={readbackSha256}");
            log.Add("LoadEngineReadonlyDiagnostics Tensors=" + string.Join("; ", tensorSummaries));
            return new OnnxLoadedEngineDiagnostics(
                attempted: true,
                succeeded: true,
                diagnosticsState: "readonly-deserialize-succeeded",
                failureReason: string.Empty,
                engineName: engine.Name,
                ioTensorCount: engine.IOTensorCount,
                layerCount: engine.LayerCount,
                optimizationProfileCount: engine.OptimizationProfileCount,
                deviceMemorySizeInBytes: engine.DeviceMemorySizeInBytes,
                auxiliaryStreamCount: engine.AuxiliaryStreamCount,
                capability: engine.Capability.ToString(),
                profilingVerbosity: engine.ProfilingVerbosity.ToString(),
                inspectorInformationLength: inspectorInformation.Length,
                ioTensorSummaries: tensorSummaries,
                readbackFingerprint: readbackFingerprint,
                readbackSha256: readbackSha256,
                evidenceBoundary: boundary);
        }
        catch (Exception exception) when (exception is TensorRtException || exception is BridgeProbeException || exception is InvalidOperationException || exception is FileNotFoundException || exception is DllNotFoundException || exception is BadImageFormatException)
        {
            log.Add($"LoadEngineReadonlyDiagnostics Attempted=True Succeeded=False Reason={exception.GetType().Name}:{exception.Message}");
            return new OnnxLoadedEngineDiagnostics(
                attempted: true,
                succeeded: false,
                diagnosticsState: "readonly-deserialize-failed",
                failureReason: exception.GetType().Name + ": " + exception.Message,
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
                evidenceBoundary: boundary);
        }
    }

    private static string CreateLoadedEngineReadbackFingerprint(
        TensorRtEngine engine,
        string inspectorInformation,
        IReadOnlyList<string> tensorSummaries)
    {
        return string.Join("|", new[]
        {
            "load-engine-readonly-diagnostics",
            "name=" + engine.Name,
            "io=" + engine.IOTensorCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "layers=" + engine.LayerCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "profiles=" + engine.OptimizationProfileCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "deviceMemory=" + engine.DeviceMemorySizeInBytes.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "auxStreams=" + engine.AuxiliaryStreamCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "capability=" + engine.Capability,
            "profilingVerbosity=" + engine.ProfilingVerbosity,
            "inspectorLength=" + (inspectorInformation ?? string.Empty).Length.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "tensors=" + string.Join(";", tensorSummaries ?? Array.Empty<string>())
        });
    }

    private static void TryCollectLayerInformationFromSerializedEngine(
        TensorRtRuntime runtime,
        string enginePath,
        OnnxEngineBuildOptions options,
        List<string> log,
        string source)
    {
        if (!options.DumpLayerInfo && string.IsNullOrWhiteSpace(options.ExportLayerInfoPath))
        {
            return;
        }

        try
        {
            using TensorRtEngine engine = runtime.DeserializeFromFile(enginePath);
            using TensorRtEngineInspector inspector = engine.CreateInspector();
            TryCollectLayerInformation(inspector, engine.LayerCount, options, log, source);
        }
        catch (Exception exception) when (exception is TensorRtException ||
                                          exception is BridgeProbeException ||
                                          exception is CudaException ||
                                          exception is DllNotFoundException ||
                                          exception is BadImageFormatException ||
                                          exception is FileNotFoundException)
        {
            log.Add($"LayerInfo Collected=False Source={source} Reason={exception.GetType().Name}:{exception.Message} EvidenceBoundary=copied-engine-inspector-diagnostics-only");
        }
    }

    private static void TryCollectLayerInformation(
        TensorRtEngineInspector inspector,
        int layerCount,
        OnnxEngineBuildOptions options,
        List<string> log,
        string source)
    {
        if (!options.DumpLayerInfo && string.IsNullOrWhiteSpace(options.ExportLayerInfoPath))
        {
            return;
        }

        string[] layerInformation;
        try
        {
            layerInformation = Enumerable.Range(0, Math.Max(0, layerCount))
                .Select(index => $"Layer[{index}] {inspector.GetLayerInformation(index, TensorRtLayerInformationFormat.Oneline)}")
                .ToArray();
        }
        catch (Exception exception) when (exception is TensorRtException ||
                                          exception is BridgeProbeException ||
                                          exception is CudaException ||
                                          exception is DllNotFoundException ||
                                          exception is BadImageFormatException)
        {
            log.Add($"LayerInfo Collected=False Source={source} Reason={exception.GetType().Name}:{exception.Message} EvidenceBoundary=copied-engine-inspector-diagnostics-only");
            return;
        }

        string content = layerInformation.Length == 0
            ? string.Empty
            : string.Join(Environment.NewLine, layerInformation) + Environment.NewLine;
        Encoding utf8NoBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
        int byteCount = utf8NoBom.GetByteCount(content);
        string sha256 = ComputeSha256(content);
        log.Add($"LayerInfo Collected=True Source={source} Layers={layerInformation.Length} Bytes={byteCount} Sha256={sha256} EvidenceBoundary=copied-engine-inspector-diagnostics-only");

        if (options.DumpLayerInfo)
        {
            foreach (string line in layerInformation)
            {
                log.Add("LayerInfo " + line);
            }
        }

        if (!string.IsNullOrWhiteSpace(options.ExportLayerInfoPath))
        {
            string fullPath = Path.GetFullPath(options.ExportLayerInfoPath);
            string? directory = Path.GetDirectoryName(fullPath);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.WriteAllText(fullPath, content, utf8NoBom);
            log.Add($"LayerInfo ExportRequested=True Written=True Path={fullPath} LengthBytes={byteCount} Sha256={sha256} EvidenceBoundary=copied-engine-inspector-diagnostics-only");
        }
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
