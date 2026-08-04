using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public static partial class TrtexecLikeParser
{
    public static TrtexecLikeOptions Parse(string[] args)
    {
        if (args == null)
        {
            throw new ArgumentNullException(nameof(args));
        }

        TensorRtApiLine tensorRtLine = TensorRtToolSupport.ResolveLine(GetValue(args, "--tensor-rt-line", "10"));
        string onnxPath = FullPathOrEmpty(FirstNonEmpty(GetValue(args, "--onnx", string.Empty), FirstNonEmpty(GetValue(args, "--model", string.Empty), GetValue(args, "--onnxFile", string.Empty))));
        string engineAliasPath = FirstNonEmpty(GetValue(args, "--engine", string.Empty), FirstNonEmpty(GetValue(args, "--plan", string.Empty), GetValue(args, "--engineFile", string.Empty)));
        bool engineAliasIsLoad = ShouldTreatEngineAliasAsLoad(args, engineAliasPath);
        string saveEnginePath = FullPathOrEmpty(FirstNonEmpty(GetValue(args, "--saveEngine", string.Empty), FirstNonEmpty(GetValue(args, "--save-engine", string.Empty), engineAliasIsLoad ? string.Empty : engineAliasPath)));
        string loadEnginePath = FullPathOrEmpty(FirstNonEmpty(GetValue(args, "--loadEngine", string.Empty), FirstNonEmpty(GetValue(args, "--load-engine", string.Empty), engineAliasIsLoad ? engineAliasPath : string.Empty)));
        bool dryRun = HasSwitch(args, "--dryRun") || HasSwitch(args, "--previewOnly");
        if (!dryRun && !string.IsNullOrWhiteSpace(onnxPath) && !File.Exists(onnxPath))
        {
            throw new FileNotFoundException("ONNX model file was not found.", onnxPath);
        }

        if (!dryRun && !string.IsNullOrWhiteSpace(loadEnginePath) && !File.Exists(loadEnginePath))
        {
            throw new FileNotFoundException("Serialized TensorRT engine file was not found.", loadEnginePath);
        }

        if (!string.IsNullOrWhiteSpace(onnxPath) && !string.IsNullOrWhiteSpace(loadEnginePath))
        {
            throw new ArgumentException("--onnx and --loadEngine are mutually exclusive in this sample stage.");
        }

        List<string> diagnostics = new List<string>();
        if (dryRun)
        {
            diagnostics.Add("Dry run requested; TensorRT runtime probing, model parsing, engine build, and inference are skipped.");
        }

        string calibrationCacheFile = FullPathOrEmpty(GetValue(args, "--calib", GetValue(args, "--calibrationCacheFile", string.Empty)));
        bool builderCache = HasSwitch(args, "--builderCache");
        bool noBuilderCache = HasSwitch(args, "--noBuilderCache");
        if (builderCache && noBuilderCache)
        {
            throw new ArgumentException("--builderCache and --noBuilderCache are mutually exclusive.");
        }
        if (HasSwitch(args, "--int8"))
        {
            diagnostics.Add("INT8 flag parsed; calibrator and calibration cache are not implemented by this sample stage.");
        }

        if (HasSwitch(args, "--useCudaGraph"))
        {
            diagnostics.Add("CUDA graph execution is requested; bounded runtime attempts per-context capture, instantiation, and launch, then records a controlled direct-enqueue fallback if capture is unavailable.");
        }

        string inputIOFormats = TrtexecLikeBuildPolicy.NormalizeIoFormats(GetValue(args, "--inputIOFormats", string.Empty), "--inputIOFormats");
        string outputIOFormats = TrtexecLikeBuildPolicy.NormalizeIoFormats(GetValue(args, "--outputIOFormats", string.Empty), "--outputIOFormats");
        string precisionConstraints = TrtexecLikeBuildPolicy.NormalizePrecisionConstraints(GetValue(args, "--precisionConstraints", string.Empty));
        string layerPrecisions = TrtexecLikeBuildPolicy.NormalizeLayerPrecisions(GetValue(args, "--layerPrecisions", string.Empty));
        string layerOutputTypes = TrtexecLikeBuildPolicy.NormalizeLayerOutputTypes(GetValue(args, "--layerOutputTypes", string.Empty));
        TrtexecLikeBuildPolicy.ValidatePolicyCombination(precisionConstraints, layerPrecisions, layerOutputTypes);

        bool stronglyTyped = HasSwitch(args, "--stronglyTyped");
        bool allowWeightStreaming = HasSwitch(args, "--allowWeightStreaming");
        bool versionCompatible = HasSwitch(args, "--versionCompatible");
        bool excludeLeanRuntime = HasSwitch(args, "--excludeLeanRuntime");
        bool stripWeights = HasSwitch(args, "--stripWeights");
        string refitFromOnnxPath = FullPathOrEmpty(GetValue(args, "--refitFromOnnx", string.Empty));
        string saveRefittedEnginePath = FullPathOrEmpty(GetValue(args, "--saveRefittedEngine", string.Empty));
        TrtexecLikeWeightStreamingBudget weightStreamingBudget =
            TrtexecLikeWeightStreamingBudget.Parse(GetValue(args, "--weightStreamingBudget", string.Empty));
        bool buildsEngine = string.IsNullOrWhiteSpace(loadEnginePath);
        bool stopsAfterBuild = dryRun || HasSwitch(args, "--buildOnly") || HasSwitch(args, "--skipInference");
        if (!dryRun && !string.IsNullOrWhiteSpace(refitFromOnnxPath) && !File.Exists(refitFromOnnxPath))
        {
            throw new FileNotFoundException("ONNX refit source file was not found.", refitFromOnnxPath);
        }

        if (!string.IsNullOrWhiteSpace(refitFromOnnxPath))
        {
            if (!buildsEngine || string.IsNullOrWhiteSpace(onnxPath))
            {
                throw new ArgumentException("--refitFromOnnx currently requires an --onnx build source; load-engine refit is not implicit.");
            }

            if (!stripWeights)
            {
                throw new ArgumentException("--refitFromOnnx requires --stripWeights so the lifecycle begins from an explicitly stripped plan.");
            }

            if (!HasSwitch(args, "--refit"))
            {
                throw new ArgumentException("--refitFromOnnx requires --refit so the stripped plan is built with the explicit refittable engine flag.");
            }

            if (!dryRun && tensorRtLine == TensorRtApiLine.TensorRt8)
            {
                throw new ArgumentException("--refitFromOnnx requires TensorRT 10 or TensorRT 11; TensorRT 8 has no ONNX parser-refitter API.");
            }
        }

        if (!string.IsNullOrWhiteSpace(saveRefittedEnginePath))
        {
            if (string.IsNullOrWhiteSpace(refitFromOnnxPath))
            {
                throw new ArgumentException("--saveRefittedEngine requires --refitFromOnnx so only a committed refitted engine can be persisted.");
            }

            if (PathsEqual(saveRefittedEnginePath, saveEnginePath))
            {
                throw new ArgumentException("--saveRefittedEngine must differ from --saveEngine so the stripped source plan and refitted output plan remain distinct.");
            }

            if (PathsEqual(saveRefittedEnginePath, onnxPath) || PathsEqual(saveRefittedEnginePath, refitFromOnnxPath))
            {
                throw new ArgumentException("--saveRefittedEngine must not overwrite an ONNX build or refit source.");
            }

            if (Directory.Exists(saveRefittedEnginePath))
            {
                throw new ArgumentException("--saveRefittedEngine must name a file, not a directory.");
            }
        }
        if (excludeLeanRuntime && !versionCompatible)
        {
            throw new ArgumentException("--excludeLeanRuntime requires --versionCompatible.");
        }

        if (buildsEngine && allowWeightStreaming && !stronglyTyped)
        {
            throw new ArgumentException("--allowWeightStreaming requires --stronglyTyped when building an engine.");
        }

        if (buildsEngine && weightStreamingBudget.IsSpecified && !allowWeightStreaming)
        {
            throw new ArgumentException("--weightStreamingBudget requires --allowWeightStreaming when building an engine.");
        }

        if (excludeLeanRuntime && !stopsAfterBuild)
        {
            throw new ArgumentException("--excludeLeanRuntime requires --buildOnly or --skipInference until an external lean runtime path is configured.");
        }

        if (stripWeights && !stopsAfterBuild && string.IsNullOrWhiteSpace(refitFromOnnxPath))
        {
            throw new ArgumentException("--stripWeights requires --buildOnly or --skipInference because stripped weights must be supplied through a separate refit lifecycle before inference.");
        }

        TrtexecLikeDeploymentOptions deploymentOptions = new TrtexecLikeDeploymentOptions(
            deviceOrdinal: ParseOptionalNonNegativeInt(GetValue(args, "--device", string.Empty), "--device"),
            builderOptimizationLevel: ParseRangeInt(GetValue(args, "--builderOptimizationLevel", "3"), "--builderOptimizationLevel", 0, 5),
            maxAuxStreams: ParseOptionalNonNegativeInt(GetValue(args, "--maxAuxStreams", string.Empty), "--maxAuxStreams"),
            dlaCore: ParseOptionalNonNegativeInt(GetValue(args, "--useDLACore", string.Empty), "--useDLACore"),
            allowGpuFallback: HasSwitch(args, "--allowGPUFallback"),
            tacticSources: NormalizeTacticSources(GetValue(args, "--tacticSources", string.Empty)),
            memoryPoolSizes: ParseMemoryPoolSizes(GetValue(args, "--memPoolSize", string.Empty)),
            inputIOFormats: inputIOFormats,
            outputIOFormats: outputIOFormats,
            calibrationCacheFile: calibrationCacheFile,
            directIO: HasSwitch(args, "--directIO"),
            sparsity: NormalizeSparsity(GetValue(args, "--sparsity", string.Empty)),
            stronglyTyped: stronglyTyped,
            minTiming: ParseOptionalPositiveInt(GetValue(args, "--minTiming", string.Empty), "--minTiming"),
            avgTiming: ParseOptionalPositiveInt(GetValue(args, "--avgTiming", string.Empty), "--avgTiming"),
            precisionConstraints: precisionConstraints,
            layerPrecisions: layerPrecisions,
            layerOutputTypes: layerOutputTypes,
            fp8: HasSwitch(args, "--fp8"),
            best: HasSwitch(args, "--best"),
            dumpRefit: HasSwitch(args, "--dumpRefit"),
            allowWeightStreaming: allowWeightStreaming,
            markDebug: GetValue(args, "--markDebug", string.Empty),
            dumpDebugTensors: HasSwitch(args, "--dumpDebugTensors"),
            versionCompatible: versionCompatible,
            excludeLeanRuntime: excludeLeanRuntime,
            stripWeights: stripWeights,
            refit: HasSwitch(args, "--refit"),
            weightStreamingBudgetBytes: weightStreamingBudget.Bytes,
            exportTimingCachePath: FullPathOrEmpty(GetValue(args, "--exportTimingCache", string.Empty)),
            safe: HasSwitch(args, "--safe"),
            consistency: HasSwitch(args, "--consistency"),
            builderCache: builderCache,
            noBuilderCache: noBuilderCache,
            maxNbTactics: ParseOptionalNonNegativeInt(GetValue(args, "--maxNbTactics", string.Empty), "--maxNbTactics"),
            tilingOptimizationLevel: ParseOptionalTilingOptimizationLevel(GetValue(args, "--tilingOptimizationLevel", string.Empty)),
            l2LimitForTilingBytes: ParseOptionalLongMemorySizeBytes(GetValue(args, "--l2LimitForTiling", string.Empty), "--l2LimitForTiling"),
            quantizationFlags: ParseOptionalQuantizationFlags(GetValue(args, "--quantizationFlags", string.Empty)),
            weightStreamingBudget: weightStreamingBudget,
            refitFromOnnxPath: refitFromOnnxPath,
            saveRefittedEnginePath: saveRefittedEnginePath);
        string shapes = FirstNonEmpty(GetValue(args, "--shapes", string.Empty), GetValue(args, "--inputShapes", string.Empty));
        string minShapes = FirstNonEmpty(GetValue(args, "--minShapes", string.Empty), shapes);
        string optShapes = FirstNonEmpty(GetValue(args, "--optShapes", string.Empty), shapes);
        string maxShapes = FirstNonEmpty(GetValue(args, "--maxShapes", string.Empty), shapes);
        TrtexecLikeRuntimeOptions runtimeOptions = new TrtexecLikeRuntimeOptions(
            noDataTransfers: HasSwitch(args, "--noDataTransfers"),
            useSpinWait: HasSwitch(args, "--useSpinWait"),
            threads: ParseThreadMode(args),
            avgRuns: ParseOptionalPositiveInt(GetValue(args, "--avgRuns", string.Empty), "--avgRuns"),
            percentile: ParseOptionalRangeFloat(GetValue(args, "--percentile", string.Empty), "--percentile", 0.0f, 100.0f),
            sleepTimeMilliseconds: ParseOptionalNonNegativeInt(GetValue(args, "--sleepTime", string.Empty), "--sleepTime"),
            idleTimeMilliseconds: ParseOptionalNonNegativeInt(GetValue(args, "--idleTime", string.Empty), "--idleTime"),
            infStreams: ParseOptionalPositiveInt(GetValue(args, "--infStreams", string.Empty), "--infStreams"),
            loadInputs: GetValue(args, "--loadInputs", string.Empty),
            dumpOutput: HasSwitch(args, "--dumpOutput"),
            dumpRawBindingsToFile: FullPathOrEmpty(GetValue(args, "--dumpRawBindingsToFile", string.Empty)),
            exportOutputPath: FullPathOrEmpty(GetValue(args, "--exportOutput", string.Empty)),
            exportTimesPath: FullPathOrEmpty(GetValue(args, "--exportTimes", string.Empty)),
            exportProfilePath: FullPathOrEmpty(GetValue(args, "--exportProfile", string.Empty)),
            saveProfilePath: FullPathOrEmpty(GetValue(args, "--saveProfile", string.Empty)),
            referenceOutputs: GetValue(args, "--referenceOutputs", string.Empty),
            referenceAbsoluteTolerance: ParseOptionalRangeFloat(GetValue(args, "--referenceAbsTolerance", string.Empty), "--referenceAbsTolerance", 0.0f, float.MaxValue) ?? 0.0f,
            referenceRelativeTolerance: ParseOptionalRangeFloat(GetValue(args, "--referenceRelTolerance", string.Empty), "--referenceRelTolerance", 0.0f, float.MaxValue) ?? 0.0f,
            referenceNaNPolicy: ParseReferenceNaNPolicy(GetValue(args, "--referenceNaNPolicy", "reject")),
            referenceInfinityPolicy: ParseReferenceInfinityPolicy(GetValue(args, "--referenceInfinityPolicy", "exact")));

        return new TrtexecLikeOptions(
            tensorRtLine,
            onnxPath,
            saveEnginePath,
            loadEnginePath,
            explicitBatch: !HasSwitch(args, "--implicitBatch"),
            fp16: HasSwitch(args, "--fp16"),
            int8: HasSwitch(args, "--int8"),
            bf16: HasSwitch(args, "--bf16"),
            tf32: !HasSwitch(args, "--noTF32"),
            workspaceBytes: ParseWorkspaceBytes(GetValue(args, "--workspace", "64")),
            timingCacheFile: FullPathOrEmpty(FirstNonEmpty(GetValue(args, "--timingCacheFile", string.Empty), GetValue(args, "--timingCache", string.Empty))),
            plugins: ParsePluginLibraries(args),
            profilingVerbosity: NormalizeProfilingVerbosity(GetValue(args, "--profilingVerbosity", HasSwitch(args, "--verbose") ? "detailed" : "layer_names_only")),
            dumpLayerInfo: HasSwitch(args, "--dumpLayerInfo"),
            exportLayerInfoPath: FullPathOrEmpty(GetValue(args, "--exportLayerInfo", string.Empty)),
            dumpProfile: HasSwitch(args, "--dumpProfile"),
            separateProfileRun: HasSwitch(args, "--separateProfileRun"),
            buildOnly: HasSwitch(args, "--buildOnly"),
            skipInference: HasSwitch(args, "--skipInference"),
            dryRun: dryRun,
            iterations: ParsePositiveInt(GetValue(args, "--iterations", "10"), "--iterations"),
            warmUpMilliseconds: ParseNonNegativeInt(GetValue(args, "--warmUp", "200"), "--warmUp"),
            durationSeconds: ParseNonNegativeInt(GetValue(args, "--duration", "3"), "--duration"),
            streams: ParsePositiveInt(GetValue(args, "--streams", "1"), "--streams"),
            batch: ParsePositiveInt(GetValue(args, "--batch", "2"), "--batch"),
            useCudaGraph: HasSwitch(args, "--useCudaGraph"),
            shapeProfile: EngineBuildProfile.Parse(
                minShapes,
                optShapes,
                maxShapes),
            exportReportPath: FullPathOrEmpty(FirstNonEmpty(GetValue(args, "--exportReport", string.Empty), GetValue(args, "--report", string.Empty))),
            evidenceSidecarPath: FullPathOrEmpty(GetValue(args, "--evidenceSidecar", string.Empty)),
            deploymentOptions: deploymentOptions,
            runtimeOptions: runtimeOptions,
            diagnostics: diagnostics);
    }
}
