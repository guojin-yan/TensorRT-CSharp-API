using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public static class TrtexecLikeParser
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
            refitFromOnnxPath: refitFromOnnxPath);
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
            saveProfilePath: FullPathOrEmpty(GetValue(args, "--saveProfile", string.Empty)));

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

    private static string GetValue(string[] args, string name, string defaultValue)
    {
        for (int index = 0; index < args.Length - 1; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }

            string prefix = name + "=";
            if (args[index].StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return args[index].Substring(prefix.Length);
            }
        }

        if (args.Length > 0)
        {
            string prefix = name + "=";
            string last = args[args.Length - 1];
            if (last.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return last.Substring(prefix.Length);
            }
        }

        return defaultValue;
    }

    private static bool HasSwitch(string[] args, string name)
    {
        return args.Any(argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase) ||
            argument.StartsWith(name + "=", StringComparison.OrdinalIgnoreCase));
    }

    private static int? ParseThreadMode(string[] args)
    {
        for (int index = 0; index < args.Length; index++)
        {
            string argument = args[index];
            if (string.Equals(argument, "--threads", StringComparison.OrdinalIgnoreCase))
            {
                if (index + 1 < args.Length && !IsOptionName(args[index + 1]))
                {
                    return ParsePositiveInt(args[index + 1], "--threads");
                }

                return 1;
            }

            const string prefix = "--threads=";
            if (argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                string value = argument.Substring(prefix.Length);
                return string.IsNullOrWhiteSpace(value) ? 1 : ParsePositiveInt(value, "--threads");
            }
        }

        return null;
    }

    private static string FullPathOrEmpty(string path)
    {
        return string.IsNullOrWhiteSpace(path) ? string.Empty : Path.GetFullPath(path);
    }

    private static IReadOnlyList<string> ParseList(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return Array.Empty<string>();
        }

        return value.Split(new[] { ';', ',' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(static item => item.Trim())
            .Where(static item => item.Length > 0)
            .ToArray();
    }

    private static IReadOnlyList<string> ParsePluginLibraries(string[] args)
    {
        List<string> values = new List<string>();
        AddValues(args, values, "--plugins");
        AddValues(args, values, "--plugin");
        AddValues(args, values, "--dynamicPlugins");
        AddValues(args, values, "--setPluginsToSerialize");

        if (values.Count == 0)
        {
            return Array.Empty<string>();
        }

        return values
            .SelectMany(static value => ParseList(value))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static void AddValues(string[] args, List<string> values, string name)
    {
        for (int index = 0; index < args.Length; index++)
        {
            string argument = args[index];
            if (string.Equals(argument, name, StringComparison.OrdinalIgnoreCase))
            {
                if (index == args.Length - 1 || IsOptionName(args[index + 1]))
                {
                    throw new ArgumentException($"{name} requires a plugin library path or path list.");
                }

                values.Add(args[index + 1]);
                index++;
                continue;
            }

            string prefix = name + "=";
            if (argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                string value = argument.Substring(prefix.Length);
                if (string.IsNullOrWhiteSpace(value))
                {
                    throw new ArgumentException($"{name} requires a plugin library path or path list.");
                }

                values.Add(value);
            }
        }
    }

    private static bool IsOptionName(string value)
    {
        return !string.IsNullOrWhiteSpace(value) && value.StartsWith("--", StringComparison.Ordinal);
    }

    private static bool ShouldTreatEngineAliasAsLoad(string[] args, string engineAliasPath)
    {
        return !string.IsNullOrWhiteSpace(engineAliasPath) &&
            string.IsNullOrWhiteSpace(GetValue(args, "--onnx", string.Empty)) &&
            string.IsNullOrWhiteSpace(GetValue(args, "--model", string.Empty)) &&
            string.IsNullOrWhiteSpace(GetValue(args, "--onnxFile", string.Empty)) &&
            !HasSwitch(args, "--buildOnly");
    }

    private static int ParsePositiveInt(string value, string argumentName)
    {
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) || parsed <= 0)
        {
            throw new ArgumentException($"{argumentName} must be a positive integer.");
        }

        return parsed;
    }

    private static int ParseNonNegativeInt(string value, string argumentName)
    {
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) || parsed < 0)
        {
            throw new ArgumentException($"{argumentName} must be a non-negative integer.");
        }

        return parsed;
    }

    private static int? ParseOptionalNonNegativeInt(string value, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        return ParseNonNegativeInt(value, argumentName);
    }

    private static int? ParseOptionalPositiveInt(string value, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        return ParsePositiveInt(value, argumentName);
    }

    private static float? ParseOptionalRangeFloat(string value, string argumentName, float minInclusive, float maxInclusive)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed) ||
            parsed < minInclusive ||
            parsed > maxInclusive)
        {
            throw new ArgumentException($"{argumentName} must be a number in the range [{minInclusive}, {maxInclusive}].");
        }

        return parsed;
    }

    private static int ParseRangeInt(string value, string argumentName, int minInclusive, int maxInclusive)
    {
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) ||
            parsed < minInclusive ||
            parsed > maxInclusive)
        {
            throw new ArgumentException($"{argumentName} must be an integer in the range [{minInclusive}, {maxInclusive}].");
        }

        return parsed;
    }

    private static ulong ParseWorkspaceBytes(string value)
    {
        return ParseMemorySizeBytes(value, "--workspace");
    }

    private static ulong? ParseOptionalMemorySizeBytes(string value, string argumentName)
    {
        return string.IsNullOrWhiteSpace(value) ? null : ParseMemorySizeBytes(value, argumentName);
    }

    private static long? ParseOptionalLongMemorySizeBytes(string value, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        ulong bytes = ParseMemorySizeBytes(value, argumentName);
        if (bytes > long.MaxValue)
        {
            throw new ArgumentException($"{argumentName} must not exceed {long.MaxValue} bytes.");
        }

        return (long)bytes;
    }

    private static TensorRtTilingOptimizationLevel? ParseOptionalTilingOptimizationLevel(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        string normalized = value.Trim().Replace("-", string.Empty).Replace("_", string.Empty).ToLowerInvariant();
        return normalized switch
        {
            "0" or "none" or "off" => TensorRtTilingOptimizationLevel.None,
            "1" or "fast" => TensorRtTilingOptimizationLevel.Fast,
            "2" or "moderate" => TensorRtTilingOptimizationLevel.Moderate,
            "3" or "full" => TensorRtTilingOptimizationLevel.Full,
            _ => throw new ArgumentException("--tilingOptimizationLevel must be none, fast, moderate, full, or 0..3.")
        };
    }

    private static TensorRtQuantizationFlags? ParseOptionalQuantizationFlags(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        TensorRtQuantizationFlags flags = TensorRtQuantizationFlags.None;
        foreach (string token in value.Split(new[] { ',', '|', ';' }, StringSplitOptions.RemoveEmptyEntries))
        {
            string normalized = token.Trim().Replace("-", string.Empty).Replace("_", string.Empty).ToLowerInvariant();
            flags |= normalized switch
            {
                "none" or "0" => TensorRtQuantizationFlags.None,
                "calibratebeforefusion" => TensorRtQuantizationFlags.CalibrateBeforeFusion,
                _ => throw new ArgumentException("--quantizationFlags supports none or calibrateBeforeFusion.")
            };
        }

        return flags;
    }

    private static IReadOnlyList<TrtexecLikeMemoryPoolSize> ParseMemoryPoolSizes(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return Array.Empty<TrtexecLikeMemoryPoolSize>();
        }

        List<TrtexecLikeMemoryPoolSize> sizes = new List<TrtexecLikeMemoryPoolSize>();
        foreach (string item in value.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
        {
            string trimmed = item.Trim();
            int separator = trimmed.IndexOf(':');
            if (separator <= 0 || separator == trimmed.Length - 1)
            {
                throw new ArgumentException("--memPoolSize entries must use poolName:sizeMiB syntax.");
            }

            string name = trimmed.Substring(0, separator);
            string sizeText = trimmed.Substring(separator + 1);
            ulong sizeMiB = ParseMemorySizeMiB(sizeText, "--memPoolSize");
            sizes.Add(new TrtexecLikeMemoryPoolSize(name, sizeMiB));
        }

        return sizes;
    }

    private static string NormalizeProfilingVerbosity(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return "layer_names_only";
        }

        string normalized = value.Trim().Replace("-", "_").ToLowerInvariant();
        return normalized switch
        {
            "none" => "none",
            "layer_names_only" or "layernamesonly" or "layer_names" or "names" => "layer_names_only",
            "detailed" or "detail" or "verbose" => "detailed",
            _ => throw new ArgumentException("--profilingVerbosity must be none, layer_names_only, or detailed.")
        };
    }

    private static string NormalizeTacticSources(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        List<string> normalizedSources = new List<string>();
        foreach (string item in value.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
        {
            string trimmed = item.Trim();
            if (trimmed.Length < 2 || (trimmed[0] != '+' && trimmed[0] != '-'))
            {
                throw new ArgumentException("--tacticSources entries must begin with + or -.");
            }

            string source = trimmed.Substring(1)
                .Replace("-", string.Empty, StringComparison.Ordinal)
                .Replace("_", string.Empty, StringComparison.Ordinal)
                .ToLowerInvariant();
            string canonicalSource = source switch
            {
                "cublas" => "CUBLAS",
                "cublaslt" => "CUBLAS_LT",
                "cudnn" => "CUDNN",
                "edgemaskconvolutions" or "edgemask" => "EDGE_MASK_CONVOLUTIONS",
                "jitconvolutions" or "jit" => "JIT_CONVOLUTIONS",
                _ => throw new ArgumentException($"Unsupported --tacticSources entry '{trimmed}'.")
            };
            normalizedSources.Add(trimmed[0] + canonicalSource);
        }

        return string.Join(",", normalizedSources);
    }

    private static string NormalizeSparsity(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        return value.Trim().ToLowerInvariant() switch
        {
            "disable" or "disabled" => "disable",
            "enable" or "enabled" => "enable",
            "force" => "force",
            _ => throw new ArgumentException("--sparsity must be disable, enable, or force.")
        };
    }

    private static ulong ParseMemorySizeMiB(string value, string argumentName)
    {
        ulong bytes = ParseMemorySizeBytes(value, argumentName);
        const ulong mib = 1024UL * 1024UL;
        if (bytes % mib != 0)
        {
            throw new ArgumentException($"{argumentName} size must resolve to a whole number of MiB.");
        }

        return bytes / mib;
    }

    private static ulong ParseMemorySizeBytes(string value, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException($"{argumentName} requires a memory size.");
        }

        string trimmed = value.Trim();
        string numberText = trimmed;
        decimal multiplier = 1024m * 1024m;
        if (TryTrimSuffix(trimmed, "gib", out numberText) || TryTrimSuffix(trimmed, "gb", out numberText))
        {
            multiplier = 1024m * 1024m * 1024m;
        }
        else if (TryTrimSuffix(trimmed, "mib", out numberText) || TryTrimSuffix(trimmed, "mb", out numberText))
        {
            multiplier = 1024m * 1024m;
        }
        else if (TryTrimSuffix(trimmed, "kib", out numberText) || TryTrimSuffix(trimmed, "kb", out numberText))
        {
            multiplier = 1024m;
        }
        else if (TryTrimSuffix(trimmed, "b", out numberText))
        {
            multiplier = 1m;
        }

        if (!decimal.TryParse(numberText.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out decimal parsed) || parsed < 0m)
        {
            throw new ArgumentException($"{argumentName} must be a non-negative memory size, defaulting to MiB when no suffix is supplied.");
        }

        decimal bytes = parsed * multiplier;
        if (bytes != decimal.Truncate(bytes) || bytes > ulong.MaxValue)
        {
            throw new ArgumentException($"{argumentName} size must resolve to a whole number of bytes.");
        }

        return checked((ulong)bytes);
    }

    private static bool TryTrimSuffix(string value, string suffix, out string withoutSuffix)
    {
        if (value.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
        {
            withoutSuffix = value.Substring(0, value.Length - suffix.Length);
            return true;
        }

        withoutSuffix = value;
        return false;
    }

    private static string FirstNonEmpty(string first, string second)
    {
        return string.IsNullOrWhiteSpace(first) ? second : first;
    }
}
