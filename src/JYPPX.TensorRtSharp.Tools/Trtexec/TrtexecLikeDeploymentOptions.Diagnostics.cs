using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class TrtexecLikeDeploymentOptions
{
    public IReadOnlyList<string> ToDiagnostics()
    {
        List<string> diagnostics = new List<string>
        {
            "BuilderOptimizationLevel=" + BuilderOptimizationLevel + " is applied to TensorRT builder config."
        };

        if (MaxAuxStreams.HasValue)
        {
            diagnostics.Add("MaxAuxStreams=" + MaxAuxStreams.Value + " is applied to TensorRT builder config.");
        }

        if (DeviceOrdinal.HasValue)
        {
            diagnostics.Add("Device=" + DeviceOrdinal.Value + " is applied on a dedicated host thread and read back before build/load/runtime work; dry-run remains parse-only.");
        }

        if (DlaCore.HasValue || AllowGpuFallback)
        {
            diagnostics.Add("DLA core and GPU fallback are applied through typed builder-config controls during a real build; the requested DLA core is validated against builder.DlaCoreCount, while TensorRT 11 additionally enables parser capability validation.");
            if (DlaCore.HasValue)
            {
                diagnostics.Add("UseDLACore=" + DlaCore.Value);
            }

            if (AllowGpuFallback)
            {
                diagnostics.Add("AllowGPUFallback=True");
            }
        }

        if (!string.IsNullOrWhiteSpace(TacticSources))
        {
            diagnostics.Add("TacticSources=" + TacticSources + " is applied relative to TensorRT's default tactic mask and read back during a real build.");
        }

        if (MemoryPoolSizes.Count > 0)
        {
            diagnostics.Add("MemPoolSize=" + MemoryPoolSizesToArgument() + " is applied during a real build through TensorRtBuilderConfig.SetMemoryPoolLimit and read back through GetMemoryPoolLimit; dry-run and load-engine remain parse-only.");
        }

        if (!string.IsNullOrWhiteSpace(InputIOFormats) || !string.IsNullOrWhiteSpace(OutputIOFormats))
        {
            diagnostics.Add("IO format options are validated against official type:format grammar, applied to parsed network I/O tensors, and read back during a real build; TensorRT 11 can constrain formats only when the requested type already matches the inferred tensor type.");
            AddDiagnostic(diagnostics, "InputIOFormats", InputIOFormats);
            AddDiagnostic(diagnostics, "OutputIOFormats", OutputIOFormats);
        }

        if (DirectIO)
        {
            diagnostics.Add("DirectIO=True is applied through the TensorRT DirectIO builder flag and read back during a real build.");
        }

        if (!string.IsNullOrWhiteSpace(CalibrationCacheFile))
        {
            diagnostics.Add("Calibration cache path is recorded; INT8 calibrator callbacks and cache lifecycle are not promoted by this build-report stage.");
            diagnostics.Add("CalibrationCacheFile=" + CalibrationCacheFile);
        }

        if (!string.IsNullOrWhiteSpace(Sparsity))
        {
            diagnostics.Add(string.Equals(Sparsity, "force", StringComparison.Ordinal)
                ? "Sparsity=force remains parse-only because official force mode rewrites model weights in addition to enabling sparse tactics."
                : "Sparsity=" + Sparsity + " is applied through the SparseWeights builder flag and read back during a real build.");
        }

        if (StronglyTyped)
        {
            diagnostics.Add("StronglyTyped=True uses the TensorRT 10 creation bit or the TensorRT 11 always-strongly-typed contract; TensorRT 8 keeps the option parse-only behind an explicit version guard.");
        }

        if (MinTiming.HasValue || AvgTiming.HasValue)
        {
            diagnostics.Add("Timing iteration controls are parsed for trtexec alignment; --avgTiming and TRT8 --minTiming are applied/read back during a real build, while TRT10/11 --minTiming and full tactic timing policy remain parse-only.");
            AddDiagnostic(diagnostics, "MinTiming", MinTiming);
            AddDiagnostic(diagnostics, "AvgTiming", AvgTiming);
        }

        if (!string.IsNullOrWhiteSpace(PrecisionConstraints) ||
            !string.IsNullOrWhiteSpace(LayerPrecisions) ||
            !string.IsNullOrWhiteSpace(LayerOutputTypes) ||
            Fp8 ||
            Best)
        {
            diagnostics.Add("Precision constraints and layer type rules are applied with exact-name precedence, single-wildcard matching, later-rule override, and typed readback on TensorRT 8/10; TensorRT 11 keeps removed setters behind explicit version guards.");
            AddDiagnostic(diagnostics, "PrecisionConstraints", PrecisionConstraints);
            AddDiagnostic(diagnostics, "LayerPrecisions", LayerPrecisions);
            AddDiagnostic(diagnostics, "LayerOutputTypes", LayerOutputTypes);
            AddDiagnostic(diagnostics, "Fp8", Fp8);
            AddDiagnostic(diagnostics, "Best", Best);
        }

        if (DumpRefit || !string.IsNullOrWhiteSpace(MarkDebug) || DumpDebugTensors)
        {
            diagnostics.Add("Refit dump and debug tensor arguments remain parse/report-only until model-specific ownership and runtime output proof are promoted.");
            AddDiagnostic(diagnostics, "DumpRefit", DumpRefit);
            AddDiagnostic(diagnostics, "MarkDebug", MarkDebug);
            AddDiagnostic(diagnostics, "DumpDebugTensors", DumpDebugTensors);
        }

        if (VersionCompatible || ExcludeLeanRuntime || StripWeights || Refit || !string.IsNullOrWhiteSpace(RefitFromOnnxPath) || !string.IsNullOrWhiteSpace(SaveRefittedEnginePath) || WeightStreamingBudget.IsSpecified || AllowWeightStreaming)
        {
            diagnostics.Add("Advanced engine packaging/refit flags are applied and read back during a real build with TensorRT-line guards; a requested weight-streaming budget is resolved and read back before execution contexts are created.");
            if (VersionCompatible)
            {
                diagnostics.Add("VersionCompatible=True");
            }

            if (ExcludeLeanRuntime)
            {
                diagnostics.Add("ExcludeLeanRuntime=True");
            }

            if (StripWeights)
            {
                diagnostics.Add("StripWeights=True");
            }

            if (Refit)
            {
                diagnostics.Add("Refit=True");
            }

            if (!string.IsNullOrWhiteSpace(RefitFromOnnxPath))
            {
                diagnostics.Add("RefitFromOnnx=" + RefitFromOnnxPath);
                diagnostics.Add("RefitFromOnnx performs copied missing/all-weight inventory and parser diagnostics before any execution context is created.");
            }

            if (!string.IsNullOrWhiteSpace(SaveRefittedEnginePath))
            {
                diagnostics.Add("SaveRefittedEngine=" + SaveRefittedEnginePath);
                diagnostics.Add("SaveRefittedEngine serializes the committed engine, disposes that engine, and independently reloads the persisted plan before any optional inference.");
            }

            if (AllowWeightStreaming)
            {
                diagnostics.Add("AllowWeightStreaming=True");
            }

            if (WeightStreamingBudget.IsSpecified)
            {
                diagnostics.Add("WeightStreamingBudget=" + WeightStreamingBudget.ArgumentValue);
                diagnostics.Add("WeightStreamingBudgetKind=" + WeightStreamingBudget.Kind);
            }
        }

        if (!string.IsNullOrWhiteSpace(ExportTimingCachePath))
        {
            diagnostics.Add("ExportTimingCache=" + ExportTimingCachePath + " is written after a successful TensorRT build through the typed timing-cache owner; the artifact remains build-cache evidence only.");
        }

        if (Safe || Consistency)
        {
            diagnostics.Add("Safety/consistency arguments are parse/report-only; they require model-specific validation before promotion.");
            if (Safe)
            {
                diagnostics.Add("Safe=True");
            }

            if (Consistency)
            {
                diagnostics.Add("Consistency=True");
            }
        }

        if (BuilderCache || NoBuilderCache)
        {
            diagnostics.Add("Builder cache policy arguments are parse/report-only; builder cache lifecycle is not promoted by this application stage.");
            if (BuilderCache)
            {
                diagnostics.Add("BuilderCache=True");
            }

            if (NoBuilderCache)
            {
                diagnostics.Add("NoBuilderCache=True");
            }
        }

        if (MaxNbTactics.HasValue || TilingOptimizationLevel.HasValue || L2LimitForTilingBytes.HasValue || QuantizationFlags.HasValue)
        {
            diagnostics.Add("Builder scalar deployment controls are applied/read back during a real build when the selected TensorRT line exposes the corresponding vendor API; unsupported lines remain controlled diagnostics.");
            AddDiagnostic(diagnostics, "MaxNbTactics", MaxNbTactics);
            AddDiagnostic(diagnostics, "TilingOptimizationLevel", TilingOptimizationLevel?.ToString() ?? string.Empty);
            AddDiagnostic(diagnostics, "L2LimitForTilingBytes", L2LimitForTilingBytes);
            AddDiagnostic(diagnostics, "QuantizationFlags", QuantizationFlags?.ToString() ?? string.Empty);
        }

        return diagnostics;
    }

    public string ScalarControlSummary => $"MaxNbTactics={MaxNbTactics?.ToString(CultureInfo.InvariantCulture) ?? string.Empty};TilingOptimizationLevel={TilingOptimizationLevel?.ToString() ?? string.Empty};L2LimitForTilingBytes={L2LimitForTilingBytes?.ToString(CultureInfo.InvariantCulture) ?? string.Empty};QuantizationFlags={QuantizationFlags?.ToString() ?? string.Empty}";

}
