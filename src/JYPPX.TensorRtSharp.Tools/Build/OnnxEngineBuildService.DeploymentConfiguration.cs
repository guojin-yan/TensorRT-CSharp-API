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
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class OnnxEngineBuildService
{
    private static void ApplyDeploymentOptions(TensorRtBuilder builder, TensorRtBuilderConfig config, OnnxEngineBuildOptions options, List<string> log)
    {
        foreach (TrtexecLikeMemoryPoolSize memoryPool in options.DeploymentOptions.MemoryPoolSizes)
        {
            TensorRtMemoryPoolType pool = memoryPool.ToTensorRtMemoryPoolType();
            config.SetMemoryPoolLimit(pool, memoryPool.SizeBytes);
            ulong readbackBytes = config.GetMemoryPoolLimit(pool);
            log.Add(
                $"TrtexecMemoryPool Applied=True Name={memoryPool.Name} Pool={pool} " +
                $"RequestedBytes={memoryPool.SizeBytes} ReadbackBytes={readbackBytes} " +
                $"ReadbackMatch={readbackBytes == memoryPool.SizeBytes}");
        }

        if (options.DeploymentOptions.MaxAuxStreams.HasValue)
        {
            config.SetMaxAuxStreams(options.DeploymentOptions.MaxAuxStreams.Value);
        }

        if (options.DeploymentOptions.AvgTiming.HasValue)
        {
            int requestedIterations = options.DeploymentOptions.AvgTiming.Value;
            config.SetAverageTimingIterations(requestedIterations);
            int readbackIterations = config.GetAverageTimingIterations();
            log.Add(
                $"TrtexecTiming AverageApplied=True RequestedIterations={requestedIterations} " +
                $"ReadbackIterations={readbackIterations} ReadbackMatch={readbackIterations == requestedIterations} " +
                "EvidenceBoundary=builder-config-readback-only");
        }

        if (options.DeploymentOptions.MinTiming.HasValue)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
            {
                int requestedIterations = options.DeploymentOptions.MinTiming.Value;
                config.SetMinTimingIterationsCompatibility(requestedIterations);
                int readbackIterations = config.MinTimingIterationsCompatibility;
                log.Add(
                    $"TrtexecTiming MinimumApplied=True VersionGuard=TRT8 RequestedIterations={requestedIterations} " +
                    $"ReadbackIterations={readbackIterations} ReadbackMatch={readbackIterations == requestedIterations} " +
                    "EvidenceBoundary=builder-config-readback-only");
            }
            else
            {
                log.Add(
                    $"TrtexecTiming MinimumApplied=False VersionGuard=TRT8 RequestedIterations={options.DeploymentOptions.MinTiming.Value} " +
                    "Reason=TensorRT 10/11 use average timing iterations; legacy minimum setter is not available on this API line.");
            }
        }

        ApplyBuilderScalarDeploymentControls(config, options, log);

        if (!string.IsNullOrWhiteSpace(options.ProfilingVerbosity))
        {
            config.SetProfilingVerbosity(ParseProfilingVerbosity(options.ProfilingVerbosity));
        }

        TrtexecLikeDeploymentOptions deployment = options.DeploymentOptions;
        if (deployment.DlaCore.HasValue)
        {
            int requestedCore = deployment.DlaCore.Value;
            int dlaCoreCount = builder.DlaCoreCount;
            if (requestedCore >= dlaCoreCount)
            {
                throw new InvalidOperationException($"--useDLACore requested core {requestedCore}, but TensorRT reports {dlaCoreCount} DLA core(s).");
            }

            config.SetDefaultDeviceType(TensorRtDeviceType.Dla);
            config.SetDlaCore(requestedCore);
            TensorRtDeviceType deviceReadback = config.GetDefaultDeviceType();
            int coreReadback = config.GetDlaCore();
            log.Add(
                $"TrtexecDeploymentControl Name=DlaCore Applied=True Requested={requestedCore} " +
                $"Readback={coreReadback} DeviceReadback={deviceReadback} DlaCoreCount={dlaCoreCount} " +
                $"ReadbackMatch={coreReadback == requestedCore && deviceReadback == TensorRtDeviceType.Dla}");
        }

        if (deployment.AllowGpuFallback)
        {
            config.SetFlag(TensorRtBuilderFlag.GpuFallback, true);
            bool readback = config.GetFlag(TensorRtBuilderFlag.GpuFallback);
            log.Add($"TrtexecDeploymentControl Name=GpuFallback Applied=True Requested=True Readback={readback} ReadbackMatch={readback}");
        }

        if (!string.IsNullOrWhiteSpace(deployment.TacticSources))
        {
            TensorRtTacticSources defaultSources = config.GetTacticSources();
            TensorRtTacticSources requestedSources = deployment.ResolveTacticSources(defaultSources);
            config.SetTacticSources(requestedSources);
            TensorRtTacticSources readbackSources = config.GetTacticSources();
            log.Add(
                $"TrtexecDeploymentControl Name=TacticSources Applied=True Requested={requestedSources} " +
                $"Readback={readbackSources} Default={defaultSources} ReadbackMatch={readbackSources == requestedSources}");
        }

        if (deployment.DirectIO)
        {
            config.SetFlag(TensorRtBuilderFlag.DirectIO, true);
            bool readback = config.GetFlag(TensorRtBuilderFlag.DirectIO);
            log.Add($"TrtexecDeploymentControl Name=DirectIO Applied=True Requested=True Readback={readback} ReadbackMatch={readback}");
        }

        if (string.Equals(deployment.Sparsity, "enable", StringComparison.Ordinal) ||
            string.Equals(deployment.Sparsity, "disable", StringComparison.Ordinal))
        {
            bool requested = string.Equals(deployment.Sparsity, "enable", StringComparison.Ordinal);
            config.SetFlag(TensorRtBuilderFlag.SparseWeights, requested);
            bool readback = config.GetFlag(TensorRtBuilderFlag.SparseWeights);
            log.Add($"TrtexecDeploymentControl Name=Sparsity Applied=True Requested={deployment.Sparsity} Readback={readback} ReadbackMatch={readback == requested}");
        }
        else if (string.Equals(deployment.Sparsity, "force", StringComparison.Ordinal))
        {
            log.Add("TrtexecDeploymentControl Name=Sparsity Applied=False Requested=force Reason=official-force-mode-rewrites-model-weights-and-is-not-implemented");
        }

        ApplyEnginePackagingOptions(config, options, log);
    }

    private static void ApplyEnginePackagingOptions(
        TensorRtBuilderConfig config,
        OnnxEngineBuildOptions options,
        List<string> log)
    {
        TrtexecLikeDeploymentOptions deployment = options.DeploymentOptions;
        ApplyBuilderFlagWithReadback(config, TensorRtBuilderFlag.VersionCompatible, deployment.VersionCompatible, "VersionCompatible", log);
        ApplyBuilderFlagWithReadback(config, TensorRtBuilderFlag.ExcludeLeanRuntime, deployment.ExcludeLeanRuntime, "ExcludeLeanRuntime", log);
        if (deployment.Refit && options.TensorRtLine == TensorRtApiLine.TensorRt8 && deployment.VersionCompatible)
        {
            log.Add("TrtexecDeploymentControl Name=Refit Applied=False Requested=True VersionGuard=TRT8 Reason=version-compatible-refit-vendor-readback-conflict");
        }
        else
        {
            ApplyBuilderFlagWithReadback(config, TensorRtBuilderFlag.Refit, deployment.Refit, "Refit", log);
        }

        if (deployment.StripWeights)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
            {
                log.Add("TrtexecDeploymentControl Name=StripWeights Applied=False Requested=True VersionGuard=TRT8 Reason=strip-plan-and-refit-identical-flags-are-not-available");
            }
            else
            {
                TensorRtBuilderFlag refitMode = deployment.Refit
                    ? TensorRtBuilderFlag.Refit
                    : TensorRtBuilderFlag.RefitIdentical;
                config.SetFlag(refitMode, true);
                config.SetFlag(TensorRtBuilderFlag.StripPlan, true);
                bool stripReadback = config.GetFlag(TensorRtBuilderFlag.StripPlan);
                bool refitReadback = config.GetFlag(refitMode);
                log.Add(
                    $"TrtexecDeploymentControl Name=StripWeights Applied=True Requested=True Readback={stripReadback} " +
                    $"RefitMode={refitMode} RefitReadback={refitReadback} ReadbackMatch={stripReadback && refitReadback}");
            }
        }

        if (deployment.AllowWeightStreaming)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
            {
                log.Add("TrtexecDeploymentControl Name=WeightStreaming Applied=False Requested=True VersionGuard=TRT8 Reason=weight-streaming-builder-flag-is-not-available");
            }
            else
            {
                ApplyBuilderFlagWithReadback(config, TensorRtBuilderFlag.WeightStreaming, true, "WeightStreaming", log);
            }
        }
    }

    private static void ApplyBuilderFlagWithReadback(
        TensorRtBuilderConfig config,
        TensorRtBuilderFlag flag,
        bool requested,
        string name,
        List<string> log)
    {
        if (!requested)
        {
            return;
        }

        config.SetFlag(flag, true);
        bool readback = config.GetFlag(flag);
        log.Add($"TrtexecDeploymentControl Name={name} Applied={readback} Requested=True Readback={readback} ReadbackMatch={readback}");
    }

    private static bool ShouldCreateStronglyTypedNetwork(OnnxEngineBuildOptions options, List<string> log)
    {
        if (!options.DeploymentOptions.StronglyTyped)
        {
            return false;
        }

        if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
        {
            log.Add("TrtexecDeploymentControl Name=StronglyTyped Applied=False Requested=True VersionGuard=TRT8 Reason=strongly-typed-network-creation-is-not-exposed-on-this-api-line");
            return false;
        }

        return true;
    }

    private static void ApplyBuilderScalarDeploymentControls(TensorRtBuilderConfig config, OnnxEngineBuildOptions options, List<string> log)
    {
        TrtexecLikeDeploymentOptions deployment = options.DeploymentOptions;
        if (deployment.MaxNbTactics.HasValue)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
            {
                log.Add($"TrtexecBuilderScalar Name=MaxNbTactics Applied=False Requested={deployment.MaxNbTactics.Value} Reason=TensorRT8Unsupported");
            }
            else
            {
                config.SetMaxTactics(deployment.MaxNbTactics.Value);
                int readback = config.GetMaxTactics();
                log.Add($"TrtexecBuilderScalar Name=MaxNbTactics Applied=True Requested={deployment.MaxNbTactics.Value} Readback={readback} ReadbackMatch={readback == deployment.MaxNbTactics.Value}");
            }
        }

        if (deployment.TilingOptimizationLevel.HasValue)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
            {
                log.Add($"TrtexecBuilderScalar Name=TilingOptimizationLevel Applied=False Requested={deployment.TilingOptimizationLevel.Value} Reason=TensorRT8Unsupported");
            }
            else
            {
                bool accepted = config.SetTilingOptimizationLevel(deployment.TilingOptimizationLevel.Value);
                TensorRtTilingOptimizationLevel readback = config.GetTilingOptimizationLevel();
                log.Add($"TrtexecBuilderScalar Name=TilingOptimizationLevel Applied={accepted} Requested={deployment.TilingOptimizationLevel.Value} Readback={readback} ReadbackMatch={readback == deployment.TilingOptimizationLevel.Value}");
            }
        }

        if (deployment.L2LimitForTilingBytes.HasValue)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
            {
                log.Add($"TrtexecBuilderScalar Name=L2LimitForTiling Applied=False RequestedBytes={deployment.L2LimitForTilingBytes.Value} Reason=TensorRT8Unsupported");
            }
            else
            {
                bool accepted = config.SetL2LimitForTiling(deployment.L2LimitForTilingBytes.Value);
                long readback = config.GetL2LimitForTiling();
                log.Add($"TrtexecBuilderScalar Name=L2LimitForTiling Applied={accepted} RequestedBytes={deployment.L2LimitForTilingBytes.Value} ReadbackBytes={readback} ReadbackMatch={readback == deployment.L2LimitForTilingBytes.Value}");
            }
        }

        if (deployment.QuantizationFlags.HasValue)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt11)
            {
                log.Add($"TrtexecBuilderScalar Name=QuantizationFlags Applied=False Requested={deployment.QuantizationFlags.Value} Reason=RemovedByTensorRT11");
            }
            else
            {
                config.SetQuantizationFlags(deployment.QuantizationFlags.Value);
                TensorRtQuantizationFlags readback = config.GetQuantizationFlags();
                log.Add($"TrtexecBuilderScalar Name=QuantizationFlags Applied=True Requested={deployment.QuantizationFlags.Value} Readback={readback} ReadbackMatch={readback == deployment.QuantizationFlags.Value}");
            }
        }
    }

    private static TensorRtProfilingVerbosity ParseProfilingVerbosity(string value)
    {
        if (string.Equals(value, "none", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtProfilingVerbosity.None;
        }

        if (string.Equals(value, "detailed", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtProfilingVerbosity.Detailed;
        }

        return TensorRtProfilingVerbosity.LayerNamesOnly;
    }

    private static string RuntimeOptionsLogLine(OnnxEngineBuildOptions options)
    {
        return $"TrtexecRuntime Iterations={options.Iterations} WarmUpMs={options.WarmUpMilliseconds} DurationSeconds={options.DurationSeconds} Streams={options.Streams} InfStreams={options.RuntimeOptions.InfStreams?.ToString() ?? ""} NoDataTransfers={options.RuntimeOptions.NoDataTransfers} UseSpinWait={options.RuntimeOptions.UseSpinWait} Threads={options.RuntimeOptions.Threads?.ToString() ?? ""} AvgRuns={options.RuntimeOptions.AvgRuns?.ToString() ?? ""} Percentile={options.RuntimeOptions.Percentile?.ToString() ?? ""} IdleTimeMs={options.RuntimeOptions.IdleTimeMilliseconds?.ToString() ?? ""} SleepTimeMs={options.RuntimeOptions.SleepTimeMilliseconds?.ToString() ?? ""} DumpOutput={options.RuntimeOptions.DumpOutput} ExportTimes={options.RuntimeOptions.ExportTimesPath} ExportProfile={options.RuntimeOptions.ExportProfilePath} ReferenceOutputs={options.RuntimeOptions.ReferenceOutputs} ReferenceAbsTolerance={options.RuntimeOptions.ReferenceAbsoluteTolerance:R} ReferenceRelTolerance={options.RuntimeOptions.ReferenceRelativeTolerance:R} ReferenceNaNPolicy={options.RuntimeOptions.ReferenceNaNPolicy} ReferenceInfinityPolicy={options.RuntimeOptions.ReferenceInfinityPolicy}";
    }
}
