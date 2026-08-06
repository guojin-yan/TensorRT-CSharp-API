using System;
using System.Collections.Concurrent;
using System.Threading;
using JYPPX.TensorRtSharp;
using JYPPX.TensorRtSharp.Shared.Interop;

internal static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            TensorRtApiLine line = ParseTensorRtLine(GetArgument(args, "--tensor-rt-line", "10"));
            string runtimePackageKey = GetArgument(args, "--runtime-package-key", "unspecified");
            if (!HasSwitch(args, "--progress-monitor-runtime-smoke-only"))
            {
                throw new ArgumentException("The --progress-monitor-runtime-smoke-only switch is required.");
            }

            TensorRtEnvironmentSnapshot environment = TensorRtEnvironmentProbe.GetCurrent();
            string expectedTensorRtMajor = ((int)line).ToString() + ".";
            if (!environment.RuntimeInfo.TensorRtAvailable ||
                !environment.RuntimeInfo.CudaToolkitAvailable ||
                !environment.BuildInfo.TensorRtVersion.StartsWith(expectedTensorRtMajor, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    "The loaded bridge does not match the requested TensorRT line. TRT=" +
                    environment.BuildInfo.TensorRtVersion + " CUDA=" + environment.BuildInfo.CudaToolkitVersion);
            }

            Console.WriteLine("PackageReferenceOnly=True");
            Console.WriteLine("ProjectReference=False");
            Console.WriteLine("SourceTreeBinary=False");
            Console.WriteLine(
                "RuntimeEnvironment" +
                " TRT=" + environment.BuildInfo.TensorRtVersion +
                " CUDA=" + environment.BuildInfo.CudaToolkitVersion +
                " TensorRtAvailable=" + environment.RuntimeInfo.TensorRtAvailable +
                " CudaAvailable=" + environment.RuntimeInfo.CudaToolkitAvailable);

            RunRealProgressMonitorRuntimeSmoke(line, runtimePackageKey);
            Console.WriteLine("ProgressMonitorPackageConsumer Passed=True Mode=ProgressMonitorRuntimeSmokeOnly");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine("ProgressMonitorPackageConsumer Passed=False Error=" + exception);
            return 1;
        }
    }

    private static void RunRealProgressMonitorRuntimeSmoke(TensorRtApiLine line, string runtimePackageKey)
    {
        if (line != TensorRtApiLine.TensorRt10 && line != TensorRtApiLine.TensorRt11)
        {
            throw new NotSupportedException("Progress monitors require TensorRT 10 or TensorRT 11.");
        }

        using TensorRtLogger logger = new(line);
        EventCounters positiveEvents = new();
        using TensorRtProgressMonitor positiveMonitor = new(
            line,
            progressEvent =>
            {
                positiveEvents.Record(progressEvent);
                return true;
            });

        bool attachedDuringBuild;
        bool detachVerified;
        using (TensorRtBuilder builder = new(logger))
        using (TensorRtBuilderConfig config = CreateConfig(builder))
        using (TensorRtNetworkDefinition network = CreateNetwork(builder, "progress_positive"))
        {
            config.SetProgressMonitor(positiveMonitor);
            attachedDuringBuild = config.HasProgressMonitor && positiveMonitor.IsAttached;
            using TensorRtHostMemory plan = builder.BuildSerializedNetwork(network, config);
            if (plan.SizeInBytes == 0)
            {
                throw new InvalidOperationException("The positive progress-monitor build returned an empty plan.");
            }

            attachedDuringBuild &= config.HasProgressMonitor && positiveMonitor.IsAttached;
            config.ClearProgressMonitor();
            detachVerified = !config.HasProgressMonitor && !positiveMonitor.IsAttached;
        }

        long positiveInvocationCount = positiveMonitor.CallbackInvocationCount;
        bool metadataCopied = positiveEvents.MetadataCopied;
        bool positivePassed =
            attachedDuringBuild &&
            detachVerified &&
            positiveInvocationCount > 0 &&
            positiveEvents.TotalCount == positiveInvocationCount &&
            positiveEvents.PhaseStartCount > 0 &&
            positiveEvents.StepCompleteCount > 0 &&
            positiveEvents.PhaseFinishCount > 0 &&
            positiveEvents.DistinctPhaseCount > 0 &&
            metadataCopied &&
            positiveMonitor.CallbackFailureCount == 0 &&
            positiveMonitor.LastCallbackException == null;
        if (!positivePassed)
        {
            throw new InvalidOperationException(
                "Progress monitor positive runtime invariants failed. Callbacks=" + positiveInvocationCount +
                " Start=" + positiveEvents.PhaseStartCount +
                " Step=" + positiveEvents.StepCompleteCount +
                " Finish=" + positiveEvents.PhaseFinishCount +
                " Failures=" + positiveMonitor.CallbackFailureCount);
        }

        EventCounters negativeEvents = new();
        int cancellationRequested = 0;
        using TensorRtProgressMonitor negativeMonitor = new(
            line,
            progressEvent =>
            {
                negativeEvents.Record(progressEvent);
                if (progressEvent.Kind == TensorRtProgressMonitorEventKind.StepComplete)
                {
                    Interlocked.Exchange(ref cancellationRequested, 1);
                    return false;
                }

                return true;
            });

        bool negativeBuildFailed = false;
        bool negativeDetachVerified;
        using (TensorRtBuilder builder = new(logger))
        using (TensorRtBuilderConfig config = CreateConfig(builder))
        using (TensorRtNetworkDefinition network = CreateNetwork(builder, "progress_cancel"))
        {
            config.SetProgressMonitor(negativeMonitor);
            try
            {
                using TensorRtHostMemory unexpected = builder.BuildSerializedNetwork(network, config);
            }
            catch (TensorRtException)
            {
                negativeBuildFailed = true;
            }
            finally
            {
                config.ClearProgressMonitor();
            }

            negativeDetachVerified = !config.HasProgressMonitor && !negativeMonitor.IsAttached;
        }

        bool negativePassed =
            Volatile.Read(ref cancellationRequested) != 0 &&
            negativeBuildFailed &&
            negativeMonitor.CallbackInvocationCount > 0 &&
            negativeEvents.StepCompleteCount > 0 &&
            negativeMonitor.CallbackFailureCount == 0 &&
            negativeMonitor.LastCallbackException == null &&
            negativeDetachVerified;
        if (!negativePassed)
        {
            throw new InvalidOperationException(
                "Progress monitor cancellation did not stop the build cleanly. Requested=" +
                (Volatile.Read(ref cancellationRequested) != 0) +
                " BuildFailed=" + negativeBuildFailed +
                " Callbacks=" + negativeMonitor.CallbackInvocationCount +
                " Failures=" + negativeMonitor.CallbackFailureCount +
                " Detached=" + negativeDetachVerified);
        }

        Console.WriteLine("ProgressMonitorRuntimeSummary");
        Console.WriteLine($"  TensorRtLine={(int)line} RuntimePackageKey={runtimePackageKey}");
        Console.WriteLine($"  Callbacks={positiveInvocationCount} Start={positiveEvents.PhaseStartCount} Step={positiveEvents.StepCompleteCount} Finish={positiveEvents.PhaseFinishCount}");
        Console.WriteLine($"  DistinctPhases={positiveEvents.DistinctPhaseCount} MetadataCopied={metadataCopied} Failures={positiveMonitor.CallbackFailureCount}");
        Console.WriteLine($"  CancellationCase=Passed Requested={Volatile.Read(ref cancellationRequested) != 0} BuildFailed={negativeBuildFailed} Callbacks={negativeMonitor.CallbackInvocationCount}");
        Console.WriteLine($"  PositiveDetached={detachVerified} NegativeDetached={negativeDetachVerified} ThreadSafeHandlerState=True");
        Console.WriteLine(
            "ProgressMonitorRealRuntime=Passed" +
            $" TensorRtLine={(int)line}" +
            $" RuntimePackageKey={runtimePackageKey}" +
            $" AttachedDuringBuild={attachedDuringBuild}" +
            $" InvocationCount={positiveInvocationCount}" +
            $" PhaseStartCount={positiveEvents.PhaseStartCount}" +
            $" StepCompleteCount={positiveEvents.StepCompleteCount}" +
            $" PhaseFinishCount={positiveEvents.PhaseFinishCount}" +
            $" DistinctPhaseCount={positiveEvents.DistinctPhaseCount}" +
            $" FailureCount={positiveMonitor.CallbackFailureCount}" +
            $" MetadataCopied={metadataCopied}" +
            $" DetachVerified={detachVerified}" +
            $" NegativeCancellationRequested={Volatile.Read(ref cancellationRequested) != 0}" +
            $" NegativeBuildFailed={negativeBuildFailed}" +
            $" NegativeInvocationCount={negativeMonitor.CallbackInvocationCount}" +
            $" NegativeFailureCount={negativeMonitor.CallbackFailureCount}" +
            $" NegativeDetachVerified={negativeDetachVerified}" +
            " RealCallbackRuntime=True");
    }

    private static TensorRtBuilderConfig CreateConfig(TensorRtBuilder builder)
    {
        TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetAverageTimingIterations(1);
        return config;
    }

    private static TensorRtNetworkDefinition CreateNetwork(TensorRtBuilder builder, string name)
    {
        TensorRtNetworkDefinition network =
            builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        network.Name = name;
        using TensorRtTensor input = network.AddInput(
            name + "_input",
            TensorRtDataType.Float,
            new TensorRtDims(new[] { 1, 1, 32, 32 }));
        TensorRtWeights kernel = TensorRtWeights.FromSingleArray(new[] { 1.0f });
        TensorRtWeights bias = TensorRtWeights.FromSingleArray(new[] { 0.0f });
        using TensorRtLayer convolution = network.AddConvolution(
            input,
            1,
            new TensorRtDims(new[] { 1, 1 }),
            kernel,
            bias);
        convolution.Name = name + "_conv";
        using TensorRtTensor output = convolution.GetOutput(0);
        output.Name = name + "_output";
        network.MarkOutput(output);
        return network;
    }

    private static bool HasSwitch(string[] args, string name)
    {
        return Array.Exists(args, value => string.Equals(value, name, StringComparison.OrdinalIgnoreCase));
    }

    private static string GetArgument(string[] args, string name, string fallback)
    {
        for (int index = 0; index + 1 < args.Length; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }
        }

        return fallback;
    }

    private static TensorRtApiLine ParseTensorRtLine(string value)
    {
        return value switch
        {
            "8" => TensorRtApiLine.TensorRt8,
            "10" => TensorRtApiLine.TensorRt10,
            "11" => TensorRtApiLine.TensorRt11,
            _ => throw new ArgumentOutOfRangeException(nameof(value), value, "TensorRT line must be 8, 10, or 11.")
        };
    }

    private sealed class EventCounters
    {
        private readonly ConcurrentDictionary<string, byte> _phaseNames =
            new(StringComparer.Ordinal);
        private long _totalCount;
        private long _phaseStartCount;
        private long _stepCompleteCount;
        private long _phaseFinishCount;
        private int _invalidMetadata;

        public long TotalCount => Interlocked.Read(ref _totalCount);

        public long PhaseStartCount => Interlocked.Read(ref _phaseStartCount);

        public long StepCompleteCount => Interlocked.Read(ref _stepCompleteCount);

        public long PhaseFinishCount => Interlocked.Read(ref _phaseFinishCount);

        public int DistinctPhaseCount => _phaseNames.Count;

        public bool MetadataCopied => Volatile.Read(ref _invalidMetadata) == 0 && _phaseNames.Count > 0;

        public void Record(TensorRtProgressMonitorEvent progressEvent)
        {
            Interlocked.Increment(ref _totalCount);
            if (string.IsNullOrWhiteSpace(progressEvent.PhaseName))
            {
                Interlocked.Exchange(ref _invalidMetadata, 1);
            }
            else
            {
                _phaseNames.TryAdd(progressEvent.PhaseName, 0);
            }

            switch (progressEvent.Kind)
            {
                case TensorRtProgressMonitorEventKind.PhaseStart:
                    Interlocked.Increment(ref _phaseStartCount);
                    break;
                case TensorRtProgressMonitorEventKind.StepComplete:
                    Interlocked.Increment(ref _stepCompleteCount);
                    break;
                case TensorRtProgressMonitorEventKind.PhaseFinish:
                    Interlocked.Increment(ref _phaseFinishCount);
                    break;
                default:
                    Interlocked.Exchange(ref _invalidMetadata, 1);
                    break;
            }
        }
    }
}
