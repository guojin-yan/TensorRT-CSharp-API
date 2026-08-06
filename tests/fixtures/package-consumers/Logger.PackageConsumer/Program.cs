using System;
using System.Collections.Concurrent;
using System.Threading;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp;
using JYPPX.TensorRtSharp.Shared.Interop;

internal static class Program
{
    private const string InputName = "logger_input";
    private const string OutputName = "logger_output";
    private const int ElementCount = 32 * 32;

    public static int Main(string[] args)
    {
        try
        {
            TensorRtApiLine line = ParseTensorRtLine(GetArgument(args, "--tensor-rt-line", "10"));
            string runtimePackageKey = GetArgument(args, "--runtime-package-key", "unspecified");
            if (!HasSwitch(args, "--logger-runtime-smoke-only"))
            {
                throw new ArgumentException("The --logger-runtime-smoke-only switch is required.");
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

            RunRealLoggerRuntimeSmoke(line, runtimePackageKey);
            Console.WriteLine("LoggerPackageConsumer Passed=True Mode=LoggerRuntimeSmokeOnly");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine("LoggerPackageConsumer Passed=False Error=" + exception);
            return 1;
        }
    }

    private static void RunRealLoggerRuntimeSmoke(TensorRtApiLine line, string runtimePackageKey)
    {
        LogRecords positiveRecords = new();
        bool builderAttached;
        bool builderDetached;
        bool runtimeAttached;
        bool runtimeDetached;
        long beforeOwnerCount;
        long afterBuildCount;

        PositiveRuntimeResult positive;
        using (TensorRtLogger logger = new TensorRtLogger(
            line,
            positiveRecords.Record,
            TensorRtLogSeverity.Verbose))
        {
            beforeOwnerCount = logger.CallbackInvocationCount;
            using TensorRtHostMemory plan = BuildPlan(logger, out builderAttached, out builderDetached);
            afterBuildCount = logger.CallbackInvocationCount;

            using (TensorRtRuntime runtime = new(logger))
            {
                runtimeAttached = logger.IsAttached;
                using TensorRtEngine engine = runtime.Deserialize(plan);
                ExecuteOnce(engine);
            }

            runtimeDetached = !logger.IsAttached;
            bool positivePassed =
                beforeOwnerCount == 0 &&
                afterBuildCount > beforeOwnerCount &&
                logger.CallbackInvocationCount == positiveRecords.TotalCount &&
                logger.CallbackInvocationCount > 0 &&
                logger.CallbackFailureCount == 0 &&
                logger.LastCallbackException == null &&
                positiveRecords.MetadataCopied &&
                positiveRecords.DistinctSeverityCount > 0 &&
                builderAttached &&
                builderDetached &&
                runtimeAttached &&
                runtimeDetached;
            if (!positivePassed)
            {
                throw new InvalidOperationException(
                    "Positive logger runtime invariants failed. Before=" + beforeOwnerCount +
                    " AfterBuild=" + afterBuildCount + " Total=" + logger.CallbackInvocationCount +
                    " Failures=" + logger.CallbackFailureCount + " Metadata=" + positiveRecords.MetadataCopied +
                    " BuilderAttached=" + builderAttached + " BuilderDetached=" + builderDetached +
                    " RuntimeAttached=" + runtimeAttached + " RuntimeDetached=" + runtimeDetached);
            }

            positive = new PositiveRuntimeResult(
                logger.CallbackInvocationCount,
                positiveRecords.DistinctSeverityCount,
                positiveRecords.FirstSeverity,
                positiveRecords.FirstMessageLength,
                logger.CallbackFailureCount,
                positiveRecords.MetadataCopied,
                builderAttached,
                builderDetached,
                runtimeAttached,
                runtimeDetached,
                beforeOwnerCount,
                afterBuildCount);
        }

        LifecycleRuntimeResult lifecycle = RunDeferredDisposeRuntime(line);
        NegativeRuntimeResult negative = RunHandlerExceptionRuntime(line);

        Console.WriteLine("LoggerRuntimeSummary");
        Console.WriteLine($"  TensorRtLine={(int)line} RuntimePackageKey={runtimePackageKey}");
        Console.WriteLine($"  PositiveCallbacks={positive.InvocationCount} Severities={positive.DistinctSeverityCount} Failures={positive.FailureCount} FirstSeverity={positive.FirstSeverity} FirstMessageLength={positive.FirstMessageLength}");
        Console.WriteLine($"  BuilderAttached={positive.BuilderAttached} BuilderDetached={positive.BuilderDetached} RuntimeAttached={positive.RuntimeAttached} RuntimeDetached={positive.RuntimeDetached}");
        Console.WriteLine($"  DeferredDispose=Passed AttachedBefore={lifecycle.AttachedBeforeDispose} AttachedAfter={lifecycle.AttachedAfterDispose} PostDisposeCallbacks={lifecycle.PostDisposeCallbacks} Detached={lifecycle.Detached} RejectsNewBorrower={lifecycle.RejectsNewBorrower}");
        Console.WriteLine($"  ExceptionCase=Passed OperationFailed={negative.OperationFailed} Callbacks={negative.InvocationCount} Failures={negative.FailureCount} Detached={negative.Detached}");
        Console.WriteLine(
            "LoggerRealRuntime=Passed" +
            $" TensorRtLine={(int)line}" +
            $" RuntimePackageKey={runtimePackageKey}" +
            $" BeforeOwnerCount={positive.BeforeOwnerCount}" +
            $" AfterBuildCount={positive.AfterBuildCount}" +
            $" PositiveInvocationCount={positive.InvocationCount}" +
            $" PositiveSeverityCount={positive.DistinctSeverityCount}" +
            $" PositiveFailureCount={positive.FailureCount}" +
            $" MetadataCopied={positive.MetadataCopied}" +
            $" BuilderAttached={positive.BuilderAttached}" +
            $" BuilderDetached={positive.BuilderDetached}" +
            $" RuntimeAttached={positive.RuntimeAttached}" +
            $" RuntimeDetached={positive.RuntimeDetached}" +
            $" LifecycleAttachedBeforeDispose={lifecycle.AttachedBeforeDispose}" +
            $" LifecycleAttachedAfterDispose={lifecycle.AttachedAfterDispose}" +
            $" LifecyclePostDisposeCallbacks={lifecycle.PostDisposeCallbacks}" +
            $" LifecycleDetached={lifecycle.Detached}" +
            $" DisposedRejectsNewBorrower={lifecycle.RejectsNewBorrower}" +
            $" NegativeOperationFailed={negative.OperationFailed}" +
            $" NegativeInvocationCount={negative.InvocationCount}" +
            $" NegativeFailureCount={negative.FailureCount}" +
            $" NegativeDetachVerified={negative.Detached}" +
            " ThreadSafeHandlerState=True" +
            " NativeFailureFlagAtomic=True" +
            " RealCallbackRuntime=True" +
            " SyntheticDiagnosticUsed=False");
    }

    private static LifecycleRuntimeResult RunDeferredDisposeRuntime(TensorRtApiLine line)
    {
        LogRecords records = new();
        TensorRtLogger logger = new(line, records.Record, TensorRtLogSeverity.Verbose);
        TensorRtBuilder? builder = null;
        bool attachedBeforeDispose = false;
        bool attachedAfterDispose = false;
        bool postDisposeCallbacks = false;
        try
        {
            builder = new TensorRtBuilder(logger);
            attachedBeforeDispose = logger.IsAttached;
            logger.Dispose();
            attachedAfterDispose = logger.IsAttached;
            long beforeBuild = logger.CallbackInvocationCount;
            using TensorRtHostMemory plan = BuildPlan(builder);
            postDisposeCallbacks = logger.CallbackInvocationCount > beforeBuild;
        }
        finally
        {
            builder?.Dispose();
            logger.Dispose();
        }

        bool detached = !logger.IsAttached;
        bool rejectsNewBorrower = false;
        try
        {
            using TensorRtRuntime _ = new(logger);
        }
        catch (ObjectDisposedException)
        {
            rejectsNewBorrower = true;
        }

        bool passed =
            attachedBeforeDispose &&
            attachedAfterDispose &&
            postDisposeCallbacks &&
            records.TotalCount > 0 &&
            records.MetadataCopied &&
            detached &&
            rejectsNewBorrower;
        if (!passed)
        {
            throw new InvalidOperationException(
                "Deferred logger disposal invariants failed. AttachedBefore=" + attachedBeforeDispose +
                " AttachedAfter=" + attachedAfterDispose + " PostDisposeCallbacks=" + postDisposeCallbacks +
                " Count=" + records.TotalCount + " Detached=" + detached + " Rejects=" + rejectsNewBorrower);
        }

        return new LifecycleRuntimeResult(
            attachedBeforeDispose,
            attachedAfterDispose,
            postDisposeCallbacks,
            detached,
            rejectsNewBorrower,
            records.TotalCount);
    }

    private static NegativeRuntimeResult RunHandlerExceptionRuntime(TensorRtApiLine line)
    {
        using TensorRtLogger logger = new TensorRtLogger(
            line,
            static (_, _) => throw new InvalidOperationException("controlled logger handler failure"),
            TensorRtLogSeverity.Verbose);
        bool operationFailed = false;
        bool attachedDuringBuild = false;
        bool detached;
        try
        {
            using TensorRtHostMemory plan = BuildPlan(logger, out attachedDuringBuild, out detached);
        }
        catch (TensorRtException)
        {
            operationFailed = true;
            detached = !logger.IsAttached;
        }

        bool passed =
            attachedDuringBuild &&
            logger.CallbackInvocationCount > 0 &&
            logger.CallbackFailureCount == logger.CallbackInvocationCount &&
            logger.LastCallbackException is InvalidOperationException &&
            detached;
        if (!passed)
        {
            throw new InvalidOperationException(
                "Logger handler exception was not isolated and detached. Attached=" + attachedDuringBuild +
                " Callbacks=" + logger.CallbackInvocationCount + " Failures=" + logger.CallbackFailureCount +
                " OperationFailed=" + operationFailed + " Detached=" + detached);
        }

        return new NegativeRuntimeResult(
            operationFailed,
            logger.CallbackInvocationCount,
            logger.CallbackFailureCount,
            detached);
    }

    private static TensorRtHostMemory BuildPlan(
        TensorRtLogger logger,
        out bool attachedDuringBuild,
        out bool detachedAfterBuild)
    {
        TensorRtHostMemory? plan = null;
        using (TensorRtBuilder builder = new(logger))
        {
            attachedDuringBuild = logger.IsAttached;
            plan = BuildPlan(builder);
        }

        detachedAfterBuild = !logger.IsAttached;
        return plan ?? throw new InvalidOperationException("TensorRT did not return a serialized plan.");
    }

    private static TensorRtHostMemory BuildPlan(TensorRtBuilder builder)
    {
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetAverageTimingIterations(1);
        using TensorRtNetworkDefinition network =
            builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        network.Name = "logger_package_consumer";
        using TensorRtTensor input = network.AddInput(
            InputName,
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
        convolution.Name = "logger_conv_1x1";
        using TensorRtTensor output = convolution.GetOutput(0);
        output.Name = OutputName;
        network.MarkOutput(output);
        return builder.BuildSerializedNetwork(network, config);
    }

    private static void ExecuteOnce(TensorRtEngine engine)
    {
        using CudaStream stream = new();
        using CudaMemory input = new(ElementCount * sizeof(float));
        using CudaMemory output = new(ElementCount * sizeof(float));
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        input.Fill(0, input.SizeInBytes);
        output.Fill(0, output.SizeInBytes);
        context.SetTensorAddress(InputName, input);
        context.SetTensorAddress(OutputName, output);
        context.EnqueueAsync(stream);
        stream.Synchronize();
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

    private sealed class LogRecords
    {
        private readonly ConcurrentDictionary<TensorRtLogSeverity, byte> _severities = new();
        private long _totalCount;
        private int _invalidMetadata;
        private int _firstRecorded;
        private int _firstMessageLength;
        private int _firstSeverity = -1;

        public long TotalCount => Interlocked.Read(ref _totalCount);

        public int DistinctSeverityCount => _severities.Count;

        public bool MetadataCopied => Volatile.Read(ref _invalidMetadata) == 0 && TotalCount > 0;

        public TensorRtLogSeverity FirstSeverity => (TensorRtLogSeverity)Volatile.Read(ref _firstSeverity);

        public int FirstMessageLength => Volatile.Read(ref _firstMessageLength);

        public void Record(TensorRtLogSeverity severity, string message)
        {
            Interlocked.Increment(ref _totalCount);
            int numericSeverity = (int)severity;
            if (numericSeverity < (int)TensorRtLogSeverity.InternalError ||
                numericSeverity > (int)TensorRtLogSeverity.Verbose ||
                string.IsNullOrWhiteSpace(message) ||
                message.IndexOf('\0') >= 0)
            {
                Interlocked.Exchange(ref _invalidMetadata, 1);
                return;
            }

            _severities.TryAdd(severity, 0);
            if (Interlocked.CompareExchange(ref _firstRecorded, 1, 0) == 0)
            {
                Volatile.Write(ref _firstSeverity, numericSeverity);
                Volatile.Write(ref _firstMessageLength, message.Length);
            }
        }
    }

    private sealed record PositiveRuntimeResult(
        long InvocationCount,
        int DistinctSeverityCount,
        TensorRtLogSeverity FirstSeverity,
        int FirstMessageLength,
        long FailureCount,
        bool MetadataCopied,
        bool BuilderAttached,
        bool BuilderDetached,
        bool RuntimeAttached,
        bool RuntimeDetached,
        long BeforeOwnerCount,
        long AfterBuildCount);

    private sealed record LifecycleRuntimeResult(
        bool AttachedBeforeDispose,
        bool AttachedAfterDispose,
        bool PostDisposeCallbacks,
        bool Detached,
        bool RejectsNewBorrower,
        long InvocationCount);

    private sealed record NegativeRuntimeResult(
        bool OperationFailed,
        long InvocationCount,
        long FailureCount,
        bool Detached);
}
