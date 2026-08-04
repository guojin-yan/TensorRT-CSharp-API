using System;
using System.Collections.Concurrent;
using System.Threading;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp;
using JYPPX.TensorRtSharp.Shared.Interop;

internal static class Program
{
    private const string InputName = "profiler_input";
    private const string OutputName = "profiler_output";
    private const int ElementCount = 32 * 32;

    public static int Main(string[] args)
    {
        try
        {
            TensorRtApiLine line = ParseTensorRtLine(GetArgument(args, "--tensor-rt-line", "10"));
            string runtimePackageKey = GetArgument(args, "--runtime-package-key", "unspecified");
            if (!HasSwitch(args, "--profiler-runtime-smoke-only"))
            {
                throw new ArgumentException("The --profiler-runtime-smoke-only switch is required.");
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

            RunRealProfilerRuntimeSmoke(line, runtimePackageKey);
            Console.WriteLine("ProfilerPackageConsumer Passed=True Mode=ProfilerRuntimeSmokeOnly");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine("ProfilerPackageConsumer Passed=False Error=" + exception);
            return 1;
        }
    }

    private static void RunRealProfilerRuntimeSmoke(TensorRtApiLine line, string runtimePackageKey)
    {
        using TensorRtLogger logger = new(line);
        using TensorRtHostMemory plan = BuildPlan(line, logger);
        using TensorRtRuntime runtime = new(logger);
        using TensorRtEngine engine = runtime.Deserialize(plan);
        using CudaStream stream = new();
        using CudaMemory input = new(ElementCount * sizeof(float));
        using CudaMemory output = new(ElementCount * sizeof(float));
        input.Fill(0, input.SizeInBytes);
        output.Fill(0, output.SizeInBytes);

        ProfileRecords immediateRecords = new();
        using TensorRtProfiler immediateProfiler = new(
            line,
            (layerName, milliseconds) => immediateRecords.Record(layerName, milliseconds));
        bool immediateMode;
        bool immediateDetachVerified;
        using (TensorRtExecutionContext context = CreateBoundContext(engine, input, output))
        {
            context.SetProfiler(immediateProfiler);
            context.EnqueueEmitsProfile = true;
            immediateMode = context.EnqueueEmitsProfile;
            bool attached = context.HasProfiler && context.HasNativeProfiler && immediateProfiler.IsAttached;
            context.EnqueueAsync(stream);
            stream.Synchronize();
            if (!attached)
            {
                throw new InvalidOperationException("The immediate profiler was not attached before enqueue.");
            }

            context.ClearProfiler();
            immediateDetachVerified = !context.HasProfiler && !immediateProfiler.IsAttached;
        }

        bool immediatePassed =
            immediateMode &&
            immediateProfiler.CallbackInvocationCount > 0 &&
            immediateRecords.TotalCount == immediateProfiler.CallbackInvocationCount &&
            immediateRecords.DistinctLayerCount > 0 &&
            immediateRecords.MetadataCopied &&
            immediateProfiler.CallbackFailureCount == 0 &&
            immediateProfiler.LastCallbackException == null &&
            immediateDetachVerified;
        if (!immediatePassed)
        {
            throw new InvalidOperationException(
                "Immediate profiler runtime invariants failed. Callbacks=" +
                immediateProfiler.CallbackInvocationCount + " Layers=" + immediateRecords.DistinctLayerCount +
                " Failures=" + immediateProfiler.CallbackFailureCount + " Detached=" + immediateDetachVerified);
        }

        ProfileRecords deferredRecords = new();
        using TensorRtProfiler deferredProfiler = new(
            line,
            (layerName, milliseconds) => deferredRecords.Record(layerName, milliseconds));
        bool deferredMode;
        long deferredBeforeReportCount;
        bool deferredReported;
        bool deferredDetachVerified;
        using (TensorRtExecutionContext context = CreateBoundContext(engine, input, output))
        {
            context.SetProfiler(deferredProfiler);
            context.EnqueueEmitsProfile = false;
            deferredMode = context.EnqueueEmitsProfile;
            context.EnqueueAsync(stream);
            stream.Synchronize();
            deferredBeforeReportCount = deferredProfiler.CallbackInvocationCount;
            deferredReported = context.ReportToProfiler();
            context.ClearProfiler();
            deferredDetachVerified = !context.HasProfiler && !deferredProfiler.IsAttached;
        }

        bool deferredPassed =
            !deferredMode &&
            deferredBeforeReportCount == 0 &&
            deferredReported &&
            deferredProfiler.CallbackInvocationCount > 0 &&
            deferredRecords.TotalCount == deferredProfiler.CallbackInvocationCount &&
            deferredRecords.DistinctLayerCount > 0 &&
            deferredRecords.MetadataCopied &&
            deferredProfiler.CallbackFailureCount == 0 &&
            deferredProfiler.LastCallbackException == null &&
            deferredDetachVerified;
        if (!deferredPassed)
        {
            throw new InvalidOperationException(
                "Deferred profiler runtime invariants failed. Mode=" + deferredMode +
                " BeforeReport=" + deferredBeforeReportCount + " Reported=" + deferredReported +
                " Callbacks=" + deferredProfiler.CallbackInvocationCount +
                " Failures=" + deferredProfiler.CallbackFailureCount + " Detached=" + deferredDetachVerified);
        }

        using TensorRtProfiler negativeProfiler = new(
            line,
            static (_, _) => throw new InvalidOperationException("controlled profiler handler failure"));
        bool negativeEnqueueFailed = false;
        bool negativeDetachVerified;
        using (TensorRtExecutionContext context = CreateBoundContext(engine, input, output))
        {
            context.SetProfiler(negativeProfiler);
            context.EnqueueEmitsProfile = true;
            try
            {
                context.EnqueueAsync(stream);
                stream.Synchronize();
            }
            catch (TensorRtException)
            {
                negativeEnqueueFailed = true;
            }
            finally
            {
                context.ClearProfiler();
            }

            negativeDetachVerified = !context.HasProfiler && !negativeProfiler.IsAttached;
        }

        bool negativePassed =
            negativeProfiler.CallbackInvocationCount > 0 &&
            negativeProfiler.CallbackFailureCount > 0 &&
            negativeProfiler.LastCallbackException is InvalidOperationException &&
            negativeDetachVerified;
        if (!negativePassed)
        {
            throw new InvalidOperationException(
                "Profiler handler exception was not recorded and detached. Callbacks=" +
                negativeProfiler.CallbackInvocationCount + " Failures=" + negativeProfiler.CallbackFailureCount +
                " EnqueueFailed=" + negativeEnqueueFailed + " Detached=" + negativeDetachVerified);
        }

        bool metadataCopied = immediateRecords.MetadataCopied && deferredRecords.MetadataCopied;
        Console.WriteLine("ProfilerRuntimeSummary");
        Console.WriteLine($"  TensorRtLine={(int)line} RuntimePackageKey={runtimePackageKey}");
        Console.WriteLine($"  ImmediateCallbacks={immediateProfiler.CallbackInvocationCount} Layers={immediateRecords.DistinctLayerCount} Failures={immediateProfiler.CallbackFailureCount}");
        Console.WriteLine($"  DeferredBeforeReport={deferredBeforeReportCount} Reported={deferredReported} Callbacks={deferredProfiler.CallbackInvocationCount} Layers={deferredRecords.DistinctLayerCount}");
        Console.WriteLine($"  ExceptionCase=Passed EnqueueFailed={negativeEnqueueFailed} Callbacks={negativeProfiler.CallbackInvocationCount} Failures={negativeProfiler.CallbackFailureCount}");
        Console.WriteLine($"  MetadataCopied={metadataCopied} ImmediateDetached={immediateDetachVerified} DeferredDetached={deferredDetachVerified} NegativeDetached={negativeDetachVerified}");
        Console.WriteLine(
            "ProfilerRealRuntime=Passed" +
            $" TensorRtLine={(int)line}" +
            $" RuntimePackageKey={runtimePackageKey}" +
            $" ImmediateMode={immediateMode}" +
            $" ImmediateInvocationCount={immediateProfiler.CallbackInvocationCount}" +
            $" ImmediateLayerCount={immediateRecords.DistinctLayerCount}" +
            $" ImmediateFailureCount={immediateProfiler.CallbackFailureCount}" +
            $" ImmediateDetachVerified={immediateDetachVerified}" +
            $" DeferredMode={deferredMode}" +
            $" DeferredBeforeReportCount={deferredBeforeReportCount}" +
            $" DeferredReported={deferredReported}" +
            $" DeferredInvocationCount={deferredProfiler.CallbackInvocationCount}" +
            $" DeferredLayerCount={deferredRecords.DistinctLayerCount}" +
            $" DeferredFailureCount={deferredProfiler.CallbackFailureCount}" +
            $" DeferredDetachVerified={deferredDetachVerified}" +
            $" MetadataCopied={metadataCopied}" +
            $" NegativeEnqueueFailed={negativeEnqueueFailed}" +
            $" NegativeInvocationCount={negativeProfiler.CallbackInvocationCount}" +
            $" NegativeFailureCount={negativeProfiler.CallbackFailureCount}" +
            $" NegativeDetachVerified={negativeDetachVerified}" +
            " RealCallbackRuntime=True");
    }

    private static TensorRtHostMemory BuildPlan(TensorRtApiLine line, TensorRtLogger logger)
    {
        using TensorRtBuilder builder = new(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetAverageTimingIterations(1);
        using TensorRtNetworkDefinition network =
            builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        network.Name = "profiler_package_consumer";
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
        convolution.Name = "profiled_conv_1x1";
        using TensorRtTensor output = convolution.GetOutput(0);
        output.Name = OutputName;
        network.MarkOutput(output);
        return builder.BuildSerializedNetwork(network, config);
    }

    private static TensorRtExecutionContext CreateBoundContext(
        TensorRtEngine engine,
        CudaMemory input,
        CudaMemory output)
    {
        TensorRtExecutionContext context = engine.CreateExecutionContext();
        try
        {
            context.SetTensorAddress(InputName, input);
            context.SetTensorAddress(OutputName, output);
            return context;
        }
        catch
        {
            context.Dispose();
            throw;
        }
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

    private sealed class ProfileRecords
    {
        private readonly ConcurrentDictionary<string, byte> _layerNames =
            new(StringComparer.Ordinal);
        private long _totalCount;
        private int _invalidMetadata;

        public long TotalCount => Interlocked.Read(ref _totalCount);

        public int DistinctLayerCount => _layerNames.Count;

        public bool MetadataCopied => Volatile.Read(ref _invalidMetadata) == 0 && _layerNames.Count > 0;

        public void Record(string layerName, float milliseconds)
        {
            Interlocked.Increment(ref _totalCount);
            if (string.IsNullOrWhiteSpace(layerName) || !float.IsFinite(milliseconds) || milliseconds < 0.0f)
            {
                Interlocked.Exchange(ref _invalidMetadata, 1);
                return;
            }

            _layerNames.TryAdd(layerName, 0);
        }
    }
}
