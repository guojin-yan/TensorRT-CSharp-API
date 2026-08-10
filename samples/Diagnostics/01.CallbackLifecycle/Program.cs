using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using JYPPX.CudaSharp;
using JYPPX.SampleSupport;
using JYPPX.TensorRtSharp;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace CallbackLifecycleSample;

internal static class Program
{
    private const string InputName = "callback_input";
    private const string OutputName = "callback_output";
    private static readonly float[] InputValues = { 1.0f, 2.0f, -3.0f, 4.0f };

    public static int Main(string[] args)
    {
        if (SampleCommandLine.HasSwitch(args, "--help") || SampleCommandLine.HasSwitch(args, "-h"))
        {
            PrintUsage();
            return 0;
        }

        try
        {
            return Run(args);
        }
        catch (SampleSkippedException exception)
        {
            WriteStatusReport(args, "skipped", exception.Message);
            return 0;
        }
        catch (Exception exception) when (TensorRtSampleSupport.IsDeploymentException(exception))
        {
            WriteStatusReport(args, "skipped", exception.Message);
            return 0;
        }
        catch (Exception exception) when (exception is ArgumentException || exception is InvalidDataException || exception is JsonException)
        {
            WriteStatusReport(args, "invalid-arguments", exception.Message);
            PrintUsage();
            return 2;
        }
        catch (Exception exception)
        {
            WriteStatusReport(args, "failed", exception.Message);
            return 1;
        }
    }

    private static int Run(string[] args)
    {
        TensorRtApiLine line = TensorRtSampleSupport.ResolveLine(
            SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "10"));
        if (line == TensorRtApiLine.TensorRt8)
        {
            throw new ArgumentException("This combined sample requires TensorRT 10 or TensorRT 11 progress and debug callbacks.");
        }

        TensorRtEnvironmentSnapshot environment = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = TensorRtSampleSupport.SelectAdapter(environment, line);
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            throw new SampleSkippedException(adapter.StatusMessage);
        }

        CallbackRecords records = new CallbackRecords();
        using TensorRtLogger logger = new TensorRtLogger(line, records.RecordLog, TensorRtLogSeverity.Verbose);
        using TensorRtProgressMonitor progressMonitor = new TensorRtProgressMonitor(line, records.RecordProgress);

        byte[] plan = BuildPlan(
            logger,
            progressMonitor,
            out bool progressAttachedDuringBuild,
            out bool progressDetachedAfterBuild);

        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtEngine engine = runtime.Deserialize(plan);
        using TensorRtProfiler profiler = new TensorRtProfiler(line, records.RecordProfile);
        TensorRtDebugTensorMetadataSnapshot debugMetadata = default;
        using TensorRtDebugListenerCallbackOwner debugListener = new TensorRtDebugListenerCallbackOwner(
            line,
            metadata =>
            {
                debugMetadata = metadata;
                return true;
            });

        float[] output;
        bool profilerAttachedDuringEnqueue;
        bool debugListenerAttachedDuringEnqueue;
        bool profilerDetachedAfterClear;
        bool debugListenerDetachedAfterClear;
        TensorRtDebugListenerRuntimeSnapshot debugAttached;
        TensorRtDebugListenerRuntimeSnapshot debugDetached;
        using (TensorRtExecutionContext context = engine.CreateExecutionContext())
        using (CudaStream stream = new CudaStream(CudaStreamCreationFlags.NonBlocking))
        using (CudaMemory input = new CudaMemory(InputValues.Length * sizeof(float)))
        using (CudaMemory outputMemory = new CudaMemory(InputValues.Length * sizeof(float)))
        {
            input.CopyFrom(InputValues);
            outputMemory.Fill(0, outputMemory.SizeInBytes);
            context.SetTensorAddress(InputName, input);
            context.SetTensorAddress(OutputName, outputMemory);

            context.SetProfiler(profiler);
            context.EnqueueEmitsProfile = true;
            context.SetDebugListener(debugListener);
            context.SetTensorDebugState(OutputName, true);
            profilerAttachedDuringEnqueue = context.HasProfiler && profiler.IsAttached;
            debugListenerAttachedDuringEnqueue = context.HasManagedDebugListener && debugListener.IsAttached;

            context.EnqueueAsync(stream);
            stream.Synchronize();
            output = outputMemory.ToSingleArray(InputValues.Length);
            debugAttached = debugListener.GetRuntimeSnapshot();

            bool debugCleared = context.ClearDebugListener();
            context.ClearProfiler();
            profilerDetachedAfterClear = !context.HasProfiler && !profiler.IsAttached;
            debugListenerDetachedAfterClear = debugCleared && !context.HasManagedDebugListener && !debugListener.IsAttached;
            debugDetached = debugListener.GetRuntimeSnapshot();
        }

        bool outputMatch = output.SequenceEqual(InputValues);
        bool progressPassed =
            progressAttachedDuringBuild &&
            progressDetachedAfterBuild &&
            records.ProgressEventCount > 0 &&
            records.DistinctProgressPhaseCount > 0 &&
            progressMonitor.CallbackFailureCount == 0;
        bool profilerPassed =
            profilerAttachedDuringEnqueue &&
            profilerDetachedAfterClear &&
            records.ProfileEventCount > 0 &&
            records.ProfileMetadataCopied &&
            profiler.CallbackFailureCount == 0;
        bool debugPassed =
            debugListenerAttachedDuringEnqueue &&
            debugListenerDetachedAfterClear &&
            debugAttached.IsRealCallbackRuntimeProof &&
            debugAttached.InFlightCallbackCount == 0 &&
            debugMetadata.MetadataCopied &&
            !debugAttached.BorrowedPointerExposed &&
            !debugDetached.IsAttached;
        bool loggerPassed = logger.HasManagedCallback && logger.CallbackFailureCount == 0;
        bool passed = plan.Length > 0 && outputMatch && progressPassed && profilerPassed && debugPassed && loggerPassed;

        WriteReport(args, new
        {
            schemaVersion = "1.0",
            sample = "Diagnostics/01.CallbackLifecycle",
            status = passed ? "passed" : "failed",
            proofClassification = "synthetic-input-runtime",
            tensorRtLine = (int)line,
            environment = new
            {
                tensorRtVersion = environment.BuildInfo.TensorRtVersion,
                cudaToolkitVersion = environment.BuildInfo.CudaToolkitVersion
            },
            logger = new
            {
                hasManagedCallback = logger.HasManagedCallback,
                invocationCount = logger.CallbackInvocationCount,
                failureCount = logger.CallbackFailureCount,
                capturedCount = records.LogCount,
                preview = records.GetLogPreview()
            },
            progressMonitor = new
            {
                attachedDuringBuild = progressAttachedDuringBuild,
                detachedAfterClear = progressDetachedAfterBuild,
                invocationCount = progressMonitor.CallbackInvocationCount,
                failureCount = progressMonitor.CallbackFailureCount,
                distinctPhaseCount = records.DistinctProgressPhaseCount
            },
            profiler = new
            {
                attachedDuringEnqueue = profilerAttachedDuringEnqueue,
                detachedAfterClear = profilerDetachedAfterClear,
                invocationCount = profiler.CallbackInvocationCount,
                failureCount = profiler.CallbackFailureCount,
                distinctLayerCount = records.DistinctProfileLayerCount,
                totalMilliseconds = records.ProfileTotalMilliseconds,
                metadataCopied = records.ProfileMetadataCopied
            },
            debugListener = new
            {
                attachedDuringEnqueue = debugListenerAttachedDuringEnqueue,
                detachedAfterClear = debugListenerDetachedAfterClear,
                invocationCount = debugAttached.InvocationCount,
                failureCount = debugAttached.FailureCount,
                inFlightCallbackCount = debugAttached.InFlightCallbackCount,
                tensorName = debugAttached.TensorName,
                shape = debugAttached.ShapeDimensions,
                metadataCopied = debugMetadata.MetadataCopied,
                borrowedPointerExposed = debugAttached.BorrowedPointerExposed,
                detachCount = debugDetached.DetachCount
            },
            execution = new
            {
                input = InputValues,
                output,
                outputMatch
            },
            lifecycleOrder = new[]
            {
                "attach progress monitor to builder config",
                "build plan and clear progress monitor",
                "attach profiler and debug listener to execution context",
                "enqueue and synchronize",
                "clear debug listener and profiler",
                "dispose callback owners after borrowers detach"
            }
        });

        Console.WriteLine("CallbackLifecycle Passed=" + passed);
        return passed ? 0 : 1;
    }

    private static byte[] BuildPlan(
        TensorRtLogger logger,
        TensorRtProgressMonitor progressMonitor,
        out bool attachedDuringBuild,
        out bool detachedAfterBuild)
    {
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        using TensorRtNetworkDefinition network =
            builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);

        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetMaxAuxStreams(0);
        config.SetProgressMonitor(progressMonitor);

        using TensorRtTensor input = network.AddInput(
            InputName,
            TensorRtDataType.Float,
            new TensorRtDims(new[] { 1, 1, 2, 2 }));
        using TensorRtLayer convolution = network.AddConvolution(
            input,
            1,
            new TensorRtDims(new[] { 1, 1 }),
            TensorRtWeights.FromSingleArray(new[] { 1.0f }),
            TensorRtWeights.FromSingleArray(new[] { 0.0f }));
        convolution.Name = "callback_conv_1x1";
        using TensorRtTensor output = convolution.GetOutput(0);
        output.Name = OutputName;
        network.MarkOutput(output);
        if (!network.MarkDebugTensor(output) || !network.IsDebugTensor(output))
        {
            throw new InvalidOperationException("TensorRT did not retain the debug tensor mark.");
        }

        attachedDuringBuild = config.HasProgressMonitor && progressMonitor.IsAttached;
        try
        {
            using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
            attachedDuringBuild &= config.HasProgressMonitor && progressMonitor.IsAttached;
            return hostMemory.ToArray();
        }
        finally
        {
            config.ClearProgressMonitor();
            detachedAfterBuild = !config.HasProgressMonitor && !progressMonitor.IsAttached;
        }
    }

    private static void WriteStatusReport(string[] args, string status, string reason)
    {
        WriteReport(args, new
        {
            schemaVersion = "1.0",
            sample = "Diagnostics/01.CallbackLifecycle",
            status,
            reason
        });
    }

    private static void WriteReport(string[] args, object report)
    {
        string json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
        string outputPath = SampleCommandLine.GetStringArgument(args, "--output-json", string.Empty);
        if (!string.IsNullOrWhiteSpace(outputPath))
        {
            string fullPath = Path.GetFullPath(outputPath);
            string? directory = Path.GetDirectoryName(fullPath);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.WriteAllText(fullPath, json);
        }

        Console.WriteLine(json);
    }

    private static void PrintUsage()
    {
        Console.WriteLine("CallbackLifecycle sample");
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project samples/Diagnostics/01.CallbackLifecycle -- --tensor-rt-line 10");
        Console.WriteLine("Options:");
        Console.WriteLine("  --tensor-rt-line <10|11>  TensorRT adapter line. Default: 10.");
        Console.WriteLine("  --output-json <path>      Also write the structured report to a JSON file.");
        Console.WriteLine("  --help, -h                Show this offline help.");
    }

    private sealed class SampleSkippedException : Exception
    {
        public SampleSkippedException(string message)
            : base(message)
        {
        }
    }
}
