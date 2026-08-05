using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp;
using JYPPX.TensorRtSharp.Shared.Interop;

internal static class Program
{
    private const string InputName = "stream_reader_input";
    private const string OutputName = "stream_reader_output";
    private const int ElementCount = 4;

    public static int Main(string[] args)
    {
        try
        {
            TensorRtApiLine line = ParseTensorRtLine(GetArgument(args, "--tensor-rt-line", "10"));
            string runtimePackageKey = GetArgument(args, "--runtime-package-key", "unspecified");
            if (!HasSwitch(args, "--stream-reader-runtime-smoke-only"))
            {
                throw new ArgumentException("The --stream-reader-runtime-smoke-only switch is required.");
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

            RunRealStreamReaderSmoke(line, runtimePackageKey);
            Console.WriteLine("StreamReaderPackageConsumer Passed=True Mode=StreamReaderRuntimeSmokeOnly");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine("StreamReaderPackageConsumer Passed=False Error=" + exception);
            return 1;
        }
    }

    private static void RunRealStreamReaderSmoke(TensorRtApiLine line, string runtimePackageKey)
    {
        using TensorRtLogger logger = new(line);
        using TensorRtHostMemory hostMemory = BuildPlan(line, logger);
        byte[] plan = hostMemory.ToArray();
        string planSha256 = Convert.ToHexString(SHA256.HashData(plan)).ToLowerInvariant();

        using TensorRtRuntime runtime = new(logger);
        using TensorRtEngine baseline = runtime.Deserialize(plan);
        EngineIdentity expected = ReadIdentity(baseline);

        byte[] mutableSource = plan.ToArray();
        using TensorRtStreamReader reader = new(line, mutableSource);
        Array.Clear(mutableSource, 0, mutableSource.Length);

        TensorRtStreamReaderRuntimeSnapshot positive;
        using (TensorRtEngine first = runtime.Deserialize(reader))
        {
            RequireIdentity(expected, first);
            ExecuteOnce(first);
        }
        using (TensorRtEngine second = runtime.Deserialize(reader))
        {
            RequireIdentity(expected, second);
            ExecuteOnce(second);
        }
        positive = reader.GetRuntimeSnapshot();

        bool metadataMatched =
            positive.Line == line &&
            positive.Length == (ulong)plan.Length &&
            positive.OwnerId > 0UL &&
            positive.DeserializeAttemptCount == 2UL &&
            positive.SuccessfulDeserializeCount == 2UL &&
            positive.FailedDeserializeCount == 0UL &&
            positive.ReadCount > 0UL &&
            positive.BytesRead > 0UL &&
            positive.HostReadCount + positive.DeviceReadCount > 0UL &&
            positive.HostReadCount + positive.DeviceReadCount <= positive.ReadCount &&
            positive.FailureCount == 0UL &&
            positive.InFlightCallbackCount == 0UL &&
            !positive.IsDeserializing &&
            positive.LastOperationSucceeded;
        if (!metadataMatched)
        {
            throw new InvalidOperationException("Positive IStreamReaderV2 snapshot was inconsistent: " + FormatSnapshot(positive));
        }

        LifecycleResult lifecycle = RunDeferredDisposeCase(line, runtime, plan, expected);
        NegativeResult truncated = RunTruncatedInputCase(line, runtime, plan);

        bool trt8Rejected = false;
        try
        {
            using TensorRtStreamReader unsupported = new(TensorRtApiLine.TensorRt8, plan);
        }
        catch (NotSupportedException)
        {
            trt8Rejected = true;
        }

        bool pointerExposed = PublicSnapshotExposesNativePointer();
        bool passed =
            metadataMatched &&
            lifecycle.DisposeDeferred &&
            lifecycle.RejectsNewDeserialize &&
            lifecycle.ReleasedAfterEngine &&
            truncated.DeserializeFailed &&
            truncated.Snapshot.DeserializeAttemptCount == 1UL &&
            truncated.Snapshot.FailedDeserializeCount == 1UL &&
            truncated.Snapshot.SuccessfulDeserializeCount == 0UL &&
            truncated.Snapshot.FailureCount > 0UL &&
            truncated.Snapshot.InFlightCallbackCount == 0UL &&
            trt8Rejected &&
            !pointerExposed;
        if (!passed)
        {
            throw new InvalidOperationException(
                "Stream reader runtime invariants failed. Positive=" + FormatSnapshot(positive) +
                " Lifecycle=" + lifecycle + " Truncated=" + FormatSnapshot(truncated.Snapshot) +
                " Trt8Rejected=" + trt8Rejected + " PointerExposed=" + pointerExposed);
        }

        Console.WriteLine("StreamReaderRuntimeSummary");
        Console.WriteLine($"  TensorRtLine={(int)line} RuntimePackageKey={runtimePackageKey}");
        Console.WriteLine($"  PlanBytes={plan.Length} PlanSha256={planSha256}");
        Console.WriteLine($"  Attempts={positive.DeserializeAttemptCount} Success={positive.SuccessfulDeserializeCount} Reads={positive.ReadCount} Seeks={positive.SeekCount}");
        Console.WriteLine($"  HostReads={positive.HostReadCount} DeviceReads={positive.DeviceReadCount} BytesRead={positive.BytesRead} Failures={positive.FailureCount}");
        Console.WriteLine($"  DeferredDispose={lifecycle.DisposeDeferred} RejectsNewDeserialize={lifecycle.RejectsNewDeserialize} ReleasedAfterEngine={lifecycle.ReleasedAfterEngine}");
        Console.WriteLine($"  TruncatedInput=Passed DeserializeFailed={truncated.DeserializeFailed} Attempts={truncated.Snapshot.DeserializeAttemptCount} Failures={truncated.Snapshot.FailureCount}");
        Console.WriteLine(
            "StreamReaderRealRuntime=Passed" +
            $" TensorRtLine={(int)line}" +
            $" RuntimePackageKey={runtimePackageKey}" +
            $" PlanBytes={plan.Length}" +
            $" PlanSha256={planSha256}" +
            $" AttemptCount={positive.DeserializeAttemptCount}" +
            $" SuccessfulDeserializeCount={positive.SuccessfulDeserializeCount}" +
            $" FailedDeserializeCount={positive.FailedDeserializeCount}" +
            $" ReadCount={positive.ReadCount}" +
            $" SeekCount={positive.SeekCount}" +
            $" HostReadCount={positive.HostReadCount}" +
            $" DeviceReadCount={positive.DeviceReadCount}" +
            $" BytesRead={positive.BytesRead}" +
            $" FailureCount={positive.FailureCount}" +
            $" InFlightCallbackCount={positive.InFlightCallbackCount}" +
            $" MetadataMatched={metadataMatched}" +
            " SourceCopied=True" +
            " Reusable=True" +
            $" LifecycleDisposeDeferred={lifecycle.DisposeDeferred}" +
            $" LifecycleRejectsNewDeserialize={lifecycle.RejectsNewDeserialize}" +
            $" LifecycleReleasedAfterEngine={lifecycle.ReleasedAfterEngine}" +
            $" TruncatedDeserializeFailed={truncated.DeserializeFailed}" +
            $" TruncatedAttemptCount={truncated.Snapshot.DeserializeAttemptCount}" +
            $" TruncatedFailedCount={truncated.Snapshot.FailedDeserializeCount}" +
            $" TruncatedFailureCount={truncated.Snapshot.FailureCount}" +
            $" Trt8Rejected={trt8Rejected}" +
            $" PointerExposed={pointerExposed}" +
            " RealCallbackRuntime=True" +
            " LegacyStreamReaderDeferred=True" +
            " StreamWriterVerified=False");
    }

    private static LifecycleResult RunDeferredDisposeCase(
        TensorRtApiLine line,
        TensorRtRuntime runtime,
        byte[] plan,
        EngineIdentity expected)
    {
        TensorRtStreamReader reader = new(line, new MemoryStream(plan, writable: false));
        TensorRtEngine engine = runtime.Deserialize(reader);
        TensorRtStreamReaderRuntimeSnapshot beforeDispose = reader.GetRuntimeSnapshot();
        reader.Dispose();

        bool disposeDeferred = reader.IsDisposed && beforeDispose.SuccessfulDeserializeCount == 1UL;
        RequireIdentity(expected, engine);
        ExecuteOnce(engine);
        TensorRtStreamReaderRuntimeSnapshot afterDispose = reader.GetRuntimeSnapshot();
        disposeDeferred = disposeDeferred && afterDispose.OwnerId == beforeDispose.OwnerId;

        bool rejectsNewDeserialize = false;
        try
        {
            using TensorRtEngine unexpected = runtime.Deserialize(reader);
        }
        catch (ObjectDisposedException)
        {
            rejectsNewDeserialize = true;
        }

        engine.Dispose();
        bool releasedAfterEngine = false;
        try
        {
            _ = reader.GetRuntimeSnapshot();
        }
        catch (ObjectDisposedException)
        {
            releasedAfterEngine = true;
        }
        finally
        {
            reader.Dispose();
        }

        return new LifecycleResult(disposeDeferred, rejectsNewDeserialize, releasedAfterEngine);
    }

    private static NegativeResult RunTruncatedInputCase(
        TensorRtApiLine line,
        TensorRtRuntime runtime,
        byte[] plan)
    {
        int truncatedLength = Math.Max(1, plan.Length / 8);
        byte[] truncatedPlan = new byte[truncatedLength];
        Buffer.BlockCopy(plan, 0, truncatedPlan, 0, truncatedLength);
        using TensorRtStreamReader reader = new(line, truncatedPlan);
        bool deserializeFailed = false;
        try
        {
            using TensorRtEngine unexpected = runtime.Deserialize(reader);
        }
        catch (TensorRtException)
        {
            deserializeFailed = true;
        }

        return new NegativeResult(deserializeFailed, reader.GetRuntimeSnapshot());
    }

    private static TensorRtHostMemory BuildPlan(TensorRtApiLine line, TensorRtLogger logger)
    {
        using TensorRtBuilder builder = new(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 32UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetAverageTimingIterations(1);
        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        network.Name = "stream_reader_package_consumer";
        using TensorRtTensor input = network.AddInput(InputName, TensorRtDataType.Float, new TensorRtDims(new[] { 1, ElementCount }));
        using TensorRtLayer identity = network.AddIdentity(input);
        identity.Name = "stream_reader_identity";
        using TensorRtTensor output = identity.GetOutput(0);
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

    private static EngineIdentity ReadIdentity(TensorRtEngine engine)
    {
        return new EngineIdentity(engine.Name, engine.IOTensorCount, engine.LayerCount);
    }

    private static void RequireIdentity(EngineIdentity expected, TensorRtEngine actual)
    {
        EngineIdentity observed = ReadIdentity(actual);
        if (observed != expected)
        {
            throw new InvalidOperationException("Deserialized engine metadata differs from the buffer baseline.");
        }
    }

    private static bool PublicSnapshotExposesNativePointer()
    {
        foreach (PropertyInfo property in typeof(TensorRtStreamReaderRuntimeSnapshot).GetProperties(BindingFlags.Instance | BindingFlags.Public))
        {
            Type type = property.PropertyType;
            if (type == typeof(IntPtr) || type == typeof(UIntPtr) || typeof(SafeHandle).IsAssignableFrom(type))
            {
                return true;
            }
        }
        return false;
    }

    private static string FormatSnapshot(TensorRtStreamReaderRuntimeSnapshot snapshot)
    {
        return $"Attempts={snapshot.DeserializeAttemptCount};Success={snapshot.SuccessfulDeserializeCount};" +
               $"Failed={snapshot.FailedDeserializeCount};Reads={snapshot.ReadCount};Seeks={snapshot.SeekCount};" +
               $"Bytes={snapshot.BytesRead};Failures={snapshot.FailureCount};InFlight={snapshot.InFlightCallbackCount};" +
               $"Diagnostic={snapshot.LastDiagnostic}";
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
            "10" => TensorRtApiLine.TensorRt10,
            "11" => TensorRtApiLine.TensorRt11,
            _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Stream reader runtime proof requires TensorRT line 10 or 11.")
        };
    }

    private sealed record EngineIdentity(string Name, int IOTensorCount, int LayerCount);
    private sealed record LifecycleResult(bool DisposeDeferred, bool RejectsNewDeserialize, bool ReleasedAfterEngine);
    private sealed record NegativeResult(bool DeserializeFailed, TensorRtStreamReaderRuntimeSnapshot Snapshot);
}
