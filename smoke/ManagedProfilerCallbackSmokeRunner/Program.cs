using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        string requestedLine = JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "auto");
        bool dependencyProbeOnly = JYPPX.SampleSupport.SampleCommandLine.HasSwitch(args, "--dependency-probe-only");

        Console.WriteLine($"ManagedProfilerCallbackSmokeRunner TensorRtLineRequest={requestedLine} DependencyProbeOnly={dependencyProbeOnly}");

        if (dependencyProbeOnly)
        {
            TensorRtApiLine probeLine = ResolveProbeLine(requestedLine);
            PrintDependencyProbe(probeLine);
            Console.WriteLine("Skipped=True Reason=DependencyProbeOnly");
            return;
        }

        TensorRtEnvironmentSnapshot snapshot;
        try
        {
            snapshot = TensorRtEnvironmentProbe.GetCurrent();
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"Skipped=True Reason=EnvironmentProbe:{exception.GetType().Name}:{exception.Message}");
            return;
        }

        TensorRtApiLine? line = ResolveTensorRtLine(snapshot, requestedLine);
        if (line == null)
        {
            Console.WriteLine("Skipped=True Reason=NoRequestedTensorRtAdapterAvailable");
            return;
        }

        TensorRtAdapterInfo adapter = GetAdapter(snapshot, line.Value);
        Console.WriteLine($"ResolvedTensorRtLine={(int)line.Value} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
        Console.WriteLine($"Adapter Runtime={adapter.RuntimeCreationSupported} Builder={adapter.BuilderCreationSupported} Message={adapter.StatusMessage}");
        PrintDependencyProbe(line.Value);

        if (!adapter.RuntimeCreationSupported)
        {
            Console.WriteLine($"Skipped=True Reason=AdapterNotReady:{adapter.StatusMessage}");
            return;
        }

        try
        {
            RunManagedProfilerCallbackSmoke(line.Value, adapter.BuilderCreationSupported);
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
            return;
        }

        Console.WriteLine("ManagedProfilerCallbackSmokeRunner Passed=True");
    }

    private static void RunManagedProfilerCallbackSmoke(TensorRtApiLine line, bool builderCreationSupported)
    {
        List<string> records = new List<string>();
        using TensorRtProfiler profiler = new TensorRtProfiler(
            line,
            (layerName, milliseconds) => records.Add($"{layerName}:{milliseconds:0.###}"));

        if (profiler.TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic))
        {
            Console.WriteLine($"ManagedProfilerInterfaceInfo Kind={interfaceInfo.Kind} Version={interfaceInfo.Major}.{interfaceInfo.Minor}");
        }
        else
        {
            Console.WriteLine($"ManagedProfilerInterfaceInfo Skipped=True Reason={diagnostic}");
        }

        if (profiler.TryGetApiLanguage(out TensorRtApiLanguage apiLanguage, out string apiLanguageDiagnostic))
        {
            Console.WriteLine($"ManagedProfilerApiLanguage Language={apiLanguage}");
        }
        else
        {
            Console.WriteLine($"ManagedProfilerApiLanguage Skipped=True Reason={apiLanguageDiagnostic}");
        }

        bool accepted = profiler.EmitDiagnostic("managed profiler callback smoke", 1.25f);
        Console.WriteLine($"ManagedProfilerCallback Accepted={accepted} Invocations={profiler.CallbackInvocationCount} Failures={profiler.CallbackFailureCount} Records={records.Count}");

        if (!accepted || profiler.CallbackInvocationCount != 1 || profiler.CallbackFailureCount != 0 || records.Count != 1)
        {
            throw new InvalidOperationException("Managed profiler callback did not receive the expected diagnostic record.");
        }

        using TensorRtProfiler throwingProfiler = new TensorRtProfiler(
            line,
            static (_, _) => throw new InvalidOperationException("managed profiler callback smoke exception"));

        bool exceptionAccepted = throwingProfiler.EmitDiagnostic("managed profiler callback exception smoke", 2.5f);
        Console.WriteLine($"ManagedProfilerCallbackException Accepted={exceptionAccepted} Invocations={throwingProfiler.CallbackInvocationCount} Failures={throwingProfiler.CallbackFailureCount} LastException={throwingProfiler.LastCallbackException?.GetType().Name}");

        if (exceptionAccepted || throwingProfiler.CallbackInvocationCount != 1 || throwingProfiler.CallbackFailureCount != 1 || throwingProfiler.LastCallbackException == null)
        {
            throw new InvalidOperationException("Managed profiler callback exception was not swallowed and recorded as expected.");
        }

        if (!builderCreationSupported)
        {
            Console.WriteLine("ManagedProfilerAttach Skipped=True Reason=BuilderUnavailable:AdapterNotReady");
            return;
        }

        if (!TensorRtEnvironmentProbe.TryCreateBuilder(line, out string builderProbeMessage))
        {
            Console.WriteLine($"ManagedProfilerAttach Skipped=True Reason=BuilderUnavailable:{builderProbeMessage}");
            return;
        }

        try
        {
            using TensorRtLogger logger = new TensorRtLogger(line);
            using TensorRtBuilder builder = new TensorRtBuilder(logger);
            using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
            using TensorRtNetworkDefinition network = builder.CreateNetwork();
            using TensorRtHostMemory serialized = builder.BuildSerializedNetwork(network, config);
            using TensorRtRuntime runtime = new TensorRtRuntime(logger);
            using TensorRtEngine engine = runtime.Deserialize(serialized);
            using TensorRtExecutionContext context = engine.CreateExecutionContext();

            bool nativeBeforeAttach = context.HasNativeProfiler;
            context.SetProfiler(profiler);
            bool nativeAfterAttach = context.HasNativeProfiler;
            bool attached = context.HasProfiler && profiler.IsAttached;
            context.ClearProfiler();
            bool nativeAfterClear = context.HasNativeProfiler;
            bool cleared = !context.HasProfiler && !profiler.IsAttached;

            Console.WriteLine($"ManagedProfilerAttach Attached={attached} Cleared={cleared} Native={nativeBeforeAttach}->{nativeAfterAttach}->{nativeAfterClear}");
            if (!attached || !cleared)
            {
                throw new InvalidOperationException("Managed profiler was not attached and cleared as expected.");
            }
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"ManagedProfilerAttach Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
        }
    }

    private static void PrintDependencyProbe(TensorRtApiLine line)
    {
        TensorRtDependencyProbeReport dependencyProbe = TensorRtEnvironmentProbe.ProbeNativeDependencies(line);
        Console.WriteLine($"DependencyProbe Line={(int)line} BridgeInitialized={dependencyProbe.BridgeInitialized} Candidates={dependencyProbe.NativeBridgeCandidates.Count} Loaded={dependencyProbe.LoadedModuleCount} SearchPathCandidates={dependencyProbe.SearchPathCandidateCount} Diagnostics={dependencyProbe.Diagnostics.Count} Message={dependencyProbe.BridgeDiagnostic}");
    }

    private static TensorRtApiLine ResolveProbeLine(string requestedLine)
    {
        if (string.Equals(requestedLine, "auto", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt11;
        }

        TensorRtApiLine? line = ResolveTensorRtLineWithoutSnapshot(requestedLine);
        if (line.HasValue)
        {
            return line.Value;
        }

        throw new ArgumentException("TensorRT line must be auto, 8, 10, or 11.", nameof(requestedLine));
    }

    private static TensorRtApiLine? ResolveTensorRtLine(TensorRtEnvironmentSnapshot snapshot, string requestedLine)
    {
        if (string.Equals(requestedLine, "auto", StringComparison.OrdinalIgnoreCase))
        {
            if (snapshot.TensorRt11.RuntimeCreationSupported)
            {
                return TensorRtApiLine.TensorRt11;
            }

            if (snapshot.TensorRt10.RuntimeCreationSupported)
            {
                return TensorRtApiLine.TensorRt10;
            }

            if (snapshot.TensorRt8.RuntimeCreationSupported)
            {
                return TensorRtApiLine.TensorRt8;
            }

            return null;
        }

        return ResolveTensorRtLineWithoutSnapshot(requestedLine);
    }

    private static TensorRtApiLine? ResolveTensorRtLineWithoutSnapshot(string requestedLine)
    {
        if (string.Equals(requestedLine, "8", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(requestedLine, "trt8", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt8;
        }

        if (string.Equals(requestedLine, "10", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(requestedLine, "trt10", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt10;
        }

        if (string.Equals(requestedLine, "11", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(requestedLine, "trt11", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt11;
        }

        throw new ArgumentException("TensorRT line must be auto, 8, 10, or 11.", nameof(requestedLine));
    }

    private static TensorRtAdapterInfo GetAdapter(TensorRtEnvironmentSnapshot snapshot, TensorRtApiLine line)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => snapshot.TensorRt8,
            TensorRtApiLine.TensorRt10 => snapshot.TensorRt10,
            TensorRtApiLine.TensorRt11 => snapshot.TensorRt11,
            _ => snapshot.TensorRt11
        };
    }

    private static bool IsSkippableEnvironmentException(Exception exception)
    {
        if (exception is DllNotFoundException || exception is BadImageFormatException)
        {
            return true;
        }

        if (exception is BridgeProbeException bridgeProbe)
        {
            return bridgeProbe.StatusCode == BridgeStatusCode.DependencyMissing ||
                bridgeProbe.StatusCode == BridgeStatusCode.NotSupported ||
                bridgeProbe.StatusCode == BridgeStatusCode.InvalidState ||
                bridgeProbe.StatusCode == BridgeStatusCode.RuntimeError;
        }

        return false;
    }
}
