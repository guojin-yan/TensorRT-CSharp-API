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

        Console.WriteLine($"ManagedProgressMonitorSmokeRunner TensorRtLineRequest={requestedLine} DependencyProbeOnly={dependencyProbeOnly}");

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

        if (!adapter.BuilderCreationSupported)
        {
            Console.WriteLine($"Skipped=True Reason=AdapterNotReady:{adapter.StatusMessage}");
            return;
        }

        try
        {
            RunManagedProgressMonitorSmoke(line.Value);
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
            return;
        }

        Console.WriteLine("ManagedProgressMonitorSmokeRunner Passed=True");
    }

    private static void RunManagedProgressMonitorSmoke(TensorRtApiLine line)
    {
        if (line == TensorRtApiLine.TensorRt8)
        {
            Console.WriteLine("Skipped=True Reason=ProgressMonitorRequiresTensorRt10Or11");
            return;
        }

        List<TensorRtProgressMonitorEventKind> events = new List<TensorRtProgressMonitorEventKind>();
        using TensorRtProgressMonitor monitor = new TensorRtProgressMonitor(
            line,
            progressEvent =>
            {
                events.Add(progressEvent.Kind);
                return progressEvent.Kind != TensorRtProgressMonitorEventKind.StepComplete || progressEvent.Step < 2;
            });

        if (monitor.TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic))
        {
            Console.WriteLine($"ManagedProgressMonitorInterfaceInfo Kind={interfaceInfo.Kind} Version={interfaceInfo.Major}.{interfaceInfo.Minor}");
        }
        else
        {
            Console.WriteLine($"ManagedProgressMonitorInterfaceInfo Skipped=True Reason={diagnostic}");
        }

        if (monitor.TryGetApiLanguage(out TensorRtApiLanguage apiLanguage, out string apiLanguageDiagnostic))
        {
            Console.WriteLine($"ManagedProgressMonitorApiLanguage Language={apiLanguage}");
        }
        else
        {
            Console.WriteLine($"ManagedProgressMonitorApiLanguage Skipped=True Reason={apiLanguageDiagnostic}");
        }

        TensorRtProgressMonitorDiagnosticResult start = monitor.EmitDiagnostic(TensorRtProgressMonitorEventKind.PhaseStart, "managed progress smoke", null, -1, 3);
        TensorRtProgressMonitorDiagnosticResult step = monitor.EmitDiagnostic(TensorRtProgressMonitorEventKind.StepComplete, "managed progress smoke", null, 2, 3);
        TensorRtProgressMonitorDiagnosticResult finish = monitor.EmitDiagnostic(TensorRtProgressMonitorEventKind.PhaseFinish, "managed progress smoke");

        Console.WriteLine($"ManagedProgressMonitorDiagnostics StartAccepted={start.CallbackAccepted} StepContinue={step.ShouldContinue} StepAccepted={step.CallbackAccepted} FinishAccepted={finish.CallbackAccepted} Invocations={monitor.CallbackInvocationCount} Failures={monitor.CallbackFailureCount} Events={events.Count}");

        if (!start.CallbackAccepted || step.ShouldContinue || !step.CallbackAccepted || !finish.CallbackAccepted || monitor.CallbackInvocationCount != 3 || monitor.CallbackFailureCount != 0 || events.Count != 3)
        {
            throw new InvalidOperationException("Managed progress monitor callback did not receive the expected diagnostic events.");
        }

        using TensorRtProgressMonitor throwingMonitor = new TensorRtProgressMonitor(
            line,
            _ => throw new InvalidOperationException("managed progress monitor callback smoke exception"));

        TensorRtProgressMonitorDiagnosticResult exceptionResult = throwingMonitor.EmitDiagnostic(TensorRtProgressMonitorEventKind.StepComplete, "managed progress exception smoke", null, 0, 1);
        Console.WriteLine($"ManagedProgressMonitorException Accepted={exceptionResult.CallbackAccepted} Continue={exceptionResult.ShouldContinue} Invocations={throwingMonitor.CallbackInvocationCount} Failures={throwingMonitor.CallbackFailureCount} LastException={throwingMonitor.LastCallbackException?.GetType().Name}");

        if (exceptionResult.CallbackAccepted || !exceptionResult.ShouldContinue || throwingMonitor.CallbackInvocationCount != 1 || throwingMonitor.CallbackFailureCount != 1 || throwingMonitor.LastCallbackException == null)
        {
            throw new InvalidOperationException("Managed progress monitor callback exception was not swallowed and recorded as expected.");
        }

        if (!TensorRtEnvironmentProbe.TryCreateBuilder(line, out string builderProbeMessage))
        {
            Console.WriteLine($"ManagedProgressMonitorAttach Skipped=True Reason=BuilderUnavailable:{builderProbeMessage}");
            return;
        }

        try
        {
            using TensorRtLogger logger = new TensorRtLogger(line);
            using TensorRtBuilder builder = new TensorRtBuilder(logger);
            using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
            config.SetProgressMonitor(monitor);
            bool attached = config.HasProgressMonitor && monitor.IsAttached;
            config.ClearProgressMonitor();
            bool cleared = !config.HasProgressMonitor && !monitor.IsAttached;

            Console.WriteLine($"ManagedProgressMonitorAttach Attached={attached} Cleared={cleared}");
            if (!attached || !cleared)
            {
                throw new InvalidOperationException("Managed progress monitor was not attached and cleared as expected.");
            }
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"ManagedProgressMonitorAttach Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
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
            if (snapshot.TensorRt11.BuilderCreationSupported)
            {
                return TensorRtApiLine.TensorRt11;
            }

            if (snapshot.TensorRt10.BuilderCreationSupported)
            {
                return TensorRtApiLine.TensorRt10;
            }

            if (snapshot.TensorRt8.BuilderCreationSupported)
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
                bridgeProbe.StatusCode == BridgeStatusCode.InvalidState;
        }

        return false;
    }
}
