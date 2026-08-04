using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        string requestedLine = JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "auto");
        bool dependencyProbeOnly = JYPPX.SampleSupport.SampleCommandLine.HasSwitch(args, "--dependency-probe-only");

        Console.WriteLine($"ManagedLoggerCallbackSmokeRunner TensorRtLineRequest={requestedLine} DependencyProbeOnly={dependencyProbeOnly}");

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
            RunManagedLoggerCallbackSmoke(line.Value);
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
            return;
        }

        Console.WriteLine("ManagedLoggerCallbackSmokeRunner Passed=True");
    }

    private static void RunManagedLoggerCallbackSmoke(TensorRtApiLine line)
    {
        List<string> messages = new List<string>();
        using TensorRtLogger logger = new TensorRtLogger(
            line,
            (severity, message) => messages.Add($"{severity}:{message}"),
            TensorRtLogSeverity.Verbose);

        if (logger.TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic))
        {
            Console.WriteLine($"ManagedLoggerInterfaceInfo Kind={interfaceInfo.Kind} Version={interfaceInfo.Major}.{interfaceInfo.Minor}");
        }
        else
        {
            Console.WriteLine($"ManagedLoggerInterfaceInfo Skipped=True Reason={diagnostic}");
        }

        if (logger.TryGetApiLanguage(out TensorRtApiLanguage apiLanguage, out string apiLanguageDiagnostic))
        {
            Console.WriteLine($"ManagedLoggerApiLanguage Language={apiLanguage}");
        }
        else
        {
            Console.WriteLine($"ManagedLoggerApiLanguage Skipped=True Reason={apiLanguageDiagnostic}");
        }

        bool accepted = logger.EmitDiagnostic(TensorRtLogSeverity.Warning, "managed logger callback smoke");
        Console.WriteLine($"ManagedLoggerCallback Accepted={accepted} Invocations={logger.CallbackInvocationCount} Failures={logger.CallbackFailureCount} Messages={messages.Count}");

        if (!accepted || logger.CallbackInvocationCount != 1 || logger.CallbackFailureCount != 0 || messages.Count != 1)
        {
            throw new InvalidOperationException("Managed logger callback did not receive the expected diagnostic message.");
        }

        using TensorRtLogger throwingLogger = new TensorRtLogger(
            line,
            static (_, _) => throw new InvalidOperationException("managed logger callback smoke exception"),
            TensorRtLogSeverity.Verbose);

        bool exceptionAccepted = throwingLogger.EmitDiagnostic(TensorRtLogSeverity.Warning, "managed logger callback exception smoke");
        Console.WriteLine($"ManagedLoggerCallbackException Accepted={exceptionAccepted} Invocations={throwingLogger.CallbackInvocationCount} Failures={throwingLogger.CallbackFailureCount} LastException={throwingLogger.LastCallbackException?.GetType().Name}");

        if (exceptionAccepted || throwingLogger.CallbackInvocationCount != 1 || throwingLogger.CallbackFailureCount != 1 || throwingLogger.LastCallbackException == null)
        {
            throw new InvalidOperationException("Managed logger callback exception was not swallowed and recorded as expected.");
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
                bridgeProbe.StatusCode == BridgeStatusCode.InvalidState;
        }

        return false;
    }
}
