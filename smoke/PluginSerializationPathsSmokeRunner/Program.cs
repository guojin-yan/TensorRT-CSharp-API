using System;
using System.Collections.Generic;
using System.IO;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        string requestedLine = JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "auto");
        bool dependencyProbeOnly = JYPPX.SampleSupport.SampleCommandLine.HasSwitch(args, "--dependency-probe-only");

        Console.WriteLine($"PluginSerializationPathsSmokeRunner TensorRtLineRequest={requestedLine} DependencyProbeOnly={dependencyProbeOnly}");

        TensorRtApiLine probeLine = ResolveProbeLine(requestedLine);
        PrintDependencyProbe(probeLine);
        if (dependencyProbeOnly)
        {
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
            Console.WriteLine("Skipped=True Reason=NoRequestedTensorRtBuilderAvailable");
            return;
        }

        TensorRtAdapterInfo adapter = GetAdapter(snapshot, line.Value);
        Console.WriteLine($"ResolvedTensorRtLine={(int)line.Value} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
        Console.WriteLine($"Adapter Runtime={adapter.RuntimeCreationSupported} Builder={adapter.BuilderCreationSupported} Message={adapter.StatusMessage}");

        try
        {
            RunPluginSerializationPathsSmoke(line.Value);
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
            return;
        }

        Console.WriteLine("PluginSerializationPathsSmokeRunner Passed=True");
    }

    private static void RunPluginSerializationPathsSmoke(TensorRtApiLine line)
    {
        if (!TensorRtEnvironmentProbe.TryCreateBuilder(line, out string builderProbeMessage))
        {
            Console.WriteLine($"Skipped=True Reason=BuilderUnavailable:{builderProbeMessage}");
            return;
        }

        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();

        string firstPath = Path.Combine(Path.GetTempPath(), "jyppx-trt-plugin-a.dll");
        string secondPath = Path.Combine(Path.GetTempPath(), "jyppx-trt-plugin-b.dll");
        string[] paths = new[] { firstPath, secondPath };

        bool set = config.SetPluginsToSerialize(paths);
        IReadOnlyList<string> copied = config.GetPluginsToSerialize();
        TensorRtBuilderConfigSerializedPluginSnapshot snapshot = config.GetSerializedPluginSnapshot();
        bool snapshotTryGet = config.TryGetSerializedPluginSnapshot(out TensorRtBuilderConfigSerializedPluginSnapshot trySnapshot, out string snapshotDiagnostic);
        bool firstMatches = copied.Count == paths.Length &&
            string.Equals(copied[0], firstPath, StringComparison.Ordinal) &&
            string.Equals(copied[1], secondPath, StringComparison.Ordinal);

        config.ClearPluginsToSerialize();
        int clearedCount = config.PluginToSerializeCount;
        bool tryGet = config.TryGetPluginsToSerialize(out IReadOnlyList<string> afterClear, out string diagnostic);
        TensorRtBuilderConfigSerializedPluginSnapshot clearedSnapshot = config.GetSerializedPluginSnapshot();

        Console.WriteLine($"PluginSerializationPaths Set={set} Count={copied.Count} FirstMatches={firstMatches} Snapshot=[{snapshot}] TrySnapshot={snapshotTryGet}/{trySnapshot.Count}/{trySnapshot.PluginLibraryPaths.Count}/{snapshotDiagnostic} ClearedCount={clearedCount} ClearedSnapshot=[{clearedSnapshot}] TryGet={tryGet} AfterClear={afterClear.Count} Diagnostic={diagnostic}");
        if (!set ||
            !firstMatches ||
            !snapshot.HasPathInventory ||
            snapshot.Count != paths.Length ||
            snapshot.PluginLibraryPaths.Count != paths.Length ||
            !snapshotTryGet ||
            trySnapshot.Count != paths.Length ||
            clearedCount != 0 ||
            clearedSnapshot.Count != 0 ||
            !tryGet ||
            afterClear.Count != 0)
        {
            throw new InvalidOperationException("Plugin serialization path set/get/clear round-trip did not match expected values.");
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
            if (snapshot.TensorRt11.RuntimeCreationSupported && snapshot.TensorRt11.BuilderCreationSupported)
            {
                return TensorRtApiLine.TensorRt11;
            }

            if (snapshot.TensorRt10.RuntimeCreationSupported && snapshot.TensorRt10.BuilderCreationSupported)
            {
                return TensorRtApiLine.TensorRt10;
            }

            if (snapshot.TensorRt8.RuntimeCreationSupported && snapshot.TensorRt8.BuilderCreationSupported)
            {
                return TensorRtApiLine.TensorRt8;
            }

            return null;
        }

        TensorRtApiLine? line = ResolveTensorRtLineWithoutSnapshot(requestedLine);
        TensorRtAdapterInfo adapter = GetAdapter(snapshot, line!.Value);
        return adapter.RuntimeCreationSupported && adapter.BuilderCreationSupported ? line : null;
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
