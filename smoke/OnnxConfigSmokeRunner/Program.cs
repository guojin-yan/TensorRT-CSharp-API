using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        string requestedLine = JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "auto");
        bool dependencyProbeOnly = JYPPX.SampleSupport.SampleCommandLine.HasSwitch(args, "--dependency-probe-only");

        Console.WriteLine($"OnnxConfigSmokeRunner TensorRtLineRequest={requestedLine} DependencyProbeOnly={dependencyProbeOnly}");

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

        try
        {
            RunOnnxConfigSmoke(line.Value);
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
            return;
        }

        Console.WriteLine("OnnxConfigSmokeRunner Passed=True");
    }

    private static void RunOnnxConfigSmoke(TensorRtApiLine line)
    {
        using TensorRtOnnxConfig config = new TensorRtOnnxConfig(line);

        TensorRtDataType initialDataType = config.ModelDataType;
        int initialVerbosity = config.VerbosityLevel;
        bool initialPrintLayerInfo = config.PrintLayerInfo;

        config.ModelDataType = TensorRtDataType.Float;
        TensorRtDataType floatDataType = config.ModelDataType;
        config.ModelDataType = TensorRtDataType.Half;
        TensorRtDataType halfDataType = config.ModelDataType;
        config.ModelDataType = TensorRtDataType.Int8;
        TensorRtDataType int8DataType = config.ModelDataType;
        config.ModelDataType = TensorRtDataType.Float;

        int targetVerbosity = Math.Max(0, initialVerbosity);
        config.VerbosityLevel = targetVerbosity;
        int roundtripVerbosity = config.VerbosityLevel;
        config.IncreaseVerbosity();
        int increasedVerbosity = config.VerbosityLevel;
        config.DecreaseVerbosity();
        int decreasedVerbosity = config.VerbosityLevel;

        config.ModelFileName = "models/yolovision.onnx";
        config.TextFileName = "artifacts/onnx-parser.txt";
        config.FullTextFileName = "artifacts/onnx-parser-full.txt";
        string modelFileName = config.ModelFileName;
        string textFileName = config.TextFileName;
        string fullTextFileName = config.FullTextFileName;

        config.PrintLayerInfo = !initialPrintLayerInfo;
        bool toggledPrintLayerInfo = config.PrintLayerInfo;
        config.PrintLayerInfo = initialPrintLayerInfo;
        bool restoredPrintLayerInfo = config.PrintLayerInfo;

        Console.WriteLine($"OnnxConfig InitialDataType={initialDataType} Float={floatDataType} Half={halfDataType} Int8={int8DataType}");
        Console.WriteLine($"OnnxConfig Verbosity={initialVerbosity}->{roundtripVerbosity}->{increasedVerbosity}->{decreasedVerbosity} PrintLayerInfo={initialPrintLayerInfo}->{toggledPrintLayerInfo}->{restoredPrintLayerInfo}");
        Console.WriteLine($"OnnxConfig FileNames Model={modelFileName} Text={textFileName} FullText={fullTextFileName}");

        if (floatDataType != TensorRtDataType.Float ||
            halfDataType != TensorRtDataType.Half ||
            int8DataType != TensorRtDataType.Int8 ||
            roundtripVerbosity != targetVerbosity ||
            increasedVerbosity != targetVerbosity + 1 ||
            decreasedVerbosity != targetVerbosity ||
            modelFileName != "models/yolovision.onnx" ||
            textFileName != "artifacts/onnx-parser.txt" ||
            fullTextFileName != "artifacts/onnx-parser-full.txt" ||
            toggledPrintLayerInfo == initialPrintLayerInfo ||
            restoredPrintLayerInfo != initialPrintLayerInfo)
        {
            throw new InvalidOperationException("ONNX config scalar controls did not roundtrip as expected.");
        }

        try
        {
            config.ModelDataType = TensorRtDataType.Bool;
            throw new InvalidOperationException("ONNX config accepted an unsupported model data type.");
        }
        catch (ArgumentOutOfRangeException)
        {
            Console.WriteLine("OnnxConfig ManagedValidation=ModelDataType");
        }

        try
        {
            config.VerbosityLevel = -1;
            throw new InvalidOperationException("ONNX config accepted a negative verbosity level.");
        }
        catch (ArgumentOutOfRangeException)
        {
            Console.WriteLine("OnnxConfig ManagedValidation=Verbosity");
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
