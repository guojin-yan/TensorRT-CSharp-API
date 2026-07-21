using System;
using System.Globalization;
using JYPPX.SampleSupport;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Tools;

namespace OnnxToEngineSample;

internal static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            return Run(args);
        }
        catch (Exception exception) when (TensorRtToolSupport.IsDeploymentException(exception))
        {
            Console.WriteLine($"OnnxToEngine=Skipped Reason={exception.Message}");
            return 0;
        }
        catch (ArgumentException exception)
        {
            Console.WriteLine($"OnnxToEngine=InvalidArguments Reason={exception.Message}");
            PrintUsage();
            return 2;
        }
    }

    private static int Run(string[] args)
    {
        if (SampleCommandLine.HasSwitch(args, "--help"))
        {
            PrintUsage();
            return 0;
        }

        if (SampleCommandLine.HasSwitch(args, "--mnist"))
        {
            return RunMnist(args);
        }

        TrtexecLikeOptions options = TrtexecLikeParser.Parse(args);
        if (options.Batch < 1 || options.Batch > 4)
        {
            throw new ArgumentOutOfRangeException(nameof(options.Batch), "Batch must be in the optimization profile range [1, 4].");
        }

        OnnxEngineBuildResult result = new OnnxEngineBuildService().Execute(OnnxEngineBuildOptions.FromTrtexecLikeOptions(options));
        foreach (string line in result.LogLines)
        {
            Console.WriteLine(line);
        }

        if (!string.IsNullOrWhiteSpace(options.ExportReportPath))
        {
            Console.WriteLine("OnnxToEngine ReportPath=" + options.ExportReportPath);
        }

        Console.WriteLine("OnnxToEngine ProofClassification=" + result.ProofClassification + " BuildEvidenceOnly=" + result.BuildEvidenceOnly + " DryRun=" + options.DryRun);
        Console.WriteLine("OnnxToEngine NormalizedCommandSha256=" + result.NormalizedCommandSha256);
        if (!string.IsNullOrWhiteSpace(result.LoadedEngineDiagnostics.DiagnosticsState))
        {
            Console.WriteLine("OnnxToEngine LoadEngineDiagnosticsState=" + result.LoadedEngineDiagnostics.DiagnosticsState + " Attempted=" + result.LoadedEngineDiagnostics.Attempted + " Succeeded=" + result.LoadedEngineDiagnostics.Succeeded);
            Console.WriteLine("OnnxToEngine LoadEngineDiagnosticsBoundary=" + result.LoadedEngineDiagnostics.EvidenceBoundary);
        }

        Console.WriteLine("OnnxToEngine WorkspaceBytes=" + result.WorkspaceBytes);
        Console.WriteLine("OnnxToEngine State=" + result.State + " Success=" + result.Success);
        return result.Success ? 0 : 2;
    }

    private static int RunMnist(string[] args)
    {
        string onnxPath = SampleCommandLine.GetStringArgument(args, "--onnx", string.Empty);
        string inputPath = SampleCommandLine.GetStringArgument(args, "--mnistInput", string.Empty);
        int expectedDigit = SampleCommandLine.GetIntArgument(args, "--expectedDigit", InferDigit(inputPath));
        string saveEnginePath = SampleCommandLine.GetStringArgument(args, "--saveEngine", string.Empty);
        string exportReportPath = SampleCommandLine.GetStringArgument(args, "--exportReport", string.Empty);
        string exportOutputPath = SampleCommandLine.GetStringArgument(args, "--exportOutput", string.Empty);
        string exportPreprocessedInputPath = SampleCommandLine.GetStringArgument(args, "--exportPreprocessedInput", string.Empty);
        int workspaceMiB = SampleCommandLine.GetPositiveIntArgument(args, "--workspace", 64);
        float minimumConfidence = GetFloatArgument(args, "--minimumConfidence", 0.9f);
        TensorRtApiLine line = ResolveLine(SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "10"));

        MnistOnnxRuntimeOptions options = new MnistOnnxRuntimeOptions(
            line,
            onnxPath,
            inputPath,
            expectedDigit,
            saveEnginePath,
            exportReportPath,
            exportOutputPath,
            exportPreprocessedInputPath,
            checked((ulong)workspaceMiB * 1024UL * 1024UL),
            minimumConfidence);
        MnistOnnxRuntimeResult result = new MnistOnnxRuntimeService().Execute(options);
        foreach (string lineItem in result.LogLines)
        {
            Console.WriteLine(lineItem);
        }

        Console.WriteLine(
            $"MnistOnnxRuntime State={result.State} Success={result.Success} Skipped={result.Skipped} " +
            $"ProofClassification={result.ProofClassification} RealModelRuntime={result.IsRealModelRuntimeProof} " +
            $"PackageConsumerRuntime={result.IsPackageConsumerRuntimeProof}");
        Console.WriteLine(
            $"MnistOnnxRuntime Expected={result.ExpectedDigit} Predicted={result.PredictedDigit} " +
            $"Confidence={result.Confidence:0.000000} OutputMatch={result.OutputMatch}");
        Console.WriteLine("MnistOnnxRuntime ProofBoundary=" + result.ProofBoundary);
        return result.Success ? 0 : 2;
    }

    private static TensorRtApiLine ResolveLine(string value)
    {
        if (string.Equals(value, "8", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "trt8", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt8;
        }

        if (string.Equals(value, "10", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "trt10", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt10;
        }

        if (string.Equals(value, "11", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "trt11", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt11;
        }

        throw new ArgumentException("TensorRT line must be 8, 10, or 11.", nameof(value));
    }

    private static int InferDigit(string path)
    {
        string fileName = System.IO.Path.GetFileNameWithoutExtension(path);
        return int.TryParse(fileName, NumberStyles.None, CultureInfo.InvariantCulture, out int digit)
            ? digit
            : -1;
    }

    private static float GetFloatArgument(string[] args, string name, float defaultValue)
    {
        string text = SampleCommandLine.GetStringArgument(
            args,
            name,
            defaultValue.ToString(CultureInfo.InvariantCulture));
        return float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value)
            ? value
            : defaultValue;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("OnnxToEngine sample");
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project samples/OnnxToEngine -- --tensor-rt-line 10 --batch 2");
        Console.WriteLine("  dotnet run --project samples/OnnxToEngine -- --onnx model.onnx --saveEngine model.plan --minShapes input:1x3x640x640 --optShapes input:1x3x640x640 --maxShapes input:4x3x640x640 --buildOnly");
        Console.WriteLine("  dotnet run --project samples/OnnxToEngine -- --mnist --tensor-rt-line 10 --onnx mnist.onnx --mnistInput 7.pgm --expectedDigit 7 --saveEngine mnist.plan --exportReport mnist-report.json --exportOutput mnist-output.json --exportPreprocessedInput mnist-input.bin");
        Console.WriteLine("Options:");
        Console.WriteLine("  --tensor-rt-line <8|10|11>  TensorRT adapter line. Default: 10.");
        Console.WriteLine("  --batch <1..4>              Runtime batch inside the optimization profile. Default: 2.");
        Console.WriteLine("  --onnx <path>               External ONNX model. External models default to build/skip-inference semantics in this sample stage.");
        Console.WriteLine("  --mnist                     Use the explicit MNIST model runner instead of generic external-model build-only behavior.");
        Console.WriteLine("  --mnistInput <path.pgm>     P5 PGM input for the MNIST runner.");
        Console.WriteLine("  --expectedDigit <0..9>      Expected class; defaults to the PGM file name when it is 0.pgm through 9.pgm.");
        Console.WriteLine("  --minimumConfidence <0..1>  Required softmax confidence for real-model-runtime. Default: 0.9.");
        Console.WriteLine("  --exportPreprocessedInput   Write the float32 input tensor bytes used for enqueue.");
        Console.WriteLine("  --saveEngine <path>         Save the serialized TensorRT engine.");
        Console.WriteLine("  --minShapes/--optShapes/--maxShapes input:1x3x640x640[,other:...]");
        Console.WriteLine("  --fp16 --int8 --bf16 --noTF32 --workspace <MiB>");
        Console.WriteLine("  --builderOptimizationLevel <0..5> --maxAuxStreams <n>");
        Console.WriteLine("  --device <ordinal> --useDLACore <n> --allowGPUFallback --tacticSources <list> --memPoolSize workspace:512,tacticDram:1024");
        Console.WriteLine("  --inputIOFormats <fmt> --outputIOFormats <fmt> --calib <cache> --directIO --sparsity <mode> --stronglyTyped");
        Console.WriteLine("  --precisionConstraints <none|prefer|obey> --layerPrecisions <pattern:type> --layerOutputTypes <pattern:type[+type]>");
        Console.WriteLine("  --shapes/--inputShapes input:1x3x640x640   Alias used as min/opt/max shapes when no explicit profile triplet is provided.");
        Console.WriteLine("  --save-engine <path> --load-engine <path> --timingCache <path> --verbose");
        Console.WriteLine("  Memory values accept MiB by default or suffixes such as 512MiB and 1GiB.");
        Console.WriteLine("  --iterations <n> --warmUp <ms> --duration <sec> --streams <n> --useCudaGraph");
        Console.WriteLine("  --noDataTransfers --useSpinWait --threads --avgRuns <n> --percentile <0..100>");
        Console.WriteLine("  --loadInputs input:file --dumpOutput --dumpRawBindingsToFile <path> --exportOutput <path> --exportTimes <path> --exportProfile <path> --saveProfile <path>");
        Console.WriteLine("  --safe --consistency --builderCache|--noBuilderCache");
        Console.WriteLine("  --buildOnly --skipInference --dryRun|--previewOnly --dumpLayerInfo --exportLayerInfo <path>");
        Console.WriteLine("  --exportReport|--report <path.json|path.md>");
    }
}
