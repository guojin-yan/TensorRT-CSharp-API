using System;
using JYPPX.TensorRtSharp.Tools;
using TensorRtExecApp.Core;

namespace TensorRtExecApp.Console;

public static class TensorRtExecCommand
{
    public static int Run(string[] args)
    {
        if (HasHelp(args))
        {
            PrintUsage();
            return 0;
        }

        if (HasSwitch(args, "--help-json") || HasSwitch(args, "--capabilities-json"))
        {
            System.Console.WriteLine(TrtexecLikeOptionCapabilities.FormatJson("TensorRtExec"));
            return 0;
        }

        try
        {
            TensorRtExecOptions options = TensorRtExecOptions.Parse(args);
            TensorRtExecReport report = new TensorRtExecService().Execute(options);
            foreach (string line in report.LogLines)
            {
                System.Console.WriteLine(line);
            }

            if (!string.IsNullOrWhiteSpace(report.ReportPath))
            {
                System.Console.WriteLine("TensorRtExec ReportPath=" + report.ReportPath);
            }

            if (!string.IsNullOrWhiteSpace(report.ProofClassification))
            {
                System.Console.WriteLine("TensorRtExec ProofClassification=" + report.ProofClassification + " BuildEvidenceOnly=" + report.BuildEvidenceOnly + " DryRun=" + report.DryRun);
            }

            if (!string.IsNullOrWhiteSpace(report.NormalizedCommandSha256))
            {
                System.Console.WriteLine("TensorRtExec NormalizedCommandSha256=" + report.NormalizedCommandSha256);
            }

            if (!string.IsNullOrWhiteSpace(report.LoadEngineDiagnosticsState))
            {
                System.Console.WriteLine("TensorRtExec LoadEngineDiagnosticsState=" + report.LoadEngineDiagnosticsState + " Attempted=" + report.LoadEngineDiagnosticsAttempted + " Succeeded=" + report.LoadEngineDiagnosticsSucceeded);
                System.Console.WriteLine("TensorRtExec LoadEngineDiagnosticsBoundary=" + report.LoadEngineDiagnosticsBoundary);
            }

            System.Console.WriteLine("TensorRtExec WorkspaceBytes=" + report.WorkspaceBytes);
            System.Console.WriteLine("TensorRtExec BuilderConfigDeploymentSnapshot=" + report.BuilderConfigDeploymentSnapshotState + " Diagnostics=" + report.BuilderConfigDeploymentDiagnosticCount);
            System.Console.WriteLine("TensorRtExec ParserPreflightSnapshot=" + report.ParserPreflightSnapshotState + " Diagnostics=" + report.ParserPreflightDiagnosticCount);
            System.Console.WriteLine("TensorRtExec RefitSnapshot=" + report.RefitState + " Attempted=" + report.RefitAttempted + " Succeeded=" + report.RefitSucceeded);
            System.Console.WriteLine("TensorRtExec RefitPersistence=" + report.RefitPersistenceState + " Attempted=" + report.RefitPersistenceAttempted + " Succeeded=" + report.RefitPersistenceSucceeded + " Plan=" + report.PersistedRefittedEnginePath);
            System.Console.WriteLine("TensorRtExec State=" + report.State + " Success=" + report.Success);
            return report.Success ? 0 : 2;
        }
        catch (Exception exception) when (exception is ArgumentException || exception is System.IO.FileNotFoundException)
        {
            System.Console.WriteLine("TensorRtExec=InvalidArguments Reason=" + exception.Message);
            PrintUsage();
            return 2;
        }
    }

    private static bool HasHelp(string[] args)
    {
        return HasSwitch(args, "--help") || HasSwitch(args, "-h");
    }

    private static bool HasSwitch(string[] args, string name)
    {
        foreach (string arg in args)
        {
            if (string.Equals(arg, name, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static void PrintUsage()
    {
        System.Console.WriteLine("TensorRtExec");
        System.Console.WriteLine("Usage:");
        System.Console.WriteLine("  TensorRtExec --onnx model.onnx --saveEngine model.plan --minShapes input:1x3x640x640 --optShapes input:1x3x640x640 --maxShapes input:4x3x640x640 --buildOnly");
        System.Console.WriteLine("  TensorRtExec --ui");
        System.Console.WriteLine("  TensorRtExec --help-json");
        System.Console.WriteLine("Input options:");
        System.Console.WriteLine("  --onnx|--model|--onnxFile <path> --saveEngine|--save-engine|--plan|--engineFile <path> --loadEngine|--load-engine <path>");
        System.Console.WriteLine("  --minShapes/--optShapes/--maxShapes input:1x3x640x640[,other:...]");
        System.Console.WriteLine("  --shapes|--inputShapes input:1x3x640x640[,other:...] --batch <n>");
        System.Console.WriteLine("Build options:");
        System.Console.WriteLine("  --tensor-rt-line <8|10|11>");
        System.Console.WriteLine("  --fp16 --int8 --bf16 --fp8 --best --noTF32 --workspace <MiB>");
        System.Console.WriteLine("  --builderOptimizationLevel <0..5> --maxAuxStreams <n>");
        System.Console.WriteLine("  --maxNbTactics <n> --tilingOptimizationLevel <none|fast|moderate|full> --l2LimitForTiling <bytes|MiB> --quantizationFlags <none|calibrateBeforeFusion>");
        System.Console.WriteLine("  --minTiming <n> --avgTiming <n> --precisionConstraints <none|prefer|obey>");
        System.Console.WriteLine("  --layerPrecisions <spec> --layerOutputTypes <spec>");
        System.Console.WriteLine("  --versionCompatible --excludeLeanRuntime --stripWeights --refit");
        System.Console.WriteLine("  --refitFromOnnx <path> (requires --onnx --stripWeights --refit; TRT10/11)");
        System.Console.WriteLine("  --saveRefittedEngine <path> (requires --refitFromOnnx; persists, disposes, reloads)");
        System.Console.WriteLine("  --allowWeightStreaming --weightStreamingBudget <-2|-1|0..100%|bytes> (requires --stronglyTyped when building)");
        System.Console.WriteLine("  --dumpRefit --markDebug <names> --dumpDebugTensors");
        System.Console.WriteLine("  --safe --consistency --builderCache|--noBuilderCache");
        System.Console.WriteLine("Runtime options:");
        System.Console.WriteLine("  --plugins|--plugin|--dynamicPlugins|--setPluginsToSerialize <dll1;dll2> --timingCacheFile|--timingCache <path>");
        System.Console.WriteLine("  --profilingVerbosity <none|layer_names_only|detailed> --verbose");
        System.Console.WriteLine("  --buildOnly --skipInference --dryRun|--previewOnly");
        System.Console.WriteLine("  --iterations <n> --warmUp <ms> --duration <sec> --streams <n> --infStreams <n> --useCudaGraph");
        System.Console.WriteLine("  --noDataTransfers --useSpinWait --threads --avgRuns <n> --percentile <0..100> --sleepTime <ms> --idleTime <ms>");
        System.Console.WriteLine("  --loadInputs input:file[,other:file] --dumpOutput --dumpRawBindingsToFile <path>");
        System.Console.WriteLine("  --referenceOutputs output:reference.json[,other:reference.json] --referenceAbsTolerance <n> --referenceRelTolerance <n>");
        System.Console.WriteLine("  --referenceNaNPolicy <reject|equal> --referenceInfinityPolicy <exact|reject>");
        System.Console.WriteLine("  --exportOutput <path> --exportTimes <path> --exportProfile <path> --saveProfile <path> --exportTimingCache <path>");
        System.Console.WriteLine("Deployment options:");
        System.Console.WriteLine("  --device <ordinal>");
        System.Console.WriteLine("  --useDLACore <n> --allowGPUFallback --tacticSources <list> --memPoolSize workspace:512,tacticDram:1024");
        System.Console.WriteLine("  --inputIOFormats <fmt> --outputIOFormats <fmt> --calib <cache> --directIO --sparsity <mode> --stronglyTyped");
        System.Console.WriteLine("Report options:");
        System.Console.WriteLine("  --dumpLayerInfo --exportLayerInfo <path> --dumpProfile --separateProfileRun");
        System.Console.WriteLine("  --exportReport|--report <path.json|path.md>");
        System.Console.WriteLine("  --help-json|--capabilities-json (machine-readable option capability surface; not runtime proof)");
        System.Console.WriteLine("Evidence options:");
        System.Console.WriteLine("  --evidenceSidecar <evidence.json>");
    }
}
