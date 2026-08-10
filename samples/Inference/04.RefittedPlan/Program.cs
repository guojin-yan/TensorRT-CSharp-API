using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using JYPPX.CudaSharp;
using JYPPX.SampleSupport;
using JYPPX.TensorRtSharp;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace RefittedPlanSample;

internal static class Program
{
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
        catch (Exception exception) when (exception is ArgumentException || exception is FileNotFoundException || exception is InvalidDataException || exception is JsonException)
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
            throw new ArgumentException("ONNX parser-refitter requires TensorRT 10 or TensorRT 11.");
        }

        bool synthetic = SampleCommandLine.HasSwitch(args, "--synthetic");
        string baselineModelPath = SampleCommandLine.GetStringArgument(args, "--baseline-model", string.Empty);
        string refitModelPath = SampleCommandLine.GetStringArgument(args, "--refit-model", string.Empty);
        if (synthetic && (!string.IsNullOrWhiteSpace(baselineModelPath) || !string.IsNullOrWhiteSpace(refitModelPath)))
        {
            throw new ArgumentException("Use --synthetic or the --baseline-model/--refit-model pair, not both.");
        }

        if (synthetic)
        {
            SyntheticRefitOnnxModel generated = SyntheticRefitOnnxModel.WriteToTemporaryDirectory();
            baselineModelPath = generated.BaselinePath;
            refitModelPath = generated.RefitPath;
        }

        if (string.IsNullOrWhiteSpace(baselineModelPath) || string.IsNullOrWhiteSpace(refitModelPath))
        {
            throw new ArgumentException("Provide --synthetic or both --baseline-model and --refit-model.");
        }

        baselineModelPath = RequireFile(baselineModelPath, "Baseline ONNX model");
        refitModelPath = RequireFile(refitModelPath, "Refit ONNX model");
        string refittedPlanPath = ResolvePlanPath(args);
        string baselinePlanPath = Path.Combine(
            Path.GetDirectoryName(refittedPlanPath)!,
            Path.GetFileNameWithoutExtension(refittedPlanPath) + ".baseline" + Path.GetExtension(refittedPlanPath));

        TensorRtEnvironmentSnapshot environment = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = TensorRtSampleSupport.SelectAdapter(environment, line);
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            throw new SampleSkippedException(adapter.StatusMessage);
        }

        float[] before;
        float[] after;
        float[] afterReload;
        int allRefittableWeightCount;
        int missingWeightCountBefore;
        int missingWeightCountAfterModelLoad;
        bool engineRefittable;
        bool parserRefitAccepted;
        bool refitCommitted;

        using TensorRtLogger logger = new TensorRtLogger(line);
        BuildBaselinePlan(line, logger, baselineModelPath, baselinePlanPath);

        using (TensorRtRuntime runtime = new TensorRtRuntime(logger))
        using (TensorRtEngine engine = runtime.DeserializeFromFile(baselinePlanPath))
        {
            engineRefittable = engine.IsRefittable;
            if (!engineRefittable)
            {
                throw new InvalidOperationException("The baseline engine is not refittable. Ensure the graph contains refittable initializers.");
            }

            before = RunInference(engine, InputValues);
            using TensorRtRefitter refitter = engine.CreateRefitter(logger);
            allRefittableWeightCount = refitter.AllRefittableWeightCount;
            missingWeightCountBefore = refitter.MissingWeightCount;
            using TensorRtOnnxParserRefitter parserRefitter = refitter.CreateOnnxParserRefitter(logger);
            parserRefitAccepted = parserRefitter.RefitFromFile(refitModelPath);
            missingWeightCountAfterModelLoad = refitter.MissingWeightCount;
            refitCommitted = parserRefitAccepted && refitter.RefitCudaEngine();
            if (!refitCommitted)
            {
                throw new InvalidOperationException("TensorRT did not commit the ONNX refit operation.");
            }

            after = RunInference(engine, InputValues);
            using TensorRtHostMemory refittedPlan = engine.Serialize();
            refittedPlan.SaveToFile(refittedPlanPath);
        }

        using (TensorRtRuntime reloadRuntime = new TensorRtRuntime(logger))
        using (TensorRtEngine reloadedEngine = reloadRuntime.DeserializeFromFile(refittedPlanPath))
        {
            afterReload = RunInference(reloadedEngine, InputValues);
        }

        float[] expectedBefore = InputValues.ToArray();
        float[] expectedAfter = InputValues.Select(static value => value * 2.0f).ToArray();
        bool beforeMatch = AreClose(before, expectedBefore);
        bool afterMatch = AreClose(after, expectedAfter);
        bool reloadedMatch = AreClose(afterReload, expectedAfter);
        bool outputChanged = !AreClose(before, after);
        string status = beforeMatch && afterMatch && reloadedMatch && outputChanged ? "passed" : "failed";

        WriteReport(args, new
        {
            schemaVersion = "1.0",
            sample = "Inference/04.RefittedPlan",
            status,
            proofClassification = "synthetic-input-runtime",
            tensorRtLine = (int)line,
            environment = new
            {
                tensorRtVersion = environment.BuildInfo.TensorRtVersion,
                cudaToolkitVersion = environment.BuildInfo.CudaToolkitVersion
            },
            models = new
            {
                source = synthetic ? "generated-synthetic-scale" : "user-provided",
                baseline = DescribeFile(baselineModelPath),
                refit = DescribeFile(refitModelPath)
            },
            plans = new
            {
                baseline = DescribeFile(baselinePlanPath),
                refitted = DescribeFile(refittedPlanPath),
                deserializedFromDisk = true
            },
            refit = new
            {
                engineRefittable,
                allRefittableWeightCount,
                missingWeightCountBefore,
                parserRefitAccepted,
                missingWeightCountAfterModelLoad,
                refitCommitted
            },
            inference = new
            {
                input = InputValues,
                before,
                after,
                afterReload,
                beforeMatch,
                afterMatch,
                reloadedMatch,
                outputChanged
            }
        });

        Console.WriteLine("RefittedPlan Passed=" + (status == "passed"));
        return status == "passed" ? 0 : 1;
    }

    private static void BuildBaselinePlan(
        TensorRtApiLine line,
        TensorRtLogger logger,
        string modelPath,
        string planPath)
    {
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtOnnxParser parser = new TensorRtOnnxParser(logger, network);

        config.SetFlag(TensorRtBuilderFlag.Refit);
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetMaxAuxStreams(0);
        if (!parser.ParseFromFile(modelPath))
        {
            throw new InvalidDataException(parser.GetErrorSummary());
        }

        using TensorRtHostMemory baselinePlan = builder.BuildSerializedNetwork(network, config);
        baselinePlan.SaveToFile(planPath);
    }

    private static float[] RunInference(TensorRtEngine engine, float[] inputValues)
    {
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        using CudaStream stream = new CudaStream(CudaStreamCreationFlags.NonBlocking);
        using CudaMemory input = new CudaMemory(inputValues.Length * sizeof(float));
        using CudaMemory output = new CudaMemory(inputValues.Length * sizeof(float));
        input.CopyFrom(inputValues);
        output.Fill(0, output.SizeInBytes);
        context.SetTensorAddress("input", input);
        context.SetTensorAddress("output", output);
        context.EnqueueAsync(stream);
        stream.Synchronize();
        return output.ToSingleArray(inputValues.Length);
    }

    private static bool AreClose(IReadOnlyList<float> actual, IReadOnlyList<float> expected)
    {
        return actual.Count == expected.Count &&
            actual.Zip(expected, static (left, right) => Math.Abs(left - right) <= 0.0001f).All(static value => value);
    }

    private static string ResolvePlanPath(string[] args)
    {
        string requested = SampleCommandLine.GetStringArgument(args, "--plan", string.Empty);
        string path = string.IsNullOrWhiteSpace(requested)
            ? Path.Combine(Path.GetTempPath(), "jyppx-tensorrt-samples", "refitted-plan", "scale-1x4.refitted.engine")
            : Path.GetFullPath(requested);
        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (string.IsNullOrWhiteSpace(directory))
        {
            throw new ArgumentException("The plan path must have a parent directory.");
        }

        Directory.CreateDirectory(directory);
        return fullPath;
    }

    private static string RequireFile(string path, string label)
    {
        string fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException(label + " was not found.", fullPath);
        }

        return fullPath;
    }

    private static object DescribeFile(string path)
    {
        return new
        {
            path,
            lengthBytes = new FileInfo(path).Length,
            sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant()
        };
    }

    private static void WriteStatusReport(string[] args, string status, string reason)
    {
        WriteReport(args, new
        {
            schemaVersion = "1.0",
            sample = "Inference/04.RefittedPlan",
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
        Console.WriteLine("RefittedPlan sample");
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project samples/Inference/04.RefittedPlan -- --synthetic --tensor-rt-line 10");
        Console.WriteLine("  dotnet run --project samples/Inference/04.RefittedPlan -- --baseline-model <baseline.onnx> --refit-model <updated.onnx>");
        Console.WriteLine("Options:");
        Console.WriteLine("  --synthetic                   Generate two deterministic 1x4 Mul models with scale 1 and 2.");
        Console.WriteLine("  --baseline-model <path>       Baseline ONNX model used to build the refittable engine.");
        Console.WriteLine("  --refit-model <path>          Structurally matching ONNX model with updated initializers.");
        Console.WriteLine("  --tensor-rt-line <10|11>      TensorRT adapter line. Default: 10.");
        Console.WriteLine("  --plan <path>                 Persist the refitted engine at this path.");
        Console.WriteLine("  --output-json <path>          Also write the structured report to a JSON file.");
        Console.WriteLine("  --help, -h                    Show this offline help.");
    }

    private sealed class SampleSkippedException : Exception
    {
        public SampleSkippedException(string message)
            : base(message)
        {
        }
    }
}
