using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using JYPPX.SampleSupport;

namespace OnnxBuildAndRunSample;

internal static class Program
{
    private const string DefaultInputShape = "1x4";

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
    }

    private static int Run(string[] args)
    {
        bool synthetic = SampleCommandLine.HasSwitch(args, "--synthetic");
        string requestedModel = SampleCommandLine.GetStringArgument(args, "--model", string.Empty);
        if (synthetic && !string.IsNullOrWhiteSpace(requestedModel))
        {
            throw new ArgumentException("Use either --synthetic or --model, not both.");
        }

        if (!synthetic && string.IsNullOrWhiteSpace(requestedModel))
        {
            throw new ArgumentException("Missing --model <path-to-model.onnx>. Use --synthetic for the deterministic Identity smoke.");
        }

        string modelPath = synthetic ? SyntheticIdentityOnnxModel.WriteToTemporaryDirectory() : requestedModel;
        string[] effectiveArgs = AddOrReplaceArgument(args, "--model", modelPath);
        if (synthetic)
        {
            effectiveArgs = AddOrReplaceArgument(effectiveArgs, "--input-shape", DefaultInputShape);
        }

        OnnxSampleOptions options = OnnxSampleOptions.FromArgs(effectiveArgs, DefaultInputShape);
        OnnxSampleResult result = TensorRtOnnxSample.RunSingleFloatInputOutput(options);
        bool identityOutputMatch = !synthetic || result.OutputValues.SequenceEqual(result.Inputs[0].Preview);
        string status = identityOutputMatch ? "passed" : "failed";

        object report = new
        {
            schemaVersion = "1.0",
            sample = "Inference/03.OnnxBuildAndRun",
            status,
            proofClassification = "synthetic-input-runtime",
            tensorRtLine = (int)result.Line,
            model = new
            {
                source = synthetic ? "generated-synthetic-identity" : "user-provided",
                path = options.ModelPath,
                sha256 = ComputeFileSha256(options.ModelPath)
            },
            input = new
            {
                name = result.InputName,
                shape = result.InputShape.Values,
                elementCount = result.Inputs[0].ElementCount,
                sha256 = result.Inputs[0].Sha256,
                preview = result.Inputs[0].Preview
            },
            output = new
            {
                name = result.OutputName,
                shape = result.OutputShape.Values,
                elementCount = result.OutputValues.Length,
                sha256 = ComputeFloatSha256(result.OutputValues),
                preview = result.OutputValues.Take(8).ToArray(),
                identityOutputMatch
            },
            engine = new
            {
                profileIndex = result.ProfileIndex,
                deviceMemoryBytes = result.EngineDeviceMemoryBytes,
                inputCount = result.Report.GetInputs().Count,
                outputCount = result.Report.GetOutputs().Count
            },
            execution = new
            {
                enqueueCount = 1,
                elapsedMilliseconds = result.ElapsedMilliseconds,
                summary = result.ExecutionSummary.ToString()
            }
        };

        WriteReport(args, report);
        Console.WriteLine("OnnxBuildAndRun Passed=" + identityOutputMatch);
        return identityOutputMatch ? 0 : 1;
    }

    private static string[] AddOrReplaceArgument(string[] args, string name, string value)
    {
        List<string> updated = new List<string>(args.Length + 2);
        bool replaced = false;
        for (int index = 0; index < args.Length; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                updated.Add(name);
                updated.Add(value);
                replaced = true;
                index++;
                continue;
            }

            updated.Add(args[index]);
        }

        if (!replaced)
        {
            updated.Add(name);
            updated.Add(value);
        }

        return updated.ToArray();
    }

    private static void WriteStatusReport(string[] args, string status, string reason)
    {
        WriteReport(args, new
        {
            schemaVersion = "1.0",
            sample = "Inference/03.OnnxBuildAndRun",
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

    private static string ComputeFileSha256(string path)
    {
        return Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    }

    private static string ComputeFloatSha256(float[] values)
    {
        byte[] bytes = new byte[checked(values.Length * sizeof(float))];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    private static void PrintUsage()
    {
        Console.WriteLine("OnnxBuildAndRun sample");
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project samples/Inference/03.OnnxBuildAndRun -- --synthetic --tensor-rt-line 10");
        Console.WriteLine("  dotnet run --project samples/Inference/03.OnnxBuildAndRun -- --model <model.onnx> --input-shape <dims>");
        Console.WriteLine("Options:");
        Console.WriteLine("  --synthetic                 Generate and run a deterministic 1x4 Identity ONNX model.");
        Console.WriteLine("  --model <path>              ONNX model path. Mutually exclusive with --synthetic.");
        Console.WriteLine("  --tensor-rt-line <8|10|11>  TensorRT adapter line. Default: 10.");
        Console.WriteLine("  --input-shape <dims>        Concrete input shape. Default: 1x4.");
        Console.WriteLine("  --input-name <name>         Input tensor name when the model cannot be inferred unambiguously.");
        Console.WriteLine("  --output-name <name>        Output tensor name. Default: the first output.");
        Console.WriteLine("  --input-pattern <pattern>   zeros, ones, or ramp. Default: ramp.");
        Console.WriteLine("  --output-json <path>        Also write the structured report to a JSON file.");
        Console.WriteLine("  --help, -h                  Show this offline help.");
    }
}
