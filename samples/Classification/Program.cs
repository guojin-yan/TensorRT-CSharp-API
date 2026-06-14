using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.SampleSupport;

namespace ClassificationSample;

internal static class Program
{
    public static int Main(string[] args)
    {
        if (SampleCommandLine.HasSwitch(args, "--help"))
        {
            PrintUsage();
            return 0;
        }

        try
        {
            OnnxSampleOptions options = OnnxSampleOptions.FromArgs(args, "1x3x224x224");
            IReadOnlyList<string> labels = TensorRtOnnxSample.ReadLabels(SampleCommandLine.GetStringArgument(args, "--labels", string.Empty));
            int topK = Math.Max(1, SampleCommandLine.GetPositiveIntArgument(args, "--top-k", 5));
            OnnxSampleResult result = TensorRtOnnxSample.RunSingleFloatInputOutput(options);

            Console.WriteLine($"Classification TensorRtLine={(int)result.Line} Model={options.ModelPath}");
            Console.WriteLine($"Input={result.InputName}:{result.InputShape} Output={result.OutputName}:{result.OutputShape}");
            Console.WriteLine($"ProfileIndex={result.ProfileIndex} EngineDeviceMemory={result.EngineDeviceMemoryBytes}");
            Console.WriteLine($"Execution {result.ExecutionSummary} ElapsedMs={result.ElapsedMilliseconds:0.###}");

            foreach ((int index, float score) in GetTopK(result.OutputValues, topK))
            {
                Console.WriteLine($"TopK Index={index} Label={TensorRtOnnxSample.LabelOrIndex(labels, index)} Score={score:0.######}");
            }

            Console.WriteLine("Classification Passed=True");
            return 0;
        }
        catch (SampleSkippedException exception)
        {
            Console.WriteLine($"Classification=Skipped Reason={exception.Message}");
            return 0;
        }
        catch (Exception exception) when (TensorRtOnnxSample.IsDeploymentException(exception))
        {
            Console.WriteLine($"Classification=Skipped Reason={exception.Message}");
            return 0;
        }
        catch (ArgumentException exception)
        {
            Console.WriteLine($"Classification=InvalidArguments Reason={exception.Message}");
            PrintUsage();
            return 2;
        }
    }

    private static IEnumerable<(int Index, float Score)> GetTopK(float[] values, int topK)
    {
        return values
            .Select(static (score, index) => (Index: index, Score: score))
            .OrderByDescending(static item => item.Score)
            .Take(topK);
    }

    private static void PrintUsage()
    {
        Console.WriteLine("Classification sample");
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project samples/Classification -- --model model.onnx --labels labels.txt --input-shape 1x3x224x224 --tensor-rt-line 10");
        Console.WriteLine("Options:");
        Console.WriteLine("  --input-name <name>       Optional input tensor name; defaults to the model's only input.");
        Console.WriteLine("  --output-name <name>      Optional output tensor name; defaults to the first output.");
        Console.WriteLine("  --min-shape <dims>        Optional dynamic profile minimum, for example 1x3x224x224.");
        Console.WriteLine("  --opt-shape <dims>        Optional dynamic profile optimum.");
        Console.WriteLine("  --max-shape <dims>        Optional dynamic profile maximum.");
        Console.WriteLine("  --input-pattern <pattern> zeros, ones, or ramp. Default: ramp.");
    }
}
