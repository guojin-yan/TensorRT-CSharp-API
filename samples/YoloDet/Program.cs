using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using JYPPX.SampleSupport;

namespace YoloDetSample;

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
            OnnxSampleOptions options = OnnxSampleOptions.FromArgs(args, "1x3x640x640");
            IReadOnlyList<string> labels = TensorRtOnnxSample.ReadLabels(SampleCommandLine.GetStringArgument(args, "--labels", string.Empty));
            float confidence = ParseFloat(SampleCommandLine.GetStringArgument(args, "--confidence", "0.25"), 0.25f);
            int topK = Math.Max(1, SampleCommandLine.GetPositiveIntArgument(args, "--top-k", 10));
            string layout = SampleCommandLine.GetStringArgument(args, "--layout", "auto");
            string objectnessMode = SampleCommandLine.GetStringArgument(args, "--has-objectness", "auto");

            OnnxSampleResult result = TensorRtOnnxSample.RunSingleFloatInputOutput(options);
            Console.WriteLine($"YoloDet TensorRtLine={(int)result.Line} Model={options.ModelPath}");
            Console.WriteLine($"Input={result.InputName}:{result.InputShape} Output={result.OutputName}:{result.OutputShape}");
            Console.WriteLine($"ProfileIndex={result.ProfileIndex} EngineDeviceMemory={result.EngineDeviceMemoryBytes}");
            Console.WriteLine($"Execution {result.ExecutionSummary} ElapsedMs={result.ElapsedMilliseconds:0.###}");

            IReadOnlyList<DetectionCandidate> detections = DecodeDetections(result.OutputValues, result.OutputShape, labels.Count, layout, objectnessMode, confidence, topK);
            if (detections.Count == 0)
            {
                Console.WriteLine($"Detections=0 Confidence={confidence:0.###} Note=The sample uses synthetic input; real images normally require preprocessing before meaningful boxes appear.");
            }
            else
            {
                foreach (DetectionCandidate detection in detections)
                {
                    Console.WriteLine(
                        $"Detection Class={TensorRtOnnxSample.LabelOrIndex(labels, detection.ClassIndex)} Score={detection.Score:0.######} " +
                        $"BoxCxCyWh={detection.CenterX:0.###},{detection.CenterY:0.###},{detection.Width:0.###},{detection.Height:0.###}");
                }
            }

            Console.WriteLine("YoloDet Passed=True");
            return 0;
        }
        catch (SampleSkippedException exception)
        {
            Console.WriteLine($"YoloDet=Skipped Reason={exception.Message}");
            return 0;
        }
        catch (Exception exception) when (TensorRtSampleSupport.IsDeploymentException(exception))
        {
            Console.WriteLine($"YoloDet=Skipped Reason={exception.Message}");
            return 0;
        }
        catch (ArgumentException exception)
        {
            Console.WriteLine($"YoloDet=InvalidArguments Reason={exception.Message}");
            PrintUsage();
            return 2;
        }
        catch (FileNotFoundException exception)
        {
            Console.WriteLine($"YoloDet=InvalidArguments Reason={exception.Message}");
            PrintUsage();
            return 2;
        }
        catch (NotSupportedException exception)
        {
            Console.WriteLine($"YoloDet=UnsupportedOutput Reason={exception.Message}");
            return 2;
        }
    }

    private static IReadOnlyList<DetectionCandidate> DecodeDetections(
        float[] values,
        JYPPX.TensorRtSharp.TensorRtDims shape,
        int labelCount,
        string layout,
        string objectnessMode,
        float confidence,
        int topK)
    {
        int[] dims = shape.Values;
        if (dims.Length != 3 || dims[0] != 1)
        {
            throw new NotSupportedException($"Expected a rank-3 YOLO output such as [1, 84, 8400] or [1, 8400, 84], got {shape}.");
        }

        bool channelsFirst = ResolveChannelsFirst(dims, layout);
        int channelCount = channelsFirst ? dims[1] : dims[2];
        int boxCount = channelsFirst ? dims[2] : dims[1];
        bool hasObjectness = ResolveHasObjectness(channelCount, labelCount, objectnessMode);
        int classOffset = hasObjectness ? 5 : 4;
        int classCount = channelCount - classOffset;
        if (classCount <= 0)
        {
            throw new NotSupportedException($"YOLO output channel count {channelCount} does not leave room for class scores.");
        }

        List<DetectionCandidate> candidates = new List<DetectionCandidate>();
        for (int box = 0; box < boxCount; box++)
        {
            float objectness = hasObjectness ? Read(values, channelsFirst, channelCount, boxCount, box, 4) : 1.0f;
            int bestClass = 0;
            float bestClassScore = float.NegativeInfinity;
            for (int classIndex = 0; classIndex < classCount; classIndex++)
            {
                float classScore = Read(values, channelsFirst, channelCount, boxCount, box, classOffset + classIndex);
                if (classScore > bestClassScore)
                {
                    bestClassScore = classScore;
                    bestClass = classIndex;
                }
            }

            float score = objectness * bestClassScore;
            if (score >= confidence)
            {
                candidates.Add(new DetectionCandidate(
                    bestClass,
                    score,
                    Read(values, channelsFirst, channelCount, boxCount, box, 0),
                    Read(values, channelsFirst, channelCount, boxCount, box, 1),
                    Read(values, channelsFirst, channelCount, boxCount, box, 2),
                    Read(values, channelsFirst, channelCount, boxCount, box, 3)));
            }
        }

        return candidates
            .OrderByDescending(static item => item.Score)
            .Take(topK)
            .ToArray();
    }

    private static bool ResolveChannelsFirst(int[] dims, string layout)
    {
        if (string.Equals(layout, "channels-first", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (string.Equals(layout, "boxes-first", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        return dims[2] > dims[1];
    }

    private static bool ResolveHasObjectness(int channelCount, int labelCount, string objectnessMode)
    {
        if (string.Equals(objectnessMode, "true", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (string.Equals(objectnessMode, "false", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (labelCount > 0)
        {
            if (channelCount == labelCount + 5)
            {
                return true;
            }

            if (channelCount == labelCount + 4)
            {
                return false;
            }
        }

        return channelCount == 85;
    }

    private static float Read(float[] values, bool channelsFirst, int channelCount, int boxCount, int box, int channel)
    {
        int index = channelsFirst
            ? channel * boxCount + box
            : box * channelCount + channel;
        return values[index];
    }

    private static float ParseFloat(string value, float defaultValue)
    {
        return float.TryParse(value, out float parsed) ? parsed : defaultValue;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("YoloDet sample");
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project samples/YoloDet -- --model yolo.onnx --labels labels.txt --input-shape 1x3x640x640 --tensor-rt-line 10");
        Console.WriteLine("Options:");
        Console.WriteLine("  --layout auto|channels-first|boxes-first");
        Console.WriteLine("  --has-objectness auto|true|false");
        Console.WriteLine("  --confidence <value>      Default: 0.25");
        Console.WriteLine("  --top-k <count>           Default: 10");
        Console.WriteLine("  --input-pattern <pattern> zeros, ones, or ramp. Default: ramp.");
    }

    private readonly struct DetectionCandidate
    {
        public DetectionCandidate(int classIndex, float score, float centerX, float centerY, float width, float height)
        {
            ClassIndex = classIndex;
            Score = score;
            CenterX = centerX;
            CenterY = centerY;
            Width = width;
            Height = height;
        }

        public int ClassIndex { get; }

        public float Score { get; }

        public float CenterX { get; }

        public float CenterY { get; }

        public float Width { get; }

        public float Height { get; }
    }
}
