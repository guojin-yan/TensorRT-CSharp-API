using System;
using System.Diagnostics;
using JYPPX.TensorRtSharp;
using YoloVisionSample;

internal static class Program
{
    public static int Main()
    {
        YoloOutputLayout parsedLayout = YoloOutputLayoutInference.Parse("channels-first");
        YoloOutputLayout inferredLayout = YoloOutputLayoutInference.InferRank3(
            new[] { 1, 84, 8400 },
            YoloOutputLayout.Auto);
        YoloPreprocessOptions preprocess = YoloPreprocessOptions.Default;
        YoloPostprocessOptions postprocess = YoloPostprocessOptions.Default;
        int capabilityCount = YoloCapabilityMatrix.Entries.Count;
        bool nativeRuntimeLoaded = IsNativeRuntimeLoaded();

        bool passed =
            parsedLayout == YoloOutputLayout.ChannelsFirst &&
            inferredLayout == YoloOutputLayout.ChannelsFirst &&
            string.Equals(preprocess.TensorLayout, "NCHW", StringComparison.Ordinal) &&
            string.Equals(preprocess.ColorOrder, "RGB", StringComparison.Ordinal) &&
            postprocess.Layout == YoloOutputLayout.Auto &&
            capabilityCount == 60 &&
            !nativeRuntimeLoaded;

        Console.WriteLine(
            $"YoloVisionManagedPackageConsumer ManagedAssembly={typeof(TensorRtEnvironmentSnapshot).Assembly.GetName().Name} " +
            $"YoloAssembly={typeof(YoloCapabilityMatrix).Assembly.GetName().Name} CapabilityCount={capabilityCount}");
        Console.WriteLine($"YoloVisionManagedPackageConsumer Passed={passed} PackageReferenceOnly=True NativeRuntimeLoaded={nativeRuntimeLoaded}");
        return passed ? 0 : 1;
    }

    private static bool IsNativeRuntimeLoaded()
    {
        string[] markers = ["jyppxtrtbridge", "nvinfer", "cudart", "cudnn", "nvrtc"];
        foreach (ProcessModule module in Process.GetCurrentProcess().Modules)
        {
            string moduleName = module.ModuleName.ToLowerInvariant();
            foreach (string marker in markers)
            {
                if (moduleName.Contains(marker, StringComparison.Ordinal))
                {
                    return true;
                }
            }
        }

        return false;
    }
}
