using System;
using JYPPX.TensorRtSharp;
using YoloVisionSample;

internal static class Program
{
    public static int Main(string[] args)
    {
        Console.WriteLine($"YoloVisionPackageConsumer ProjectReference=False CoreAssembly={typeof(YoloModelProfile).Assembly.GetName().Name}");
        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        Console.WriteLine($"YoloVisionPackageConsumer BridgeTensorRt={snapshot.BuildInfo.TensorRtVersion} BridgeCuda={snapshot.BuildInfo.CudaToolkitVersion}");
        return YoloVisionCommand.Run(args);
    }
}
