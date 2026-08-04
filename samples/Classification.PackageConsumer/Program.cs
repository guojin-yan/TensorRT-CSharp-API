using System;
using ClassificationSample;
using JYPPX.TensorRtSharp;

internal static class Program
{
    public static int Main(string[] args)
    {
        Console.WriteLine($"ClassificationPackageConsumer ProjectReference=False CoreAssembly={typeof(TensorRtEnvironmentProbe).Assembly.GetName().Name}");
        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        Console.WriteLine($"ClassificationPackageConsumer BridgeTensorRt={snapshot.BuildInfo.TensorRtVersion} BridgeCuda={snapshot.BuildInfo.CudaToolkitVersion}");
        return ClassificationCommand.Run(args);
    }
}
