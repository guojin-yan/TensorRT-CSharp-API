using System;
using YoloVisionSample;

internal static class Program
{
    public static int Main(string[] args)
    {
        Console.WriteLine($"YoloVisionPackageConsumer ProjectReference=False CoreAssembly={typeof(YoloModelProfile).Assembly.GetName().Name}");
        return YoloVisionCommand.Run(args);
    }
}
