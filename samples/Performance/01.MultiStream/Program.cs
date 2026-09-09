using System;
using System.Linq;
using JYPPX.CudaSharp;

namespace MultiStreamSample;

internal static class Program
{
    private const int ByteCount = 4096;

    private static int Main(string[] args)
    {
        if (args.Any(static argument => string.Equals(argument, "--help", StringComparison.OrdinalIgnoreCase) || string.Equals(argument, "-h", StringComparison.OrdinalIgnoreCase)))
        {
            Console.WriteLine("MultiStream sample");
            Console.WriteLine("Usage:");
            Console.WriteLine("  dotnet run --project samples/Performance/01.MultiStream");
            Console.WriteLine("Options:");
            Console.WriteLine("  --help, -h                 Show this offline help.");
            return 0;
        }

        CudaEnvironmentSnapshot snapshot;
        try
        {
            snapshot = CudaEnvironmentProbe.GetCurrent();
        }
        catch (CudaException exception)
        {
            Console.WriteLine($"MultiStream=Skipped Reason={exception.Message}");
            return 0;
        }

        Console.WriteLine($"Bridge={snapshot.BuildInfo.BridgeName} CUDA Toolkit={snapshot.BuildInfo.CudaToolkitVersion} DeviceCount={snapshot.CudaRuntimeInfo.DeviceCount}");

        if (!snapshot.CudaRuntimeInfo.VendorDependencyAvailable)
        {
            Console.WriteLine($"MultiStream=Skipped Reason={snapshot.CudaRuntimeInfo.StatusMessage}");
            return 0;
        }

        using CudaStream streamA = new CudaStream(CudaStreamCreationFlags.NonBlocking);
        using CudaStream streamB = new CudaStream(CudaStreamCreationFlags.NonBlocking);
        using CudaMemory deviceA = new CudaMemory(ByteCount);
        using CudaMemory deviceB = new CudaMemory(ByteCount);
        using CudaPinnedMemory hostA = new CudaPinnedMemory(ByteCount);
        using CudaPinnedMemory hostB = new CudaPinnedMemory(ByteCount);
        using CudaEvent eventA = new CudaEvent();
        using CudaEvent eventB = new CudaEvent();

        deviceA.FillAsync(0x11, ByteCount, streamA);
        deviceB.FillAsync(0x22, ByteCount, streamB);
        deviceA.CopyToAsync(hostA, ByteCount, streamA);
        deviceB.CopyToAsync(hostB, ByteCount, streamB);
        eventA.Record(streamA);
        eventB.Record(streamB);
        eventA.Synchronize();
        eventB.Synchronize();

        bool streamAOk = hostA.ToArray(ByteCount).All(static value => value == 0x11);
        bool streamBOk = hostB.ToArray(ByteCount).All(static value => value == 0x22);
        Console.WriteLine($"IndependentStreams={streamAOk && streamBOk} A={streamAOk} B={streamBOk} Bytes={ByteCount}");

        using CudaPinnedMemory orderedHost = new CudaPinnedMemory(ByteCount);
        using CudaEvent orderingEvent = new CudaEvent();

        deviceA.FillAsync(0x33, ByteCount, streamA);
        orderingEvent.Record(streamA);
        streamB.WaitFor(orderingEvent);
        deviceA.CopyToAsync(orderedHost, ByteCount, streamB);
        streamB.Synchronize();

        bool crossStreamWaitOk = orderedHost.ToArray(ByteCount).All(static value => value == 0x33);
        Console.WriteLine($"CrossStreamWait={crossStreamWaitOk} ProducerStream={streamA.Flags} ConsumerStream={streamB.Flags}");
        Console.WriteLine($"StreamIds A={TryGetStreamId(streamA)} B={TryGetStreamId(streamB)}");
        Console.WriteLine($"MultiStream Passed={streamAOk && streamBOk && crossStreamWaitOk}");

        return 0;
    }

    private static string TryGetStreamId(CudaStream stream)
    {
        try
        {
            return stream.Id.ToString();
        }
        catch (CudaException exception)
        {
            return $"unavailable:{exception.Message}";
        }
    }
}
