using System.IO;
using System.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RuntimeSerializationStreamIoTests
{
    [Fact]
    public void ManagedStreamSerializationApisCopyThroughManagedBuffers()
    {
        string hostMemory = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtHostMemory.cs");
        string hostMemoryMetadata = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtHostMemory.Trt11Metadata.cs");
        string runtime = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRuntime.cs");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.cs");
        string hostMemoryInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11FourteenthBatch.cs");

        Assert.Contains("public void CopyTo(Stream destination)", hostMemory);
        Assert.Contains("byte[] buffer = ToArray();", hostMemory);
        Assert.Contains("destination.Write(buffer, 0, buffer.Length);", hostMemory);
        Assert.Contains("public MemoryStream OpenRead()", hostMemory);
        Assert.Contains("return new MemoryStream(ToArray(), writable: false);", hostMemory);
        Assert.Contains("public TensorRtDataType DataType", hostMemoryMetadata);
        Assert.Contains("TensorRT 8, TensorRT 10, and TensorRT 11", hostMemoryMetadata);
        Assert.Contains("不暴露 host-memory 指针", hostMemoryMetadata);
        Assert.Contains("NativeBridgeApi.GetHostMemoryDataType(Line, _handle)", hostMemoryMetadata);

        Assert.Contains("public TensorRtEngine Deserialize(ArraySegment<byte> serializedEngine)", runtime);
        Assert.Contains("Buffer.BlockCopy(serializedEngine.Array, serializedEngine.Offset, buffer, 0, serializedEngine.Count);", runtime);
        Assert.Contains("public TensorRtEngine Deserialize(ReadOnlySpan<byte> serializedEngine)", runtime);
        Assert.Contains("return Deserialize(serializedEngine.ToArray());", runtime);
        Assert.Contains("public TensorRtEngine Deserialize(Stream serializedEngineStream)", runtime);
        Assert.Contains("serializedEngineStream.CopyTo(copy);", runtime);
        Assert.Contains("return Deserialize(copy.ToArray());", runtime);
        Assert.Contains("GCHandle pinned = GCHandle.Alloc(engineData, GCHandleType.Pinned);", interop);
        Assert.Contains("pinned.Free();", interop);
        Assert.Contains("public static TensorRtDataType GetHostMemoryDataType", hostMemoryInterop);
        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_host_memory_get_type", hostMemoryInterop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_host_memory_get_type", hostMemoryInterop);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_host_memory_get_type", hostMemoryInterop);

        Assert.DoesNotContain("public IntPtr", hostMemory + hostMemoryMetadata + runtime);
        Assert.DoesNotContain("public nint", hostMemory + hostMemoryMetadata + runtime);
    }

    [Fact]
    public void OnnxSmokeCoversManagedStreamRoundTrip()
    {
        string program = ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs");

        Assert.Contains("hostMemory.CopyTo(copiedEngineStream);", program);
        Assert.Contains("streamEngineBytes = copiedEngineStream.ToArray();", program);
        Assert.Contains("engineBytes.SequenceEqual(streamEngineBytes)", program);
        Assert.Contains("using TensorRtEngine engine = runtime.Deserialize(copiedEngineStream);", program);
        Assert.Contains("StreamRoundTrip=True", program);
    }

    [Fact]
    public void SmokeOutputsHostMemoryTypeAndSerializationConfigCopiedEvidence()
    {
        string tensorRtSmoke = ReadSource("smoke", "TensorRtSmokeRunner", "Program.cs");
        string onnxSmoke = ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs");

        Assert.Contains("HostMemory={hostMemory.SizeInBytes}/{hostMemory.DataType}", tensorRtSmoke);
        Assert.Contains("EngineSerialized={serializedEngine.SizeInBytes}/{serializedEngine.DataType}", tensorRtSmoke);
        Assert.Contains("ConfigPlan={configuredPlan.SizeInBytes}/{configuredPlan.DataType}", tensorRtSmoke);
        Assert.Contains("Serialized={serializedDefault.SizeInBytes}/{serializedDefault.DataType}->{serializedWithConfig.SizeInBytes}/{serializedWithConfig.DataType}", tensorRtSmoke);
        Assert.Contains("SerializationConfigSummary=", tensorRtSmoke);
        Assert.Contains("RuntimeConfigSummary=", tensorRtSmoke);
        Assert.Contains("DependencyProbeOnly", tensorRtSmoke);
        Assert.Contains("PrintDependencyProbe(probeLine);", tensorRtSmoke);
        Assert.Contains("Skipped=True Reason=DependencyProbeOnly", tensorRtSmoke);

        Assert.Contains("HostMemory={hostMemory.SizeInBytes}/{hostMemory.DataType}", onnxSmoke);
        Assert.Contains("TimingCacheBytes={serializedTimingCache.SizeInBytes}/{serializedTimingCache.DataType}", onnxSmoke);
        Assert.Contains("EngineFileRoundTrip=True", onnxSmoke);
        Assert.Contains("StreamRoundTrip=True", onnxSmoke);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
