namespace JYPPX.CudaSharp;

/// <summary>
/// Describes one CUDA device as reported by the bridge.
/// </summary>
public sealed class CudaDeviceInfo
{
    public CudaDeviceInfo(
        int ordinal,
        string name,
        int major,
        int minor,
        int multiProcessorCount,
        int warpSize,
        int maxThreadsPerBlock,
        bool canMapHostMemory,
        bool integrated,
        ulong totalGlobalMemory)
    {
        Ordinal = ordinal;
        Name = name;
        Major = major;
        Minor = minor;
        MultiProcessorCount = multiProcessorCount;
        WarpSize = warpSize;
        MaxThreadsPerBlock = maxThreadsPerBlock;
        CanMapHostMemory = canMapHostMemory;
        Integrated = integrated;
        TotalGlobalMemory = totalGlobalMemory;
    }

    public int Ordinal { get; }
    public string Name { get; }
    public int Major { get; }
    public int Minor { get; }
    public int MultiProcessorCount { get; }
    public int WarpSize { get; }
    public int MaxThreadsPerBlock { get; }
    public bool CanMapHostMemory { get; }
    public bool Integrated { get; }
    public ulong TotalGlobalMemory { get; }
}

