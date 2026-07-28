namespace JYPPX.CudaSharp;

using JYPPX.CudaSharp.Internal.Interop;

/// <summary>
/// Minimum CUDA device properties used by <c>cudaChooseDevice</c>.
/// <c>cudaChooseDevice</c> 使用的最小 CUDA 设备约束。
/// </summary>
/// <remarks>
/// Unspecified numeric members should remain zero. CUDA treats the structure as a set of desired minimums.
/// 未指定的数值成员应保持为零；CUDA 会把该结构解释为期望的最小能力集合。
/// </remarks>
public sealed class CudaDeviceSelectionRequirements
{
    /// <summary>
    /// Gets or sets the minimum CUDA compute capability major version.
    /// 获取或设置最低 CUDA compute capability major 版本。
    /// </summary>
    public int Major { get; set; }

    /// <summary>
    /// Gets or sets the minimum CUDA compute capability minor version.
    /// 获取或设置最低 CUDA compute capability minor 版本。
    /// </summary>
    public int Minor { get; set; }

    /// <summary>
    /// Gets or sets the desired minimum multiprocessor count.
    /// 获取或设置期望的最低 multiprocessor 数量。
    /// </summary>
    public int MultiProcessorCount { get; set; }

    /// <summary>
    /// Gets or sets the desired warp size. Leave zero when not constrained.
    /// 获取或设置期望 warp size；不约束时保持为零。
    /// </summary>
    public int WarpSize { get; set; }

    /// <summary>
    /// Gets or sets the desired minimum threads per block.
    /// 获取或设置期望的最低 block 线程数。
    /// </summary>
    public int MaxThreadsPerBlock { get; set; }

    /// <summary>
    /// Gets or sets whether host memory mapping support is required.
    /// 获取或设置是否要求支持 host memory mapping。
    /// </summary>
    public bool RequireHostMemoryMapping { get; set; }

    /// <summary>
    /// Gets or sets whether an integrated GPU is required.
    /// 获取或设置是否要求 integrated GPU。
    /// </summary>
    public bool RequireIntegratedGpu { get; set; }

    /// <summary>
    /// Gets or sets the desired minimum total global memory in bytes.
    /// 获取或设置期望的最低 global memory 字节数。
    /// </summary>
    public ulong TotalGlobalMemory { get; set; }

    internal NativeCudaDeviceSelectionRequirements ToNative()
    {
        return new NativeCudaDeviceSelectionRequirements
        {
            Major = Major,
            Minor = Minor,
            MultiProcessorCount = MultiProcessorCount,
            WarpSize = WarpSize,
            MaxThreadsPerBlock = MaxThreadsPerBlock,
            CanMapHostMemory = RequireHostMemoryMapping ? 1 : 0,
            Integrated = RequireIntegratedGpu ? 1 : 0,
            TotalGlobalMemory = TotalGlobalMemory
        };
    }
}
