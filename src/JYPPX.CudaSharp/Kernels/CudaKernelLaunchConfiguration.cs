using System;

namespace JYPPX.CudaSharp;

/// <summary>Describes an owner-bound CUDA kernel launch. 描述 owner-bound CUDA kernel launch。</summary>
public readonly struct CudaKernelLaunchConfiguration
{
    /// <summary>Creates a launch configuration. 创建 launch 配置。</summary>
    public CudaKernelLaunchConfiguration(CudaDim3 gridDimensions, CudaDim3 blockDimensions, int dynamicSharedMemoryBytes = 0)
    {
        gridDimensions.Validate();
        blockDimensions.Validate();
        if (dynamicSharedMemoryBytes < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dynamicSharedMemoryBytes));
        }

        GridDimensions = gridDimensions;
        BlockDimensions = blockDimensions;
        DynamicSharedMemoryBytes = dynamicSharedMemoryBytes;
    }

    /// <summary>Gets grid dimensions. 获取 grid 维度。</summary>
    public CudaDim3 GridDimensions { get; }

    /// <summary>Gets block dimensions. 获取 block 维度。</summary>
    public CudaDim3 BlockDimensions { get; }

    /// <summary>Gets dynamic shared-memory bytes per block. 获取每个 block 的动态共享内存字节数。</summary>
    public int DynamicSharedMemoryBytes { get; }

    internal void Validate()
    {
        GridDimensions.Validate();
        BlockDimensions.Validate();
        if (DynamicSharedMemoryBytes < 0)
        {
            throw new InvalidOperationException("CUDA dynamic shared-memory size must not be negative.");
        }
    }
}
