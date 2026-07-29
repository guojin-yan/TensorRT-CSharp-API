using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Memory-advice commands accepted by CUDA managed memory.
/// CUDA managed memory 接受的内存建议命令。
/// </summary>
public enum CudaMemoryAdvice
{
    /// <summary>
    /// Marks the memory range as read-mostly.
    /// 将该内存范围标记为主要用于读取。
    /// </summary>
    SetReadMostly = 1,
    /// <summary>
    /// Clears the read-mostly advice.
    /// 清除主要读取的建议标记。
    /// </summary>
    UnsetReadMostly = 2,
    /// <summary>
    /// Sets a preferred location for the memory range.
    /// 为该内存范围设置首选位置。
    /// </summary>
    SetPreferredLocation = 3,
    /// <summary>
    /// Clears the preferred location advice.
    /// 清除首选位置建议。
    /// </summary>
    UnsetPreferredLocation = 4,
    /// <summary>
    /// Marks the memory range as accessed by a device.
    /// 将该内存范围标记为会被某个设备访问。
    /// </summary>
    SetAccessedBy = 5,
    /// <summary>
    /// Clears the accessed-by advice.
    /// 清除 accessed-by 建议。
    /// </summary>
    UnsetAccessedBy = 6
}
