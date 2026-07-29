using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Identifies CUDA memory-pool attributes used by deployment memory management.
/// 标识部署内存管理中常用的 CUDA 内存池属性。
/// </summary>
public enum CudaMemoryPoolAttribute
{
    /// <summary>
    /// Enables reuse that follows event dependencies.
    /// 启用遵循事件依赖关系的内存复用。
    /// </summary>
    ReuseFollowEventDependencies = 1,

    /// <summary>
    /// Allows opportunistic allocation reuse.
    /// 允许机会性的分配复用。
    /// </summary>
    ReuseAllowOpportunistic = 2,

    /// <summary>
    /// Allows CUDA to insert internal dependencies for reuse.
    /// 允许 CUDA 为复用插入内部依赖。
    /// </summary>
    ReuseAllowInternalDependencies = 3,

    /// <summary>
    /// Controls the amount of memory retained before release.
    /// 控制释放前保留的缓存内存量。
    /// </summary>
    ReleaseThreshold = 4,

    /// <summary>
    /// Reports current reserved memory.
    /// 报告当前已保留的内存。
    /// </summary>
    ReservedMemoryCurrent = 5,

    /// <summary>
    /// Reports high-water reserved memory.
    /// 报告已保留内存的峰值。
    /// </summary>
    ReservedMemoryHigh = 6,

    /// <summary>
    /// Reports current used memory.
    /// 报告当前已使用的内存。
    /// </summary>
    UsedMemoryCurrent = 7,

    /// <summary>
    /// Reports high-water used memory.
    /// 报告已使用内存的峰值。
    /// </summary>
    UsedMemoryHigh = 8
}

/// <summary>
/// Describes CUDA memory-pool access permissions for a device location.
/// 描述某个 CUDA 设备位置对 memory pool 的访问权限。
/// </summary>
[Flags]
public enum CudaMemoryPoolAccessFlags : uint
{
    /// <summary>
    /// No access is allowed.
    /// 不允许访问。
    /// </summary>
    None = 0,

    /// <summary>
    /// Read-only access is allowed.
    /// 允许只读访问。
    /// </summary>
    Read = 1,

    /// <summary>
    /// Read-write access is allowed.
    /// 允许读写访问。
    /// </summary>
    ReadWrite = 3
}
