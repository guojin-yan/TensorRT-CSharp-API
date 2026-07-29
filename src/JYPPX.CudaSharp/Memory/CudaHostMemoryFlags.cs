using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Flags that control pinned host-memory allocation behavior.
/// 控制 pinned host memory 分配行为的标志。
/// </summary>
[Flags]
public enum CudaPinnedMemoryAllocationFlags : uint
{
    /// <summary>
    /// Uses CUDA default pinned-memory allocation behavior.
    /// 使用 CUDA 默认的 pinned memory 分配行为。
    /// </summary>
    Default = 0,
    /// <summary>
    /// Makes the allocation portable across CUDA contexts.
    /// 使该分配在多个 CUDA context 之间可移植。
    /// </summary>
    Portable = 1,
    /// <summary>
    /// Maps the pinned allocation into the device address space.
    /// 将该 pinned 分配映射到设备地址空间。
    /// </summary>
    Mapped = 2,
    /// <summary>
    /// Requests write-combined host memory.
    /// 请求 write-combined host memory。
    /// </summary>
    WriteCombined = 4
}
/// <summary>
/// Flags that control registered host-memory behavior.
/// 控制 registered host memory 行为的标志。
/// </summary>
[Flags]
public enum CudaHostRegistrationFlags : uint
{
    /// <summary>
    /// Uses CUDA default host-registration behavior.
    /// 使用 CUDA 默认的 host registration 行为。
    /// </summary>
    Default = 0,
    /// <summary>
    /// Makes the registration portable across CUDA contexts.
    /// 使该注册在多个 CUDA context 之间可移植。
    /// </summary>
    Portable = 1,
    /// <summary>
    /// Maps the registered memory into the device address space.
    /// 将注册的内存映射到设备地址空间。
    /// </summary>
    Mapped = 2,
    /// <summary>
    /// Marks the registration as I/O memory.
    /// 将该注册标记为 I/O memory。
    /// </summary>
    IoMemory = 4,
    /// <summary>
    /// Registers the host memory as read-only from the device perspective.
    /// 从设备视角将 host memory 注册为只读。
    /// </summary>
    ReadOnly = 8
}
