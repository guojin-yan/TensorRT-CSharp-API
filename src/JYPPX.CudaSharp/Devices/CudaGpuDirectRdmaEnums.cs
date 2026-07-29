using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Selects the target for GPU Direct RDMA write flushing.
/// 选择 GPU Direct RDMA 写入刷新的目标。
/// </summary>
public enum CudaGpuDirectRdmaWritesTarget
{
    /// <summary>
    /// Flushes writes for the current CUDA device.
    /// 为当前 CUDA 设备刷新写入。
    /// </summary>
    CurrentDevice = 0
}
/// <summary>
/// Selects the visibility scope for GPU Direct RDMA write flushing.
/// 选择 GPU Direct RDMA 写入刷新的可见性范围。
/// </summary>
public enum CudaGpuDirectRdmaWritesScope
{
    /// <summary>
    /// Flushes writes so the owner can observe them.
    /// 刷新写入以便拥有者可见。
    /// </summary>
    ToOwner = 100,
    /// <summary>
    /// Flushes writes so all devices can observe them.
    /// 刷新写入以便所有设备都可见。
    /// </summary>
    ToAllDevices = 200
}
