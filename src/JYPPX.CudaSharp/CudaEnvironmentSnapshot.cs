using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// High-level environment snapshot for CUDA bridge diagnostics.
/// 用于 CUDA bridge 诊断的高层环境快照。
/// </summary>
public sealed class CudaEnvironmentSnapshot
{
    /// <summary>
    /// Creates a high-level CUDA environment snapshot.
    /// 创建高层 CUDA 环境快照。
    /// </summary>
    public CudaEnvironmentSnapshot(
        BridgeBuildInfo buildInfo,
        BridgeRuntimeInfo bridgeRuntimeInfo,
        CudaRuntimeInfo cudaRuntimeInfo,
        IReadOnlyList<CudaDeviceInfo> devices)
    {
        BuildInfo = buildInfo;
        BridgeRuntimeInfo = bridgeRuntimeInfo;
        CudaRuntimeInfo = cudaRuntimeInfo;
        Devices = devices;
    }

    /// <summary>
    /// Gets bridge build information.
    /// 获取 bridge 构建信息。
    /// </summary>
    public BridgeBuildInfo BuildInfo { get; }
    /// <summary>
    /// Gets bridge runtime information.
    /// 获取 bridge 运行时信息。
    /// </summary>
    public BridgeRuntimeInfo BridgeRuntimeInfo { get; }
    /// <summary>
    /// Gets CUDA runtime information.
    /// 获取 CUDA runtime 信息。
    /// </summary>
    public CudaRuntimeInfo CudaRuntimeInfo { get; }
    /// <summary>
    /// Gets the visible CUDA device snapshots.
    /// 获取可见 CUDA 设备快照集合。
    /// </summary>
    public IReadOnlyList<CudaDeviceInfo> Devices { get; }
}
