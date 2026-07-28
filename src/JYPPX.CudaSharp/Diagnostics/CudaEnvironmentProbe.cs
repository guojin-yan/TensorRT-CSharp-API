using System.Collections.Generic;
using JYPPX.CudaSharp.Internal;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Probes the current CUDA bridge state and available devices.
/// 探测当前 CUDA bridge 状态和可用设备。
/// </summary>
public static class CudaEnvironmentProbe
{
    /// <summary>
    /// Collects the current CUDA bridge, runtime, and device snapshot.
    /// 收集当前 CUDA bridge、runtime 与设备快照。
    /// </summary>
    /// <returns>The current CUDA environment snapshot. 当前 CUDA 环境快照。</returns>
    public static CudaEnvironmentSnapshot GetCurrent()
    {
        NativeBridgeLoader.EnsureInitialized();

        var buildInfo = BridgeInfoMapper.ToManaged(NativeBridgeApi.GetBuildInfo());
        var bridgeRuntimeInfo = BridgeInfoMapper.ToManaged(NativeBridgeApi.GetRuntimeInfo());
        var runtimeInfo = CudaInfoMapper.ToManaged(NativeCudaApi.GetRuntimeInfo());

        List<CudaDeviceInfo> devices = new List<CudaDeviceInfo>();
        if (runtimeInfo.VendorDependencyAvailable && runtimeInfo.DeviceCount > 0)
        {
            for (int ordinal = 0; ordinal < runtimeInfo.DeviceCount; ordinal++)
            {
                devices.Add(CudaInfoMapper.ToManaged(NativeCudaApi.GetDeviceInfo(ordinal)));
            }
        }

        return new CudaEnvironmentSnapshot(buildInfo, bridgeRuntimeInfo, runtimeInfo, devices);
    }

    /// <summary>
    /// Gets the CUDA device count visible to the current process.
    /// 获取当前进程可见的 CUDA 设备数量。
    /// </summary>
    /// <returns>The number of visible CUDA devices. 可见 CUDA 设备数量。</returns>
    public static int GetDeviceCount()
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceCount();
    }

    /// <summary>
    /// Tries to create a CUDA stream to confirm runtime usability.
    /// 尝试创建 CUDA stream 以确认 runtime 可用性。
    /// </summary>
    /// <param name="message">Returns a success or diagnostic message. 返回成功信息或诊断信息。</param>
    /// <returns><see langword="true"/> when a CUDA stream was created successfully. 成功创建 CUDA stream 时返回 <see langword="true"/>。</returns>
    public static bool TryCreateStream(out string message)
    {
        NativeBridgeLoader.EnsureInitialized();

        using SafeCudaStreamHandle stream = NativeCudaApi.CreateStream();
        if (!stream.IsInvalid)
        {
            message = "CUDA stream handle created successfully.";
            return true;
        }

        message = NativeBridgeApi.GetLastErrorMessageOrFallback("CUDA stream creation failed.");
        return false;
    }
}
