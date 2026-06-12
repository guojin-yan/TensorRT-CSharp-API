using System.Collections.Generic;
using JYPPX.CudaSharp.Internal;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Probes the current CUDA bridge state and available devices.
/// </summary>
public static class CudaEnvironmentProbe
{
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

    public static int GetDeviceCount()
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceCount();
    }

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

