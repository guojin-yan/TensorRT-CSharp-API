using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// High-level environment snapshot for CUDA bridge diagnostics.
/// </summary>
public sealed class CudaEnvironmentSnapshot
{
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

    public BridgeBuildInfo BuildInfo { get; }
    public BridgeRuntimeInfo BridgeRuntimeInfo { get; }
    public CudaRuntimeInfo CudaRuntimeInfo { get; }
    public IReadOnlyList<CudaDeviceInfo> Devices { get; }
}

