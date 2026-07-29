using System;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Captured runtime environment for one MNIST proof attempt.
/// 一次 MNIST proof 尝试所采集的运行环境。
/// </summary>
public sealed class MnistRuntimeEnvironment
{
    public MnistRuntimeEnvironment(
        string hostOs,
        string processArchitecture,
        string machineName,
        string gpuName,
        string computeCapability,
        ulong gpuMemoryBytes,
        int cudaDriverVersion,
        int cudaRuntimeVersion,
        string cudaToolkitVersion,
        string tensorRtVersion,
        string bridgeVersion)
    {
        HostOs = hostOs;
        ProcessArchitecture = processArchitecture;
        MachineName = machineName;
        GpuName = gpuName;
        ComputeCapability = computeCapability;
        GpuMemoryBytes = gpuMemoryBytes;
        CudaDriverVersion = cudaDriverVersion;
        CudaRuntimeVersion = cudaRuntimeVersion;
        CudaToolkitVersion = cudaToolkitVersion;
        TensorRtVersion = tensorRtVersion;
        BridgeVersion = bridgeVersion;
    }

    public string HostOs { get; }

    public string ProcessArchitecture { get; }

    public string MachineName { get; }

    public string GpuName { get; }

    public string ComputeCapability { get; }

    public ulong GpuMemoryBytes { get; }

    public int CudaDriverVersion { get; }

    public int CudaRuntimeVersion { get; }

    public string CudaToolkitVersion { get; }

    public string TensorRtVersion { get; }

    public string BridgeVersion { get; }

    public static MnistRuntimeEnvironment Capture(TensorRtEnvironmentSnapshot snapshot)
    {
        CudaDeviceProperties device = CudaDevice.CurrentProperties;
        BridgeBuildInfo build = snapshot.BuildInfo;
        return new MnistRuntimeEnvironment(
            RuntimeInformation.OSDescription,
            RuntimeInformation.ProcessArchitecture.ToString(),
            Environment.MachineName,
            device.Info.Name,
            device.ComputeCapabilityLabel,
            device.Info.TotalGlobalMemory,
            CudaDevice.DriverVersion,
            CudaDevice.RuntimeVersion,
            build.CudaToolkitVersion,
            build.TensorRtVersion,
            $"{build.BridgeVersionMajor}.{build.BridgeVersionMinor}.{build.BridgeVersionPatch}");
    }
}
