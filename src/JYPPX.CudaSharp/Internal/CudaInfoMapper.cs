using System.Text;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp.Internal;

internal static class CudaInfoMapper
{
    public static CudaRuntimeInfo ToManaged(NativeCudaRuntimeInfo value)
    {
        return new CudaRuntimeInfo(
            vendorDependencyAvailable: value.VendorDependencyAvailable != 0,
            supportsStreams: value.SupportsStreams != 0,
            supportsEvents: value.SupportsEvents != 0,
            supportsMemory: value.SupportsMemory != 0,
            runtimeVersion: value.RuntimeVersion,
            driverVersion: value.DriverVersion,
            deviceCount: value.DeviceCount,
            statusMessage: Utf8Interop.ReadString(value.StatusMessage));
    }

    public static CudaDeviceInfo ToManaged(NativeCudaDeviceInfo value)
    {
        int terminator = System.Array.IndexOf(value.Name, (byte)0);
        int length = terminator >= 0 ? terminator : value.Name.Length;
        string name = length == 0 ? string.Empty : Encoding.ASCII.GetString(value.Name, 0, length);

        return new CudaDeviceInfo(
            ordinal: value.Ordinal,
            name: name,
            major: value.Major,
            minor: value.Minor,
            multiProcessorCount: value.MultiProcessorCount,
            warpSize: value.WarpSize,
            maxThreadsPerBlock: value.MaxThreadsPerBlock,
            canMapHostMemory: value.CanMapHostMemory != 0,
            integrated: value.Integrated != 0,
            totalGlobalMemory: value.TotalGlobalMemory);
    }
}

