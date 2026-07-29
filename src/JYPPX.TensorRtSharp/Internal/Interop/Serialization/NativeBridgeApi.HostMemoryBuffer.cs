using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static ulong GetHostMemorySize(TensorRtApiLine line, SafeTensorRtObjectHandle hostMemory)
    {
        UIntPtr size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_host_memory_get_size(hostMemory, out size),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_host_memory_get_size(hostMemory, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_host_memory_get_size(hostMemory, out size),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return size.ToUInt64();
    }

    public static byte[] CopyHostMemoryToArray(TensorRtApiLine line, SafeTensorRtObjectHandle hostMemory)
    {
        ulong size = GetHostMemorySize(line, hostMemory);
        if (size > int.MaxValue)
        {
            throw new InvalidOperationException("TensorRT host memory is too large to copy into a single managed byte array.");
        }

        byte[] data = new byte[checked((int)size)];
        UIntPtr bytesWritten;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_host_memory_copy_to_buffer(hostMemory, data, (UIntPtr)data.Length, out bytesWritten);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_host_memory_copy_to_buffer(hostMemory, data, (UIntPtr)data.Length, out bytesWritten);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_host_memory_copy_to_buffer(hostMemory, data, (UIntPtr)data.Length, out bytesWritten);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        if (bytesWritten.ToUInt64() != size)
        {
            throw new InvalidOperationException("TensorRT host memory copy returned an unexpected byte count.");
        }

        return data;
    }

}
