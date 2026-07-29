using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtDataType GetHostMemoryDataType(TensorRtApiLine line, SafeTensorRtObjectHandle hostMemory)
    {
        int dataType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_host_memory_get_type(hostMemory, out dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_host_memory_get_type(hostMemory, out dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_host_memory_get_type(hostMemory, out dataType),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

}
