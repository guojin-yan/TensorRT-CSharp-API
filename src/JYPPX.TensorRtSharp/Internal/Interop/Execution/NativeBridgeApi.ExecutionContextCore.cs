using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateExecutionContext(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.ExecutionContextCreate(engine, out SafeTensorRtObjectHandle context);
        NativeStatus.ThrowIfFailed(status);
        return context;
    }

    public static SafeTensorRtObjectHandle CreateExecutionContextWithoutDeviceMemory(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        SafeTensorRtObjectHandle context;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_create_execution_context_without_device_memory(engine, out context),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_create_execution_context_without_device_memory(engine, out context),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_create_execution_context_without_device_memory(engine, out context),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return context;
    }

}
