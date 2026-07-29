using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateRuntimeConfig(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        SafeTensorRtObjectHandle config;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_create_runtime_config(engine, out config),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_create_runtime_config(engine, out config),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Runtime config is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return config;
    }

    public static SafeTensorRtObjectHandle CreateExecutionContext(TensorRtApiLine line, SafeTensorRtObjectHandle engine, TensorRtExecutionContextAllocationStrategy strategy)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(CreateExecutionContext));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_engine_create_execution_context_with_strategy(engine, (int)strategy, out SafeTensorRtObjectHandle context);
        NativeStatus.ThrowIfFailed(status);
        return context;
    }

    public static SafeTensorRtObjectHandle CreateExecutionContext(TensorRtApiLine line, SafeTensorRtObjectHandle engine, SafeTensorRtObjectHandle runtimeConfig)
    {
        SafeTensorRtObjectHandle context;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_create_execution_context_with_runtime_config(engine, runtimeConfig, out context),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_create_execution_context_with_runtime_config(engine, runtimeConfig, out context),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Runtime config is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return context;
    }

    public static void SetRuntimeConfigAllocationStrategy(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtExecutionContextAllocationStrategy strategy)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_config_set_execution_context_allocation_strategy(config, (int)strategy),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_config_set_execution_context_allocation_strategy(config, (int)strategy),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Runtime config is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtExecutionContextAllocationStrategy GetRuntimeConfigAllocationStrategy(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int strategy;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_config_get_execution_context_allocation_strategy(config, out strategy),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_config_get_execution_context_allocation_strategy(config, out strategy),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Runtime config is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtExecutionContextAllocationStrategy)strategy;
    }

}
