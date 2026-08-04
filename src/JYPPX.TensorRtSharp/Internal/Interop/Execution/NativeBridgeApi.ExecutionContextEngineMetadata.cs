using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static ulong GetExecutionContextInputConsumedEventAddressValue(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextInputConsumedEventAddressValue));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_input_consumed_event_value(context, out UIntPtr address);
        NativeStatus.ThrowIfFailed(status);
        return address.ToUInt64();
    }

    public static TensorRtExecutionContextAllocationStrategy GetExecutionContextRuntimeConfigAllocationStrategy(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextRuntimeConfigAllocationStrategy));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_runtime_config_allocation_strategy(context, out int strategy);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtExecutionContextAllocationStrategy)strategy;
    }

    public static string GetExecutionContextEngineName(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextEngineName));
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_engine_name(context, buffer, size, out required),
            "Execution context engine name is too large for the managed buffer.");
    }

    public static int GetExecutionContextEngineIOTensorCount(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextEngineIOTensorCount));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_engine_io_tensor_count(context, out int count);
        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int GetExecutionContextEngineLayerCount(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextEngineLayerCount));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_engine_layer_count(context, out int count);
        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int GetExecutionContextEngineOptimizationProfileCount(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextEngineOptimizationProfileCount));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_engine_optimization_profile_count(context, out int count);
        NativeStatus.ThrowIfFailed(status);
        return count;
    }
}
