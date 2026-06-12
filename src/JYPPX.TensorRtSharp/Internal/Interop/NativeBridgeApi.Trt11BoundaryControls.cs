using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static int GetBuilderMaxDlaBatchSize(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetBuilderMaxDlaBatchSize));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_get_max_dla_batch_size(builder, out int size);
        NativeStatus.ThrowIfFailed(status);
        return size;
    }

    public static bool SetBuilderMaxThreads(TensorRtApiLine line, SafeTensorRtObjectHandle builder, int maxThreads)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetBuilderMaxThreads));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_set_max_threads(builder, maxThreads, out int set);
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static int GetBuilderMaxThreads(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetBuilderMaxThreads));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_get_max_threads(builder, out int maxThreads);
        NativeStatus.ThrowIfFailed(status);
        return maxThreads;
    }

    public static void ClearBuilderGpuAllocator(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearBuilderGpuAllocator));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_builder_clear_gpu_allocator(builder));
    }

    public static bool HasBuilderErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(HasBuilderErrorRecorder));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_has_error_recorder(builder, out int hasRecorder);
        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    public static void ClearBuilderErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearBuilderErrorRecorder));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_builder_clear_error_recorder(builder));
    }

    public static void ResetBuilder(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ResetBuilder));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_builder_reset(builder));
    }

    public static bool IsNetworkSupported(TensorRtApiLine line, SafeTensorRtObjectHandle builder, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle config)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(IsNetworkSupported));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_is_network_supported(builder, network, config, out int supported);
        NativeStatus.ThrowIfFailed(status);
        return supported != 0;
    }

    public static bool HasEngineErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(HasEngineErrorRecorder));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_engine_has_error_recorder(engine, out int hasRecorder);
        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    public static void ClearEngineErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearEngineErrorRecorder));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_engine_clear_error_recorder(engine));
    }

    public static string GetEngineAliasedInputTensor(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetEngineAliasedInputTensor));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(tensorName);
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_engine_get_aliased_input_tensor(engine, nameUtf8.Pointer, buffer, size, out required),
            "Aliased input tensor name is too large for the managed buffer.");
    }

    public static bool HasExecutionContextErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(HasExecutionContextErrorRecorder));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_has_error_recorder(context, out int hasRecorder);
        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    public static void ClearExecutionContextErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearExecutionContextErrorRecorder));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_error_recorder(context));
    }

    public static bool HasNetworkErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(HasNetworkErrorRecorder));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_has_error_recorder(network, out int hasRecorder);
        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    public static void ClearNetworkErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearNetworkErrorRecorder));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_network_clear_error_recorder(network));
    }

    public static void RemoveNetworkTensor(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(RemoveNetworkTensor));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_network_remove_tensor(network, tensor));
    }

    public static SafeTensorRtObjectHandle AddTopKV2Layer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        TensorRtTopKOperation operation,
        int k,
        uint axes,
        TensorRtDataType indicesType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(AddTopKV2Layer));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_add_topk_v2(network, input, (int)operation, k, axes, (int)indicesType, out SafeTensorRtObjectHandle layer);
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }
}
