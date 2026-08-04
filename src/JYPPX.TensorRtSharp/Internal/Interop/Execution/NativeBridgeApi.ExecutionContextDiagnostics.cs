using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static ulong GetExecutionContextTensorAddressValue(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextTensorAddressValue));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_tensor_address_value(context, tensorNameUtf8.Pointer, out UIntPtr address);
        NativeStatus.ThrowIfFailed(status);
        return address.ToUInt64();
    }

    public static ulong GetExecutionContextOutputTensorAddressValue(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextOutputTensorAddressValue));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_output_tensor_address_value(context, tensorNameUtf8.Pointer, out UIntPtr address);
        NativeStatus.ThrowIfFailed(status);
        return address.ToUInt64();
    }

    public static bool ClearExecutionContextOutputAllocator(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int cleared = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_clear_output_allocator(context, tensorNameUtf8.Pointer, out cleared),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_clear_output_allocator(context, tensorNameUtf8.Pointer, out cleared),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_output_allocator(context, tensorNameUtf8.Pointer, out cleared),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static bool ClearExecutionContextTemporaryStorageAllocator(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int cleared = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_clear_temporary_storage_allocator(context, out cleared),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_clear_temporary_storage_allocator(context, out cleared),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_temporary_storage_allocator(context, out cleared),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static bool ClearExecutionContextDebugListener(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int cleared;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_clear_debug_listener(context, out cleared),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_debug_listener(context, out cleared),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(ClearExecutionContextDebugListener)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static bool HasExecutionContextDebugListener(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int hasListener;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_has_debug_listener(context, out hasListener),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_has_debug_listener(context, out hasListener),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(HasExecutionContextDebugListener)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasListener != 0;
    }

    public static void ClearExecutionContextProfiler(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_clear_profiler(context),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_clear_profiler(context),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_profiler(context),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool HasExecutionContextProfiler(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int hasProfiler;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_has_profiler(context, out hasProfiler),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_has_profiler(context, out hasProfiler),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_has_profiler(context, out hasProfiler),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasProfiler != 0;
    }

    public static void SetExecutionContextProfiler(TensorRtApiLine line, SafeTensorRtObjectHandle context, SafeTensorRtObjectHandle profiler)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_profiler(context, profiler),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_profiler(context, profiler),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_profiler(context, profiler),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool HasExecutionContextRuntimeConfig(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(HasExecutionContextRuntimeConfig));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_has_runtime_config(context, out int hasRuntimeConfig);
        NativeStatus.ThrowIfFailed(status);
        return hasRuntimeConfig != 0;
    }

    public static bool SetExecutionContextNvtxVerbosity(TensorRtApiLine line, SafeTensorRtObjectHandle context, TensorRtProfilingVerbosity verbosity)
    {
        int set = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_nvtx_verbosity(context, (int)verbosity, out set),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_nvtx_verbosity(context, (int)verbosity, out set),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_nvtx_verbosity(context, (int)verbosity, out set),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static TensorRtProfilingVerbosity GetExecutionContextNvtxVerbosity(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int verbosity = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_get_nvtx_verbosity(context, out verbosity),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_nvtx_verbosity(context, out verbosity),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_nvtx_verbosity(context, out verbosity),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtProfilingVerbosity)verbosity;
    }

    public static void ClearExecutionContextAuxStreams(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_clear_aux_streams(context),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_clear_aux_streams(context),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_aux_streams(context),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool SetExecutionContextUnfusedTensorsDebugState(TensorRtApiLine line, SafeTensorRtObjectHandle context, bool enabled)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetExecutionContextUnfusedTensorsDebugState));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_set_unfused_tensors_debug_state(context, enabled ? 1 : 0, out int set);
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static bool GetExecutionContextUnfusedTensorsDebugState(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextUnfusedTensorsDebugState));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_unfused_tensors_debug_state(context, out int debugState);
        NativeStatus.ThrowIfFailed(status);
        return debugState != 0;
    }
}
