using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool IsExecutionContextInputConsumedEventSet(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(IsExecutionContextInputConsumedEventSet));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_is_input_consumed_event_set(context, out int isSet);
        NativeStatus.ThrowIfFailed(status);
        return isSet != 0;
    }

    public static bool IsExecutionContextOutputTensorAddressSet(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(IsExecutionContextOutputTensorAddressSet));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_is_output_tensor_address_set(context, tensorNameUtf8.Pointer, out int isSet);
        NativeStatus.ThrowIfFailed(status);
        return isSet != 0;
    }

    public static bool HasExecutionContextOutputAllocator(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int hasAllocator = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_has_output_allocator(context, tensorNameUtf8.Pointer, out hasAllocator),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_has_output_allocator(context, tensorNameUtf8.Pointer, out hasAllocator),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_has_output_allocator(context, tensorNameUtf8.Pointer, out hasAllocator),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasAllocator != 0;
    }

    public static bool HasExecutionContextTemporaryStorageAllocator(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int hasAllocator = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_has_temporary_storage_allocator(context, out hasAllocator),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_has_temporary_storage_allocator(context, out hasAllocator),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_has_temporary_storage_allocator(context, out hasAllocator),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasAllocator != 0;
    }

}
