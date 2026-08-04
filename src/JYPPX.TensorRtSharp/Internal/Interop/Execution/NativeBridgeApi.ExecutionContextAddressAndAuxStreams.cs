using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool ClearExecutionContextTensorAddress(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearExecutionContextTensorAddress));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_tensor_address(context, tensorNameUtf8.Pointer, out int cleared);
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static bool ClearExecutionContextInputTensorAddress(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearExecutionContextInputTensorAddress));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_input_tensor_address(context, tensorNameUtf8.Pointer, out int cleared);
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static bool ClearExecutionContextOutputTensorAddress(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearExecutionContextOutputTensorAddress));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_output_tensor_address(context, tensorNameUtf8.Pointer, out int cleared);
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static void ClearExecutionContextDeviceMemory(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_clear_device_memory_v2(context),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_device_memory(context),
            TensorRtApiLine.TensorRt8 => throw new NotSupportedException("TensorRT 8 does not expose a size-aware device-memory clear operation."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool ClearExecutionContextInputConsumedEvent(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearExecutionContextInputConsumedEvent));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_input_consumed_event(context, out int cleared);
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static void SetExecutionContextAuxStreams(TensorRtApiLine line, SafeTensorRtObjectHandle context, IReadOnlyList<SafeCudaStreamHandle> streams)
    {
        if (streams == null)
        {
            throw new ArgumentNullException(nameof(streams));
        }

        if (streams.Count == 0)
        {
            NativeStatus.ThrowIfFailed(InvokeExecutionContextAuxStreams(line, context, IntPtr.Zero, 0));
            return;
        }

        IntPtr[] streamPointers = new IntPtr[streams.Count];
        for (int i = 0; i < streams.Count; i++)
        {
            if (streams[i] == null || streams[i].IsInvalid)
            {
                throw new ArgumentException("Auxiliary streams must contain valid CUDA stream handles.", nameof(streams));
            }

            streamPointers[i] = streams[i].DangerousGetHandle();
        }

        GCHandle pinned = GCHandle.Alloc(streamPointers, GCHandleType.Pinned);
        try
        {
            NativeStatus.ThrowIfFailed(InvokeExecutionContextAuxStreams(line, context, pinned.AddrOfPinnedObject(), streamPointers.Length));
        }
        finally
        {
            pinned.Free();
        }
    }

    private static BridgeStatusCode InvokeExecutionContextAuxStreams(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle context,
        IntPtr streams,
        int streamCount)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_aux_streams(context, streams, streamCount),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_aux_streams(context, streams, streamCount),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_aux_streams(context, streams, streamCount),
            _ => throw UnsupportedLine()
        };
    }
}
