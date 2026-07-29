using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static void SetInputShape(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName, TensorRtDims dims)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
        }

        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        NativeTensorRtDims nativeDims = dims.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_input_shape(context, tensorNameUtf8.Pointer, ref nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_input_shape(context, tensorNameUtf8.Pointer, ref nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_input_shape(context, tensorNameUtf8.Pointer, ref nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetBindingDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle context, int bindingIndex, TensorRtDims dims)
    {
        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Binding-index dimension updates are only exposed for the TensorRT 8 adapter.");
        }

        NativeTensorRtDims nativeDims = dims.ToNative();
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_execution_context_set_binding_dimensions(context, bindingIndex, ref nativeDims);
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetTensorAddress(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName, SafeCudaMemoryHandle memory)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
        }

        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_tensor_address(context, tensorNameUtf8.Pointer, memory),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_tensor_address(context, tensorNameUtf8.Pointer, memory),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_tensor_address(context, tensorNameUtf8.Pointer, memory),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetInputTensorAddress(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName, SafeCudaMemoryHandle memory)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
        }

        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_input_tensor_address(context, tensorNameUtf8.Pointer, memory),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_input_tensor_address(context, tensorNameUtf8.Pointer, memory),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_input_tensor_address(context, tensorNameUtf8.Pointer, memory),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetOutputTensorAddress(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName, SafeCudaMemoryHandle memory)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
        }

        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_output_tensor_address(context, tensorNameUtf8.Pointer, memory),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_output_tensor_address(context, tensorNameUtf8.Pointer, memory),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_output_tensor_address(context, tensorNameUtf8.Pointer, memory),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void EnqueueAsync(TensorRtApiLine line, SafeTensorRtObjectHandle context, SafeCudaStreamHandle stream)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_enqueue_async(context, stream),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_enqueue_async(context, stream),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_enqueue_async(context, stream),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

}
