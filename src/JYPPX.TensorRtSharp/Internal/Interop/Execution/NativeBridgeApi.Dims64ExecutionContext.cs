using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtDims64 GetExecutionContextTensorShape64(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextTensorShape64));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_tensor_shape64(context, tensorNameUtf8.Pointer, out NativeTensorRtDims64 shape);
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims64.FromNative(shape);
    }

    public static long GetExecutionContextTensorShapeDimensionExtent64(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName, int dimensionIndex)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextTensorShapeDimensionExtent64));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_tensor_shape_dimension_extent64(context, tensorNameUtf8.Pointer, dimensionIndex, out long extent);
        NativeStatus.ThrowIfFailed(status);
        return extent;
    }

    public static TensorRtDims64 GetExecutionContextTensorStrides64(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextTensorStrides64));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_tensor_strides64(context, tensorNameUtf8.Pointer, out NativeTensorRtDims64 strides);
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims64.FromNative(strides);
    }

    public static long GetExecutionContextTensorStrideDimensionExtent64(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName, int dimensionIndex)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextTensorStrideDimensionExtent64));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_tensor_stride_dimension_extent64(context, tensorNameUtf8.Pointer, dimensionIndex, out long extent);
        NativeStatus.ThrowIfFailed(status);
        return extent;
    }

}
