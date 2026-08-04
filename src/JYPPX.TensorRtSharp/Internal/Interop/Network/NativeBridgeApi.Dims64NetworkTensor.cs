using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtDims64 GetTensorShape64(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetTensorShape64));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_tensor_get_shape64(tensor, out NativeTensorRtDims64 dims);
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims64.FromNative(dims);
    }

    public static long GetTensorDimensionExtent64(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, int dimensionIndex)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetTensorDimensionExtent64));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_tensor_get_dimension_extent64(tensor, dimensionIndex, out long extent);
        NativeStatus.ThrowIfFailed(status);
        return extent;
    }

    public static TensorRtDims64 GetNetworkInputTensorShape64(TensorRtApiLine line, SafeTensorRtObjectHandle network, int index)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetNetworkInputTensorShape64));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_input_get_tensor_shape64(network, index, out NativeTensorRtDims64 shape);
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims64.FromNative(shape);
    }

    public static TensorRtDims64 GetNetworkOutputTensorShape64(TensorRtApiLine line, SafeTensorRtObjectHandle network, int index)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetNetworkOutputTensorShape64));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_output_get_tensor_shape64(network, index, out NativeTensorRtDims64 shape);
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims64.FromNative(shape);
    }

    public static long GetNetworkInputTensorDimensionExtent64(TensorRtApiLine line, SafeTensorRtObjectHandle network, int index, int dimensionIndex)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetNetworkInputTensorDimensionExtent64));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_input_get_tensor_dimension_extent64(network, index, dimensionIndex, out long extent);
        NativeStatus.ThrowIfFailed(status);
        return extent;
    }

    public static long GetNetworkOutputTensorDimensionExtent64(TensorRtApiLine line, SafeTensorRtObjectHandle network, int index, int dimensionIndex)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetNetworkOutputTensorDimensionExtent64));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_output_get_tensor_dimension_extent64(network, index, dimensionIndex, out long extent);
        NativeStatus.ThrowIfFailed(status);
        return extent;
    }

}
