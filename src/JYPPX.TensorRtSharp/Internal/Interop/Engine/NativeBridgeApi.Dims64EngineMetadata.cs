using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtDims64 GetEngineTensorShape64(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetEngineTensorShape64));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_shape64(engine, tensorNameUtf8.Pointer, out NativeTensorRtDims64 shape);
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims64.FromNative(shape);
    }

    public static long GetEngineTensorDimensionExtent64(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, int dimensionIndex)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetEngineTensorDimensionExtent64));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_dimension_extent64(engine, tensorNameUtf8.Pointer, dimensionIndex, out long extent);
        NativeStatus.ThrowIfFailed(status);
        return extent;
    }

    public static TensorRtDims64 GetEngineProfileShape64(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetEngineProfileShape64));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_engine_get_profile_shape64(engine, tensorNameUtf8.Pointer, profileIndex, (int)selector, out NativeTensorRtDims64 shape);
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims64.FromNative(shape);
    }

    public static long GetEngineProfileShapeDimensionExtent64(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector, int dimensionIndex)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetEngineProfileShapeDimensionExtent64));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_engine_get_profile_shape_dimension_extent64(engine, tensorNameUtf8.Pointer, profileIndex, (int)selector, dimensionIndex, out long extent);
        NativeStatus.ThrowIfFailed(status);
        return extent;
    }

}
