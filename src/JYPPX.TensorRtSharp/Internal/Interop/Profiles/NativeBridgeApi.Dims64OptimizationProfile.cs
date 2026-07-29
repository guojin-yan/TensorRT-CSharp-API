using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtDims64 GetOptimizationProfileShape64(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetOptimizationProfileShape64));
        ValidateTensorName(inputName);
        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_shape64(profile, inputNameUtf8.Pointer, (int)selector, out NativeTensorRtDims64 dims);
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims64.FromNative(dims);
    }

    public static long GetOptimizationProfileShapeDimensionExtent64(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector, int dimensionIndex)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetOptimizationProfileShapeDimensionExtent64));
        ValidateTensorName(inputName);
        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_shape_dimension_extent64(profile, inputNameUtf8.Pointer, (int)selector, dimensionIndex, out long extent);
        NativeStatus.ThrowIfFailed(status);
        return extent;
    }

}
