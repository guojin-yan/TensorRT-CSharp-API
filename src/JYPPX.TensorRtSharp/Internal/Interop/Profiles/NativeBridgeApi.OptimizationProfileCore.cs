using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateOptimizationProfile(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        SafeTensorRtObjectHandle profile;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_builder_create_optimization_profile(builder, out profile);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_create_optimization_profile(builder, out profile);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_create_optimization_profile(builder, out profile);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return profile;
    }

    public static void SetOptimizationProfileShape(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector, TensorRtDims dims)
    {
        if (string.IsNullOrWhiteSpace(inputName))
        {
            throw new ArgumentException("Input name must not be null or empty.", nameof(inputName));
        }

        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        NativeTensorRtDims nativeDims = dims.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_optimization_profile_set_shape(profile, inputNameUtf8.Pointer, (int)selector, ref nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_set_shape(profile, inputNameUtf8.Pointer, (int)selector, ref nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_set_shape(profile, inputNameUtf8.Pointer, (int)selector, ref nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetOptimizationProfileShape(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector)
    {
        if (string.IsNullOrWhiteSpace(inputName))
        {
            throw new ArgumentException("Input name must not be null or empty.", nameof(inputName));
        }

        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        NativeTensorRtDims nativeDims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_optimization_profile_get_shape(profile, inputNameUtf8.Pointer, (int)selector, out nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_get_shape(profile, inputNameUtf8.Pointer, (int)selector, out nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_shape(profile, inputNameUtf8.Pointer, (int)selector, out nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(nativeDims);
    }

    public static void SetOptimizationProfileShapeValues(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector, IReadOnlyList<int> values)
    {
        if (string.IsNullOrWhiteSpace(inputName))
        {
            throw new ArgumentException("Input name must not be null or empty.", nameof(inputName));
        }

        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Count == 0)
        {
            throw new ArgumentException("Shape values must not be empty.", nameof(values));
        }

        int[] valueArray = new int[values.Count];
        for (int i = 0; i < values.Count; i++)
        {
            valueArray[i] = values[i];
        }

        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        GCHandle pinned = GCHandle.Alloc(valueArray, GCHandleType.Pinned);
        try
        {
            IntPtr valuePointer = pinned.AddrOfPinnedObject();
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_optimization_profile_set_shape_values(profile, inputNameUtf8.Pointer, (int)selector, valuePointer, valueArray.Length),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_set_shape_values(profile, inputNameUtf8.Pointer, (int)selector, valuePointer, valueArray.Length),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_set_shape_values(profile, inputNameUtf8.Pointer, (int)selector, valuePointer, valueArray.Length),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
        }
        finally
        {
            pinned.Free();
        }
    }

    public static int GetOptimizationProfileShapeValueCount(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName)
    {
        if (string.IsNullOrWhiteSpace(inputName))
        {
            throw new ArgumentException("Input name must not be null or empty.", nameof(inputName));
        }

        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_optimization_profile_get_shape_value_count(profile, inputNameUtf8.Pointer, out count);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_optimization_profile_get_shape_value_count(profile, inputNameUtf8.Pointer, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_shape_value_count(profile, inputNameUtf8.Pointer, out count);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int[] GetOptimizationProfileShapeValues(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector)
    {
        int count = GetOptimizationProfileShapeValueCount(line, profile, inputName);
        if (count <= 0)
        {
            return Array.Empty<int>();
        }

        int[] values = new int[count];
        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        GCHandle pinned = GCHandle.Alloc(values, GCHandleType.Pinned);
        try
        {
            int actualCount;
            BridgeStatusCode status;
            switch (line)
            {
                case TensorRtApiLine.TensorRt8:
                    status = NativeMethodsTensorRt.jyppx_trt8_optimization_profile_get_shape_values(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), values.Length, out actualCount);
                    break;
                case TensorRtApiLine.TensorRt10:
                    status = NativeMethodsTensorRt.jyppx_trt10_optimization_profile_get_shape_values(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), values.Length, out actualCount);
                    break;
                case TensorRtApiLine.TensorRt11:
                    status = NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_shape_values(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), values.Length, out actualCount);
                    break;
                default:
                    throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
            }

            NativeStatus.ThrowIfFailed(status);
            if (actualCount == values.Length)
            {
                return values;
            }

            int[] trimmed = new int[Math.Max(actualCount, 0)];
            Array.Copy(values, trimmed, trimmed.Length);
            return trimmed;
        }
        finally
        {
            pinned.Free();
        }
    }

    public static void SetOptimizationProfileExtraMemoryTarget(TensorRtApiLine line, SafeTensorRtObjectHandle profile, float target)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_optimization_profile_set_extra_memory_target(profile, target),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_set_extra_memory_target(profile, target),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_set_extra_memory_target(profile, target),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static float GetOptimizationProfileExtraMemoryTarget(TensorRtApiLine line, SafeTensorRtObjectHandle profile)
    {
        float target;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_optimization_profile_get_extra_memory_target(profile, out target);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_optimization_profile_get_extra_memory_target(profile, out target);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_extra_memory_target(profile, out target);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return target;
    }

    public static bool IsOptimizationProfileValid(TensorRtApiLine line, SafeTensorRtObjectHandle profile)
    {
        int isValid;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_optimization_profile_is_valid(profile, out isValid);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_optimization_profile_is_valid(profile, out isValid);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_optimization_profile_is_valid(profile, out isValid);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return isValid != 0;
    }

}
