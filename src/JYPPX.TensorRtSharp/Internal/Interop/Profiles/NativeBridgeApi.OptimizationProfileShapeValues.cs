using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool SetOptimizationProfileShapeValuesV2(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector, IReadOnlyList<long> values)
    {
        EnsureTensorRt10Or11(line, nameof(SetOptimizationProfileShapeValuesV2));
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

        long[] valueArray = new long[values.Count];
        for (int i = 0; i < values.Count; i++)
        {
            valueArray[i] = values[i];
        }

        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        GCHandle pinned = GCHandle.Alloc(valueArray, GCHandleType.Pinned);
        try
        {
            int set;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_set_shape_values_v2(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), valueArray.Length, out set),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_set_shape_values_v2(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), valueArray.Length, out set),
                _ => throw UnsupportedLine()
            };
            NativeStatus.ThrowIfFailed(status);
            return set != 0;
        }
        finally
        {
            pinned.Free();
        }
    }

    public static int GetOptimizationProfileShapeValueCountV2(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName)
    {
        EnsureTensorRt10Or11(line, nameof(GetOptimizationProfileShapeValueCountV2));
        if (string.IsNullOrWhiteSpace(inputName))
        {
            throw new ArgumentException("Input name must not be null or empty.", nameof(inputName));
        }

        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_get_shape_value_count_v2(profile, inputNameUtf8.Pointer, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_shape_value_count_v2(profile, inputNameUtf8.Pointer, out count),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static long[] GetOptimizationProfileShapeValuesV2(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector)
    {
        int count = GetOptimizationProfileShapeValueCountV2(line, profile, inputName);
        if (count <= 0)
        {
            return Array.Empty<long>();
        }

        long[] values = new long[count];
        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        GCHandle pinned = GCHandle.Alloc(values, GCHandleType.Pinned);
        try
        {
            int actualCount;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_get_shape_values_v2(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), values.Length, out actualCount),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_shape_values_v2(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), values.Length, out actualCount),
                _ => throw UnsupportedLine()
            };
            NativeStatus.ThrowIfFailed(status);
            if (actualCount == values.Length)
            {
                return values;
            }

            long[] trimmed = new long[Math.Max(actualCount, 0)];
            Array.Copy(values, trimmed, trimmed.Length);
            return trimmed;
        }
        finally
        {
            pinned.Free();
        }
    }

}
