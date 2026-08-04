using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static string GetTensorName(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        BridgeStatusCode status = GetTensorNameNative(line, tensor, IntPtr.Zero, UIntPtr.Zero, out UIntPtr requiredSize);
        NativeStatus.ThrowIfFailed(status);

        ulong required = requiredSize.ToUInt64();
        if (required == 0)
        {
            return string.Empty;
        }

        if (required > int.MaxValue)
        {
            throw new BridgeProbeException(BridgeStatusCode.BufferTooSmall, BridgeErrorCategory.TensorRt, "Tensor name is too large for the managed buffer.");
        }

        IntPtr buffer = Marshal.AllocHGlobal(checked((int)required));
        try
        {
            status = GetTensorNameNative(line, tensor, buffer, requiredSize, out _);
            NativeStatus.ThrowIfFailed(status);
            return Utf8Interop.ReadString(buffer);
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }

    public static void SetTensorName(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(name));
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_name(tensor, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_name(tensor, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_set_name(tensor, nameUtf8.Pointer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDataType GetTensorDataType(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int dataType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_data_type(tensor, out dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_data_type(tensor, out dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_data_type(tensor, out dataType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static TensorRtDims GetTensorShape(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        NativeTensorRtDims nativeDims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_shape(tensor, out nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_shape(tensor, out nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_shape(tensor, out nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(nativeDims);
    }

    public static void SetTensorShape(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, TensorRtDims dims)
    {
        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        NativeTensorRtDims nativeDims = dims.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_shape(tensor, ref nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_shape(tensor, ref nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetTensorDataType(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, TensorRtDataType dataType)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_data_type(tensor, (int)dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_data_type(tensor, (int)dataType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtTensorLocation GetTensorLocation(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int location;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_location(tensor, out location),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_location(tensor, out location),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_location(tensor, out location),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTensorLocation)location;
    }

    public static void SetTensorLocation(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, TensorRtTensorLocation location)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_location(tensor, (int)location),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_location(tensor, (int)location),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_set_location(tensor, (int)location),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtTensorFormats GetTensorAllowedFormats(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        uint formats;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_allowed_formats(tensor, out formats),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_allowed_formats(tensor, out formats),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_allowed_formats(tensor, out formats),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTensorFormats)formats;
    }

    public static void SetTensorAllowedFormats(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, TensorRtTensorFormats formats)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_allowed_formats(tensor, (uint)formats),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_allowed_formats(tensor, (uint)formats),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_set_allowed_formats(tensor, (uint)formats),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool IsTensorShapeTensor(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int isShapeTensor = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_is_shape_tensor(tensor, out isShapeTensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_is_shape_tensor(tensor, out isShapeTensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_is_shape_tensor(tensor, out isShapeTensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return isShapeTensor != 0;
    }

    public static bool IsTensorExecutionTensor(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int isExecutionTensor = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_is_execution_tensor(tensor, out isExecutionTensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_is_execution_tensor(tensor, out isExecutionTensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_is_execution_tensor(tensor, out isExecutionTensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return isExecutionTensor != 0;
    }

    public static bool GetTensorBroadcastAcrossBatch(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int value = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_broadcast_across_batch(tensor, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_broadcast_across_batch(tensor, out value),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_broadcast_across_batch(tensor, out value),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }

    public static void SetTensorBroadcastAcrossBatch(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, bool value)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_broadcast_across_batch(tensor, value ? 1 : 0),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_broadcast_across_batch(tensor, value ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_set_broadcast_across_batch(tensor, value ? 1 : 0),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static string GetTensorDimensionName(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, int dimensionIndex)
    {
        if (dimensionIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensionIndex), "Dimension index must be greater than or equal to zero.");
        }

        return line switch
        {
            TensorRtApiLine.TensorRt8 => ReadUtf8Buffer((byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt8_tensor_get_dimension_name(tensor, dimensionIndex, buffer, size, out required), "Tensor dimension name is too large for the managed buffer."),
            TensorRtApiLine.TensorRt10 => ReadUtf8Buffer((byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt10_tensor_get_dimension_name(tensor, dimensionIndex, buffer, size, out required), "Tensor dimension name is too large for the managed buffer."),
            TensorRtApiLine.TensorRt11 => ReadUtf8Buffer((byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_tensor_get_dimension_name(tensor, dimensionIndex, buffer, size, out required), "Tensor dimension name is too large for the managed buffer."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
    }

    public static void SetTensorDimensionName(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, int dimensionIndex, string name)
    {
        if (dimensionIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensionIndex), "Dimension index must be greater than or equal to zero.");
        }

        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Dimension name must not be null or empty. Use ClearTensorDimensionName to remove a name.", nameof(name));
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_dimension_name(tensor, dimensionIndex, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_dimension_name(tensor, dimensionIndex, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_set_dimension_name(tensor, dimensionIndex, nameUtf8.Pointer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void ClearTensorDimensionName(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, int dimensionIndex)
    {
        if (dimensionIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensionIndex), "Dimension index must be greater than or equal to zero.");
        }

        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_clear_dimension_name(tensor, dimensionIndex),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_clear_dimension_name(tensor, dimensionIndex),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_clear_dimension_name(tensor, dimensionIndex),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetTensorDynamicRange(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, float minimum, float maximum)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_dynamic_range(tensor, minimum, maximum),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_dynamic_range(tensor, minimum, maximum),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_set_dynamic_range(tensor, minimum, maximum),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool IsTensorDynamicRangeSet(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int value = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_dynamic_range_is_set(tensor, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_dynamic_range_is_set(tensor, out value),
            TensorRtApiLine.TensorRt11 => BridgeStatusCode.NotSupported,
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }

    public static float GetTensorDynamicRangeMin(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        float value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_dynamic_range_min(tensor, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_dynamic_range_min(tensor, out value),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_dynamic_range_min(tensor, out value),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static float GetTensorDynamicRangeMax(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        float value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_dynamic_range_max(tensor, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_dynamic_range_max(tensor, out value),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_dynamic_range_max(tensor, out value),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static void ResetTensorDynamicRange(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_reset_dynamic_range(tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_reset_dynamic_range(tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_reset_dynamic_range(tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    private static BridgeStatusCode GetTensorNameNative(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle tensor,
        IntPtr outputBuffer,
        UIntPtr outputBufferSize,
        out UIntPtr requiredSize)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_name(tensor, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_name(tensor, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_name(tensor, outputBuffer, outputBufferSize, out requiredSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
    }

}
