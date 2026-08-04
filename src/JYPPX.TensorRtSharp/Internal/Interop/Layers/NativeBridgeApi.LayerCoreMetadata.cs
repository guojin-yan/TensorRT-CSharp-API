using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle GetLayerOutput(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        SafeTensorRtObjectHandle tensor;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_output(layer, index, out tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_output(layer, index, out tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_output(layer, index, out tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return tensor;
    }

    public static SafeTensorRtObjectHandle GetLayerInput(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        SafeTensorRtObjectHandle tensor;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_input(layer, index, out tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_input(layer, index, out tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_input(layer, index, out tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return tensor;
    }

    public static int GetLayerInputCount(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_input_count(layer, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_input_count(layer, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_input_count(layer, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int GetLayerOutputCount(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_output_count(layer, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_output_count(layer, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_output_count(layer, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static TensorRtLayerType GetLayerType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int type;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_type(layer, out type),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_type(layer, out type),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_type(layer, out type),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return MapLayerType(line, type);
    }

    private static TensorRtLayerType MapLayerType(TensorRtApiLine line, int nativeType)
    {
        if (line == TensorRtApiLine.TensorRt8)
        {
            switch (nativeType)
            {
                case 21:
                    return TensorRtLayerType.IdentityTrt8;
                case 23:
                    return TensorRtLayerType.SliceTrt8;
                case 24:
                    return TensorRtLayerType.ShapeTrt8;
                case 26:
                    return TensorRtLayerType.ResizeTrt8;
                case 31:
                    return TensorRtLayerType.SelectTrt8;
                case 32:
                    return TensorRtLayerType.FillTrt8;
                case 33:
                    return TensorRtLayerType.QuantizeTrt8;
                case 34:
                    return TensorRtLayerType.DequantizeTrt8;
                case 39:
                    return TensorRtLayerType.Einsum;
                case 40:
                    return TensorRtLayerType.Assertion;
                case 41:
                    return TensorRtLayerType.OneHot;
                case 43:
                    return TensorRtLayerType.GridSample;
                case 44:
                    return TensorRtLayerType.Nms;
                case 45:
                    return TensorRtLayerType.ReverseSequence;
                case 46:
                    return TensorRtLayerType.NormalizationTrt8;
                case 47:
                    return TensorRtLayerType.Cast;
            }
        }

        if (line == TensorRtApiLine.TensorRt10)
        {
            switch (nativeType)
            {
                case 20:
                    return TensorRtLayerType.IdentityTrt10;
                case 22:
                    return TensorRtLayerType.SliceTrt10;
                case 23:
                    return TensorRtLayerType.ShapeTrt10;
                case 25:
                    return TensorRtLayerType.ResizeTrt10;
                case 30:
                    return TensorRtLayerType.SelectTrt10;
                case 31:
                    return TensorRtLayerType.FillTrt10;
                case 32:
                    return TensorRtLayerType.QuantizeTrt10;
                case 33:
                    return TensorRtLayerType.DequantizeTrt10;
                case 45:
                    return TensorRtLayerType.NormalizationTrt10;
            }
        }

        return Enum.IsDefined(typeof(TensorRtLayerType), nativeType)
            ? (TensorRtLayerType)nativeType
            : TensorRtLayerType.Unknown;
    }

    public static string GetLayerName(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        BridgeStatusCode status = GetLayerNameNative(line, layer, IntPtr.Zero, UIntPtr.Zero, out UIntPtr requiredSize);
        NativeStatus.ThrowIfFailed(status);

        ulong required = requiredSize.ToUInt64();
        if (required == 0)
        {
            return string.Empty;
        }

        if (required > int.MaxValue)
        {
            throw new BridgeProbeException(BridgeStatusCode.BufferTooSmall, BridgeErrorCategory.TensorRt, "Layer name is too large for the managed buffer.");
        }

        IntPtr buffer = Marshal.AllocHGlobal(checked((int)required));
        try
        {
            status = GetLayerNameNative(line, layer, buffer, requiredSize, out _);
            NativeStatus.ThrowIfFailed(status);
            return Utf8Interop.ReadString(buffer);
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }

    public static void SetLayerName(TensorRtApiLine line, SafeTensorRtObjectHandle layer, string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Layer name must not be null or empty.", nameof(name));
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_set_name(layer, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_set_name(layer, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_set_name(layer, nameUtf8.Pointer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetLayerPrecision(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDataType dataType)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_set_precision(layer, (int)dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_set_precision(layer, (int)dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_set_precision(layer, (int)dataType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDataType GetLayerPrecision(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int dataType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_precision(layer, out dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_precision(layer, out dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_precision(layer, out dataType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static bool IsLayerPrecisionSet(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int isSet;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_precision_is_set(layer, out isSet),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_precision_is_set(layer, out isSet),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_precision_is_set(layer, out isSet),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return isSet != 0;
    }

    public static void ResetLayerPrecision(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_reset_precision(layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_reset_precision(layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_reset_precision(layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetLayerOutputType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int outputIndex, TensorRtDataType dataType)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_set_output_type(layer, outputIndex, (int)dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_set_output_type(layer, outputIndex, (int)dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_set_output_type(layer, outputIndex, (int)dataType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDataType GetLayerOutputType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int outputIndex)
    {
        int dataType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_output_type(layer, outputIndex, out dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_output_type(layer, outputIndex, out dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_output_type(layer, outputIndex, out dataType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static bool IsLayerOutputTypeSet(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int outputIndex)
    {
        int isSet;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_output_type_is_set(layer, outputIndex, out isSet),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_output_type_is_set(layer, outputIndex, out isSet),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_output_type_is_set(layer, outputIndex, out isSet),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return isSet != 0;
    }

    public static void ResetLayerOutputType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int outputIndex)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_reset_output_type(layer, outputIndex),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_reset_output_type(layer, outputIndex),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_reset_output_type(layer, outputIndex),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    private static BridgeStatusCode GetLayerNameNative(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        IntPtr outputBuffer,
        UIntPtr outputBufferSize,
        out UIntPtr requiredSize)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_name(layer, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_name(layer, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_name(layer, outputBuffer, outputBufferSize, out requiredSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
    }

}
