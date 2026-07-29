using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static int GetEngineIOTensorCount(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_io_tensor_count(engine, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_io_tensor_count(engine, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_io_tensor_count(engine, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static NativeTensorRtTensorInfo GetEngineIOTensorInfo(TensorRtApiLine line, SafeTensorRtObjectHandle engine, int index)
    {
        NativeTensorRtTensorInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_io_tensor_info(engine, index, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_io_tensor_info(engine, index, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_io_tensor_info(engine, index, out info),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return info;
    }

    public static ulong GetEngineDeviceMemorySize(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        UIntPtr size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_device_memory_size(engine, out size),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_device_memory_size(engine, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_device_memory_size(engine, out size),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return size.ToUInt64();
    }

    public static ulong GetEngineDeviceMemorySizeForProfile(TensorRtApiLine line, SafeTensorRtObjectHandle engine, int profileIndex)
    {
        UIntPtr size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_device_memory_size_for_profile(engine, profileIndex, out size),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_device_memory_size_for_profile(engine, profileIndex, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_device_memory_size_for_profile(engine, profileIndex, out size),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return size.ToUInt64();
    }

    public static ulong GetEngineDeviceMemorySizeV2(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        UIntPtr size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_device_memory_size_v2(engine, out size),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_device_memory_size_v2(engine, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_device_memory_size_v2(engine, out size),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return size.ToUInt64();
    }

    public static ulong GetEngineDeviceMemorySizeForProfileV2(TensorRtApiLine line, SafeTensorRtObjectHandle engine, int profileIndex)
    {
        UIntPtr size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_device_memory_size_for_profile_v2(engine, profileIndex, out size),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_device_memory_size_for_profile_v2(engine, profileIndex, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_device_memory_size_for_profile_v2(engine, profileIndex, out size),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return size.ToUInt64();
    }

    public static int GetEngineAuxiliaryStreamCount(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_nb_aux_streams(engine, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_nb_aux_streams(engine, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_nb_aux_streams(engine, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static bool IsEngineDebugTensor(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
        }

        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int isDebugTensor;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_is_debug_tensor(engine, tensorNameUtf8.Pointer, out isDebugTensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_is_debug_tensor(engine, tensorNameUtf8.Pointer, out isDebugTensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_is_debug_tensor(engine, tensorNameUtf8.Pointer, out isDebugTensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return isDebugTensor != 0;
    }

    public static int GetEngineOptimizationProfileCount(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_optimization_profile_count(engine, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_optimization_profile_count(engine, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_optimization_profile_count(engine, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static string GetEngineIOTensorName(TensorRtApiLine line, SafeTensorRtObjectHandle engine, int index)
    {
        BridgeStatusCode status = GetEngineIOTensorNameNative(line, engine, index, IntPtr.Zero, UIntPtr.Zero, out UIntPtr requiredSize);
        NativeStatus.ThrowIfFailed(status);

        ulong required = requiredSize.ToUInt64();
        if (required == 0)
        {
            return string.Empty;
        }

        if (required > int.MaxValue)
        {
            throw new BridgeProbeException(BridgeStatusCode.BufferTooSmall, BridgeErrorCategory.TensorRt, "Engine tensor name is too large for the managed buffer.");
        }

        IntPtr buffer = Marshal.AllocHGlobal(checked((int)required));
        try
        {
            status = GetEngineIOTensorNameNative(line, engine, index, buffer, requiredSize, out _);
            NativeStatus.ThrowIfFailed(status);
            return Utf8Interop.ReadString(buffer);
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }

    public static int GetEngineTensorIndex(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
        }

        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int index;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_index(engine, tensorNameUtf8.Pointer, out index),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_index(engine, tensorNameUtf8.Pointer, out index),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_index(engine, tensorNameUtf8.Pointer, out index),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return index;
    }

    public static TensorRtDataType GetEngineTensorDataType(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
        }

        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int dataType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_data_type(engine, tensorNameUtf8.Pointer, out dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_data_type(engine, tensorNameUtf8.Pointer, out dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_data_type(engine, tensorNameUtf8.Pointer, out dataType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static TensorRtDims GetEngineTensorShape(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
        }

        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        NativeTensorRtDims shape;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_shape(engine, tensorNameUtf8.Pointer, out shape),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_shape(engine, tensorNameUtf8.Pointer, out shape),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_shape(engine, tensorNameUtf8.Pointer, out shape),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(shape);
    }

    public static TensorRtIOMode GetEngineTensorIOMode(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
        }

        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int ioMode;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_io_mode(engine, tensorNameUtf8.Pointer, out ioMode),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_io_mode(engine, tensorNameUtf8.Pointer, out ioMode),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_io_mode(engine, tensorNameUtf8.Pointer, out ioMode),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtIOMode)ioMode;
    }

    private static BridgeStatusCode GetEngineIOTensorNameNative(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle engine,
        int index,
        IntPtr outputBuffer,
        UIntPtr outputBufferSize,
        out UIntPtr requiredSize)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_io_tensor_name(engine, index, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_io_tensor_name(engine, index, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_io_tensor_name(engine, index, outputBuffer, outputBufferSize, out requiredSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
    }

}
