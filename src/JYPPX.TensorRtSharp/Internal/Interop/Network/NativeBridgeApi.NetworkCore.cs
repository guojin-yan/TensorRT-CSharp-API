using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddNetworkInput(TensorRtApiLine line, SafeTensorRtObjectHandle network, string name, TensorRtDataType dataType, TensorRtDims dims)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(name));
        }

        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name);
        NativeTensorRtDims nativeDims = dims.ToNative();
        SafeTensorRtObjectHandle tensor;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_input(network, nameUtf8.Pointer, (int)dataType, ref nativeDims, out tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_input(network, nameUtf8.Pointer, (int)dataType, ref nativeDims, out tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_input(network, nameUtf8.Pointer, (int)dataType, ref nativeDims, out tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return tensor;
    }

    public static void MarkNetworkOutput(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_mark_output(network, tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_mark_output(network, tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_mark_output(network, tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void UnmarkNetworkOutput(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_unmark_output(network, tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_unmark_output(network, tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_unmark_output(network, tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetNetworkInputCount(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_input_count(network, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_input_count(network, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_input_count(network, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int GetNetworkOutputCount(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_output_count(network, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_output_count(network, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_output_count(network, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int GetNetworkLayerCount(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_layer_count(network, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_layer_count(network, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_layer_count(network, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static SafeTensorRtObjectHandle GetNetworkLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, int index)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_layer(network, index, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_layer(network, index, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_layer(network, index, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static string GetNetworkName(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        BridgeStatusCode status = GetNetworkNameNative(line, network, IntPtr.Zero, UIntPtr.Zero, out UIntPtr requiredSize);
        NativeStatus.ThrowIfFailed(status);

        ulong required = requiredSize.ToUInt64();
        if (required == 0)
        {
            return string.Empty;
        }

        if (required > int.MaxValue)
        {
            throw new BridgeProbeException(BridgeStatusCode.BufferTooSmall, BridgeErrorCategory.TensorRt, "Network name is too large for the managed buffer.");
        }

        IntPtr buffer = Marshal.AllocHGlobal(checked((int)required));
        try
        {
            status = GetNetworkNameNative(line, network, buffer, requiredSize, out _);
            NativeStatus.ThrowIfFailed(status);
            return Utf8Interop.ReadString(buffer);
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }

    public static void SetNetworkName(TensorRtApiLine line, SafeTensorRtObjectHandle network, string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Network name must not be null or empty.", nameof(name));
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_set_name(network, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_set_name(network, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_set_name(network, nameUtf8.Pointer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtNetworkDefinitionCreationFlags GetNetworkFlags(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        uint flags;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_flags(network, out flags),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_flags(network, out flags),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_flags(network, out flags),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtNetworkDefinitionCreationFlags)flags;
    }

    public static bool HasImplicitBatchDimension(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        int hasImplicitBatchDimension;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_has_implicit_batch_dimension(network, out hasImplicitBatchDimension),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_has_implicit_batch_dimension(network, out hasImplicitBatchDimension),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_has_implicit_batch_dimension(network, out hasImplicitBatchDimension),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return hasImplicitBatchDimension != 0;
    }

    public static bool GetNetworkFlag(TensorRtApiLine line, SafeTensorRtObjectHandle network, TensorRtNetworkDefinitionCreationFlags flag)
    {
        int flagIndex = GetSingleBitFlagIndex((uint)flag, nameof(flag));
        int enabled;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_flag(network, flagIndex, out enabled),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_flag(network, flagIndex, out enabled),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_flag(network, flagIndex, out enabled),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    public static SafeTensorRtObjectHandle GetNetworkInput(TensorRtApiLine line, SafeTensorRtObjectHandle network, int index)
    {
        SafeTensorRtObjectHandle tensor;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_input(network, index, out tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_input(network, index, out tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_input(network, index, out tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return tensor;
    }

    public static SafeTensorRtObjectHandle GetNetworkOutput(TensorRtApiLine line, SafeTensorRtObjectHandle network, int index)
    {
        SafeTensorRtObjectHandle tensor;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_output(network, index, out tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_output(network, index, out tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_output(network, index, out tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return tensor;
    }

    private static BridgeStatusCode GetNetworkNameNative(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        IntPtr outputBuffer,
        UIntPtr outputBufferSize,
        out UIntPtr requiredSize)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_name(network, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_name(network, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_name(network, outputBuffer, outputBufferSize, out requiredSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
    }

}
