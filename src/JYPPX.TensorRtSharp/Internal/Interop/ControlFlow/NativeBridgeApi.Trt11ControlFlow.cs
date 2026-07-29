using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddLoop(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        SafeTensorRtObjectHandle loop;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_loop(network, out loop),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_loop(network, out loop),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_loop(network, out loop),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return loop;
    }

    public static SafeTensorRtObjectHandle AddIfConditional(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        SafeTensorRtObjectHandle conditional;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_if_conditional(network, out conditional),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_if_conditional(network, out conditional),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_if_conditional(network, out conditional),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return conditional;
    }

    public static string GetLoopName(TensorRtApiLine line, SafeTensorRtObjectHandle loop)
    {
        Utf8BufferGetter getter = line switch
        {
            TensorRtApiLine.TensorRt8 => WrapUtf8PointerGetter((IntPtr buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt8_loop_get_name(loop, buffer, size, out required)),
            TensorRtApiLine.TensorRt10 => WrapUtf8PointerGetter((IntPtr buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt10_loop_get_name(loop, buffer, size, out required)),
            TensorRtApiLine.TensorRt11 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_loop_get_name(loop, buffer, size, out required),
            _ => throw UnsupportedLine()
        };
        return ReadUtf8Buffer(getter, "Loop name is too large for the managed buffer.");
    }

    public static void SetLoopName(TensorRtApiLine line, SafeTensorRtObjectHandle loop, string name)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name ?? string.Empty);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_loop_set_name(loop, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_loop_set_name(loop, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_loop_set_name(loop, nameUtf8.Pointer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static SafeTensorRtObjectHandle AddLoopRecurrence(TensorRtApiLine line, SafeTensorRtObjectHandle loop, SafeTensorRtObjectHandle initialValue)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_loop_add_recurrence(loop, initialValue, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_loop_add_recurrence(loop, initialValue, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_loop_add_recurrence(loop, initialValue, out layer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddLoopTripLimit(TensorRtApiLine line, SafeTensorRtObjectHandle loop, SafeTensorRtObjectHandle tensor, TensorRtTripLimitKind kind)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_loop_add_trip_limit(loop, tensor, (int)kind, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_loop_add_trip_limit(loop, tensor, (int)kind, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_loop_add_trip_limit(loop, tensor, (int)kind, out layer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddLoopIterator(TensorRtApiLine line, SafeTensorRtObjectHandle loop, SafeTensorRtObjectHandle tensor, int axis, bool reverse)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_loop_add_iterator(loop, tensor, axis, reverse ? 1 : 0, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_loop_add_iterator(loop, tensor, axis, reverse ? 1 : 0, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_loop_add_iterator(loop, tensor, axis, reverse ? 1 : 0, out layer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddLoopOutput(TensorRtApiLine line, SafeTensorRtObjectHandle loop, SafeTensorRtObjectHandle tensor, TensorRtLoopOutputKind kind, int axis)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_loop_add_output(loop, tensor, (int)kind, axis, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_loop_add_output(loop, tensor, (int)kind, axis, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_loop_add_output(loop, tensor, (int)kind, axis, out layer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static string GetIfConditionalName(TensorRtApiLine line, SafeTensorRtObjectHandle conditional)
    {
        Utf8BufferGetter getter = line switch
        {
            TensorRtApiLine.TensorRt8 => WrapUtf8PointerGetter((IntPtr buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt8_if_conditional_get_name(conditional, buffer, size, out required)),
            TensorRtApiLine.TensorRt10 => WrapUtf8PointerGetter((IntPtr buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt10_if_conditional_get_name(conditional, buffer, size, out required)),
            TensorRtApiLine.TensorRt11 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_if_conditional_get_name(conditional, buffer, size, out required),
            _ => throw UnsupportedLine()
        };
        return ReadUtf8Buffer(getter, "Conditional name is too large for the managed buffer.");
    }

    public static void SetIfConditionalName(TensorRtApiLine line, SafeTensorRtObjectHandle conditional, string name)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name ?? string.Empty);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_if_conditional_set_name(conditional, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_if_conditional_set_name(conditional, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_if_conditional_set_name(conditional, nameUtf8.Pointer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static SafeTensorRtObjectHandle SetIfConditionalCondition(TensorRtApiLine line, SafeTensorRtObjectHandle conditional, SafeTensorRtObjectHandle condition)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_if_conditional_set_condition(conditional, condition, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_if_conditional_set_condition(conditional, condition, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_if_conditional_set_condition(conditional, condition, out layer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddIfConditionalInput(TensorRtApiLine line, SafeTensorRtObjectHandle conditional, SafeTensorRtObjectHandle input)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_if_conditional_add_input(conditional, input, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_if_conditional_add_input(conditional, input, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_if_conditional_add_input(conditional, input, out layer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddIfConditionalOutput(TensorRtApiLine line, SafeTensorRtObjectHandle conditional, SafeTensorRtObjectHandle trueOutput, SafeTensorRtObjectHandle falseOutput)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_if_conditional_add_output(conditional, trueOutput, falseOutput, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_if_conditional_add_output(conditional, trueOutput, falseOutput, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_if_conditional_add_output(conditional, trueOutput, falseOutput, out layer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static string GetLoopBoundaryLoopName(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        Utf8BufferGetter getter = line switch
        {
            TensorRtApiLine.TensorRt8 => WrapUtf8PointerGetter((IntPtr buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt8_loop_boundary_layer_get_loop_name(layer, buffer, size, out required)),
            TensorRtApiLine.TensorRt10 => WrapUtf8PointerGetter((IntPtr buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt10_loop_boundary_layer_get_loop_name(layer, buffer, size, out required)),
            TensorRtApiLine.TensorRt11 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_loop_boundary_layer_get_loop_name(layer, buffer, size, out required),
            _ => throw UnsupportedLine()
        };
        return ReadUtf8Buffer(getter, "Loop boundary name is too large for the managed buffer.");
    }

    public static string GetIfConditionalBoundaryName(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        Utf8BufferGetter getter = line switch
        {
            TensorRtApiLine.TensorRt8 => WrapUtf8PointerGetter((IntPtr buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt8_if_conditional_boundary_layer_get_conditional_name(layer, buffer, size, out required)),
            TensorRtApiLine.TensorRt10 => WrapUtf8PointerGetter((IntPtr buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt10_if_conditional_boundary_layer_get_conditional_name(layer, buffer, size, out required)),
            TensorRtApiLine.TensorRt11 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_if_conditional_boundary_layer_get_conditional_name(layer, buffer, size, out required),
            _ => throw UnsupportedLine()
        };
        return ReadUtf8Buffer(getter, "Conditional boundary name is too large for the managed buffer.");
    }

    public static TensorRtLoopOutputKind GetLoopOutputKind(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtLoopOutputKind)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_loop_output_layer_get_kind, NativeMethodsTensorRt.jyppx_trt10_loop_output_layer_get_kind, NativeMethodsTensorRt.jyppx_trt11_loop_output_layer_get_kind);
    }

    public static void SetLoopOutputAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int axis)
    {
        SetLayerInt(line, layer, axis, NativeMethodsTensorRt.jyppx_trt8_loop_output_layer_set_axis, NativeMethodsTensorRt.jyppx_trt10_loop_output_layer_set_axis, NativeMethodsTensorRt.jyppx_trt11_loop_output_layer_set_axis);
    }

    public static int GetLoopOutputAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_loop_output_layer_get_axis, NativeMethodsTensorRt.jyppx_trt10_loop_output_layer_get_axis, NativeMethodsTensorRt.jyppx_trt11_loop_output_layer_get_axis);
    }

    public static TensorRtTripLimitKind GetTripLimitKind(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtTripLimitKind)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_trip_limit_layer_get_kind, NativeMethodsTensorRt.jyppx_trt10_trip_limit_layer_get_kind, NativeMethodsTensorRt.jyppx_trt11_trip_limit_layer_get_kind);
    }

    public static void SetIteratorAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int axis)
    {
        SetLayerInt(line, layer, axis, NativeMethodsTensorRt.jyppx_trt8_iterator_layer_set_axis, NativeMethodsTensorRt.jyppx_trt10_iterator_layer_set_axis, NativeMethodsTensorRt.jyppx_trt11_iterator_layer_set_axis);
    }

    public static int GetIteratorAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_iterator_layer_get_axis, NativeMethodsTensorRt.jyppx_trt10_iterator_layer_get_axis, NativeMethodsTensorRt.jyppx_trt11_iterator_layer_get_axis);
    }

    public static void SetIteratorReverse(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool reverse)
    {
        SetLayerInt(line, layer, reverse ? 1 : 0, NativeMethodsTensorRt.jyppx_trt8_iterator_layer_set_reverse, NativeMethodsTensorRt.jyppx_trt10_iterator_layer_set_reverse, NativeMethodsTensorRt.jyppx_trt11_iterator_layer_set_reverse);
    }

    public static bool GetIteratorReverse(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_iterator_layer_get_reverse, NativeMethodsTensorRt.jyppx_trt10_iterator_layer_get_reverse, NativeMethodsTensorRt.jyppx_trt11_iterator_layer_get_reverse) != 0;
    }

    private delegate BridgeStatusCode Utf8PointerBufferGetter(IntPtr outputBuffer, UIntPtr outputBufferSize, out UIntPtr requiredSize);

    private static Utf8BufferGetter WrapUtf8PointerGetter(Utf8PointerBufferGetter getter)
    {
        return (byte[] outputBuffer, UIntPtr outputBufferSize, out UIntPtr requiredSize) =>
        {
            if (outputBuffer.Length == 0)
            {
                return getter(IntPtr.Zero, outputBufferSize, out requiredSize);
            }

            GCHandle pinned = GCHandle.Alloc(outputBuffer, GCHandleType.Pinned);
            try
            {
                return getter(pinned.AddrOfPinnedObject(), outputBufferSize, out requiredSize);
            }
            finally
            {
                pinned.Free();
            }
        };
    }
}
