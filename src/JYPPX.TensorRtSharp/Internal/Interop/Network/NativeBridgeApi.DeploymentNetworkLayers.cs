using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddScatterLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle data,
        SafeTensorRtObjectHandle indices,
        SafeTensorRtObjectHandle updates,
        TensorRtScatterMode mode)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_scatter(network, data, indices, updates, (int)mode, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_scatter(network, data, indices, updates, (int)mode, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_scatter(network, data, indices, updates, (int)mode, out layer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddOneHotLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle indices,
        SafeTensorRtObjectHandle values,
        SafeTensorRtObjectHandle depth,
        int axis)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_network_add_one_hot(network, indices, values, depth, axis, out layer);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_network_add_one_hot(network, indices, values, depth, axis, out layer);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_network_add_one_hot(network, indices, values, depth, axis, out layer);
                break;
            default:
                throw UnsupportedLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddCumulativeLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        SafeTensorRtObjectHandle axis,
        TensorRtCumulativeOperation operation,
        bool exclusive,
        bool reverse)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_cumulative(network, input, axis, (int)operation, exclusive ? 1 : 0, reverse ? 1 : 0, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_cumulative(network, input, axis, (int)operation, exclusive ? 1 : 0, reverse ? 1 : 0, out layer),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "AddCumulativeLayer is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddAssertionLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle condition, string message)
    {
        using Utf8Interop.Utf8StringScope messageUtf8 = Utf8Interop.ToNativeString(message ?? string.Empty);
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_network_add_assertion(network, condition, messageUtf8.Pointer, out layer);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_network_add_assertion(network, condition, messageUtf8.Pointer, out layer);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_network_add_assertion(network, condition, messageUtf8.Pointer, out layer);
                break;
            default:
                throw UnsupportedLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddGridSampleLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, SafeTensorRtObjectHandle grid)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_network_add_grid_sample(network, input, grid, out layer);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_network_add_grid_sample(network, input, grid, out layer);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_network_add_grid_sample(network, input, grid, out layer);
                break;
            default:
                throw UnsupportedLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddNormalizationV2Layer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        SafeTensorRtObjectHandle scale,
        SafeTensorRtObjectHandle bias,
        uint axes)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_network_add_normalization_v2(network, input, scale, bias, axes, out layer);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_network_add_normalization_v2(network, input, scale, bias, axes, out layer);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_network_add_normalization_v2(network, input, scale, bias, axes, out layer);
                break;
            default:
                throw UnsupportedLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddSqueezeLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, SafeTensorRtObjectHandle axes)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_squeeze(network, input, axes, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_squeeze(network, input, axes, out layer),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(AddSqueezeLayer)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddUnsqueezeLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, SafeTensorRtObjectHandle axes)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_unsqueeze(network, input, axes, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_unsqueeze(network, input, axes, out layer),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(AddUnsqueezeLayer)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddDynamicQuantizeV2Layer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        TensorRtDims blockShape,
        TensorRtDataType outputType,
        TensorRtDataType scaleType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(AddDynamicQuantizeV2Layer));
        if (blockShape == null)
        {
            throw new ArgumentNullException(nameof(blockShape));
        }

        NativeTensorRtDims nativeBlockShape = blockShape.ToNative();
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_add_dynamic_quantize_v2(
            network,
            input,
            ref nativeBlockShape,
            (int)outputType,
            (int)scaleType,
            out SafeTensorRtObjectHandle layer);
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddDynamicQuantizeLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        int axis,
        int blockSize,
        TensorRtDataType outputType,
        TensorRtDataType scaleType)
    {
        if (blockSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(blockSize), "Dynamic quantize block size must be positive.");
        }

        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_dynamic_quantize(network, input, axis, blockSize, (int)outputType, (int)scaleType, out layer),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "AddDynamicQuantizeLayer is available for the TensorRT 10 adapter. Use AddDynamicQuantizeV2Layer for TensorRT 11."),
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "TensorRT 11 uses AddDynamicQuantizeV2Layer with a block shape instead of axis/blockSize."),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static bool MarkWeightsRefittable(TensorRtApiLine line, SafeTensorRtObjectHandle network, string weightsName)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(RequireNonEmpty(weightsName, nameof(weightsName)));
        int marked;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_mark_weights_refittable(network, nameUtf8.Pointer, out marked),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_mark_weights_refittable(network, nameUtf8.Pointer, out marked),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(MarkWeightsRefittable)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return marked != 0;
    }

    public static bool UnmarkWeightsRefittable(TensorRtApiLine line, SafeTensorRtObjectHandle network, string weightsName)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(RequireNonEmpty(weightsName, nameof(weightsName)));
        int unmarked;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_unmark_weights_refittable(network, nameUtf8.Pointer, out unmarked),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_unmark_weights_refittable(network, nameUtf8.Pointer, out unmarked),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(UnmarkWeightsRefittable)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return unmarked != 0;
    }

    public static bool AreWeightsMarkedRefittable(TensorRtApiLine line, SafeTensorRtObjectHandle network, string weightsName)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(RequireNonEmpty(weightsName, nameof(weightsName)));
        int marked;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_are_weights_marked_refittable(network, nameUtf8.Pointer, out marked),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_are_weights_marked_refittable(network, nameUtf8.Pointer, out marked),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(AreWeightsMarkedRefittable)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return marked != 0;
    }

    public static bool SetWeightsName(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle constantLayer, string weightsName)
    {
        if (constantLayer == null)
        {
            throw new ArgumentNullException(nameof(constantLayer));
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(RequireNonEmpty(weightsName, nameof(weightsName)));
        int set;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_set_weights_name(network, constantLayer, nameUtf8.Pointer, out set),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_set_weights_name(network, constantLayer, nameUtf8.Pointer, out set),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_set_weights_name(network, constantLayer, nameUtf8.Pointer, out set),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static SafeTensorRtObjectHandle AddCastLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, TensorRtDataType toType)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_network_add_cast(network, input, (int)toType, out layer);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_network_add_cast(network, input, (int)toType, out layer);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_network_add_cast(network, input, (int)toType, out layer);
                break;
            default:
                throw UnsupportedLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddNonZeroLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, TensorRtDataType indicesType)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                if (indicesType != TensorRtDataType.Int32)
                {
                    throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "TensorRT 8 addNonZero does not expose an indices type parameter; use Int32.");
                }

                status = NativeMethodsTensorRt.jyppx_trt8_network_add_non_zero(network, input, out layer);
                break;
            case TensorRtApiLine.TensorRt10:
                if (indicesType != TensorRtDataType.Int32)
                {
                    throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "TensorRT 10 addNonZero does not expose an indices type parameter; use Int32.");
                }

                status = NativeMethodsTensorRt.jyppx_trt10_network_add_non_zero(network, input, out layer);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_network_add_non_zero(network, input, (int)indicesType, out layer);
                break;
            default:
                throw UnsupportedLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddRaggedSoftMaxLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, SafeTensorRtObjectHandle bounds)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_ragged_softmax(network, input, bounds, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_ragged_softmax(network, input, bounds, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_ragged_softmax(network, input, bounds, out layer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddNmsLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle boxes, SafeTensorRtObjectHandle scores, SafeTensorRtObjectHandle maxOutputBoxesPerClass, TensorRtDataType indicesType)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_network_add_nms(network, boxes, scores, maxOutputBoxesPerClass, (int)indicesType, out layer);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_network_add_nms(network, boxes, scores, maxOutputBoxesPerClass, (int)indicesType, out layer);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_network_add_nms(network, boxes, scores, maxOutputBoxesPerClass, (int)indicesType, out layer);
                break;
            default:
                throw UnsupportedLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddReverseSequenceLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, SafeTensorRtObjectHandle sequenceLengths)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_reverse_sequence(network, input, sequenceLengths, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_reverse_sequence(network, input, sequenceLengths, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_reverse_sequence(network, input, sequenceLengths, out layer),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddEinsumLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle[] inputs, string equation)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (inputs.Length == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputs), "Einsum requires at least one input tensor.");
        }

        IntPtr[] inputHandles = new IntPtr[inputs.Length];
        for (int index = 0; index < inputs.Length; index++)
        {
            if (inputs[index] == null || inputs[index].IsInvalid)
            {
                throw new ArgumentException("Input tensor handles must not be null or invalid.", nameof(inputs));
            }

            inputHandles[index] = inputs[index].DangerousGetHandle();
        }

        using Utf8Interop.Utf8StringScope equationUtf8 = Utf8Interop.ToNativeString(RequireNonEmpty(equation, nameof(equation)));
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_network_add_einsum(network, inputHandles, inputHandles.Length, equationUtf8.Pointer, out layer);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_network_add_einsum(network, inputHandles, inputHandles.Length, equationUtf8.Pointer, out layer);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_network_add_einsum(network, inputHandles, inputHandles.Length, equationUtf8.Pointer, out layer);
                break;
            default:
                throw UnsupportedLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

}
