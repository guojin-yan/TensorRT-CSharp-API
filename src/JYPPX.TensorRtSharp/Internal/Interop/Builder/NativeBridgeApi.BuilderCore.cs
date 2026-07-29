using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateBuilder(TensorRtApiLine line, SafeTensorRtObjectHandle logger)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.BuilderCreate(logger, out SafeTensorRtObjectHandle builder);
        NativeStatus.ThrowIfFailed(status);
        return builder;
    }

    public static bool BuilderPlatformHasFastFp16(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int supported;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_platform_has_fast_fp16(builder, out supported),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_platform_has_fast_fp16(builder, out supported),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_platform_has_fast_fp16(builder, out supported),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return supported != 0;
    }

    public static bool BuilderPlatformHasFastInt8(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int supported;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_platform_has_fast_int8(builder, out supported),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_platform_has_fast_int8(builder, out supported),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_platform_has_fast_int8(builder, out supported),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return supported != 0;
    }

    public static bool BuilderPlatformHasTf32(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int supported;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_platform_has_tf32(builder, out supported),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_platform_has_tf32(builder, out supported),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_platform_has_tf32(builder, out supported),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return supported != 0;
    }

    public static int GetBuilderDlaCoreCount(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_get_dla_core_count(builder, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_get_dla_core_count(builder, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_get_dla_core_count(builder, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static SafeTensorRtObjectHandle CreateBuilderConfig(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.ConfigCreate(builder, out SafeTensorRtObjectHandle config);
        NativeStatus.ThrowIfFailed(status);
        return config;
    }

}
