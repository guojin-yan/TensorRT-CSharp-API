using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle SerializeEngine(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        BridgeStatusCode status;
        SafeTensorRtObjectHandle hostMemory;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_engine_serialize(engine, out hostMemory);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_engine_serialize(engine, out hostMemory);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_engine_serialize(engine, out hostMemory);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return hostMemory;
    }

    public static SafeTensorRtObjectHandle CreateSerializationConfig(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        SafeTensorRtObjectHandle config;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_create_serialization_config(engine, out config),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_create_serialization_config(engine, out config),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Serialization config is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return config;
    }

    public static SafeTensorRtObjectHandle SerializeEngineWithConfig(TensorRtApiLine line, SafeTensorRtObjectHandle engine, SafeTensorRtObjectHandle config)
    {
        SafeTensorRtObjectHandle hostMemory;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_serialize_with_config(engine, config, out hostMemory),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_serialize_with_config(engine, config, out hostMemory),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Serialization config is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return hostMemory;
    }

    public static bool SetSerializationConfigFlags(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtSerializationFlags flags)
    {
        int set;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_serialization_config_set_flags(config, (uint)flags, out set),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_serialization_config_set_flags(config, (uint)flags, out set),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Serialization config is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static TensorRtSerializationFlags GetSerializationConfigFlags(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        uint flags;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_serialization_config_get_flags(config, out flags),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_serialization_config_get_flags(config, out flags),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Serialization config is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtSerializationFlags)flags;
    }

    public static bool SetSerializationConfigFlag(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtSerializationFlag flag)
    {
        int set;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_serialization_config_set_flag(config, (int)flag, out set),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_serialization_config_set_flag(config, (int)flag, out set),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Serialization config is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static bool ClearSerializationConfigFlag(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtSerializationFlag flag)
    {
        int cleared;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_serialization_config_clear_flag(config, (int)flag, out cleared),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_serialization_config_clear_flag(config, (int)flag, out cleared),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Serialization config is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static bool GetSerializationConfigFlag(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtSerializationFlag flag)
    {
        int enabled;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_serialization_config_get_flag(config, (int)flag, out enabled),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_serialization_config_get_flag(config, (int)flag, out enabled),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Serialization config is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

}
