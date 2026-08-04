using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static void SetBuilderConfigFlags(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtBuilderFlags flags)
    {
        uint nativeFlags = TensorRtBuilderFlagMapper.ToNativeFlags(line, flags);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_flags(config, nativeFlags),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_flags(config, nativeFlags),
            TensorRtApiLine.TensorRt10 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "IBuilderConfig::setFlags is exposed by this bridge for TensorRT 8 and 11; TensorRT 10 does not provide this builder-config API."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtBuilderFlags GetBuilderConfigFlags(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        uint flags;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_flags(config, out flags),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_flags(config, out flags),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_flags(config, out flags),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return TensorRtBuilderFlagMapper.FromNativeFlags(line, flags);
    }

    public static void SetBuilderConfigDefaultDeviceType(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtDeviceType deviceType)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_default_device_type(config, (int)deviceType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_default_device_type(config, (int)deviceType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_default_device_type(config, (int)deviceType),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDeviceType GetBuilderConfigDefaultDeviceType(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int deviceType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_default_device_type(config, out deviceType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_default_device_type(config, out deviceType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_default_device_type(config, out deviceType),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDeviceType)deviceType;
    }

    public static void SetBuilderConfigDlaCore(TensorRtApiLine line, SafeTensorRtObjectHandle config, int dlaCore)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_dla_core(config, dlaCore),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_dla_core(config, dlaCore),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_dla_core(config, dlaCore),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetBuilderConfigDlaCore(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int dlaCore;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_dla_core(config, out dlaCore),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_dla_core(config, out dlaCore),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_dla_core(config, out dlaCore),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return dlaCore;
    }

    public static bool SetBuilderConfigTilingOptimizationLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtTilingOptimizationLevel level)
    {
        int set;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_tiling_optimization_level(config, (int)level, out set),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_tiling_optimization_level(config, (int)level, out set),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Builder config tiling optimization controls are available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static TensorRtTilingOptimizationLevel GetBuilderConfigTilingOptimizationLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int level;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_tiling_optimization_level(config, out level),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_tiling_optimization_level(config, out level),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Builder config tiling optimization controls are available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTilingOptimizationLevel)level;
    }

    public static bool SetBuilderConfigL2LimitForTiling(TensorRtApiLine line, SafeTensorRtObjectHandle config, long bytes)
    {
        int set;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_l2_limit_for_tiling(config, bytes, out set),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_l2_limit_for_tiling(config, bytes, out set),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Builder config L2 tiling controls are available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static long GetBuilderConfigL2LimitForTiling(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        long bytes;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_l2_limit_for_tiling(config, out bytes),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_l2_limit_for_tiling(config, out bytes),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Builder config L2 tiling controls are available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return bytes;
    }

    public static void SetBuilderConfigMaxTactics(TensorRtApiLine line, SafeTensorRtObjectHandle config, int maxTactics)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_max_nb_tactics(config, maxTactics),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_max_nb_tactics(config, maxTactics),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Builder config max tactics controls are available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetBuilderConfigMaxTactics(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int maxTactics;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_max_nb_tactics(config, out maxTactics),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_max_nb_tactics(config, out maxTactics),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Builder config max tactics controls are available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return maxTactics;
    }

    public static void SetBuilderConfigQuantizationFlags(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtQuantizationFlags flags)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_quantization_flags(config, (uint)flags),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_quantization_flags(config, (uint)flags),
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Builder config quantization flags are exposed by this bridge for TensorRT 8 and 10; TensorRT 11 removed this deprecated API."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtQuantizationFlags GetBuilderConfigQuantizationFlags(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        uint flags;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_quantization_flags(config, out flags),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_quantization_flags(config, out flags),
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Builder config quantization flags are exposed by this bridge for TensorRT 8 and 10; TensorRT 11 removed this deprecated API."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtQuantizationFlags)flags;
    }

    public static void ClearBuilderConfigQuantizationFlag(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtQuantizationFlag flag)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_clear_quantization_flag(config, (int)flag),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_clear_quantization_flag(config, (int)flag),
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Builder config quantization flags are exposed by this bridge for TensorRT 8 and 10; TensorRT 11 removed this deprecated API."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetBuilderConfigQuantizationFlag(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtQuantizationFlag flag)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_quantization_flag(config, (int)flag),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_quantization_flag(config, (int)flag),
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Builder config quantization flags are exposed by this bridge for TensorRT 8 and 10; TensorRT 11 removed this deprecated API."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool GetBuilderConfigQuantizationFlag(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtQuantizationFlag flag)
    {
        int enabled;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_quantization_flag(config, (int)flag, out enabled),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_quantization_flag(config, (int)flag, out enabled),
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Builder config quantization flags are exposed by this bridge for TensorRT 8 and 10; TensorRT 11 removed this deprecated API."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    public static bool SetBuilderConfigRemoteAutoTuningConfig(TensorRtApiLine line, SafeTensorRtObjectHandle config, string configText)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetBuilderConfigRemoteAutoTuningConfig));
        using Utf8Interop.Utf8StringScope configUtf8 = Utf8Interop.ToNativeString(configText ?? string.Empty);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_set_remote_auto_tuning_config(config, configUtf8.Pointer, out int set);
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static string GetBuilderConfigRemoteAutoTuningConfig(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetBuilderConfigRemoteAutoTuningConfig));
        return ReadUtf8Buffer((byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_remote_auto_tuning_config(config, buffer, size, out required), "Remote auto-tuning config is too large for the managed buffer.");
    }
}
