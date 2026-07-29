using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static int AddOptimizationProfile(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle profile)
    {
        int index;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_builder_config_add_optimization_profile(config, profile, out index);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_config_add_optimization_profile(config, profile, out index);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_config_add_optimization_profile(config, profile, out index);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return index;
    }

    public static void SetBuilderConfigProfileStream(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeCudaStreamHandle stream)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_profile_stream(config, stream),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_profile_stream(config, stream),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_profile_stream(config, stream),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool IsBuilderConfigProfileStreamSet(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int isSet;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_is_profile_stream_set(config, out isSet),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_is_profile_stream_set(config, out isSet),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_is_profile_stream_set(config, out isSet),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return isSet != 0;
    }

    public static int GetBuilderConfigOptimizationProfileCount(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_optimization_profile_count(config, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_optimization_profile_count(config, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_optimization_profile_count(config, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static void SetBuilderConfigCalibrationProfile(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle profile)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_calibration_profile(config, profile),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_calibration_profile(config, profile),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_calibration_profile(config, profile),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool HasBuilderConfigCalibrationProfile(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int hasProfile;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_has_calibration_profile(config, out hasProfile),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_has_calibration_profile(config, out hasProfile),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_has_calibration_profile(config, out hasProfile),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return hasProfile != 0;
    }

    public static bool HasBuilderConfigAlgorithmSelectorCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int hasSelector;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_has_algorithm_selector(config, out hasSelector),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_has_algorithm_selector(config, out hasSelector),
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "IBuilderConfig::getAlgorithmSelector presence is exposed by this bridge for TensorRT 8 and 10; TensorRT 11 callback ownership remains deferred."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return hasSelector != 0;
    }

    public static bool HasBuilderConfigInt8CalibratorCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int hasCalibrator;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_has_int8_calibrator(config, out hasCalibrator),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_has_int8_calibrator(config, out hasCalibrator),
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "IBuilderConfig::getInt8Calibrator presence is exposed by this bridge for TensorRT 8 and 10; TensorRT 11 callback ownership remains deferred."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return hasCalibrator != 0;
    }

    public static void SetBuilderConfigFlag(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtBuilderFlag flag, bool enabled)
    {
        int nativeFlag = TensorRtBuilderFlagMapper.ToNativeFlag(line, flag);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_flag(config, nativeFlag, enabled ? 1 : 0),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_flag(config, nativeFlag, enabled ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_flag(config, nativeFlag, enabled ? 1 : 0),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool GetBuilderConfigFlag(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtBuilderFlag flag)
    {
        int nativeFlag = TensorRtBuilderFlagMapper.ToNativeFlag(line, flag);
        int enabled;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_flag(config, nativeFlag, out enabled),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_flag(config, nativeFlag, out enabled),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_flag(config, nativeFlag, out enabled),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    public static void SetBuilderConfigEngineCapability(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtEngineCapability capability)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_engine_capability(config, (int)capability),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_engine_capability(config, (int)capability),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_engine_capability(config, (int)capability),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtEngineCapability GetBuilderConfigEngineCapability(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int capability;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_engine_capability(config, out capability),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_engine_capability(config, out capability),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_engine_capability(config, out capability),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtEngineCapability)capability;
    }

    public static void SetBuilderConfigPreviewFeature(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtPreviewFeature feature, bool enabled)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_preview_feature(config, (int)feature, enabled ? 1 : 0),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_preview_feature(config, (int)feature, enabled ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_preview_feature(config, (int)feature, enabled ? 1 : 0),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool GetBuilderConfigPreviewFeature(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtPreviewFeature feature)
    {
        int enabled;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_preview_feature(config, (int)feature, out enabled),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_preview_feature(config, (int)feature, out enabled),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_preview_feature(config, (int)feature, out enabled),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    public static void SetBuilderConfigHardwareCompatibilityLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtHardwareCompatibilityLevel level)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_hardware_compatibility_level(config, (int)level),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_hardware_compatibility_level(config, (int)level),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_hardware_compatibility_level(config, (int)level),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtHardwareCompatibilityLevel GetBuilderConfigHardwareCompatibilityLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int level;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_hardware_compatibility_level(config, out level),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_hardware_compatibility_level(config, out level),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_hardware_compatibility_level(config, out level),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtHardwareCompatibilityLevel)level;
    }

    public static void SetBuilderConfigRuntimePlatform(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtRuntimePlatform platform)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_runtime_platform(config, (int)platform),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_runtime_platform(config, (int)platform),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_runtime_platform(config, (int)platform),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtRuntimePlatform GetBuilderConfigRuntimePlatform(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int platform;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_runtime_platform(config, out platform),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_runtime_platform(config, out platform),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_runtime_platform(config, out platform),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtRuntimePlatform)platform;
    }

    public static void SetLayerDeviceType(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle layer, TensorRtDeviceType deviceType)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_layer_device_type(config, layer, (int)deviceType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_layer_device_type(config, layer, (int)deviceType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_layer_device_type(config, layer, (int)deviceType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDeviceType GetLayerDeviceType(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle layer)
    {
        int deviceType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_layer_device_type(config, layer, out deviceType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_layer_device_type(config, layer, out deviceType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_layer_device_type(config, layer, out deviceType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDeviceType)deviceType;
    }

    public static bool IsLayerDeviceTypeSet(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle layer)
    {
        int isSet;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_is_layer_device_type_set(config, layer, out isSet),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_is_layer_device_type_set(config, layer, out isSet),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_is_layer_device_type_set(config, layer, out isSet),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return isSet != 0;
    }

    public static void ResetLayerDeviceType(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle layer)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_reset_layer_device_type(config, layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_reset_layer_device_type(config, layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_reset_layer_device_type(config, layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetMemoryPoolLimit(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtMemoryPoolType pool, ulong bytes)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_memory_pool_limit(config, (int)pool, new UIntPtr(bytes)),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_memory_pool_limit(config, (int)pool, new UIntPtr(bytes)),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_memory_pool_limit(config, (int)pool, new UIntPtr(bytes)),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static ulong GetMemoryPoolLimit(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtMemoryPoolType pool)
    {
        UIntPtr bytes;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_memory_pool_limit(config, (int)pool, out bytes),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_memory_pool_limit(config, (int)pool, out bytes),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_memory_pool_limit(config, (int)pool, out bytes),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return bytes.ToUInt64();
    }

    public static void SetBuilderOptimizationLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config, int level)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_optimization_level(config, level),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_optimization_level(config, level),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_optimization_level(config, level),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetBuilderOptimizationLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int level;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_optimization_level(config, out level),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_optimization_level(config, out level),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_optimization_level(config, out level),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return level;
    }

    public static void SetProfilingVerbosity(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtProfilingVerbosity verbosity)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_profiling_verbosity(config, (int)verbosity),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_profiling_verbosity(config, (int)verbosity),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_profiling_verbosity(config, (int)verbosity),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtProfilingVerbosity GetProfilingVerbosity(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int verbosity;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_profiling_verbosity(config, out verbosity),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_profiling_verbosity(config, out verbosity),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_profiling_verbosity(config, out verbosity),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtProfilingVerbosity)verbosity;
    }

    public static void SetMaxAuxStreams(TensorRtApiLine line, SafeTensorRtObjectHandle config, int maxStreams)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_max_aux_streams(config, maxStreams),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_max_aux_streams(config, maxStreams),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_max_aux_streams(config, maxStreams),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetMaxAuxStreams(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int maxStreams;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_max_aux_streams(config, out maxStreams),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_max_aux_streams(config, out maxStreams),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_max_aux_streams(config, out maxStreams),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return maxStreams;
    }

    public static void SetAverageTimingIterations(TensorRtApiLine line, SafeTensorRtObjectHandle config, int iterations)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_average_timing_iterations(config, iterations),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_average_timing_iterations(config, iterations),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_average_timing_iterations(config, iterations),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetAverageTimingIterations(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int iterations;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_average_timing_iterations(config, out iterations),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_average_timing_iterations(config, out iterations),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_average_timing_iterations(config, out iterations),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return iterations;
    }

    public static ulong GetMaxWorkspaceSizeCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "IBuilderConfig::getMaxWorkspaceSize is a TensorRT 8 legacy compatibility API. Use GetMemoryPoolLimit(Workspace) for portable TensorRT 8/10/11 diagnostics.");
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_builder_config_get_max_workspace_size(config, out UIntPtr workspaceSize);
        NativeStatus.ThrowIfFailed(status);
        return workspaceSize.ToUInt64();
    }

    public static void SetMaxWorkspaceSizeCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle config, ulong workspaceSize)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "IBuilderConfig::setMaxWorkspaceSize is a TensorRT 8 legacy compatibility API. Use SetMemoryPoolLimit(Workspace) for portable TensorRT 8/10/11 configuration.");
        }

        if (UIntPtr.Size == 4 && workspaceSize > uint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(workspaceSize), "Workspace size exceeds the native size_t range.");
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_builder_config_set_max_workspace_size(config, (UIntPtr)workspaceSize);
        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetMinTimingIterationsCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "IBuilderConfig::getMinTimingIterations is a TensorRT 8 legacy compatibility API. Use GetAverageTimingIterations for portable TensorRT 8/10/11 diagnostics.");
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_builder_config_get_min_timing_iterations(config, out int iterations);
        NativeStatus.ThrowIfFailed(status);
        return iterations;
    }

    public static void SetMinTimingIterationsCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle config, int iterations)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "IBuilderConfig::setMinTimingIterations is a TensorRT 8 legacy compatibility API. Use SetAverageTimingIterations for portable TensorRT 8/10/11 configuration.");
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_builder_config_set_min_timing_iterations(config, iterations);
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetTacticSources(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtTacticSources sources)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_tactic_sources(config, (uint)sources),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_tactic_sources(config, (uint)sources),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_tactic_sources(config, (uint)sources),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtTacticSources GetTacticSources(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        uint tacticSources;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_tactic_sources(config, out tacticSources),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_tactic_sources(config, out tacticSources),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_tactic_sources(config, out tacticSources),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTacticSources)tacticSources;
    }

    private static int GetSingleBitFlagIndex(uint value, string parameterName)
    {
        if (value == 0 || (value & (value - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(parameterName, "Flag query requires exactly one bit flag.");
        }

        int index = 0;
        while ((value >>= 1) != 0)
        {
            index++;
        }

        return index;
    }

}
