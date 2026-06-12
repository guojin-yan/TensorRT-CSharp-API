using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static long GetEngineStreamableWeightsSize(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        long size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_streamable_weights_size(engine, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_streamable_weights_size(engine, out size),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Engine weight streaming is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return size;
    }

    public static bool SetEngineWeightStreamingBudgetV2(TensorRtApiLine line, SafeTensorRtObjectHandle engine, long budget)
    {
        int set;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_set_weight_streaming_budget_v2(engine, budget, out set),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_set_weight_streaming_budget_v2(engine, budget, out set),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Engine weight streaming is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static long GetEngineWeightStreamingBudgetV2(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        long budget;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_weight_streaming_budget_v2(engine, out budget),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_weight_streaming_budget_v2(engine, out budget),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Engine weight streaming is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return budget;
    }

    public static long GetEngineWeightStreamingAutomaticBudget(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        long budget;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_weight_streaming_automatic_budget(engine, out budget),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_weight_streaming_automatic_budget(engine, out budget),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Engine weight streaming is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return budget;
    }

    public static long GetEngineWeightStreamingScratchMemorySize(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        long size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_weight_streaming_scratch_memory_size(engine, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_weight_streaming_scratch_memory_size(engine, out size),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Engine weight streaming is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return size;
    }

    public static long GetEngineStat(TensorRtApiLine line, SafeTensorRtObjectHandle engine, TensorRtEngineStat stat)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetEngineStat));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_engine_get_engine_stat(engine, (int)stat, out long value);
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static TensorRtHardwareCompatibilityLevel GetEngineHardwareCompatibilityLevel(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        int level;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_hardware_compatibility_level(engine, out level),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_hardware_compatibility_level(engine, out level),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Engine hardware compatibility level is available through this bridge for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtHardwareCompatibilityLevel)level;
    }

    public static bool IsExecutionContextInputConsumedEventSet(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(IsExecutionContextInputConsumedEventSet));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_is_input_consumed_event_set(context, out int isSet);
        NativeStatus.ThrowIfFailed(status);
        return isSet != 0;
    }

    public static bool IsExecutionContextOutputTensorAddressSet(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(IsExecutionContextOutputTensorAddressSet));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_is_output_tensor_address_set(context, tensorNameUtf8.Pointer, out int isSet);
        NativeStatus.ThrowIfFailed(status);
        return isSet != 0;
    }

    public static bool HasExecutionContextOutputAllocator(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(HasExecutionContextOutputAllocator));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_has_output_allocator(context, tensorNameUtf8.Pointer, out int hasAllocator);
        NativeStatus.ThrowIfFailed(status);
        return hasAllocator != 0;
    }

    public static bool HasExecutionContextTemporaryStorageAllocator(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(HasExecutionContextTemporaryStorageAllocator));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_has_temporary_storage_allocator(context, out int hasAllocator);
        NativeStatus.ThrowIfFailed(status);
        return hasAllocator != 0;
    }

    public static void SetBuilderConfigFlags(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtBuilderFlags flags)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetBuilderConfigFlags));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_set_flags(config, (uint)flags);
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtBuilderFlags GetBuilderConfigFlags(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetBuilderConfigFlags));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_get_flags(config, out uint flags);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtBuilderFlags)flags;
    }

    public static void SetBuilderConfigDefaultDeviceType(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtDeviceType deviceType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetBuilderConfigDefaultDeviceType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_set_default_device_type(config, (int)deviceType);
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDeviceType GetBuilderConfigDefaultDeviceType(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetBuilderConfigDefaultDeviceType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_get_default_device_type(config, out int deviceType);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDeviceType)deviceType;
    }

    public static void SetBuilderConfigDlaCore(TensorRtApiLine line, SafeTensorRtObjectHandle config, int dlaCore)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetBuilderConfigDlaCore));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_set_dla_core(config, dlaCore);
        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetBuilderConfigDlaCore(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetBuilderConfigDlaCore));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_get_dla_core(config, out int dlaCore);
        NativeStatus.ThrowIfFailed(status);
        return dlaCore;
    }

    public static bool SetBuilderConfigTilingOptimizationLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtTilingOptimizationLevel level)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetBuilderConfigTilingOptimizationLevel));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_set_tiling_optimization_level(config, (int)level, out int set);
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static TensorRtTilingOptimizationLevel GetBuilderConfigTilingOptimizationLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetBuilderConfigTilingOptimizationLevel));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_get_tiling_optimization_level(config, out int level);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTilingOptimizationLevel)level;
    }

    public static bool SetBuilderConfigL2LimitForTiling(TensorRtApiLine line, SafeTensorRtObjectHandle config, long bytes)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetBuilderConfigL2LimitForTiling));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_set_l2_limit_for_tiling(config, bytes, out int set);
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static long GetBuilderConfigL2LimitForTiling(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetBuilderConfigL2LimitForTiling));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_get_l2_limit_for_tiling(config, out long bytes);
        NativeStatus.ThrowIfFailed(status);
        return bytes;
    }

    public static void SetBuilderConfigMaxTactics(TensorRtApiLine line, SafeTensorRtObjectHandle config, int maxTactics)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetBuilderConfigMaxTactics));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_set_max_nb_tactics(config, maxTactics);
        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetBuilderConfigMaxTactics(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetBuilderConfigMaxTactics));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_get_max_nb_tactics(config, out int maxTactics);
        NativeStatus.ThrowIfFailed(status);
        return maxTactics;
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
