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

    public static long GetEngineMinimumWeightStreamingBudget(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        if (line != TensorRtApiLine.TensorRt10)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "The legacy minimum weight-streaming budget query is exposed by this bridge only for TensorRT 10.");
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt10_engine_get_minimum_weight_streaming_budget(engine, out long budget);
        NativeStatus.ThrowIfFailed(status);
        return budget;
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
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_hardware_compatibility_level(engine, out level),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_hardware_compatibility_level(engine, out level),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_hardware_compatibility_level(engine, out level),
            _ => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Engine hardware compatibility level is available through this bridge for TensorRT 8, 10, and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtHardwareCompatibilityLevel)level;
    }

    public static bool HasEngineImplicitBatchDimensionCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        int hasImplicitBatch;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_cuda_engine_has_implicit_batch_dimension(engine, out hasImplicitBatch),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_cuda_engine_has_implicit_batch_dimension(engine, out hasImplicitBatch),
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Engine implicit-batch compatibility query is available through this bridge for TensorRT 8 and TensorRT 10."),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return hasImplicitBatch != 0;
    }

}
