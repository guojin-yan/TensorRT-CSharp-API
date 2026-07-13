using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static int GetBuilderMaxDlaBatchSize(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_get_max_dla_batch_size(builder, out size),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_get_max_dla_batch_size(builder, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_get_max_dla_batch_size(builder, out size),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return size;
    }

    public static int GetBuilderMaxBatchSizeCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Builder max batch size is a legacy TensorRT 8 compatibility query.");
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_builder_get_max_batch_size(builder, out int maxBatchSize);
        NativeStatus.ThrowIfFailed(status);
        return maxBatchSize;
    }

    public static bool SetBuilderMaxThreads(TensorRtApiLine line, SafeTensorRtObjectHandle builder, int maxThreads)
    {
        int set;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_set_max_threads(builder, maxThreads, out set),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_set_max_threads(builder, maxThreads, out set),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_set_max_threads(builder, maxThreads, out set),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static int GetBuilderMaxThreads(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int maxThreads;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_get_max_threads(builder, out maxThreads),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_get_max_threads(builder, out maxThreads),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_get_max_threads(builder, out maxThreads),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return maxThreads;
    }

    public static void ClearBuilderGpuAllocator(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_clear_gpu_allocator(builder),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_clear_gpu_allocator(builder),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_clear_gpu_allocator(builder),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool HasBuilderErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int hasRecorder;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_has_error_recorder(builder, out hasRecorder),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_has_error_recorder(builder, out hasRecorder),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_has_error_recorder(builder, out hasRecorder),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    public static bool HasBuilderLogger(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int hasLogger;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_has_logger(builder, out hasLogger),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_has_logger(builder, out hasLogger),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_has_logger(builder, out hasLogger),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasLogger != 0;
    }

    public static void ClearBuilderErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_clear_error_recorder(builder),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_clear_error_recorder(builder),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_clear_error_recorder(builder),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static void ResetBuilder(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ResetBuilder));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_builder_reset(builder));
    }

    public static bool IsNetworkSupported(TensorRtApiLine line, SafeTensorRtObjectHandle builder, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle config)
    {
        int supported;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_is_network_supported(builder, network, config, out supported),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_is_network_supported(builder, network, config, out supported),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_is_network_supported(builder, network, config, out supported),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return supported != 0;
    }

    public static bool HasEngineErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        int hasRecorder;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_has_error_recorder(engine, out hasRecorder),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_has_error_recorder(engine, out hasRecorder),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_has_error_recorder(engine, out hasRecorder),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    public static void ClearEngineErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_clear_error_recorder(engine),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_clear_error_recorder(engine),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_clear_error_recorder(engine),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtErrorRecorderSnapshot GetEngineErrorRecorderSnapshot(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        NativeTensorRtErrorRecorderSnapshotInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_error_recorder_snapshot_info(engine, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_error_recorder_snapshot_info(engine, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_error_recorder_snapshot_info(engine, out info),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);

        bool hasRecorder = info.HasRecorder != 0;
        int errorCount = Math.Max(0, info.ErrorCount);
        if (!hasRecorder || errorCount == 0)
        {
            return BridgeInfoMapper.ToManaged(line, info, Array.Empty<TensorRtErrorRecord>());
        }

        List<TensorRtErrorRecord> records = new List<TensorRtErrorRecord>(errorCount);
        for (int index = 0; index < errorCount; index++)
        {
            NativeTensorRtErrorRecordInfo error;
            status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_error_recorder_error(engine, index, out error),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_error_recorder_error(engine, index, out error),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_error_recorder_error(engine, index, out error),
                _ => throw UnsupportedLine()
            };
            NativeStatus.ThrowIfFailed(status);
            records.Add(BridgeInfoMapper.ToManaged(error));
        }

        return BridgeInfoMapper.ToManaged(line, info, records);
    }

    public static string GetEngineAliasedInputTensor(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetEngineAliasedInputTensor));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(tensorName);
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_engine_get_aliased_input_tensor(engine, nameUtf8.Pointer, buffer, size, out required),
            "Aliased input tensor name is too large for the managed buffer.");
    }

    public static bool HasExecutionContextErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int hasRecorder;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_has_error_recorder(context, out hasRecorder),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_has_error_recorder(context, out hasRecorder),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_has_error_recorder(context, out hasRecorder),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    public static void ClearExecutionContextErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_clear_error_recorder(context),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_clear_error_recorder(context),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_error_recorder(context),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtErrorRecorderSnapshot GetExecutionContextErrorRecorderSnapshot(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        NativeTensorRtErrorRecorderSnapshotInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_get_error_recorder_snapshot_info(context, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_error_recorder_snapshot_info(context, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_error_recorder_snapshot_info(context, out info),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);

        bool hasRecorder = info.HasRecorder != 0;
        int errorCount = Math.Max(0, info.ErrorCount);
        if (!hasRecorder || errorCount == 0)
        {
            return BridgeInfoMapper.ToManaged(line, info, Array.Empty<TensorRtErrorRecord>());
        }

        List<TensorRtErrorRecord> records = new List<TensorRtErrorRecord>(errorCount);
        for (int index = 0; index < errorCount; index++)
        {
            NativeTensorRtErrorRecordInfo error;
            status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_get_error_recorder_error(context, index, out error),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_error_recorder_error(context, index, out error),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_error_recorder_error(context, index, out error),
                _ => throw UnsupportedLine()
            };
            NativeStatus.ThrowIfFailed(status);
            records.Add(BridgeInfoMapper.ToManaged(error));
        }

        return BridgeInfoMapper.ToManaged(line, info, records);
    }

    public static bool HasNetworkErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        int hasRecorder;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_has_error_recorder(network, out hasRecorder),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_has_error_recorder(network, out hasRecorder),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_has_error_recorder(network, out hasRecorder),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    public static void ClearNetworkErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_clear_error_recorder(network),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_clear_error_recorder(network),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_clear_error_recorder(network),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static void RemoveNetworkTensor(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(RemoveNetworkTensor));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_network_remove_tensor(network, tensor));
    }

    public static SafeTensorRtObjectHandle AddTopKV2Layer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        TensorRtTopKOperation operation,
        int k,
        uint axes,
        TensorRtDataType indicesType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(AddTopKV2Layer));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_add_topk_v2(network, input, (int)operation, k, axes, (int)indicesType, out SafeTensorRtObjectHandle layer);
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }
}
