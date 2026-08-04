using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;
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

    public static void SetBuilderMaxBatchSizeCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle builder, int maxBatchSize)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "Builder max batch size is a legacy TensorRT 8 compatibility control.");
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_builder_set_max_batch_size(builder, maxBatchSize);
        NativeStatus.ThrowIfFailed(status);
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

    public static TensorRtErrorRecorderSnapshot GetBuilderErrorRecorderSnapshot(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        NativeTensorRtErrorRecorderSnapshotInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_get_error_recorder_snapshot_info(builder, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_get_error_recorder_snapshot_info(builder, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_get_error_recorder_snapshot_info(builder, out info),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);

        return ReadOwnerErrorRecorderSnapshot(
            line,
            info,
            index =>
            {
                NativeTensorRtErrorRecordInfo error;
                BridgeStatusCode errorStatus = line switch
                {
                    TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_get_error_recorder_error(builder, index, out error),
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_get_error_recorder_error(builder, index, out error),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_get_error_recorder_error(builder, index, out error),
                    _ => throw UnsupportedLine()
                };
                NativeStatus.ThrowIfFailed(errorStatus);
                return error;
            });
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

}
