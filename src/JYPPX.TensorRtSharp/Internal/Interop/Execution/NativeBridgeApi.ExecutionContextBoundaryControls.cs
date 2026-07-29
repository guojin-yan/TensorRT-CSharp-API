using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
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

}
