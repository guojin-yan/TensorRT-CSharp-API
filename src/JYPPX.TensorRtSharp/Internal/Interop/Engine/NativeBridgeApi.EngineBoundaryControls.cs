using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
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

    public static TensorRtErrorRecorderSnapshot GetEngineInspectorErrorRecorderSnapshot(TensorRtApiLine line, SafeTensorRtObjectHandle inspector)
    {
        NativeTensorRtErrorRecorderSnapshotInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_inspector_get_error_recorder_snapshot_info(inspector, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_get_error_recorder_snapshot_info(inspector, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_get_error_recorder_snapshot_info(inspector, out info),
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
                    TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_inspector_get_error_recorder_error(inspector, index, out error),
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_get_error_recorder_error(inspector, index, out error),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_get_error_recorder_error(inspector, index, out error),
                    _ => throw UnsupportedLine()
                };
                NativeStatus.ThrowIfFailed(errorStatus);
                return error;
            });
    }

}
