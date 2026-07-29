using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static void SetRuntimeDlaCore(TensorRtApiLine line, SafeTensorRtObjectHandle runtime, int dlaCore)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_set_dla_core(runtime, dlaCore),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_set_dla_core(runtime, dlaCore),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_set_dla_core(runtime, dlaCore),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetRuntimeDlaCore(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_get_dla_core(runtime, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_get_dla_core(runtime, out value),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_get_dla_core(runtime, out value),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static int GetRuntimeDlaCoreCount(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_get_dla_core_count(runtime, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_get_dla_core_count(runtime, out value),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_get_dla_core_count(runtime, out value),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static bool SetRuntimeMaxThreads(TensorRtApiLine line, SafeTensorRtObjectHandle runtime, int maxThreads)
    {
        int set;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_set_max_threads(runtime, maxThreads, out set),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_set_max_threads(runtime, maxThreads, out set),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_set_max_threads(runtime, maxThreads, out set),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static int GetRuntimeMaxThreads(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_get_max_threads(runtime, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_get_max_threads(runtime, out value),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_get_max_threads(runtime, out value),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static void SetRuntimeTemporaryDirectory(TensorRtApiLine line, SafeTensorRtObjectHandle runtime, string directoryPath)
    {
        if (string.IsNullOrWhiteSpace(directoryPath))
        {
            throw new ArgumentException("Temporary directory must not be null or empty.", nameof(directoryPath));
        }

        using Utf8Interop.Utf8StringScope pathUtf8 = Utf8Interop.ToNativeString(directoryPath);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_set_temporary_directory(runtime, pathUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_set_temporary_directory(runtime, pathUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_set_temporary_directory(runtime, pathUtf8.Pointer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static string GetRuntimeTemporaryDirectory(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_get_temporary_directory(runtime, buffer, size, out required),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_get_temporary_directory(runtime, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_get_temporary_directory(runtime, buffer, size, out required),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            },
            "Runtime temporary directory is too large for the managed buffer.");
    }

    public static void ClearRuntimeTemporaryDirectory(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_clear_temporary_directory(runtime),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_clear_temporary_directory(runtime),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_clear_temporary_directory(runtime),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetRuntimeTempfileControlFlags(TensorRtApiLine line, SafeTensorRtObjectHandle runtime, TensorRtTempfileControlFlags flags)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_set_tempfile_control_flags(runtime, (uint)flags),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_set_tempfile_control_flags(runtime, (uint)flags),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_set_tempfile_control_flags(runtime, (uint)flags),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtTempfileControlFlags GetRuntimeTempfileControlFlags(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        uint flags;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_get_tempfile_control_flags(runtime, out flags),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_get_tempfile_control_flags(runtime, out flags),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_get_tempfile_control_flags(runtime, out flags),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTempfileControlFlags)flags;
    }

    public static void SetRuntimeEngineHostCodeAllowed(TensorRtApiLine line, SafeTensorRtObjectHandle runtime, bool allowed)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_set_engine_host_code_allowed(runtime, allowed ? 1 : 0),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_set_engine_host_code_allowed(runtime, allowed ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_set_engine_host_code_allowed(runtime, allowed ? 1 : 0),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool GetRuntimeEngineHostCodeAllowed(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        int allowed;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_get_engine_host_code_allowed(runtime, out allowed),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_get_engine_host_code_allowed(runtime, out allowed),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_get_engine_host_code_allowed(runtime, out allowed),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
        return allowed != 0;
    }

    public static bool HasRuntimeErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        int hasRecorder;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_has_error_recorder(runtime, out hasRecorder),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_has_error_recorder(runtime, out hasRecorder),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_has_error_recorder(runtime, out hasRecorder),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    public static bool HasRuntimeLogger(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        int hasLogger;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_has_logger(runtime, out hasLogger),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_has_logger(runtime, out hasLogger),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_has_logger(runtime, out hasLogger),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
        return hasLogger != 0;
    }

    public static TensorRtErrorRecorderSnapshot GetRuntimeErrorRecorderSnapshot(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        NativeTensorRtErrorRecorderSnapshotInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_get_error_recorder_snapshot_info(runtime, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_get_error_recorder_snapshot_info(runtime, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_get_error_recorder_snapshot_info(runtime, out info),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
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
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_get_error_recorder_error(runtime, index, out error),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_get_error_recorder_error(runtime, index, out error),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_get_error_recorder_error(runtime, index, out error),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };
            NativeStatus.ThrowIfFailed(status);
            records.Add(BridgeInfoMapper.ToManaged(error));
        }

        return BridgeInfoMapper.ToManaged(line, info, records);
    }

    public static void ClearRuntimeErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_clear_error_recorder(runtime),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_clear_error_recorder(runtime),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_clear_error_recorder(runtime),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static void ClearRuntimeGpuAllocator(TensorRtApiLine line, SafeTensorRtObjectHandle runtime)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_runtime_clear_gpu_allocator(runtime),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_clear_gpu_allocator(runtime),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_clear_gpu_allocator(runtime),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
        NativeStatus.ThrowIfFailed(status);
    }

}
