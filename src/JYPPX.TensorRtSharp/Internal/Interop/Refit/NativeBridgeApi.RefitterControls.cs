using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool RefitCudaEngineAsync(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, SafeCudaStreamHandle stream)
    {
        int refitted;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_refit_cuda_engine_async(refitter, stream, out refitted),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_refit_cuda_engine_async(refitter, stream, out refitted),
            _ => throw UnsupportedRefitterFeature(nameof(RefitCudaEngineAsync), "TensorRT 10 and 11")
        };
        NativeStatus.ThrowIfFailed(status);
        return refitted != 0;
    }

    public static bool SetRefitterMaxThreads(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, int maxThreads)
    {
        int set;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_set_max_threads(refitter, maxThreads, out set),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_set_max_threads(refitter, maxThreads, out set),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_set_max_threads(refitter, maxThreads, out set),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static int GetRefitterMaxThreads(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        int maxThreads;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_max_threads(refitter, out maxThreads),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_max_threads(refitter, out maxThreads),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_get_max_threads(refitter, out maxThreads),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return maxThreads;
    }

    public static void SetRefitterWeightsValidation(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, bool enabled)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_set_weights_validation(refitter, enabled ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_set_weights_validation(refitter, enabled ? 1 : 0),
            _ => throw UnsupportedRefitterFeature(nameof(SetRefitterWeightsValidation), "TensorRT 10 and 11")
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool GetRefitterWeightsValidation(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        int enabled;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_weights_validation(refitter, out enabled),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_get_weights_validation(refitter, out enabled),
            _ => throw UnsupportedRefitterFeature(nameof(GetRefitterWeightsValidation), "TensorRT 10 and 11")
        };
        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    public static bool UnsetRefitterNamedWeights(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, string weightsName)
    {
        ValidateTensorName(weightsName);
        using Utf8Interop.Utf8StringScope weightsNameUtf8 = Utf8Interop.ToNativeString(weightsName);
        int unset;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_unset_named_weights(refitter, weightsNameUtf8.Pointer, out unset),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_unset_named_weights(refitter, weightsNameUtf8.Pointer, out unset),
            _ => throw UnsupportedRefitterFeature(nameof(UnsetRefitterNamedWeights), "TensorRT 10 and 11")
        };
        NativeStatus.ThrowIfFailed(status);
        return unset != 0;
    }

    public static TensorRtTensorLocation GetRefitterWeightsLocation(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, string weightsName)
    {
        ValidateTensorName(weightsName);
        using Utf8Interop.Utf8StringScope weightsNameUtf8 = Utf8Interop.ToNativeString(weightsName);
        int location;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_weights_location(refitter, weightsNameUtf8.Pointer, out location),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_get_weights_location(refitter, weightsNameUtf8.Pointer, out location),
            _ => throw UnsupportedRefitterFeature(nameof(GetRefitterWeightsLocation), "TensorRT 10 and 11")
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTensorLocation)location;
    }

    public static NativeTensorRtWeightsInfo GetRefitterNamedWeightsInfo(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, string weightsName)
    {
        ValidateTensorName(weightsName);
        using Utf8Interop.Utf8StringScope weightsNameUtf8 = Utf8Interop.ToNativeString(weightsName);
        NativeTensorRtWeightsInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_named_weights_info(refitter, weightsNameUtf8.Pointer, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_get_named_weights_info(refitter, weightsNameUtf8.Pointer, out info),
            _ => throw UnsupportedRefitterFeature(nameof(GetRefitterNamedWeightsInfo), "TensorRT 10 and 11")
        };
        NativeStatus.ThrowIfFailed(status);
        return info;
    }

    public static NativeTensorRtWeightsInfo GetRefitterWeightsPrototypeInfo(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, string weightsName)
    {
        ValidateTensorName(weightsName);
        using Utf8Interop.Utf8StringScope weightsNameUtf8 = Utf8Interop.ToNativeString(weightsName);
        NativeTensorRtWeightsInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_weights_prototype_info(refitter, weightsNameUtf8.Pointer, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_get_weights_prototype_info(refitter, weightsNameUtf8.Pointer, out info),
            _ => throw UnsupportedRefitterFeature(nameof(GetRefitterWeightsPrototypeInfo), "TensorRT 10 and 11")
        };
        NativeStatus.ThrowIfFailed(status);
        return info;
    }

    public static bool HasRefitterErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        int hasRecorder;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_has_error_recorder(refitter, out hasRecorder),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_has_error_recorder(refitter, out hasRecorder),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_has_error_recorder(refitter, out hasRecorder),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    public static TensorRtErrorRecorderSnapshot GetRefitterErrorRecorderSnapshot(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        NativeTensorRtErrorRecorderSnapshotInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_error_recorder_snapshot_info(refitter, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_error_recorder_snapshot_info(refitter, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_get_error_recorder_snapshot_info(refitter, out info),
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
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_error_recorder_error(refitter, index, out error),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_error_recorder_error(refitter, index, out error),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_get_error_recorder_error(refitter, index, out error),
                _ => throw UnsupportedLine()
            };
            NativeStatus.ThrowIfFailed(status);
            records.Add(BridgeInfoMapper.ToManaged(error));
        }

        return BridgeInfoMapper.ToManaged(line, info, records);
    }

    public static void ClearRefitterErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_clear_error_recorder(refitter),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_clear_error_recorder(refitter),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_clear_error_recorder(refitter),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool HasRefitterLogger(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        int hasLogger;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_has_logger(refitter, out hasLogger),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_has_logger(refitter, out hasLogger),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_has_logger(refitter, out hasLogger),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasLogger != 0;
    }

    public static bool SetRefitterDynamicRange(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, string tensorName, float minimum, float maximum)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int set;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_set_dynamic_range(refitter, tensorNameUtf8.Pointer, minimum, maximum, out set),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_set_dynamic_range(refitter, tensorNameUtf8.Pointer, minimum, maximum, out set),
            _ => throw UnsupportedRefitterFeature(nameof(SetRefitterDynamicRange), "TensorRT 8 and 10")
        };
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static float GetRefitterDynamicRangeMinimum(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        float minimum;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_dynamic_range_min(refitter, tensorNameUtf8.Pointer, out minimum),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_dynamic_range_min(refitter, tensorNameUtf8.Pointer, out minimum),
            _ => throw UnsupportedRefitterFeature(nameof(GetRefitterDynamicRangeMinimum), "TensorRT 8 and 10")
        };
        NativeStatus.ThrowIfFailed(status);
        return minimum;
    }

    public static float GetRefitterDynamicRangeMaximum(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        float maximum;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_dynamic_range_max(refitter, tensorNameUtf8.Pointer, out maximum),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_dynamic_range_max(refitter, tensorNameUtf8.Pointer, out maximum),
            _ => throw UnsupportedRefitterFeature(nameof(GetRefitterDynamicRangeMaximum), "TensorRT 8 and 10")
        };
        NativeStatus.ThrowIfFailed(status);
        return maximum;
    }

    public static int GetRefitterDynamicRangeTensorCount(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_dynamic_range_tensor_count(refitter, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_dynamic_range_tensor_count(refitter, out count),
            _ => throw UnsupportedRefitterFeature(nameof(GetRefitterDynamicRangeTensorCount), "TensorRT 8 and 10")
        };
        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static NativeTensorRtRefitEntryInfo[] GetRefitterDynamicRangeTensorEntries(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        return GetRefitterEntries(
            line,
            refitter,
            GetRefitterDynamicRangeTensorCount(line, refitter),
            NativeMethodsTensorRt.jyppx_trt8_refitter_get_dynamic_range_tensor_entries,
            NativeMethodsTensorRt.jyppx_trt10_refitter_get_dynamic_range_tensor_entries);
    }

    public static bool SetRefitterNamedWeights(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, string weightsName, TensorRtRefitWeightsBuffer weights)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        ValidateTensorName(weightsName);
        using Utf8Interop.Utf8StringScope weightsNameUtf8 = Utf8Interop.ToNativeString(weightsName);
        int set;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_set_named_weights(refitter, weightsNameUtf8.Pointer, (int)weights.DataType, weights.Pointer, weights.ElementCount, out set),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_set_named_weights(refitter, weightsNameUtf8.Pointer, (int)weights.DataType, weights.Pointer, weights.ElementCount, out set),
            _ => throw UnsupportedRefitterFeature(nameof(SetRefitterNamedWeights), "TensorRT 8 and 10")
        };
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static int GetRefitterMissingWeightsCount(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_missing_weights_count(refitter, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_missing_weights_count(refitter, out count),
            _ => throw UnsupportedRefitterFeature(nameof(GetRefitterMissingWeightsCount), "TensorRT 8 and 10")
        };
        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int GetRefitterAllWeightsCount(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_all_weights_count(refitter, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_all_weights_count(refitter, out count),
            _ => throw UnsupportedRefitterFeature(nameof(GetRefitterAllWeightsCount), "TensorRT 8 and 10")
        };
        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static NativeTensorRtRefitEntryInfo[] GetRefitterMissingWeightsEntries(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        return GetRefitterEntries(
            line,
            refitter,
            GetRefitterMissingWeightsCount(line, refitter),
            NativeMethodsTensorRt.jyppx_trt8_refitter_get_missing_weights_entries,
            NativeMethodsTensorRt.jyppx_trt10_refitter_get_missing_weights_entries);
    }

    public static NativeTensorRtRefitEntryInfo[] GetRefitterAllWeightsEntries(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        return GetRefitterEntries(
            line,
            refitter,
            GetRefitterAllWeightsCount(line, refitter),
            NativeMethodsTensorRt.jyppx_trt8_refitter_get_all_weights_entries,
            NativeMethodsTensorRt.jyppx_trt10_refitter_get_all_weights_entries);
    }

    private static BridgeProbeException UnsupportedRefitterFeature(string featureName, string supportedLines)
    {
        return new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{featureName} is available through this bridge for {supportedLines}.");
    }
}
