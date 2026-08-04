using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static int GetRefitterMissingCount(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_missing_count(refitter, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_missing_count(refitter, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_get_missing_count(refitter, out count),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int GetRefitterAllCount(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_all_count(refitter, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_all_count(refitter, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_get_all_count(refitter, out count),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static NativeTensorRtRefitEntryInfo[] GetRefitterMissingEntries(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        return GetRefitterEntries(
            line,
            refitter,
            GetRefitterMissingCount(line, refitter),
            NativeMethodsTensorRt.jyppx_trt8_refitter_get_missing_entries,
            NativeMethodsTensorRt.jyppx_trt10_refitter_get_missing_entries,
            NativeMethodsTensorRt.jyppx_trt11_refitter_get_missing_entries);
    }

    public static NativeTensorRtRefitEntryInfo[] GetRefitterAllEntries(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        return GetRefitterEntries(
            line,
            refitter,
            GetRefitterAllCount(line, refitter),
            NativeMethodsTensorRt.jyppx_trt8_refitter_get_all_entries,
            NativeMethodsTensorRt.jyppx_trt10_refitter_get_all_entries,
            NativeMethodsTensorRt.jyppx_trt11_refitter_get_all_entries);
    }

    public static bool SetRefitterWeights(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, string layerName, TensorRtWeightsRole role, TensorRtRefitWeightsBuffer weights)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        if (string.IsNullOrWhiteSpace(layerName))
        {
            throw new ArgumentException("Layer name must not be null or empty.", nameof(layerName));
        }

        using Utf8Interop.Utf8StringScope layerNameUtf8 = Utf8Interop.ToNativeString(layerName);
        int set;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_set_weights(refitter, layerNameUtf8.Pointer, (int)role, (int)weights.DataType, weights.Pointer, weights.ElementCount, out set),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_set_weights(refitter, layerNameUtf8.Pointer, (int)role, (int)weights.DataType, weights.Pointer, weights.ElementCount, out set),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_set_weights(refitter, layerNameUtf8.Pointer, (int)role, (int)weights.DataType, weights.Pointer, weights.ElementCount, out set),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static bool RefitCudaEngine(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        int refitted;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_refit_cuda_engine(refitter, out refitted),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_refit_cuda_engine(refitter, out refitted),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_refit_cuda_engine(refitter, out refitted),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return refitted != 0;
    }

}
