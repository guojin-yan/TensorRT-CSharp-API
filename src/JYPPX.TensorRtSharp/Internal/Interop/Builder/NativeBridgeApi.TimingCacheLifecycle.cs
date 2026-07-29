using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateTimingCache(TensorRtApiLine line, SafeTensorRtObjectHandle config, byte[]? serializedCache)
    {
        GCHandle pinned = default;
        IntPtr blob = IntPtr.Zero;
        UIntPtr blobSize = UIntPtr.Zero;
        try
        {
            if (serializedCache != null && serializedCache.Length > 0)
            {
                pinned = GCHandle.Alloc(serializedCache, GCHandleType.Pinned);
                blob = pinned.AddrOfPinnedObject();
                blobSize = new UIntPtr((ulong)serializedCache.Length);
            }

            BridgeStatusCode status;
            SafeTensorRtObjectHandle cache;
            switch (line)
            {
                case TensorRtApiLine.TensorRt8:
                    status = NativeMethodsTensorRt.jyppx_trt8_builder_config_create_timing_cache(config, blob, blobSize, out cache);
                    break;
                case TensorRtApiLine.TensorRt10:
                    status = NativeMethodsTensorRt.jyppx_trt10_builder_config_create_timing_cache(config, blob, blobSize, out cache);
                    break;
                case TensorRtApiLine.TensorRt11:
                    status = NativeMethodsTensorRt.jyppx_trt11_builder_config_create_timing_cache(config, blob, blobSize, out cache);
                    break;
                default:
                    throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
            }

            NativeStatus.ThrowIfFailed(status);
            return cache;
        }
        finally
        {
            if (pinned.IsAllocated)
            {
                pinned.Free();
            }
        }
    }

    public static void SetTimingCache(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle cache, bool ignoreMismatch)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_timing_cache(config, cache, ignoreMismatch ? 1 : 0),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_timing_cache(config, cache, ignoreMismatch ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_timing_cache(config, cache, ignoreMismatch ? 1 : 0),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static SafeTensorRtObjectHandle SerializeTimingCache(TensorRtApiLine line, SafeTensorRtObjectHandle cache)
    {
        BridgeStatusCode status;
        SafeTensorRtObjectHandle hostMemory;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_timing_cache_serialize(cache, out hostMemory);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_timing_cache_serialize(cache, out hostMemory);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_timing_cache_serialize(cache, out hostMemory);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return hostMemory;
    }

}
