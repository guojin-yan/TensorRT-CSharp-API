using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    private const int TimingCacheKeySizeInBytes = 16;

    public static bool CombineTimingCaches(TensorRtApiLine line, SafeTensorRtObjectHandle cache, SafeTensorRtObjectHandle inputCache, bool ignoreMismatch)
    {
        int combined;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_timing_cache_combine(cache, inputCache, ignoreMismatch ? 1 : 0, out combined),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_timing_cache_combine(cache, inputCache, ignoreMismatch ? 1 : 0, out combined),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_timing_cache_combine(cache, inputCache, ignoreMismatch ? 1 : 0, out combined),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return combined != 0;
    }

    public static bool ResetTimingCache(TensorRtApiLine line, SafeTensorRtObjectHandle cache)
    {
        int reset;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_timing_cache_reset(cache, out reset),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_timing_cache_reset(cache, out reset),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_timing_cache_reset(cache, out reset),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return reset != 0;
    }

    public static long GetTimingCacheKeyCount(TensorRtApiLine line, SafeTensorRtObjectHandle cache)
    {
        long keyCount;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_timing_cache_query_key_count(cache, out keyCount),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_timing_cache_query_key_count(cache, out keyCount),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(GetTimingCacheKeyCount)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return keyCount;
    }

    public static byte[] CopyTimingCacheKeys(TensorRtApiLine line, SafeTensorRtObjectHandle cache)
    {
        EnsureTensorRt10Or11TimingCache(line, nameof(CopyTimingCacheKeys));
        long keyCount = GetTimingCacheKeyCount(line, cache);
        if (keyCount == 0)
        {
            return Array.Empty<byte>();
        }

        while (true)
        {
            byte[] buffer = AllocateTimingCacheKeyBuffer(keyCount);
            long availableKeyCount;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_timing_cache_copy_keys(cache, buffer, keyCount, out availableKeyCount),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_timing_cache_copy_keys(cache, buffer, keyCount, out availableKeyCount),
                _ => throw UnsupportedLine()
            };
            NativeStatus.ThrowIfFailed(status);
            if (availableKeyCount < 0)
            {
                throw new InvalidOperationException("TensorRT timing cache returned a negative key count.");
            }

            if (availableKeyCount <= keyCount)
            {
                int byteCount = GetTimingCacheKeyByteCount(availableKeyCount);
                if (byteCount == buffer.Length)
                {
                    return buffer;
                }

                byte[] result = new byte[byteCount];
                Array.Copy(buffer, result, byteCount);
                return result;
            }

            keyCount = availableKeyCount;
        }
    }

    public static bool TryQueryTimingCache(TensorRtApiLine line, SafeTensorRtObjectHandle cache, byte[] key, out ulong tacticHash, out float timingMilliseconds)
    {
        EnsureTensorRt10Or11TimingCache(line, nameof(TryQueryTimingCache));
        ValidateTimingCacheKey(key);
        int found;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_timing_cache_query(cache, key, key.Length, out tacticHash, out timingMilliseconds, out found),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_timing_cache_query(cache, key, key.Length, out tacticHash, out timingMilliseconds, out found),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return found != 0;
    }

    public static bool UpdateTimingCache(TensorRtApiLine line, SafeTensorRtObjectHandle cache, byte[] key, ulong tacticHash, float timingMilliseconds)
    {
        EnsureTensorRt10Or11TimingCache(line, nameof(UpdateTimingCache));
        ValidateTimingCacheKey(key);
        int updated;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_timing_cache_update(cache, key, key.Length, tacticHash, timingMilliseconds, out updated),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_timing_cache_update(cache, key, key.Length, tacticHash, timingMilliseconds, out updated),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return updated != 0;
    }

    private static void EnsureTensorRt10Or11TimingCache(TensorRtApiLine line, string apiName)
    {
        if (line == TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{apiName} is available for TensorRT 10 and TensorRT 11 adapters.");
        }

        if (line != TensorRtApiLine.TensorRt10 && line != TensorRtApiLine.TensorRt11)
        {
            throw UnsupportedLine();
        }
    }

    private static void ValidateTimingCacheKey(byte[] key)
    {
        if (key == null)
        {
            throw new ArgumentNullException(nameof(key));
        }

        if (key.Length != TimingCacheKeySizeInBytes)
        {
            throw new ArgumentException("TensorRT timing cache keys must be exactly 16 bytes.", nameof(key));
        }
    }

    private static byte[] AllocateTimingCacheKeyBuffer(long keyCount)
    {
        if (keyCount < 0)
        {
            throw new InvalidOperationException("TensorRT timing cache returned a negative key count.");
        }

        return new byte[GetTimingCacheKeyByteCount(keyCount)];
    }

    private static int GetTimingCacheKeyByteCount(long keyCount)
    {
        checked
        {
            long byteCount = keyCount * TimingCacheKeySizeInBytes;
            if (byteCount > int.MaxValue)
            {
                throw new InvalidOperationException("TensorRT timing cache key data is too large for a managed byte array.");
            }

            return (int)byteCount;
        }
    }
}
