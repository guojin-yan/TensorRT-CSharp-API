using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed class TensorRtTimingCache : IDisposable
{
    public const int KeySizeInBytes = 16;

    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtTimingCache(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    public TensorRtApiLine Line { get; }

    public TensorRtHostMemory Serialize()
    {
        return new TensorRtHostMemory(Line, NativeBridgeApi.SerializeTimingCache(Line, _handle));
    }

    public bool Combine(TensorRtTimingCache inputCache, bool ignoreMismatch = false)
    {
        ValidateCompatible(inputCache, nameof(inputCache));
        return NativeBridgeApi.CombineTimingCaches(Line, _handle, inputCache.Handle, ignoreMismatch);
    }

    public bool Reset()
    {
        return NativeBridgeApi.ResetTimingCache(Line, _handle);
    }

    public long QueryKeyCount()
    {
        return NativeBridgeApi.GetTimingCacheKeyCount(Line, _handle);
    }

    public byte[] QueryKeyBytes()
    {
        return NativeBridgeApi.CopyTimingCacheKeys(Line, _handle);
    }

    public IReadOnlyList<byte[]> QueryKeys()
    {
        byte[] flatKeys = QueryKeyBytes();
        if (flatKeys.Length == 0)
        {
            return Array.Empty<byte[]>();
        }

        byte[][] keys = new byte[flatKeys.Length / KeySizeInBytes][];
        for (int index = 0; index < keys.Length; ++index)
        {
            byte[] key = new byte[KeySizeInBytes];
            Array.Copy(flatKeys, index * KeySizeInBytes, key, 0, KeySizeInBytes);
            keys[index] = key;
        }

        return keys;
    }

    public bool TryQuery(byte[] key, out TensorRtTimingCacheValue value)
    {
        ValidateKey(key);
        bool found = NativeBridgeApi.TryQueryTimingCache(Line, _handle, key, out ulong tacticHash, out float timingMilliseconds);
        value = new TensorRtTimingCacheValue(tacticHash, timingMilliseconds);
        return found;
    }

    public bool Update(byte[] key, TensorRtTimingCacheValue value)
    {
        return Update(key, value.TacticHash, value.TimingMilliseconds);
    }

    public bool Update(byte[] key, ulong tacticHash, float timingMilliseconds)
    {
        ValidateKey(key);
        return NativeBridgeApi.UpdateTimingCache(Line, _handle, key, tacticHash, timingMilliseconds);
    }

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ValidateCompatible(TensorRtTimingCache cache, string parameterName)
    {
        if (cache == null)
        {
            throw new ArgumentNullException(parameterName);
        }

        if (cache.Line != Line)
        {
            throw new ArgumentException("Timing cache must belong to the same TensorRT API line.", parameterName);
        }
    }

    private static void ValidateKey(byte[] key)
    {
        if (key == null)
        {
            throw new ArgumentNullException(nameof(key));
        }

        if (key.Length != KeySizeInBytes)
        {
            throw new ArgumentException("TensorRT timing cache keys must be exactly 16 bytes.", nameof(key));
        }
    }
}

public readonly struct TensorRtTimingCacheValue
{
    public const ulong InvalidTacticHash = ulong.MaxValue;

    public TensorRtTimingCacheValue(ulong tacticHash, float timingMilliseconds)
    {
        TacticHash = tacticHash;
        TimingMilliseconds = timingMilliseconds;
    }

    public ulong TacticHash { get; }

    public float TimingMilliseconds { get; }

    public bool IsValid => TacticHash != InvalidTacticHash && !float.IsNaN(TimingMilliseconds) && TimingMilliseconds >= 0.0F;
}
