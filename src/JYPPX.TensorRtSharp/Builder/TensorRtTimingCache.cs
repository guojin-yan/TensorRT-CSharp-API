using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT timing cache.
/// TensorRT timing cache 的托管封装。
/// </summary>
public sealed class TensorRtTimingCache : IDisposable
{
    /// <summary>
    /// The fixed size of one TensorRT timing-cache key in bytes.
    /// 单个 TensorRT timing cache key 的固定字节长度。
    /// </summary>
    public const int KeySizeInBytes = 16;

    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtTimingCache(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this timing cache.
    /// 获取当前 timing cache 使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Serializes the timing cache into TensorRT host memory.
    /// 将 timing cache 序列化为 TensorRT host memory。
    /// </summary>
    /// <returns>The serialized timing-cache payload. 序列化的 timing cache 负载。</returns>
    public TensorRtHostMemory Serialize()
    {
        return new TensorRtHostMemory(Line, NativeBridgeApi.SerializeTimingCache(Line, _handle));
    }

    /// <summary>
    /// Merges another timing cache into this timing cache.
    /// 将另一个 timing cache 合并到当前 timing cache 中。
    /// </summary>
    /// <param name="inputCache">The timing cache to merge. 要合并的 timing cache。</param>
    /// <param name="ignoreMismatch">Whether TensorRT should ignore cache mismatches. TensorRT 是否忽略 cache 不匹配。</param>
    /// <returns><see langword="true"/> when TensorRT applied the merge. TensorRT 应用合并时返回 <see langword="true"/>。</returns>
    public bool Combine(TensorRtTimingCache inputCache, bool ignoreMismatch = false)
    {
        ValidateCompatible(inputCache, nameof(inputCache));
        return NativeBridgeApi.CombineTimingCaches(Line, _handle, inputCache.Handle, ignoreMismatch);
    }

    /// <summary>
    /// Resets the timing cache contents.
    /// 重置 timing cache 内容。
    /// </summary>
    /// <returns><see langword="true"/> when the cache was reset successfully. 成功重置 cache 时返回 <see langword="true"/>。</returns>
    public bool Reset()
    {
        return NativeBridgeApi.ResetTimingCache(Line, _handle);
    }

    /// <summary>
    /// Gets the number of timing-cache keys stored in the cache.
    /// 获取当前 cache 中存储的 timing cache key 数量。
    /// </summary>
    /// <returns>The key count. key 数量。</returns>
    public long QueryKeyCount()
    {
        return NativeBridgeApi.GetTimingCacheKeyCount(Line, _handle);
    }

    /// <summary>
    /// Copies the flattened timing-cache key bytes.
    /// 复制扁平化的 timing cache key 字节数据。
    /// </summary>
    /// <returns>The concatenated key bytes. 拼接后的 key 字节数组。</returns>
    public byte[] QueryKeyBytes()
    {
        return NativeBridgeApi.CopyTimingCacheKeys(Line, _handle);
    }

    /// <summary>
    /// Returns timing-cache keys as individual 16-byte arrays.
    /// 将 timing cache key 作为独立的 16 字节数组返回。
    /// </summary>
    /// <returns>A read-only list of keys. key 的只读列表。</returns>
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

    /// <summary>
    /// Tries to query one timing-cache entry by key.
    /// 尝试按 key 查询一个 timing cache 条目。
    /// </summary>
    /// <param name="key">The 16-byte timing-cache key. 16 字节 timing cache key。</param>
    /// <param name="value">The queried value. 查询到的值。</param>
    /// <returns><see langword="true"/> when the key exists. key 存在时返回 <see langword="true"/>。</returns>
    public bool TryQuery(byte[] key, out TensorRtTimingCacheValue value)
    {
        ValidateKey(key);
        bool found = NativeBridgeApi.TryQueryTimingCache(Line, _handle, key, out ulong tacticHash, out float timingMilliseconds);
        value = new TensorRtTimingCacheValue(tacticHash, timingMilliseconds);
        return found;
    }

    /// <summary>
    /// Updates one timing-cache entry with a prebuilt value object.
    /// 使用预构建值对象更新一个 timing cache 条目。
    /// </summary>
    /// <param name="key">The 16-byte timing-cache key. 16 字节 timing cache key。</param>
    /// <param name="value">The value to store. 要写入的值。</param>
    /// <returns><see langword="true"/> when TensorRT accepted the update. TensorRT 接受更新时返回 <see langword="true"/>。</returns>
    public bool Update(byte[] key, TensorRtTimingCacheValue value)
    {
        return Update(key, value.TacticHash, value.TimingMilliseconds);
    }

    /// <summary>
    /// Updates one timing-cache entry with explicit tactic and timing data.
    /// 使用显式 tactic 与 timing 数据更新一个 timing cache 条目。
    /// </summary>
    /// <param name="key">The 16-byte timing-cache key. 16 字节 timing cache key。</param>
    /// <param name="tacticHash">The TensorRT tactic hash. TensorRT tactic 哈希值。</param>
    /// <param name="timingMilliseconds">The observed timing in milliseconds. 观测到的 timing，单位为毫秒。</param>
    /// <returns><see langword="true"/> when TensorRT accepted the update. TensorRT 接受更新时返回 <see langword="true"/>。</returns>
    public bool Update(byte[] key, ulong tacticHash, float timingMilliseconds)
    {
        ValidateKey(key);
        return NativeBridgeApi.UpdateTimingCache(Line, _handle, key, tacticHash, timingMilliseconds);
    }

    /// <summary>
    /// Releases the TensorRT timing-cache handle.
    /// 释放 TensorRT timing cache 句柄。
    /// </summary>
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

/// <summary>
/// Represents one TensorRT timing-cache entry value.
/// 表示一个 TensorRT timing cache 条目值。
/// </summary>
public readonly struct TensorRtTimingCacheValue
{
    /// <summary>
    /// Sentinel tactic hash used for invalid or missing entries.
    /// 用于无效或缺失条目的哨兵 tactic 哈希值。
    /// </summary>
    public const ulong InvalidTacticHash = ulong.MaxValue;

    /// <summary>
    /// Initializes one TensorRT timing-cache value.
    /// 初始化一个 TensorRT timing cache 值。
    /// </summary>
    /// <param name="tacticHash">The TensorRT tactic hash. TensorRT tactic 哈希值。</param>
    /// <param name="timingMilliseconds">The timing in milliseconds. timing，单位为毫秒。</param>
    public TensorRtTimingCacheValue(ulong tacticHash, float timingMilliseconds)
    {
        TacticHash = tacticHash;
        TimingMilliseconds = timingMilliseconds;
    }

    /// <summary>
    /// Gets the TensorRT tactic hash.
    /// 获取 TensorRT tactic 哈希值。
    /// </summary>
    public ulong TacticHash { get; }

    /// <summary>
    /// Gets the timing in milliseconds.
    /// 获取 timing，单位为毫秒。
    /// </summary>
    public float TimingMilliseconds { get; }

    /// <summary>
    /// Gets whether the timing-cache value looks valid.
    /// 获取当前 timing cache 值是否看起来有效。
    /// </summary>
    public bool IsValid => TacticHash != InvalidTacticHash && !float.IsNaN(TimingMilliseconds) && TimingMilliseconds >= 0.0F;
}
