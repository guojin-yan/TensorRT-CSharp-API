using System;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT builder configuration.
/// TensorRT builder 配置的托管封装。
/// </summary>
public sealed partial class TensorRtBuilderConfig
{
    /// <summary>
    /// Creates a TensorRT timing cache from optional serialized bytes.
    /// 使用可选的序列化字节创建一个 TensorRT timing cache。
    /// </summary>
    /// <param name="serializedCache">Optional serialized timing-cache payload. 可选的序列化 timing cache 负载。</param>
    /// <returns>A TensorRT timing-cache wrapper. TensorRT timing cache 封装。</returns>
    public TensorRtTimingCache CreateTimingCache(byte[]? serializedCache = null)
    {
        return new TensorRtTimingCache(Line, NativeBridgeApi.CreateTimingCache(Line, _handle, serializedCache));
    }

    /// <summary>
    /// Attaches a TensorRT timing cache to this builder configuration.
    /// 将一个 TensorRT timing cache 附加到当前 builder 配置。
    /// </summary>
    /// <param name="cache">The timing cache to attach. 要附加的 timing cache。</param>
    /// <param name="ignoreMismatch">Whether TensorRT should ignore cache mismatches. TensorRT 是否忽略 cache 不匹配。</param>
    public void SetTimingCache(TensorRtTimingCache cache, bool ignoreMismatch = false)
    {
        if (cache == null)
        {
            throw new ArgumentNullException(nameof(cache));
        }

        if (cache.Line != Line)
        {
            throw new ArgumentException("Timing cache must belong to the same TensorRT API line as the builder config.");
        }

        NativeBridgeApi.SetTimingCache(Line, _handle, cache.Handle, ignoreMismatch);
    }

}
