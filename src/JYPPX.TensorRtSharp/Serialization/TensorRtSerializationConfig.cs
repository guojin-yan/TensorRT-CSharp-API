using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Wraps TensorRT <c>ISerializationConfig</c> for engine serialization options.
/// 封装 TensorRT <c>ISerializationConfig</c>，用于控制 engine 序列化选项。
/// </summary>
/// <remarks>
/// This wrapper is currently supported for TensorRT 10 and TensorRT 11 serialization-config paths.
/// 当前封装支持 TensorRT 10 和 TensorRT 11 的 serialization-config 路径。
/// </remarks>
public sealed class TensorRtSerializationConfig : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtSerializationConfig(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line that owns this serialization config.
    /// 获取拥有当前 serialization config 的 TensorRT API 版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets or sets all serialization flags as a bitmask.
    /// 获取或设置全部 serialization flags 位掩码。
    /// </summary>
    public TensorRtSerializationFlags Flags
    {
        get => NativeBridgeApi.GetSerializationConfigFlags(Line, _handle);
        set => NativeBridgeApi.SetSerializationConfigFlags(Line, _handle, value);
    }

    /// <summary>
    /// Enables a single serialization flag.
    /// 启用单个 serialization flag。
    /// </summary>
    /// <param name="flag">The flag to enable. / 要启用的 flag。</param>
    /// <returns><c>true</c> when TensorRT accepted the flag. / TensorRT 接受该 flag 时返回 <c>true</c>。</returns>
    public bool SetFlag(TensorRtSerializationFlag flag)
    {
        return NativeBridgeApi.SetSerializationConfigFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Clears a single serialization flag.
    /// 清除单个 serialization flag。
    /// </summary>
    /// <param name="flag">The flag to clear. / 要清除的 flag。</param>
    /// <returns><c>true</c> when TensorRT accepted the clear operation. / TensorRT 接受清除操作时返回 <c>true</c>。</returns>
    public bool ClearFlag(TensorRtSerializationFlag flag)
    {
        return NativeBridgeApi.ClearSerializationConfigFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Gets whether a single serialization flag is enabled.
    /// 获取单个 serialization flag 是否已启用。
    /// </summary>
    /// <param name="flag">The flag to query. / 要查询的 flag。</param>
    /// <returns><c>true</c> when the flag is enabled. / 当 flag 已启用时返回 <c>true</c>。</returns>
    public bool GetFlag(TensorRtSerializationFlag flag)
    {
        return NativeBridgeApi.GetSerializationConfigFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Gets a compact copied summary of this serialization config.
    /// 获取当前 serialization config 的紧凑复制型摘要。
    /// </summary>
    /// <remarks>
    /// This method reads scalar serialization flags only. It does not expose the native config handle and does not promote runtime proof.
    /// 该方法只读取 serialization flags 标量值；不暴露原生 config handle，也不会晋级 runtime proof。
    /// </remarks>
    public TensorRtSerializationConfigSummary ToSummary()
    {
        return new TensorRtSerializationConfigSummary(Line, Flags);
    }

    /// <summary>
    /// Releases the native serialization config handle.
    /// 释放原生 serialization config 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
