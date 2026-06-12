using System;
using System.Collections.Generic;
using System.Text;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Provides a safe managed wrapper over TensorRT refitter operations.
/// 提供 TensorRT refitter 操作的安全托管封装。
/// </summary>
public sealed partial class TensorRtRefitter : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtRefitter(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    /// <summary>
    /// Gets the TensorRT API line that owns this refitter.
    /// 获取拥有当前 refitter 的 TensorRT API 版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the number of required weights still missing before refit can complete.
    /// 获取执行 refit 前仍缺失的必要权重数量。
    /// </summary>
    public int MissingWeightCount => NativeBridgeApi.GetRefitterMissingCount(Line, _handle);

    /// <summary>
    /// Gets the number of weights TensorRT reports as refittable for the engine.
    /// 获取 TensorRT 报告的当前 engine 可 refit 权重数量。
    /// </summary>
    public int AllRefittableWeightCount => NativeBridgeApi.GetRefitterAllCount(Line, _handle);

    /// <summary>
    /// Gets the layer/role pairs still missing before TensorRT can complete refit.
    /// 获取 TensorRT 完成 refit 前仍缺失的 layer/role 组合。
    /// </summary>
    /// <returns>The missing refit entries. 缺失的 refit 条目。</returns>
    public IReadOnlyList<TensorRtRefitEntry> GetMissingEntries()
    {
        return ConvertEntries(NativeBridgeApi.GetRefitterMissingEntries(Line, _handle));
    }

    /// <summary>
    /// Gets all layer/role pairs that TensorRT reports as refittable.
    /// 获取 TensorRT 报告的全部可 refit layer/role 组合。
    /// </summary>
    /// <returns>All refittable entries. 全部可 refit 条目。</returns>
    public IReadOnlyList<TensorRtRefitEntry> GetAllEntries()
    {
        return ConvertEntries(NativeBridgeApi.GetRefitterAllEntries(Line, _handle));
    }

    /// <summary>
    /// Sets weights for a layer/role pair using a pinned refit weights buffer.
    /// 使用 pinned refit 权重缓冲区为指定 layer/role 设置权重。
    /// </summary>
    /// <param name="layerName">The TensorRT layer name. TensorRT 层名称。</param>
    /// <param name="role">The TensorRT weight role. TensorRT 权重角色。</param>
    /// <param name="weights">The pinned refit weights buffer. pinned refit 权重缓冲区。</param>
    /// <returns><c>true</c> if TensorRT accepted the weights; otherwise <c>false</c>. 如果 TensorRT 接受该权重则返回 <c>true</c>，否则返回 <c>false</c>。</returns>
    public bool SetWeights(string layerName, TensorRtWeightsRole role, TensorRtRefitWeightsBuffer weights)
    {
        return NativeBridgeApi.SetRefitterWeights(Line, _handle, layerName, role, weights);
    }

    /// <summary>
    /// Sets weights for a refit entry using a pinned refit weights buffer.
    /// 使用 pinned refit 权重缓冲区为指定 refit 条目设置权重。
    /// </summary>
    /// <param name="entry">The layer/role entry to update. 要更新的 layer/role 条目。</param>
    /// <param name="weights">The pinned refit weights buffer. pinned refit 权重缓冲区。</param>
    /// <returns><c>true</c> if TensorRT accepted the weights; otherwise <c>false</c>. 如果 TensorRT 接受该权重则返回 <c>true</c>，否则返回 <c>false</c>。</returns>
    public bool SetWeights(TensorRtRefitEntry entry, TensorRtRefitWeightsBuffer weights)
    {
        return SetWeights(entry.LayerName, entry.Role, weights);
    }

    /// <summary>
    /// Applies the currently configured weights to the CUDA engine.
    /// 将当前已配置的权重应用到 CUDA engine。
    /// </summary>
    /// <returns>
    /// <c>true</c> when TensorRT reports the refit operation succeeded; otherwise <c>false</c>.
    /// 当 TensorRT 报告 refit 操作成功时返回 <c>true</c>，否则返回 <c>false</c>。
    /// </returns>
    public bool RefitCudaEngine()
    {
        return NativeBridgeApi.RefitCudaEngine(Line, _handle);
    }

    /// <summary>
    /// Releases the native refitter handle.
    /// 释放原生 refitter 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private static IReadOnlyList<TensorRtRefitEntry> ConvertEntries(NativeTensorRtRefitEntryInfo[] nativeEntries)
    {
        if (nativeEntries.Length == 0)
        {
            return Array.Empty<TensorRtRefitEntry>();
        }

        TensorRtRefitEntry[] entries = new TensorRtRefitEntry[nativeEntries.Length];
        for (int index = 0; index < nativeEntries.Length; index++)
        {
            entries[index] = new TensorRtRefitEntry(DecodeFixedUtf8(nativeEntries[index].LayerName), (TensorRtWeightsRole)nativeEntries[index].Role);
        }

        return entries;
    }

    private static string DecodeFixedUtf8(byte[] value)
    {
        if (value == null || value.Length == 0)
        {
            return string.Empty;
        }

        int terminator = Array.IndexOf(value, (byte)0);
        int length = terminator >= 0 ? terminator : value.Length;
        return length == 0 ? string.Empty : Encoding.UTF8.GetString(value, 0, length);
    }
}
