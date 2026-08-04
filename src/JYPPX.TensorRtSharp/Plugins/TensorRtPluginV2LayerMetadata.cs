using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Pointer-free metadata copied from a TensorRT PluginV2 layer.
/// 从 TensorRT PluginV2 layer 复制出的无指针元数据。
/// </summary>
/// <remarks>
/// The native bridge reads the borrowed plugin only while the owning network lease is held and copies every value
/// before returning. This type never exposes or owns an <c>IPluginV2*</c> pointer.
/// Native bridge 只在 owning network lease 有效期间读取 borrowed plugin，并在返回前复制全部值；
/// 本类型从不暴露或持有 <c>IPluginV2*</c> 指针。
/// </remarks>
public sealed class TensorRtPluginV2LayerMetadata
{
    internal TensorRtPluginV2LayerMetadata(
        TensorRtApiLine line,
        string pluginType,
        string pluginVersion,
        string pluginNamespace,
        ulong serializationSize,
        int packedTensorRtVersion,
        int outputCount,
        bool hasExtCapability,
        bool hasIoExtCapability,
        bool hasDynamicExtCapability)
    {
        Line = line;
        PluginType = pluginType ?? string.Empty;
        PluginVersion = pluginVersion ?? string.Empty;
        PluginNamespace = pluginNamespace ?? string.Empty;
        SerializationSize = serializationSize;
        PackedTensorRtVersion = packedTensorRtVersion;
        OutputCount = outputCount;
        HasExtCapability = hasExtCapability;
        HasIoExtCapability = hasIoExtCapability;
        HasDynamicExtCapability = hasDynamicExtCapability;
    }

    /// <summary>Gets the TensorRT API line used for the query. 获取查询使用的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied <c>IPluginV2::getPluginType</c> value. 获取复制后的 plugin type。</summary>
    public string PluginType { get; }

    /// <summary>Gets the copied <c>IPluginV2::getPluginVersion</c> value. 获取复制后的 plugin version。</summary>
    public string PluginVersion { get; }

    /// <summary>Gets the copied <c>IPluginV2::getPluginNamespace</c> value. 获取复制后的 plugin namespace。</summary>
    public string PluginNamespace { get; }

    /// <summary>Gets the copied serialized payload size in bytes. 获取复制后的序列化载荷字节数。</summary>
    public ulong SerializationSize { get; }

    /// <summary>
    /// Gets the packed value returned by <c>IPluginV2::getTensorRTVersion</c>.
    /// 获取 <c>IPluginV2::getTensorRTVersion</c> 返回的 packed 值。
    /// </summary>
    public int PackedTensorRtVersion { get; }

    /// <summary>Gets the output count reported by <c>IPluginV2::getNbOutputs</c>. 获取 plugin 报告的输出数量。</summary>
    public int OutputCount { get; }

    /// <summary>Gets whether the plugin implements <c>IPluginV2Ext</c>. 获取 plugin 是否实现 IPluginV2Ext。</summary>
    public bool HasExtCapability { get; }

    /// <summary>Gets whether the plugin implements <c>IPluginV2IOExt</c>. 获取 plugin 是否实现 IPluginV2IOExt。</summary>
    public bool HasIoExtCapability { get; }

    /// <summary>Gets whether the plugin implements <c>IPluginV2DynamicExt</c>. 获取 plugin 是否实现 IPluginV2DynamicExt。</summary>
    public bool HasDynamicExtCapability { get; }

    /// <summary>Gets the PluginV2 API tag stored in the upper byte. 获取高字节中的 PluginV2 API tag。</summary>
    public byte PluginApiVersionTag => (byte)((uint)PackedTensorRtVersion >> 24);

    /// <summary>Gets the TensorRT numeric version stored in the lower 24 bits. 获取低 24 位中的 TensorRT 数字版本。</summary>
    public int TensorRtVersion => PackedTensorRtVersion & 0x00FFFFFF;

    /// <summary>Gets the decoded TensorRT major version. 获取解码后的 TensorRT major 版本。</summary>
    public int TensorRtMajor => TensorRtVersion / 10000;

    /// <summary>Gets the decoded TensorRT minor version. 获取解码后的 TensorRT minor 版本。</summary>
    public int TensorRtMinor => (TensorRtVersion / 100) % 100;

    /// <summary>Gets the decoded TensorRT patch version. 获取解码后的 TensorRT patch 版本。</summary>
    public int TensorRtPatch => TensorRtVersion % 100;

    /// <summary>Gets whether the copied identity and version values are internally usable. 获取复制值是否具备可用的一致性。</summary>
    public bool IsConsistent =>
        !string.IsNullOrWhiteSpace(PluginType) &&
        !string.IsNullOrWhiteSpace(PluginVersion) &&
        TensorRtVersion > 0 &&
        OutputCount > 0 &&
        (!HasIoExtCapability || HasExtCapability) &&
        (!HasDynamicExtCapability || HasExtCapability);

    /// <summary>Returns a compact pointer-free diagnostic string. 返回简短的无指针诊断字符串。</summary>
    public override string ToString() =>
        $"{PluginType}:{PluginVersion}:{PluginNamespace}:outputs={OutputCount}:ext={HasExtCapability}/{HasIoExtCapability}/{HasDynamicExtCapability}:bytes={SerializationSize}:trt={TensorRtMajor}.{TensorRtMinor}.{TensorRtPatch}:tag={PluginApiVersionTag}";
}
