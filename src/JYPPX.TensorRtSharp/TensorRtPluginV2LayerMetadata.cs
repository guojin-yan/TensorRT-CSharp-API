using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
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
        int packedTensorRtVersion)
    {
        Line = line;
        PluginType = pluginType ?? string.Empty;
        PluginVersion = pluginVersion ?? string.Empty;
        PluginNamespace = pluginNamespace ?? string.Empty;
        SerializationSize = serializationSize;
        PackedTensorRtVersion = packedTensorRtVersion;
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
        TensorRtVersion > 0;

    /// <summary>Returns a compact pointer-free diagnostic string. 返回简短的无指针诊断字符串。</summary>
    public override string ToString() =>
        $"{PluginType}:{PluginVersion}:{PluginNamespace}:bytes={SerializationSize}:trt={TensorRtMajor}.{TensorRtMinor}.{TensorRtPatch}:tag={PluginApiVersionTag}";
}

public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets copied PluginV2 identity, serialization-size, and TensorRT-version metadata for this layer.
    /// 获取当前 layer 的 PluginV2 identity、serialization size 和 TensorRT version 元数据副本。
    /// </summary>
    /// <remarks>
    /// Retrieve the layer through <see cref="TensorRtNetworkDefinition.GetLayer(int)"/> so the network owner lease
    /// remains alive. The raw plugin pointer never crosses the native ABI.
    /// 必须通过 <see cref="TensorRtNetworkDefinition.GetLayer(int)"/> 获取 layer，以保持 network owner lease；
    /// 原始 plugin 指针不会跨越 native ABI。
    /// </remarks>
    /// <returns>Copied PluginV2 metadata. 已复制的 PluginV2 元数据。</returns>
    public TensorRtPluginV2LayerMetadata GetPluginV2Metadata()
    {
        if (_ownerLease == null)
        {
            throw new InvalidOperationException(
                "PluginV2 metadata requires a network-owned layer. Retrieve the layer through TensorRtNetworkDefinition.GetLayer.");
        }

        return NativeBridgeApi.GetPluginV2LayerMetadata(Line, _handle);
    }

    /// <summary>
    /// Tries to get copied PluginV2 metadata without exposing the borrowed plugin pointer.
    /// 尝试获取 PluginV2 元数据副本，不暴露 borrowed plugin 指针。
    /// </summary>
    /// <param name="metadata">Copied metadata when the layer is PluginV2. 当 layer 为 PluginV2 时返回元数据副本。</param>
    /// <param name="diagnostic">Success or rejection diagnostic. 成功或拒绝原因。</param>
    /// <returns><see langword="true"/> when metadata was copied. 成功复制元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetPluginV2Metadata(out TensorRtPluginV2LayerMetadata? metadata, out string diagnostic)
    {
        try
        {
            metadata = GetPluginV2Metadata();
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsPluginV2MetadataProbeException(exception))
        {
            metadata = null;
            diagnostic = exception.Message;
            return false;
        }
    }

    private static bool IsPluginV2MetadataProbeException(Exception exception)
    {
        return exception is BridgeProbeException ||
               exception is InvalidOperationException ||
               exception is ArgumentException ||
               exception is NotSupportedException ||
               exception is DllNotFoundException ||
               exception is BadImageFormatException ||
               exception is EntryPointNotFoundException ||
               exception is SEHException ||
               exception is AccessViolationException;
    }
}
