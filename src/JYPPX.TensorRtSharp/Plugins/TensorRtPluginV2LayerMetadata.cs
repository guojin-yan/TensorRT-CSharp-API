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
        EnsurePluginV2OwnerLease();
        return NativeBridgeApi.GetPluginV2LayerMetadata(Line, _handle);
    }

    /// <summary>
    /// Evaluates the legacy <c>IPluginV2::getOutputDimensions</c> callback using dimensions copied from this layer's inputs.
    /// 使用当前 layer 输入维度副本执行旧版 IPluginV2::getOutputDimensions 查询。
    /// </summary>
    public TensorRtDims GetPluginV2LegacyOutputDimensions(int outputIndex)
    {
        EnsurePluginV2OwnerLease();
        if (outputIndex < 0) { throw new ArgumentOutOfRangeException(nameof(outputIndex)); }
        return NativeBridgeApi.GetPluginV2LegacyOutputDimensions(Line, _handle, outputIndex);
    }

    /// <summary>
    /// Gets the legacy workspace-size requirement for a positive maximum batch size without exposing device memory.
    /// 获取给定正数最大 batch size 的旧版 workspace 字节数，不暴露设备内存。
    /// </summary>
    public ulong GetPluginV2LegacyWorkspaceSize(int maxBatchSize)
    {
        EnsurePluginV2OwnerLease();
        if (maxBatchSize <= 0) { throw new ArgumentOutOfRangeException(nameof(maxBatchSize)); }
        return NativeBridgeApi.GetPluginV2LegacyWorkspaceSize(Line, _handle, maxBatchSize);
    }

    /// <summary>
    /// Evaluates legacy <c>IPluginV2::supportsFormat</c> for a copied data type and tensor format.
    /// 使用复制的 data type 与 tensor format 执行旧版 IPluginV2::supportsFormat 查询。
    /// </summary>
    public bool SupportsPluginV2LegacyFormat(TensorRtDataType dataType, TensorRtTensorFormat tensorFormat)
    {
        EnsurePluginV2OwnerLease();
        if (!Enum.IsDefined(typeof(TensorRtDataType), dataType)) { throw new ArgumentOutOfRangeException(nameof(dataType)); }
        if (!Enum.IsDefined(typeof(TensorRtTensorFormat), tensorFormat) || tensorFormat == TensorRtTensorFormat.Unknown)
        {
            throw new ArgumentOutOfRangeException(nameof(tensorFormat));
        }
        return NativeBridgeApi.SupportsPluginV2LegacyFormat(Line, _handle, dataType, tensorFormat);
    }

    /// <summary>
    /// Gets the output data type reported by <c>IPluginV2Ext</c> using types copied from this layer's inputs.
    /// 使用当前 layer 输入类型副本获取 IPluginV2Ext 报告的输出类型。
    /// </summary>
    public TensorRtDataType GetPluginV2OutputDataType(int outputIndex)
    {
        EnsurePluginV2OwnerLease();
        if (outputIndex < 0) { throw new ArgumentOutOfRangeException(nameof(outputIndex)); }
        return NativeBridgeApi.GetPluginV2OutputDataType(Line, _handle, outputIndex);
    }

    /// <summary>
    /// Queries the deprecated implicit-batch input broadcast capability available on TensorRT 8 and 10.
    /// 查询 TensorRT 8/10 提供的旧版 implicit-batch 输入广播能力。
    /// </summary>
    public bool CanPluginV2BroadcastInputAcrossBatch(int inputIndex)
    {
        EnsurePluginV2OwnerLease();
        if (inputIndex < 0) { throw new ArgumentOutOfRangeException(nameof(inputIndex)); }
        return NativeBridgeApi.CanPluginV2BroadcastInputAcrossBatch(Line, _handle, inputIndex);
    }

    /// <summary>
    /// Queries deprecated implicit-batch output broadcasting from caller-owned input broadcast flags on TensorRT 8 and 10.
    /// 使用 caller-owned 输入广播标志查询 TensorRT 8/10 的旧版输出广播行为。
    /// </summary>
    public bool IsPluginV2OutputBroadcastAcrossBatch(int outputIndex, System.Collections.Generic.IReadOnlyList<bool> inputIsBroadcasted)
    {
        EnsurePluginV2OwnerLease();
        if (outputIndex < 0) { throw new ArgumentOutOfRangeException(nameof(outputIndex)); }
        if (inputIsBroadcasted == null) { throw new ArgumentNullException(nameof(inputIsBroadcasted)); }
        byte[] flags = new byte[inputIsBroadcasted.Count];
        for (int index = 0; index < flags.Length; ++index) { flags[index] = inputIsBroadcasted[index] ? (byte)1 : (byte)0; }
        return NativeBridgeApi.IsPluginV2OutputBroadcastAcrossBatch(Line, _handle, outputIndex, flags);
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

    private void EnsurePluginV2OwnerLease()
    {
        if (_ownerLease == null)
        {
            throw new InvalidOperationException(
                "PluginV2 queries require a network-owned layer. Retrieve the layer through TensorRtNetworkDefinition.GetLayer.");
        }
    }
}
