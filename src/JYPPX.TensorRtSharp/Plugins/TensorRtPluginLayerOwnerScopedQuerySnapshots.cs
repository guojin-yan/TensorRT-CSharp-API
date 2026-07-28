using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>Identifies a pointer-free plugin format capability snapshot. 标识无指针 plugin 格式能力快照。</summary>
public enum TensorRtPluginFormatCapabilityKind
{
    /// <summary>Represents IPluginV2IOExt format-combination decisions. 表示 IPluginV2IOExt 格式组合结果。</summary>
    PluginV2IoExt = 0,

    /// <summary>Represents IPluginV2DynamicExt format-combination decisions. 表示 IPluginV2DynamicExt 格式组合结果。</summary>
    PluginV2DynamicExt = 1,

    /// <summary>Represents IPluginV3OneBuild format-combination decisions. 表示 IPluginV3OneBuild 格式组合结果。</summary>
    PluginV3OneBuild = 2
}

/// <summary>
/// Copied format-combination decisions for the data types and first allowed formats declared by a network-owned layer.
/// 针对 network-owned layer 声明的数据类型与首个允许格式复制出的格式组合结果。
/// </summary>
public sealed class TensorRtPluginFormatSupportSnapshot
{
    internal TensorRtPluginFormatSupportSnapshot(
        TensorRtApiLine line,
        TensorRtPluginFormatCapabilityKind capability,
        int inputCount,
        int outputCount,
        IReadOnlyList<bool> support)
    {
        Line = line;
        Capability = capability;
        InputCount = inputCount;
        OutputCount = outputCount;
        Support = support ?? Array.Empty<bool>();
    }

    /// <summary>Gets the TensorRT API line used for the query. 获取查询使用的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the queried plugin capability. 获取被查询的 plugin capability。</summary>
    public TensorRtPluginFormatCapabilityKind Capability { get; }

    /// <summary>Gets the copied input count. 获取复制出的输入数量。</summary>
    public int InputCount { get; }

    /// <summary>Gets the copied output count. 获取复制出的输出数量。</summary>
    public int OutputCount { get; }

    /// <summary>Gets input-then-output support flags. 获取按输入后输出排列的支持标志。</summary>
    public IReadOnlyList<bool> Support { get; }

    /// <summary>Gets whether every declared position is supported as one prefix-consistent combination. 获取全部声明位置是否形成前缀一致的受支持组合。</summary>
    public bool AllSupported => Support.Count == InputCount + OutputCount && Support.All(value => value);

    /// <summary>Gets whether the snapshot is bounded, copied, and internally consistent. 获取快照是否有界、已复制且内部一致。</summary>
    public bool IsConsistent => InputCount >= 0 && OutputCount > 0 && Support.Count == InputCount + OutputCount;

    /// <summary>Gets support for an input position. 获取指定输入位置的支持状态。</summary>
    public bool IsInputSupported(int inputIndex)
    {
        if ((uint)inputIndex >= (uint)InputCount) { throw new ArgumentOutOfRangeException(nameof(inputIndex)); }
        return Support[inputIndex];
    }

    /// <summary>Gets support for an output position. 获取指定输出位置的支持状态。</summary>
    public bool IsOutputSupported(int outputIndex)
    {
        if ((uint)outputIndex >= (uint)OutputCount) { throw new ArgumentOutOfRangeException(nameof(outputIndex)); }
        return Support[InputCount + outputIndex];
    }

    /// <summary>Returns compact diagnostics. 返回简短诊断。</summary>
    public override string ToString() => $"{Line}:{Capability}:io={InputCount}/{OutputCount}:all={AllSupported}";
}

/// <summary>Copied PluginV3 build IO query results. PluginV3 build IO 查询结果副本。</summary>
public sealed class TensorRtPluginV3BuildIoSnapshot
{
    internal TensorRtPluginV3BuildIoSnapshot(
        TensorRtApiLine line,
        int inputCount,
        int outputCount,
        IReadOnlyList<TensorRtDataType> outputDataTypes,
        IReadOnlyList<int> aliasedInputIndices,
        bool aliasMetadataAvailable,
        TensorRtPluginFormatSupportSnapshot formatSupport)
    {
        Line = line;
        InputCount = inputCount;
        OutputCount = outputCount;
        OutputDataTypes = outputDataTypes ?? Array.Empty<TensorRtDataType>();
        AliasedInputIndices = aliasedInputIndices ?? Array.Empty<int>();
        AliasMetadataAvailable = aliasMetadataAvailable;
        FormatSupport = formatSupport ?? throw new ArgumentNullException(nameof(formatSupport));
    }

    /// <summary>Gets the TensorRT API line used for the query. 获取查询使用的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied input count. 获取复制出的输入数量。</summary>
    public int InputCount { get; }

    /// <summary>Gets the copied output count. 获取复制出的输出数量。</summary>
    public int OutputCount { get; }

    /// <summary>Gets output data types copied from IPluginV3OneBuild. 获取从 IPluginV3OneBuild 复制出的输出数据类型。</summary>
    public IReadOnlyList<TensorRtDataType> OutputDataTypes { get; }

    /// <summary>Gets aliased input indices, where -1 means no alias. 获取 alias 输入索引，其中 -1 表示无 alias。</summary>
    public IReadOnlyList<int> AliasedInputIndices { get; }

    /// <summary>Gets whether IPluginV3OneBuildV2 alias metadata was available. 获取 IPluginV3OneBuildV2 alias 元数据是否可用。</summary>
    public bool AliasMetadataAvailable { get; }

    /// <summary>Gets copied current format-combination decisions. 获取复制出的当前格式组合结果。</summary>
    public TensorRtPluginFormatSupportSnapshot FormatSupport { get; }

    /// <summary>Gets whether all copied values are internally consistent and pointer-free. 获取复制值是否内部一致且无指针。</summary>
    public bool IsConsistent =>
        InputCount >= 0 &&
        OutputCount > 0 &&
        OutputDataTypes.Count == OutputCount &&
        AliasedInputIndices.Count == OutputCount &&
        AliasedInputIndices.All(index => index >= -1 && index < InputCount) &&
        FormatSupport.IsConsistent;

    /// <summary>Returns compact diagnostics. 返回简短诊断。</summary>
    public override string ToString() =>
        $"{Line}:io={InputCount}/{OutputCount}:types={OutputDataTypes.Count}:aliases={AliasMetadataAvailable}:formats={FormatSupport.AllSupported}";
}

/// <summary>Pointer-free inventory copied from IPluginV3OneRuntime serialization fields. 从 IPluginV3OneRuntime 序列化字段复制出的无指针 inventory。</summary>
public sealed class TensorRtPluginV3SerializationFieldInventory
{
    internal TensorRtPluginV3SerializationFieldInventory(TensorRtApiLine line, IReadOnlyList<TensorRtPluginFieldInfo> fields)
    {
        Line = line;
        Fields = fields ?? Array.Empty<TensorRtPluginFieldInfo>();
    }

    /// <summary>Gets the TensorRT API line used for the query. 获取查询使用的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets copied field names, types, lengths, and data-presence flags. 获取复制出的字段名、类型、长度和 data presence 标志。</summary>
    public IReadOnlyList<TensorRtPluginFieldInfo> Fields { get; }

    /// <summary>Gets whether no native field or data pointer is retained. 获取是否未持有任何原生 field 或 data 指针。</summary>
    public bool PointerFreeCopiedInventory => true;

    /// <summary>Gets whether copied field metadata is internally consistent. 获取字段元数据副本是否内部一致。</summary>
    public bool IsConsistent => Fields.All(item => item != null && item.Length >= 0);

    /// <summary>Returns compact diagnostics. 返回简短诊断。</summary>
    public override string ToString() => $"{Line}:serializationFields={Fields.Count}:pointerFree={PointerFreeCopiedInventory}";
}

public sealed partial class TensorRtLayer
{
    /// <summary>Copies IPluginV2DynamicExt decisions for the layer's currently declared types and formats. 复制 layer 当前声明类型与格式的 IPluginV2DynamicExt 结果。</summary>
    public TensorRtPluginFormatSupportSnapshot GetPluginV2DynamicFormatSupportSnapshot()
    {
        EnsurePluginV2OwnerLease();
        return NativeBridgeApi.GetPluginV2CurrentFormatSupport(
            Line, _handle, TensorRtPluginFormatCapabilityKind.PluginV2DynamicExt, InputCount, OutputCount);
    }

    /// <summary>Copies IPluginV2IOExt decisions for the layer's currently declared types and formats. 复制 layer 当前声明类型与格式的 IPluginV2IOExt 结果。</summary>
    public TensorRtPluginFormatSupportSnapshot GetPluginV2IoExtFormatSupportSnapshot()
    {
        EnsurePluginV2OwnerLease();
        return NativeBridgeApi.GetPluginV2CurrentFormatSupport(
            Line, _handle, TensorRtPluginFormatCapabilityKind.PluginV2IoExt, InputCount, OutputCount);
    }

    /// <summary>Tries to copy IPluginV2DynamicExt current format decisions. 尝试复制 IPluginV2DynamicExt 当前格式结果。</summary>
    public bool TryGetPluginV2DynamicFormatSupportSnapshot(
        out TensorRtPluginFormatSupportSnapshot? snapshot,
        out string diagnostic)
    {
        try
        {
            snapshot = GetPluginV2DynamicFormatSupportSnapshot();
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsPluginV2MetadataProbeException(exception))
        {
            snapshot = null;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>Tries to copy IPluginV2IOExt current format decisions. 尝试复制 IPluginV2IOExt 当前格式结果。</summary>
    public bool TryGetPluginV2IoExtFormatSupportSnapshot(
        out TensorRtPluginFormatSupportSnapshot? snapshot,
        out string diagnostic)
    {
        try
        {
            snapshot = GetPluginV2IoExtFormatSupportSnapshot();
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsPluginV2MetadataProbeException(exception))
        {
            snapshot = null;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>Copies PluginV3 output types, optional alias indices, and current format decisions. 复制 PluginV3 输出类型、可选 alias 索引与当前格式结果。</summary>
    public TensorRtPluginV3BuildIoSnapshot GetPluginV3BuildIoSnapshot()
    {
        if (_ownerLease == null)
        {
            throw new InvalidOperationException("PluginV3 build IO queries require a network-owned layer.");
        }
        return NativeBridgeApi.GetPluginV3BuildIoSnapshot(Line, _handle);
    }

    /// <summary>Copies PluginV3 runtime serialization field metadata without reading field data pointers. 复制 PluginV3 runtime 序列化字段元数据，不读取 field data 指针。</summary>
    public TensorRtPluginV3SerializationFieldInventory GetPluginV3RuntimeSerializationFields()
    {
        if (_ownerLease == null)
        {
            throw new InvalidOperationException("PluginV3 runtime serialization-field queries require a network-owned layer.");
        }
        return NativeBridgeApi.GetPluginV3RuntimeSerializationFields(Line, _handle);
    }

    /// <summary>Tries to copy PluginV3 build IO metadata. 尝试复制 PluginV3 build IO 元数据。</summary>
    public bool TryGetPluginV3BuildIoSnapshot(out TensorRtPluginV3BuildIoSnapshot? snapshot, out string diagnostic)
    {
        try
        {
            snapshot = GetPluginV3BuildIoSnapshot();
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsPluginV3MetadataProbeException(exception))
        {
            snapshot = null;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>Tries to copy PluginV3 runtime serialization fields. 尝试复制 PluginV3 runtime 序列化字段。</summary>
    public bool TryGetPluginV3RuntimeSerializationFields(
        out TensorRtPluginV3SerializationFieldInventory? inventory,
        out string diagnostic)
    {
        try
        {
            inventory = GetPluginV3RuntimeSerializationFields();
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsPluginV3MetadataProbeException(exception))
        {
            inventory = null;
            diagnostic = exception.Message;
            return false;
        }
    }
}
