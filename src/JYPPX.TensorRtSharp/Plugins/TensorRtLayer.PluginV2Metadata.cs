using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

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
