using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Copied versioned-interface metadata for a PluginV3 object or capability.
/// PluginV3 对象或 capability 的 versioned-interface 元数据副本。
/// </summary>
public readonly struct TensorRtPluginV3InterfaceMetadata
{
    internal TensorRtPluginV3InterfaceMetadata(TensorRtInterfaceInfo interfaceInfo, TensorRtApiLanguage apiLanguage)
    {
        InterfaceInfo = interfaceInfo;
        ApiLanguage = apiLanguage;
    }

    /// <summary>Gets the copied interface kind and version. 获取复制出的 interface kind 与版本。</summary>
    public TensorRtInterfaceInfo InterfaceInfo { get; }

    /// <summary>Gets the copied implementation language. 获取复制出的实现语言。</summary>
    public TensorRtApiLanguage ApiLanguage { get; }

    /// <summary>Gets whether the copied interface metadata is usable. 获取复制出的 interface 元数据是否可用。</summary>
    public bool IsConsistent =>
        !string.IsNullOrWhiteSpace(InterfaceInfo.Kind) &&
        InterfaceInfo.Major > 0 &&
        ApiLanguage != TensorRtApiLanguage.Unknown;

    /// <summary>Returns compact interface diagnostics. 返回简短的 interface 诊断。</summary>
    public override string ToString() => $"{InterfaceInfo}:{ApiLanguage}";
}

/// <summary>Copied PluginV3 core capability metadata. PluginV3 core capability 元数据副本。</summary>
public sealed class TensorRtPluginV3CoreMetadata
{
    internal TensorRtPluginV3CoreMetadata(
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        TensorRtPluginV3InterfaceMetadata interfaceMetadata)
    {
        PluginName = pluginName ?? string.Empty;
        PluginVersion = pluginVersion ?? string.Empty;
        PluginNamespace = pluginNamespace ?? string.Empty;
        InterfaceMetadata = interfaceMetadata;
    }

    /// <summary>Gets the copied plugin name. 获取复制出的 plugin name。</summary>
    public string PluginName { get; }

    /// <summary>Gets the copied plugin version. 获取复制出的 plugin version。</summary>
    public string PluginVersion { get; }

    /// <summary>Gets the copied plugin namespace. 获取复制出的 plugin namespace。</summary>
    public string PluginNamespace { get; }

    /// <summary>Gets copied core interface metadata. 获取复制出的 core interface 元数据。</summary>
    public TensorRtPluginV3InterfaceMetadata InterfaceMetadata { get; }

    /// <summary>Gets whether identity and interface metadata are internally consistent. 获取 identity 与 interface 元数据是否一致。</summary>
    public bool IsConsistent =>
        !string.IsNullOrWhiteSpace(PluginName) &&
        !string.IsNullOrWhiteSpace(PluginVersion) &&
        InterfaceMetadata.IsConsistent;

    /// <summary>Returns compact core diagnostics. 返回简短的 core 诊断。</summary>
    public override string ToString() => $"{PluginName}:{PluginVersion}:{PluginNamespace}:{InterfaceMetadata}";
}

/// <summary>Copied safe-query metadata from a PluginV3 build capability. PluginV3 build capability 的安全查询元数据副本。</summary>
public sealed class TensorRtPluginV3BuildMetadata
{
    internal TensorRtPluginV3BuildMetadata(
        TensorRtPluginV3InterfaceMetadata interfaceMetadata,
        int outputCount,
        int tacticCount,
        int formatCombinationLimit,
        string timingCacheId,
        string metadataString)
    {
        InterfaceMetadata = interfaceMetadata;
        OutputCount = outputCount;
        TacticCount = tacticCount;
        FormatCombinationLimit = formatCombinationLimit;
        TimingCacheId = timingCacheId ?? string.Empty;
        MetadataString = metadataString ?? string.Empty;
    }

    /// <summary>Gets copied build interface metadata. 获取复制出的 build interface 元数据。</summary>
    public TensorRtPluginV3InterfaceMetadata InterfaceMetadata { get; }

    /// <summary>Gets the number of plugin outputs. 获取 plugin output 数量。</summary>
    public int OutputCount { get; }

    /// <summary>Gets the advertised custom tactic count. 获取声明的自定义 tactic 数量。</summary>
    public int TacticCount { get; }

    /// <summary>Gets the maximum timed format-combination count. 获取最大 format combination 计时数量。</summary>
    public int FormatCombinationLimit { get; }

    /// <summary>Gets the copied optional timing-cache identifier. 获取复制出的可选 timing-cache ID。</summary>
    public string TimingCacheId { get; }

    /// <summary>Gets the copied optional plugin creation metadata. 获取复制出的可选 plugin 创建元数据。</summary>
    public string MetadataString { get; }

    /// <summary>Gets whether copied build metadata is internally consistent. 获取 build 元数据是否一致。</summary>
    public bool IsConsistent =>
        InterfaceMetadata.IsConsistent &&
        OutputCount > 0 &&
        TacticCount >= 0 &&
        FormatCombinationLimit > 0;

    /// <summary>Returns compact build diagnostics. 返回简短的 build 诊断。</summary>
    public override string ToString() =>
        $"{InterfaceMetadata}:outputs={OutputCount}:tactics={TacticCount}:formats={FormatCombinationLimit}:cache={TimingCacheId}:metadata={MetadataString}";
}

/// <summary>Copied PluginV3 runtime capability metadata. PluginV3 runtime capability 元数据副本。</summary>
public sealed class TensorRtPluginV3RuntimeMetadata
{
    internal TensorRtPluginV3RuntimeMetadata(TensorRtPluginV3InterfaceMetadata interfaceMetadata)
    {
        InterfaceMetadata = interfaceMetadata;
    }

    /// <summary>Gets copied runtime interface metadata. 获取复制出的 runtime interface 元数据。</summary>
    public TensorRtPluginV3InterfaceMetadata InterfaceMetadata { get; }

    /// <summary>Gets whether copied runtime metadata is internally consistent. 获取 runtime 元数据是否一致。</summary>
    public bool IsConsistent => InterfaceMetadata.IsConsistent;

    /// <summary>Returns compact runtime diagnostics. 返回简短的 runtime 诊断。</summary>
    public override string ToString() => InterfaceMetadata.ToString();
}

/// <summary>
/// Pointer-free metadata copied from a TensorRT PluginV3 layer and its capability interfaces.
/// 从 TensorRT PluginV3 layer 及其 capability interface 复制出的无指针元数据。
/// </summary>
/// <remarks>
/// Borrowed plugin and capability pointers remain inside one native owner-scoped call. Only strings, scalars, enums,
/// and interface versions cross the ABI. Borrowed plugin 与 capability 指针只存在于 native owner-scoped 调用内；
/// 跨 ABI 的只有字符串、标量、枚举和 interface version。
/// </remarks>
public sealed class TensorRtPluginV3LayerMetadata
{
    internal TensorRtPluginV3LayerMetadata(
        TensorRtApiLine line,
        TensorRtPluginV3InterfaceMetadata pluginInterface,
        bool hasCoreCapability,
        TensorRtPluginV3CoreMetadata core,
        TensorRtPluginV3BuildMetadata? build,
        TensorRtPluginV3RuntimeMetadata? runtime)
    {
        Line = line;
        PluginInterface = pluginInterface;
        HasCoreCapability = hasCoreCapability;
        Core = core ?? throw new ArgumentNullException(nameof(core));
        Build = build;
        Runtime = runtime;
    }

    /// <summary>Gets the TensorRT API line used for the query. 获取查询使用的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets copied metadata for the IPluginV3 wrapper interface. 获取 IPluginV3 wrapper interface 元数据副本。</summary>
    public TensorRtPluginV3InterfaceMetadata PluginInterface { get; }

    /// <summary>Gets whether the required core capability was reported. 获取是否报告了必需的 core capability。</summary>
    public bool HasCoreCapability { get; }

    /// <summary>Gets copied core identity and interface metadata. 获取 core identity 与 interface 元数据副本。</summary>
    public TensorRtPluginV3CoreMetadata Core { get; }

    /// <summary>Gets copied build metadata when the build capability is available. 获取 build capability 可用时的元数据副本。</summary>
    public TensorRtPluginV3BuildMetadata? Build { get; }

    /// <summary>Gets copied runtime metadata when the runtime capability is available. 获取 runtime capability 可用时的元数据副本。</summary>
    public TensorRtPluginV3RuntimeMetadata? Runtime { get; }

    /// <summary>Gets whether the build capability is available. 获取 build capability 是否可用。</summary>
    public bool HasBuildCapability => Build != null;

    /// <summary>Gets whether the runtime capability is available. 获取 runtime capability 是否可用。</summary>
    public bool HasRuntimeCapability => Runtime != null;

    /// <summary>Gets whether required copied metadata is internally consistent. 获取必需元数据是否一致。</summary>
    public bool IsConsistent =>
        PluginInterface.IsConsistent &&
        HasCoreCapability &&
        Core.IsConsistent &&
        (Build == null || Build.IsConsistent) &&
        (Runtime == null || Runtime.IsConsistent);

    /// <summary>Returns compact pointer-free diagnostics. 返回简短的无指针诊断。</summary>
    public override string ToString() =>
        $"{Core}:plugin={PluginInterface}:build={Build?.ToString() ?? "n/a"}:runtime={Runtime?.ToString() ?? "n/a"}";
}

public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets copied PluginV3 core, build, runtime, and versioned-interface metadata for this layer.
    /// 获取当前 layer 的 PluginV3 core、build、runtime 和 versioned-interface 元数据副本。
    /// </summary>
    /// <remarks>
    /// Retrieve the layer through <see cref="TensorRtNetworkDefinition.GetLayer(int)"/> to retain the network owner.
    /// Raw plugin and capability pointers never cross the native ABI.
    /// 必须通过 <see cref="TensorRtNetworkDefinition.GetLayer(int)"/> 获取 layer 以保持 network owner；
    /// 原始 plugin 与 capability 指针不会跨越 native ABI。
    /// </remarks>
    public TensorRtPluginV3LayerMetadata GetPluginV3Metadata()
    {
        if (_ownerLease == null)
        {
            throw new InvalidOperationException(
                "PluginV3 metadata requires a network-owned layer. Retrieve the layer through TensorRtNetworkDefinition.GetLayer.");
        }

        return NativeBridgeApi.GetPluginV3LayerMetadata(Line, _handle);
    }

    /// <summary>
    /// Tries to get copied PluginV3 metadata without exposing borrowed plugin or capability pointers.
    /// 尝试获取 PluginV3 元数据副本，不暴露 borrowed plugin 或 capability 指针。
    /// </summary>
    /// <param name="metadata">Copied metadata on success. 成功时返回元数据副本。</param>
    /// <param name="diagnostic">Success or rejection diagnostic. 成功或拒绝诊断。</param>
    /// <returns><see langword="true"/> when metadata was copied. 成功复制元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetPluginV3Metadata(out TensorRtPluginV3LayerMetadata? metadata, out string diagnostic)
    {
        try
        {
            metadata = GetPluginV3Metadata();
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsPluginV3MetadataProbeException(exception))
        {
            metadata = null;
            diagnostic = exception.Message;
            return false;
        }
    }

    private static bool IsPluginV3MetadataProbeException(Exception exception)
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
