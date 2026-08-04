using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtBuilder
{
    /// <summary>
    /// Gets a read-only snapshot of TensorRT plugin creators registered for this builder.
    /// 获取当前 builder 可见的 TensorRT plugin creator 只读快照。
    /// </summary>
    /// <remarks>
    /// This method does not create plugins, deserialize plugins, load plugin libraries, or take ownership of creator objects.
    /// 该方法不会创建 plugin、反序列化 plugin、加载 plugin library，也不会接管 creator 对象所有权。
    /// </remarks>
    /// <returns>A plugin registry inventory snapshot. Plugin registry inventory 快照。</returns>
    public TensorRtPluginRegistryInventory GetPluginRegistryInventory()
    {
        bool registryAvailable = NativeBridgeApi.IsBuilderPluginRegistryAvailable(Line, _handle);
        if (!registryAvailable)
        {
            return new TensorRtPluginRegistryInventory(Line, TensorRtPluginRegistrySource.Builder, hasErrorRecorder: false, parentSearchEnabled: false, recursiveCreatorCount: null, Array.Empty<TensorRtPluginCreatorInfo>());
        }

        int creatorCount = NativeBridgeApi.GetBuilderPluginRegistryCreatorCount(Line, _handle);
        int? recursiveCreatorCount = Line == TensorRtApiLine.TensorRt8
            ? null
            : NativeBridgeApi.GetBuilderPluginRegistryRecursiveCreatorCount(Line, _handle);
        bool hasErrorRecorder = NativeBridgeApi.HasBuilderPluginRegistryErrorRecorder(Line, _handle);
        bool parentSearchEnabled = NativeBridgeApi.IsBuilderPluginRegistryParentSearchEnabled(Line, _handle);
        List<TensorRtPluginCreatorInfo> creators = new List<TensorRtPluginCreatorInfo>(creatorCount);

        for (int creatorIndex = 0; creatorIndex < creatorCount; creatorIndex++)
        {
            string name = NativeBridgeApi.GetBuilderPluginCreatorName(Line, _handle, creatorIndex);
            string version = NativeBridgeApi.GetBuilderPluginCreatorVersion(Line, _handle, creatorIndex);
            string pluginNamespace = NativeBridgeApi.GetBuilderPluginCreatorNamespace(Line, _handle, creatorIndex);
            string interfaceKind = NativeBridgeApi.GetBuilderPluginCreatorInterfaceKind(Line, _handle, creatorIndex, out int interfaceMajor, out int interfaceMinor);
            TensorRtApiLanguage apiLanguage = NativeBridgeApi.GetBuilderPluginCreatorApiLanguage(Line, _handle, creatorIndex);
            int? tensorRtVersion = Line == TensorRtApiLine.TensorRt8
                ? NativeBridgeApi.GetBuilderPluginCreatorTensorRtVersion(Line, _handle, creatorIndex)
                : null;
            int fieldCount = NativeBridgeApi.GetBuilderPluginCreatorFieldCount(Line, _handle, creatorIndex);
            List<TensorRtPluginFieldInfo> fields = new List<TensorRtPluginFieldInfo>(fieldCount);

            for (int fieldIndex = 0; fieldIndex < fieldCount; fieldIndex++)
            {
                string fieldName = NativeBridgeApi.GetBuilderPluginCreatorFieldName(Line, _handle, creatorIndex, fieldIndex);
                NativeBridgeApi.GetBuilderPluginCreatorFieldMetadata(Line, _handle, creatorIndex, fieldIndex, out TensorRtPluginFieldType fieldType, out int length, out bool hasData);
                fields.Add(new TensorRtPluginFieldInfo(fieldName, fieldType, length, hasData));
            }

            creators.Add(new TensorRtPluginCreatorInfo(
                creatorIndex,
                name,
                version,
                pluginNamespace,
                interfaceKind,
                interfaceMajor,
                interfaceMinor,
                apiLanguage,
                fields,
                tensorRtVersion));
        }

        return new TensorRtPluginRegistryInventory(Line, TensorRtPluginRegistrySource.Builder, hasErrorRecorder, parentSearchEnabled, recursiveCreatorCount: recursiveCreatorCount, creators);
    }

    /// <summary>
    /// Checks whether this builder exposes a local plugin registry through the safe inventory bridge.
    /// 检查当前 builder 是否能通过安全 inventory 桥接访问本地 plugin registry。
    /// </summary>
    /// <returns><see langword="true"/> when the builder plugin registry can be queried. 可查询 builder plugin registry 时返回 <see langword="true"/>。</returns>
    public bool IsPluginRegistryAvailable()
    {
        return NativeBridgeApi.IsBuilderPluginRegistryAvailable(Line, _handle);
    }

    /// <summary>
    /// Tries to check builder plugin registry availability without throwing probe exceptions.
    /// 尝试检查 builder plugin registry 可用性；探针异常会转为诊断字符串。
    /// </summary>
    /// <param name="exists">Set to <see langword="true"/> when the builder plugin registry can be queried. 可查询时设为 <see langword="true"/>。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the availability check completed successfully. 可用性检查成功完成时返回 <see langword="true"/>。</returns>
    public bool TryIsPluginRegistryAvailable(out bool exists, out string diagnostic)
    {
        try
        {
            exists = IsPluginRegistryAvailable();
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsPluginInventoryProbeException(exception))
        {
            exists = false;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to get a plugin registry inventory without throwing for unsupported TensorRT lines.
    /// 尝试获取 plugin registry inventory；当 TensorRT line 不支持时不抛出异常。
    /// </summary>
    /// <param name="inventory">The inventory snapshot when the query succeeds. 查询成功时的 inventory 快照。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the inventory was collected successfully. 成功收集 inventory 时返回 <see langword="true"/>。</returns>
    public bool TryGetPluginRegistryInventory(out TensorRtPluginRegistryInventory inventory, out string diagnostic)
    {
        try
        {
            inventory = GetPluginRegistryInventory();
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            inventory = new TensorRtPluginRegistryInventory(Line, TensorRtPluginRegistrySource.Builder, hasErrorRecorder: false, parentSearchEnabled: false, recursiveCreatorCount: null, Array.Empty<TensorRtPluginCreatorInfo>());
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Checks whether this builder-visible plugin registry contains a creator matching the supplied metadata.
    /// 检查当前 builder 可见的 plugin registry 是否包含匹配给定元数据的 creator。
    /// </summary>
    /// <remarks>
    /// This lookup does not create a plugin, does not return the native creator pointer, and does not take ownership of TensorRT objects.
    /// 该 lookup 不会创建 plugin、不会返回 native creator 指针，也不会接管 TensorRT 对象所有权。
    /// </remarks>
    /// <param name="pluginName">The plugin creator name. plugin creator 名称。</param>
    /// <param name="pluginVersion">The plugin creator version. plugin creator 版本。</param>
    /// <param name="pluginNamespace">The plugin creator namespace. plugin creator 命名空间。</param>
    /// <returns><see langword="true"/> when TensorRT finds a matching creator. TensorRT 找到匹配 creator 时返回 <see langword="true"/>。</returns>
    public bool IsPluginCreatorRegistered(string pluginName, string pluginVersion, string pluginNamespace)
    {
        return NativeBridgeApi.IsBuilderPluginCreatorRegistered(Line, _handle, pluginName, pluginVersion, pluginNamespace);
    }

    /// <summary>
    /// Tries to check whether this builder-visible plugin registry contains a matching creator without throwing probe exceptions.
    /// 尝试检查当前 builder 可见的 plugin registry 是否存在匹配 creator；探针异常会转为诊断字符串。
    /// </summary>
    /// <remarks>
    /// This lookup does not create a plugin and does not return or own the native creator pointer.
    /// 该 lookup 不会创建 plugin，也不会返回或持有 native creator 指针。
    /// </remarks>
    /// <param name="pluginName">The plugin creator name. plugin creator 名称。</param>
    /// <param name="pluginVersion">The plugin creator version. plugin creator 版本。</param>
    /// <param name="pluginNamespace">The plugin creator namespace. plugin creator 命名空间。</param>
    /// <param name="found">Set to <see langword="true"/> when TensorRT finds a matching creator. 找到匹配 creator 时设为 <see langword="true"/>。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the lookup completed successfully. lookup 成功完成时返回 <see langword="true"/>。</returns>
    public bool TryIsPluginCreatorRegistered(
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        out bool found,
        out string diagnostic)
    {
        try
        {
            found = IsPluginCreatorRegistered(pluginName, pluginVersion, pluginNamespace);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsPluginInventoryProbeException(exception))
        {
            found = false;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to read metadata for a plugin creator from this builder-visible plugin registry.
    /// 尝试从当前 builder 可见的 plugin registry 读取某个 plugin creator 的元数据。
    /// </summary>
    /// <remarks>
    /// The lookup does not create a plugin, does not return the native creator pointer, and does not take ownership of TensorRT objects.
    /// The returned metadata is copied into managed objects.
    /// 该 lookup 不会创建 plugin、不会返回 native creator 指针，也不会接管 TensorRT 对象所有权；返回的元数据会复制到托管对象中。
    /// </remarks>
    /// <param name="pluginName">The plugin creator name. plugin creator 名称。</param>
    /// <param name="pluginVersion">The plugin creator version. plugin creator 版本。</param>
    /// <param name="pluginNamespace">The plugin creator namespace. plugin creator 命名空间。</param>
    /// <param name="creator">The copied creator metadata when a matching creator is found. 找到匹配 creator 时复制出的 creator 元数据。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when a matching creator was found and copied. 找到并复制匹配 creator 时返回 <see langword="true"/>。</returns>
    public bool TryGetPluginCreator(
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        out TensorRtPluginCreatorInfo? creator,
        out string diagnostic)
    {
        try
        {
            bool found = NativeBridgeApi.TryGetBuilderPluginCreator(Line, _handle, pluginName, pluginVersion, pluginNamespace, out creator);
            diagnostic = found ? "OK" : "Plugin creator was not found.";
            return found;
        }
        catch (Exception exception) when (IsPluginInventoryProbeException(exception))
        {
            creator = null;
            diagnostic = exception.Message;
            return false;
        }
    }

    private static bool IsPluginInventoryProbeException(Exception exception)
    {
        return exception is BridgeProbeException ||
               exception is NotSupportedException ||
               exception is InvalidOperationException ||
               exception is DllNotFoundException ||
               exception is BadImageFormatException ||
               exception is EntryPointNotFoundException ||
               exception is SEHException ||
               exception is AccessViolationException;
    }
}
