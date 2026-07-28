using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtRuntime
{
    /// <summary>
    /// Checks whether this runtime exposes a runtime-local plugin registry.
    /// 检查当前 runtime 是否暴露 runtime-local plugin registry。
    /// </summary>
    /// <remarks>
    /// This method only checks registry availability. It does not return or own the native registry pointer.
    /// TensorRT 8, 10, and 11 are queried through copied metadata probes.
    /// 该方法只检查 registry 可用性；不会返回或持有 native registry 指针。TensorRT 8、10 和 11 均通过复制元数据探针查询。
    /// </remarks>
    /// <returns><see langword="true"/> when TensorRT reports a runtime-local plugin registry. TensorRT 报告存在 runtime-local plugin registry 时返回 <see langword="true"/>。</returns>
    public bool IsPluginRegistryAvailable()
    {
        return NativeBridgeApi.RuntimePluginRegistryExists(Line, _handle);
    }

    /// <summary>
    /// Tries to check whether this runtime exposes a runtime-local plugin registry without throwing probe exceptions.
    /// 尝试检查当前 runtime 是否暴露 runtime-local plugin registry；探针异常会转为诊断字符串。
    /// </summary>
    /// <param name="exists">Set to <see langword="true"/> when the runtime-local plugin registry exists. 存在 runtime-local plugin registry 时设为 <see langword="true"/>。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the availability query completed successfully. 可用性查询成功完成时返回 <see langword="true"/>。</returns>
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
    /// Gets a read-only snapshot of plugin creators registered in this runtime-local plugin registry.
    /// 获取当前 runtime-local plugin registry 中已注册 plugin creator 的只读快照。
    /// </summary>
    /// <remarks>
    /// This method does not create plugins, deserialize plugins, load plugin libraries, or take ownership of creator objects.
    /// Returned creator and field metadata is copied into managed objects.
    /// TensorRT 8 does not expose a separate recursive creator count through this bridge; <see cref="TensorRtPluginRegistryInventory.RecursiveCreatorCount"/> is <see langword="null"/> for that line.
    /// 该方法不会创建 plugin、反序列化 plugin、加载 plugin library，也不会接管 creator 对象所有权；返回的 creator 和字段元数据会复制到托管对象中。TensorRT 8 不通过该桥接暴露单独的递归 creator count，因此 <see cref="TensorRtPluginRegistryInventory.RecursiveCreatorCount"/> 为 <see langword="null"/>。
    /// </remarks>
    /// <returns>A plugin registry inventory snapshot. Plugin registry inventory 快照。</returns>
    public TensorRtPluginRegistryInventory GetPluginRegistryInventory()
    {
        return NativeBridgeApi.GetRuntimePluginRegistryInventory(Line, _handle);
    }

    /// <summary>
    /// Tries to get a runtime-local plugin registry inventory without throwing probe exceptions.
    /// 尝试获取 runtime-local plugin registry inventory；探针异常会转为诊断字符串。
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
        catch (Exception exception) when (IsPluginInventoryProbeException(exception))
        {
            inventory = new TensorRtPluginRegistryInventory(Line, TensorRtPluginRegistrySource.Runtime, hasErrorRecorder: false, parentSearchEnabled: false, recursiveCreatorCount: null, Array.Empty<TensorRtPluginCreatorInfo>());
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Checks whether this runtime-local plugin registry contains a creator matching the supplied metadata.
    /// 检查当前 runtime-local plugin registry 是否包含匹配给定元数据的 creator。
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
        return NativeBridgeApi.IsRuntimePluginCreatorRegistered(Line, _handle, pluginName, pluginVersion, pluginNamespace);
    }

    /// <summary>
    /// Tries to check whether this runtime-local plugin registry contains a matching creator without throwing probe exceptions.
    /// 尝试检查当前 runtime-local plugin registry 是否存在匹配 creator；探针异常会转为诊断字符串。
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
    /// Tries to read metadata for a plugin creator from this runtime-local plugin registry.
    /// 尝试从当前 runtime-local plugin registry 读取某个 plugin creator 的元数据。
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
            bool found = NativeBridgeApi.TryGetRuntimePluginCreator(Line, _handle, pluginName, pluginVersion, pluginNamespace, out creator);
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
