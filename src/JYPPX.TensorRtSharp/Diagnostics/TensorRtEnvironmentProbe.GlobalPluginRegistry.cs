using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Queries the current bridge state without requiring TensorRT inference to succeed.
/// 在不要求 TensorRT 推理成功的前提下查询当前 bridge 状态。
/// </summary>
public static partial class TensorRtEnvironmentProbe
{
    /// <summary>
    /// Gets a read-only snapshot of TensorRT's global plugin registry.
    /// 获取 TensorRT 全局 plugin registry 的只读快照。
    /// </summary>
    /// <remarks>
    /// This method does not initialize plugin libraries, create plugins, deserialize plugins, or take ownership of creator objects.
    /// 该方法不会初始化 plugin library、创建 plugin、反序列化 plugin，也不会接管 creator 对象所有权。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <returns>A global plugin registry inventory snapshot. 全局 plugin registry inventory 快照。</returns>
    public static TensorRtPluginRegistryInventory GetGlobalPluginRegistryInventory(TensorRtApiLine line)
    {
        return GetGlobalPluginRegistryInventory(line, includeCreatorFields: true);
    }

    /// <summary>
    /// Gets a read-only global plugin registry snapshot with optional creator-field collection.
    /// 获取全局 plugin registry 只读快照，并可选择是否采集 creator 字段。
    /// </summary>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="includeCreatorFields">Whether to invoke creator field metadata APIs. 是否调用 creator 字段元数据 API。</param>
    public static TensorRtPluginRegistryInventory GetGlobalPluginRegistryInventory(
        TensorRtApiLine line,
        bool includeCreatorFields)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeBridgeApi.GetGlobalPluginRegistryInventory(line, includeCreatorFields);
    }

    /// <summary>
    /// Tries to get TensorRT's global plugin registry inventory without throwing for unsupported lines.
    /// 尝试获取 TensorRT 全局 plugin registry inventory；不支持时不抛出异常。
    /// </summary>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="inventory">The inventory snapshot when the query succeeds. 查询成功时的 inventory 快照。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the inventory was collected successfully. 成功采集 inventory 时返回 <see langword="true"/>。</returns>
    public static bool TryGetGlobalPluginRegistryInventory(TensorRtApiLine line, out TensorRtPluginRegistryInventory? inventory, out string diagnostic)
    {
        return TryGetGlobalPluginRegistryInventory(line, includeCreatorFields: true, out inventory, out diagnostic);
    }

    /// <summary>
    /// Tries to copy global plugin registry metadata with optional creator-field collection.
    /// 尝试复制全局 plugin registry 元数据，并可选择是否采集 creator 字段。
    /// </summary>
    public static bool TryGetGlobalPluginRegistryInventory(
        TensorRtApiLine line,
        bool includeCreatorFields,
        out TensorRtPluginRegistryInventory? inventory,
        out string diagnostic)
    {
        try
        {
            inventory = GetGlobalPluginRegistryInventory(line, includeCreatorFields);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            inventory = null;
            diagnostic = FormatProbeException("Global plugin registry inventory query", exception);
            return false;
        }
    }

    /// <summary>
    /// Gets whether TensorRT's global plugin registry searches its parent registry.
    /// 获取 TensorRT 全局 plugin registry 是否搜索 parent registry。
    /// </summary>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <returns><see langword="true"/> when parent search is enabled. parent search 已启用时返回 <see langword="true"/>。</returns>
    public static bool IsGlobalPluginRegistryParentSearchEnabled(TensorRtApiLine line)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeBridgeApi.IsGlobalPluginRegistryParentSearchEnabled(line);
    }

    /// <summary>
    /// Tries to get the global plugin registry parent-search state.
    /// 尝试获取全局 plugin registry 的 parent-search 状态。
    /// </summary>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="enabled">The copied parent-search state. 复制出的 parent-search 状态。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the query completed successfully. 查询成功完成时返回 <see langword="true"/>。</returns>
    public static bool TryIsGlobalPluginRegistryParentSearchEnabled(TensorRtApiLine line, out bool enabled, out string diagnostic)
    {
        try
        {
            enabled = IsGlobalPluginRegistryParentSearchEnabled(line);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            enabled = false;
            diagnostic = FormatProbeException("Global plugin registry parent-search query", exception);
            return false;
        }
    }

    /// <summary>
    /// Sets the global plugin registry parent-search state and verifies the value by reading it back.
    /// 设置全局 plugin registry 的 parent-search 状态，并通过读回进行校验。
    /// </summary>
    /// <param name="line">The TensorRT API line to update. 要更新的 TensorRT API line。</param>
    /// <param name="enabled">The requested parent-search state. 请求的 parent-search 状态。</param>
    public static void SetGlobalPluginRegistryParentSearchEnabled(TensorRtApiLine line, bool enabled)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeBridgeApi.SetGlobalPluginRegistryParentSearchEnabled(line, enabled);
        bool actual = NativeBridgeApi.IsGlobalPluginRegistryParentSearchEnabled(line);
        if (actual != enabled)
        {
            throw new InvalidOperationException(
                $"TensorRT global plugin registry parent-search readback mismatch: requested={enabled}, actual={actual}.");
        }
    }

    /// <summary>
    /// Tries to set and read back the global plugin registry parent-search state.
    /// 尝试设置并读回全局 plugin registry 的 parent-search 状态。
    /// </summary>
    /// <param name="line">The TensorRT API line to update. 要更新的 TensorRT API line。</param>
    /// <param name="enabled">The requested parent-search state. 请求的 parent-search 状态。</param>
    /// <param name="actualEnabled">The read-back state when the update succeeds. 更新成功时读回的状态。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the update and readback both succeed. 更新及读回均成功时返回 <see langword="true"/>。</returns>
    public static bool TrySetGlobalPluginRegistryParentSearchEnabled(
        TensorRtApiLine line,
        bool enabled,
        out bool actualEnabled,
        out string diagnostic)
    {
        try
        {
            SetGlobalPluginRegistryParentSearchEnabled(line, enabled);
            actualEnabled = enabled;
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            actualEnabled = false;
            diagnostic = FormatProbeException("Global plugin registry parent-search update", exception);
            return false;
        }
    }

    /// <summary>
    /// Checks whether TensorRT exposes a non-null global plugin registry.
    /// 检查 TensorRT 是否暴露非空全局 plugin registry。
    /// </summary>
    /// <remarks>
    /// This method only checks whether <c>getPluginRegistry</c> returns a non-null registry pointer.
    /// It does not return the registry pointer, register creators, unregister creators, load plugin libraries, or take ownership of native objects.
    /// 该方法只检查 <c>getPluginRegistry</c> 是否返回非空 registry 指针；不会返回 registry 指针、注册或注销 creator、加载 plugin library，也不会接管 native 对象所有权。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <returns><see langword="true"/> when TensorRT reports a global plugin registry. TensorRT 报告存在全局 plugin registry 时返回 <see langword="true"/>。</returns>
    public static bool IsGlobalPluginRegistryAvailable(TensorRtApiLine line)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeBridgeApi.GlobalPluginRegistryExists(line);
    }

    /// <summary>
    /// Tries to check whether TensorRT exposes a global plugin registry without throwing probe exceptions.
    /// 尝试检查 TensorRT 是否暴露全局 plugin registry；探针异常会转为诊断字符串。
    /// </summary>
    /// <remarks>
    /// This method only checks whether <c>getPluginRegistry</c> returns a non-null registry pointer.
    /// It does not return the registry pointer or take ownership of native objects.
    /// 该方法只检查 <c>getPluginRegistry</c> 是否返回非空 registry 指针；不会返回 registry 指针，也不会接管 native 对象所有权。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="available">Set to <see langword="true"/> when TensorRT reports a registry. TensorRT 报告存在 registry 时设为 <see langword="true"/>。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the probe completed successfully. 探针成功完成时返回 <see langword="true"/>。</returns>
    public static bool TryIsGlobalPluginRegistryAvailable(TensorRtApiLine line, out bool available, out string diagnostic)
    {
        try
        {
            available = IsGlobalPluginRegistryAvailable(line);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            available = false;
            diagnostic = FormatProbeException("Global plugin registry availability query", exception);
            return false;
        }
    }

    /// <summary>
    /// Checks whether TensorRT's global plugin registry contains a creator matching the supplied metadata.
    /// 检查 TensorRT 全局 plugin registry 是否存在匹配给定元数据的 creator。
    /// </summary>
    /// <remarks>
    /// This lookup does not create a plugin and does not return or own the native creator pointer.
    /// 该 lookup 不会创建 plugin，也不会返回或持有 native creator 指针。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="pluginName">The plugin creator name. plugin creator 名称。</param>
    /// <param name="pluginVersion">The plugin creator version. plugin creator 版本。</param>
    /// <param name="pluginNamespace">The plugin creator namespace. plugin creator 命名空间。</param>
    /// <returns><see langword="true"/> when TensorRT finds a matching creator. TensorRT 找到匹配 creator 时返回 <see langword="true"/>。</returns>
    public static bool IsGlobalPluginCreatorRegistered(TensorRtApiLine line, string pluginName, string pluginVersion, string pluginNamespace)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeBridgeApi.IsGlobalPluginCreatorRegistered(line, pluginName, pluginVersion, pluginNamespace);
    }

    /// <summary>
    /// Tries to check whether TensorRT's global plugin registry contains a matching creator without throwing probe exceptions.
    /// 尝试检查 TensorRT 全局 plugin registry 是否存在匹配 creator；探针异常会转为诊断字符串。
    /// </summary>
    /// <remarks>
    /// This lookup does not create a plugin and does not return or own the native creator pointer.
    /// 该 lookup 不会创建 plugin，也不会返回或持有 native creator 指针。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="pluginName">The plugin creator name. plugin creator 名称。</param>
    /// <param name="pluginVersion">The plugin creator version. plugin creator 版本。</param>
    /// <param name="pluginNamespace">The plugin creator namespace. plugin creator 命名空间。</param>
    /// <param name="found">Set to <see langword="true"/> when TensorRT finds a matching creator. 找到匹配 creator 时设为 <see langword="true"/>。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the lookup completed successfully. lookup 成功完成时返回 <see langword="true"/>。</returns>
    public static bool TryIsGlobalPluginCreatorRegistered(
        TensorRtApiLine line,
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        out bool found,
        out string diagnostic)
    {
        try
        {
            found = IsGlobalPluginCreatorRegistered(line, pluginName, pluginVersion, pluginNamespace);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            found = false;
            diagnostic = FormatProbeException("Global plugin creator lookup", exception);
            return false;
        }
    }

    /// <summary>
    /// Tries to read metadata for a plugin creator from TensorRT's global plugin registry.
    /// 尝试从 TensorRT 全局 plugin registry 读取某个 plugin creator 的元数据。
    /// </summary>
    /// <remarks>
    /// The lookup does not create a plugin, does not return the native creator pointer, and does not take ownership of TensorRT objects.
    /// The returned metadata is copied into managed objects.
    /// 该 lookup 不会创建 plugin、不会返回 native creator 指针，也不会接管 TensorRT 对象所有权；返回的元数据会复制到托管对象中。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="pluginName">The plugin creator name. plugin creator 名称。</param>
    /// <param name="pluginVersion">The plugin creator version. plugin creator 版本。</param>
    /// <param name="pluginNamespace">The plugin creator namespace. plugin creator 命名空间。</param>
    /// <param name="creator">The copied creator metadata when a matching creator is found. 找到匹配 creator 时复制出的 creator 元数据。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when a matching creator was found and copied. 找到并复制匹配 creator 时返回 <see langword="true"/>。</returns>
    public static bool TryGetGlobalPluginCreator(
        TensorRtApiLine line,
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        out TensorRtPluginCreatorInfo? creator,
        out string diagnostic)
    {
        return TryGetGlobalPluginCreator(
            line,
            pluginName,
            pluginVersion,
            pluginNamespace,
            includeCreatorFields: true,
            out creator,
            out diagnostic);
    }

    /// <summary>
    /// Tries to copy global plugin creator metadata with optional field collection.
    /// 尝试复制全局 plugin creator 元数据，并可选择是否采集字段。
    /// </summary>
    public static bool TryGetGlobalPluginCreator(
        TensorRtApiLine line,
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        bool includeCreatorFields,
        out TensorRtPluginCreatorInfo? creator,
        out string diagnostic)
    {
        try
        {
            NativeBridgeLoader.EnsureInitialized();
            bool found = NativeBridgeApi.TryGetGlobalPluginCreator(
                line,
                pluginName,
                pluginVersion,
                pluginNamespace,
                includeCreatorFields,
                out creator);
            diagnostic = found ? "OK" : "Plugin creator was not found.";
            return found;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            creator = null;
            diagnostic = FormatProbeException("Global plugin creator metadata query", exception);
            return false;
        }
    }

}
