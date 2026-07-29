using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
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
    /// Checks whether TensorRT exposes a builder capability plugin registry for the requested engine capability.
    /// 检查 TensorRT 是否为指定 engine capability 暴露 builder capability plugin registry。
    /// </summary>
    /// <remarks>
    /// This method only checks whether <c>getBuilderPluginRegistry</c> returns a non-null registry pointer.
    /// It does not return the registry pointer, register creators, unregister creators, load plugin libraries, or take ownership of native objects.
    /// 该方法只检查 <c>getBuilderPluginRegistry</c> 是否返回非空 registry 指针；不会返回 registry 指针、注册或注销 creator、加载 plugin library，也不会接管 native 对象所有权。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="capability">The engine capability whose builder registry should be checked. 要检查 builder registry 的 engine capability。</param>
    /// <returns><see langword="true"/> when TensorRT reports a builder capability plugin registry. TensorRT 报告存在 builder capability plugin registry 时返回 <see langword="true"/>。</returns>
    public static bool IsBuilderCapabilityPluginRegistryAvailable(TensorRtApiLine line, TensorRtEngineCapability capability)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeBridgeApi.BuilderCapabilityPluginRegistryExists(line, capability);
    }

    /// <summary>
    /// Tries to check whether TensorRT exposes a builder capability plugin registry without throwing probe exceptions.
    /// 尝试检查 TensorRT 是否暴露 builder capability plugin registry；探针异常会转为诊断字符串。
    /// </summary>
    /// <remarks>
    /// This method only checks whether <c>getBuilderPluginRegistry</c> returns a non-null registry pointer.
    /// It does not return the registry pointer or take ownership of native objects.
    /// 该方法只检查 <c>getBuilderPluginRegistry</c> 是否返回非空 registry 指针；不会返回 registry 指针，也不会接管 native 对象所有权。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="capability">The engine capability whose builder registry should be checked. 要检查 builder registry 的 engine capability。</param>
    /// <param name="available">Set to <see langword="true"/> when TensorRT reports a registry. TensorRT 报告存在 registry 时设为 <see langword="true"/>。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the probe completed successfully. 探针成功完成时返回 <see langword="true"/>。</returns>
    public static bool TryIsBuilderCapabilityPluginRegistryAvailable(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
        out bool available,
        out string diagnostic)
    {
        try
        {
            available = IsBuilderCapabilityPluginRegistryAvailable(line, capability);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            available = false;
            diagnostic = FormatProbeException("Builder capability plugin registry existence query", exception);
            return false;
        }
    }

    /// <summary>
    /// Checks whether TensorRT exposes a builder safe plugin registry for the requested engine capability.
    /// 检查 TensorRT 是否为指定 engine capability 暴露 builder safe plugin registry。
    /// </summary>
    /// <remarks>
    /// This method calls the deprecated TensorRT safe-registry getter only to test for a non-null registry pointer.
    /// It does not return the registry pointer, register creators, unregister creators, load plugin libraries, or take ownership of native objects.
    /// 该方法只调用已弃用的 TensorRT safe registry getter 判断返回指针是否非空；不会返回 registry 指针、注册或注销 creator、加载 plugin library，也不会接管 native 对象所有权。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="capability">The engine capability whose safe builder registry should be checked. 要检查 safe builder registry 的 engine capability。</param>
    /// <returns><see langword="true"/> when TensorRT reports a safe builder plugin registry. TensorRT 报告存在 safe builder plugin registry 时返回 <see langword="true"/>。</returns>
    public static bool IsBuilderSafePluginRegistryAvailable(TensorRtApiLine line, TensorRtEngineCapability capability)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeBridgeApi.BuilderSafePluginRegistryExists(line, capability);
    }

    /// <summary>
    /// Tries to check whether TensorRT exposes a builder safe plugin registry without throwing probe exceptions.
    /// 尝试检查 TensorRT 是否暴露 builder safe plugin registry；探针异常会转为诊断字符串。
    /// </summary>
    /// <remarks>
    /// This method only checks whether TensorRT returns a non-null safe registry pointer.
    /// It does not return the registry pointer or take ownership of native objects.
    /// 该方法只检查 TensorRT 是否返回非空 safe registry 指针；不会返回 registry 指针，也不会接管 native 对象所有权。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="capability">The engine capability whose safe builder registry should be checked. 要检查 safe builder registry 的 engine capability。</param>
    /// <param name="available">Set to <see langword="true"/> when TensorRT reports a safe registry. TensorRT 报告存在 safe registry 时设为 <see langword="true"/>。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the probe completed successfully. 探针成功完成时返回 <see langword="true"/>。</returns>
    public static bool TryIsBuilderSafePluginRegistryAvailable(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
        out bool available,
        out string diagnostic)
    {
        try
        {
            available = IsBuilderSafePluginRegistryAvailable(line, capability);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            available = false;
            diagnostic = FormatProbeException("Builder safe plugin registry existence query", exception);
            return false;
        }
    }

    /// <summary>
    /// Gets a read-only snapshot of TensorRT's builder capability plugin registry.
    /// 获取 TensorRT builder capability plugin registry 的只读快照。
    /// </summary>
    /// <remarks>
    /// This method queries <c>getBuilderPluginRegistry</c> for the requested engine capability. It does not create plugins,
    /// register plugins, unregister plugins, initialize plugin libraries, or take ownership of creator objects.
    /// Creator identity, interface info, and field metadata are copied into managed value objects.
    /// 该方法会针对请求的 engine capability 查询 <c>getBuilderPluginRegistry</c>；不会创建 plugin、注册或注销 plugin、初始化 plugin library，也不会接管 creator 对象所有权。
    /// Creator 标识、interface 信息和字段元数据会复制到托管值对象中。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="capability">The engine capability whose builder-visible registry should be inspected. 要检查其 builder-visible registry 的 engine capability。</param>
    /// <returns>A builder capability plugin registry inventory snapshot. builder capability plugin registry inventory 快照。</returns>
    public static TensorRtPluginRegistryInventory GetBuilderCapabilityPluginRegistryInventory(TensorRtApiLine line, TensorRtEngineCapability capability)
    {
        return GetBuilderCapabilityPluginRegistryInventory(line, capability, includeCreatorFields: true);
    }

    /// <summary>
    /// Gets a builder capability plugin registry snapshot with optional creator-field collection.
    /// 获取 builder capability plugin registry 快照，并可选择是否采集 creator 字段。
    /// </summary>
    public static TensorRtPluginRegistryInventory GetBuilderCapabilityPluginRegistryInventory(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
        bool includeCreatorFields)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeBridgeApi.GetBuilderCapabilityPluginRegistryInventory(line, capability, includeCreatorFields);
    }

    /// <summary>
    /// Tries to get TensorRT's builder capability plugin registry inventory without throwing for unsupported lines.
    /// 尝试获取 TensorRT builder capability plugin registry inventory；不支持的 API line 不会抛出异常。
    /// </summary>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="capability">The engine capability whose builder-visible registry should be inspected. 要检查其 builder-visible registry 的 engine capability。</param>
    /// <param name="inventory">The inventory snapshot when the query succeeds. 查询成功时的 inventory 快照。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the inventory was collected successfully. 成功采集 inventory 时返回 <see langword="true"/>。</returns>
    public static bool TryGetBuilderCapabilityPluginRegistryInventory(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
        out TensorRtPluginRegistryInventory? inventory,
        out string diagnostic)
    {
        return TryGetBuilderCapabilityPluginRegistryInventory(
            line,
            capability,
            includeCreatorFields: true,
            out inventory,
            out diagnostic);
    }

    /// <summary>
    /// Tries to copy builder capability registry metadata with optional creator-field collection.
    /// 尝试复制 builder capability registry 元数据，并可选择是否采集 creator 字段。
    /// </summary>
    public static bool TryGetBuilderCapabilityPluginRegistryInventory(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
        bool includeCreatorFields,
        out TensorRtPluginRegistryInventory? inventory,
        out string diagnostic)
    {
        try
        {
            inventory = GetBuilderCapabilityPluginRegistryInventory(line, capability, includeCreatorFields);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            inventory = null;
            diagnostic = FormatProbeException("Builder capability plugin registry inventory query", exception);
            return false;
        }
    }

    /// <summary>
    /// Checks whether TensorRT's builder capability plugin registry contains a creator matching the supplied metadata.
    /// 检查 TensorRT builder capability plugin registry 是否包含匹配给定元数据的 creator。
    /// </summary>
    /// <remarks>
    /// This lookup does not create a plugin and does not return or own the native creator pointer.
    /// 该 lookup 不会创建 plugin，也不会返回或持有 native creator 指针。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="capability">The engine capability whose builder-visible registry should be inspected. 要检查其 builder-visible registry 的 engine capability。</param>
    /// <param name="pluginName">The plugin creator name. plugin creator 名称。</param>
    /// <param name="pluginVersion">The plugin creator version. plugin creator 版本。</param>
    /// <param name="pluginNamespace">The plugin creator namespace. plugin creator 命名空间。</param>
    /// <returns><see langword="true"/> when TensorRT finds a matching creator. TensorRT 找到匹配 creator 时返回 <see langword="true"/>。</returns>
    public static bool IsBuilderCapabilityPluginCreatorRegistered(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
        string pluginName,
        string pluginVersion,
        string pluginNamespace)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeBridgeApi.IsBuilderCapabilityPluginCreatorRegistered(line, capability, pluginName, pluginVersion, pluginNamespace);
    }

    /// <summary>
    /// Tries to check whether TensorRT's builder capability plugin registry contains a matching creator without throwing probe exceptions.
    /// 尝试检查 TensorRT builder capability plugin registry 是否存在匹配 creator；探针异常会转为诊断字符串。
    /// </summary>
    /// <remarks>
    /// This lookup does not create a plugin and does not return or own the native creator pointer.
    /// 该 lookup 不会创建 plugin，也不会返回或持有 native creator 指针。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="capability">The engine capability whose builder-visible registry should be inspected. 要检查其 builder-visible registry 的 engine capability。</param>
    /// <param name="pluginName">The plugin creator name. plugin creator 名称。</param>
    /// <param name="pluginVersion">The plugin creator version. plugin creator 版本。</param>
    /// <param name="pluginNamespace">The plugin creator namespace. plugin creator 命名空间。</param>
    /// <param name="found">Set to <see langword="true"/> when TensorRT finds a matching creator. 找到匹配 creator 时设为 <see langword="true"/>。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the lookup completed successfully. lookup 成功完成时返回 <see langword="true"/>。</returns>
    public static bool TryIsBuilderCapabilityPluginCreatorRegistered(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        out bool found,
        out string diagnostic)
    {
        try
        {
            found = IsBuilderCapabilityPluginCreatorRegistered(line, capability, pluginName, pluginVersion, pluginNamespace);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            found = false;
            diagnostic = FormatProbeException("Builder capability plugin creator lookup", exception);
            return false;
        }
    }

    /// <summary>
    /// Tries to read metadata for a plugin creator from TensorRT's builder capability plugin registry.
    /// 尝试从 TensorRT builder capability plugin registry 读取某个 plugin creator 的元数据。
    /// </summary>
    /// <remarks>
    /// The returned metadata is copied into managed objects. The native creator pointer is never exposed or owned by managed code.
    /// 返回的元数据会复制到托管对象中；native creator 指针不会暴露给托管代码，也不会由托管代码持有。
    /// </remarks>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="capability">The engine capability whose builder-visible registry should be inspected. 要检查其 builder-visible registry 的 engine capability。</param>
    /// <param name="pluginName">The plugin creator name. plugin creator 名称。</param>
    /// <param name="pluginVersion">The plugin creator version. plugin creator 版本。</param>
    /// <param name="pluginNamespace">The plugin creator namespace. plugin creator 命名空间。</param>
    /// <param name="creator">The copied creator metadata when a matching creator is found. 找到匹配 creator 时复制出的 creator 元数据。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when a matching creator was found and copied. 找到并复制匹配 creator 时返回 <see langword="true"/>。</returns>
    public static bool TryGetBuilderCapabilityPluginCreator(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
        string pluginName,
        string pluginVersion,
        string pluginNamespace,
        out TensorRtPluginCreatorInfo? creator,
        out string diagnostic)
    {
        return TryGetBuilderCapabilityPluginCreator(
            line,
            capability,
            pluginName,
            pluginVersion,
            pluginNamespace,
            includeCreatorFields: true,
            out creator,
            out diagnostic);
    }

    /// <summary>
    /// Tries to copy builder capability plugin creator metadata with optional field collection.
    /// 尝试复制 builder capability plugin creator 元数据，并可选择是否采集字段。
    /// </summary>
    public static bool TryGetBuilderCapabilityPluginCreator(
        TensorRtApiLine line,
        TensorRtEngineCapability capability,
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
            bool found = NativeBridgeApi.TryGetBuilderCapabilityPluginCreator(
                line,
                capability,
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
            diagnostic = FormatProbeException("Builder capability plugin creator metadata query", exception);
            return false;
        }
    }

}
