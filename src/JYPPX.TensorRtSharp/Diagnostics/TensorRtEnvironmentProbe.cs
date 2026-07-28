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
public static class TensorRtEnvironmentProbe
{
    /// <summary>
    /// Initializes and registers TensorRT built-in plugins for the logger's API line.
    /// 使用 logger 所属版本线初始化并注册 TensorRT 内置 plugin。
    /// </summary>
    /// <param name="logger">The logger used synchronously by vendor plugin initialization. vendor 初始化期间同步使用的 logger。</param>
    /// <param name="libNamespace">Optional namespace for the built-in plugin registrations. 内置 plugin 注册使用的可选 namespace。</param>
    /// <returns><see langword="true"/> when the vendor reports successful initialization. vendor 报告初始化成功时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// This is an explicit process-global registration operation. It does not create, return, or own plugin objects,
    /// and the logger is borrowed only for the synchronous vendor call.
    /// 这是显式的进程级注册操作；不会创建、返回或接管 plugin 对象，logger 只在同步 vendor 调用期间被借用。
    /// </remarks>
    public static bool InitializeBuiltInPlugins(TensorRtLogger logger, string? libNamespace = null)
    {
        if (logger == null)
        {
            throw new ArgumentNullException(nameof(logger));
        }

        NativeBridgeLoader.EnsureInitialized();
        return NativeBridgeApi.InitializeLibNvInferPlugins(logger.Line, logger.Handle, libNamespace);
    }

    /// <summary>
    /// Tries to initialize TensorRT built-in plugins and returns a bounded diagnostic.
    /// 尝试初始化 TensorRT 内置 plugin，并返回受控诊断。
    /// </summary>
    public static bool TryInitializeBuiltInPlugins(
        TensorRtLogger logger,
        string? libNamespace,
        out bool initialized,
        out string diagnostic)
    {
        try
        {
            initialized = InitializeBuiltInPlugins(logger, libNamespace);
            diagnostic = initialized ? "OK" : "TensorRT vendor reported plugin initialization failure.";
            return initialized;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            initialized = false;
            diagnostic = FormatProbeException("TensorRT built-in plugin initialization", exception);
            return false;
        }
    }

    /// <summary>
    /// Gets the current high-level bridge environment snapshot.
    /// 获取当前高层 bridge 环境快照。
    /// </summary>
    /// <returns>The current bridge environment snapshot. 当前 bridge 环境快照。</returns>
    public static TensorRtEnvironmentSnapshot GetCurrent()
    {
        NativeBridgeLoader.EnsureInitialized();

        var buildInfo = BridgeInfoMapper.ToManaged(NativeBridgeApi.GetBuildInfo());
        var runtimeInfo = BridgeInfoMapper.ToManaged(NativeBridgeApi.GetRuntimeInfo());
        var capabilityInfo = BridgeInfoMapper.ToManaged(NativeBridgeApi.GetCapabilityInfo());
        var trt8 = GetAdapterInfoOrFallback(TensorRtApiLine.TensorRt8, buildInfo);
        var trt10 = GetAdapterInfoOrFallback(TensorRtApiLine.TensorRt10, buildInfo);
        var trt11 = GetAdapterInfoOrFallback(TensorRtApiLine.TensorRt11, buildInfo);

        return new TensorRtEnvironmentSnapshot(buildInfo, runtimeInfo, capabilityInfo, trt8, trt10, trt11);
    }

    /// <summary>
    /// Gets TensorRT global runtime version details without creating a runtime.
    /// 无需创建 runtime 即可获取 TensorRT 全局 runtime 版本信息。
    /// </summary>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <returns>A global runtime version snapshot. 全局 runtime 版本快照。</returns>
    public static TensorRtGlobalRuntimeVersion GetGlobalRuntimeVersion(TensorRtApiLine line)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeBridgeApi.GetGlobalRuntimeVersion(line);
    }

    /// <summary>
    /// Tries to get TensorRT global runtime version details without throwing for unsupported lines.
    /// 尝试获取 TensorRT 全局 runtime 版本信息；不支持时不抛出异常。
    /// </summary>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="version">The version snapshot when the query succeeds. 查询成功时的版本快照。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the version was collected successfully. 成功采集版本信息时返回 <see langword="true"/>。</returns>
    public static bool TryGetGlobalRuntimeVersion(TensorRtApiLine line, out TensorRtGlobalRuntimeVersion? version, out string diagnostic)
    {
        try
        {
            version = GetGlobalRuntimeVersion(line);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            version = null;
            diagnostic = FormatProbeException("Global runtime version query", exception);
            return false;
        }
    }

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

    /// <summary>
    /// Probes native bridge, TensorRT, CUDA, cuDNN, and parser DLL resolution using managed OS APIs only.
    /// 仅使用托管 OS API 探测 native bridge、TensorRT、CUDA、cuDNN 与 parser DLL 的解析情况。
    /// </summary>
    /// <remarks>
    /// This diagnostic does not call TensorRT global version functions, global registry functions, runtime creation, or builder creation.
    /// It is intended to inspect DLL search-path drift before riskier vendor entry points are used.
    /// 该诊断不会调用 TensorRT 全局版本函数、全局 registry 函数、runtime 创建或 builder 创建；用于在调用风险更高的 vendor 入口点前检查 DLL 搜索路径漂移。
    /// </remarks>
    /// <param name="line">The TensorRT API line used to select expected TensorRT DLL file names. 用于选择预期 TensorRT DLL 文件名的 TensorRT API line。</param>
    /// <returns>A non-throwing native dependency probe report. 非抛异常的 native 依赖探测报告。</returns>
    public static TensorRtDependencyProbeReport ProbeNativeDependencies(TensorRtApiLine line)
    {
        List<string> diagnostics = new List<string>();
        bool bridgeInitialized;
        string bridgeDiagnostic;

        try
        {
            NativeBridgeLoader.EnsureInitialized();
            bridgeInitialized = true;
            bridgeDiagnostic = "Bridge loader initialized.";
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            bridgeInitialized = false;
            bridgeDiagnostic = FormatProbeException("Bridge initialization", exception);
        }

        IReadOnlyList<TensorRtNativeDependencyInfo> bridgeCandidates = EnumerateNativeBridgeCandidates(diagnostics);
        IReadOnlyList<TensorRtNativeDependencyInfo> loadedModules = EnumerateLoadedDependencyModules(diagnostics);
        IReadOnlyList<TensorRtNativeDependencyInfo> searchPathCandidates = EnumerateSearchPathDependencyCandidates(line, diagnostics);

        return new TensorRtDependencyProbeReport(line, bridgeInitialized, bridgeDiagnostic, bridgeCandidates, loadedModules, searchPathCandidates, diagnostics);
    }

    /// <summary>
    /// Runs staged TensorRT runtime probes and returns diagnostics for each stage.
    /// 运行 TensorRT runtime 分阶段探针，并返回每个阶段的诊断信息。
    /// </summary>
    /// <remarks>
    /// The probe intentionally separates global version, global registry, logger creation, builder creation, and runtime creation.
    /// 探针会刻意分离全局版本、全局 registry、logger 创建、builder 创建和 runtime 创建阶段。
    /// </remarks>
    /// <param name="line">The TensorRT API line to probe. 要探测的 TensorRT API line。</param>
    /// <returns>A staged runtime probe report. 分阶段 runtime 探针报告。</returns>
    public static TensorRtRuntimeProbeReport ProbeRuntime(TensorRtApiLine line)
    {
        List<TensorRtRuntimeProbeStage> stages = new List<TensorRtRuntimeProbeStage>();
        TensorRtGlobalRuntimeVersion? version = null;
        TensorRtPluginRegistryInventory? globalRegistry = null;

        try
        {
            NativeBridgeLoader.EnsureInitialized();
            stages.Add(new TensorRtRuntimeProbeStage("BridgeInitialized", succeeded: true, "Bridge loader initialized."));
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            stages.Add(new TensorRtRuntimeProbeStage("BridgeInitialized", succeeded: false, FormatProbeException("Bridge initialization", exception)));
            return new TensorRtRuntimeProbeReport(line, version, globalRegistry, stages);
        }

        bool inferVersionOk = TryAddProbeStage(stages, "GlobalInferLibVersion", () => NativeBridgeApi.GetGlobalInferLibVersion(line), value => $"Packed={value}", out int inferLibVersion);
        bool inferMajorOk = TryAddProbeStage(stages, "GlobalInferLibMajorVersion", () => NativeBridgeApi.GetGlobalInferLibMajorVersion(line), value => $"Major={value}", out int inferMajor);
        bool inferMinorOk = TryAddProbeStage(stages, "GlobalInferLibMinorVersion", () => NativeBridgeApi.GetGlobalInferLibMinorVersion(line), value => $"Minor={value}", out int inferMinor);
        bool inferPatchOk = TryAddProbeStage(stages, "GlobalInferLibPatchVersion", () => NativeBridgeApi.GetGlobalInferLibPatchVersion(line), value => $"Patch={value}", out int inferPatch);
        bool inferBuildOk = TryAddProbeStage(stages, "GlobalInferLibBuildVersion", () => NativeBridgeApi.GetGlobalInferLibBuildVersion(line), value => $"Build={value}", out int inferBuild);
        bool onnxParserOk = TryAddProbeStage(stages, "GlobalOnnxParserVersion", () => NativeBridgeApi.GetGlobalOnnxParserVersion(line), value => $"OnnxParser={value}", out int onnxParserVersion);
        bool globalLoggerOk = TryAddProbeStage(stages, "GlobalLogger", () => NativeBridgeApi.GlobalHasLogger(line), value => $"HasGlobalLogger={value}", out bool hasGlobalLogger);
        if (inferVersionOk && inferMajorOk && inferMinorOk && inferPatchOk && inferBuildOk && onnxParserOk && globalLoggerOk)
        {
            version = new TensorRtGlobalRuntimeVersion(line, inferLibVersion, inferMajor, inferMinor, inferPatch, inferBuild, onnxParserVersion, hasGlobalLogger);
        }

        if (TryGetGlobalPluginRegistryInventory(line, includeCreatorFields: false, out globalRegistry, out string registryDiagnostic))
        {
            string recursiveCount = globalRegistry?.RecursiveCreatorCount?.ToString() ?? "n/a";
            stages.Add(new TensorRtRuntimeProbeStage("GlobalPluginRegistry", succeeded: true, $"Creators={globalRegistry?.CreatorCount ?? 0} Recursive={recursiveCount} ParentSearch={globalRegistry?.ParentSearchEnabled ?? false} ErrorRecorder={globalRegistry?.HasErrorRecorder ?? false}"));
        }
        else
        {
            stages.Add(new TensorRtRuntimeProbeStage("GlobalPluginRegistry", succeeded: false, registryDiagnostic));
        }

        bool loggerOk = TryCreateLogger(line, out string loggerMessage);
        stages.Add(new TensorRtRuntimeProbeStage("LoggerCreate", loggerOk, loggerMessage));

        bool builderOk = TryCreateBuilder(line, out string builderMessage);
        stages.Add(new TensorRtRuntimeProbeStage("BuilderCreate", builderOk, builderMessage));

        bool runtimeOk = TryCreateRuntime(line, out string runtimeMessage);
        stages.Add(new TensorRtRuntimeProbeStage("RuntimeCreate", runtimeOk, runtimeMessage));

        return new TensorRtRuntimeProbeReport(line, version, globalRegistry, stages);
    }

    private static IReadOnlyList<TensorRtNativeDependencyInfo> EnumerateNativeBridgeCandidates(List<string> diagnostics)
    {
        List<TensorRtNativeDependencyInfo> results = new List<TensorRtNativeDependencyInfo>();
        HashSet<string> seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        try
        {
            foreach (string candidate in NativeBridgePathResolver.EnumerateCandidatePaths(typeof(NativeMethodsCommon).Assembly))
            {
                string path = NormalizeProbePath(candidate);
                if (string.IsNullOrWhiteSpace(path) || !seen.Add(path))
                {
                    continue;
                }

                results.Add(CreateNativeDependencyInfo(TensorRtNativeDependencySource.NativeBridgeCandidate, path, string.Empty));
            }
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            AddDependencyProbeDiagnostic(diagnostics, "Native bridge candidate enumeration failed", exception);
        }

        return results;
    }

    private static IReadOnlyList<TensorRtNativeDependencyInfo> EnumerateLoadedDependencyModules(List<string> diagnostics)
    {
        List<TensorRtNativeDependencyInfo> results = new List<TensorRtNativeDependencyInfo>();
        HashSet<string> seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        try
        {
            using Process process = Process.GetCurrentProcess();
            foreach (ProcessModule? module in process.Modules)
            {
                if (module is null)
                {
                    continue;
                }

                string moduleName = SafeGetModuleName(module, diagnostics);
                if (!IsInterestingDependencyName(moduleName))
                {
                    continue;
                }

                string modulePath = NormalizeProbePath(SafeGetModulePath(module, diagnostics));
                string key = !string.IsNullOrWhiteSpace(modulePath) ? modulePath : moduleName;
                if (string.IsNullOrWhiteSpace(key) || !seen.Add(key))
                {
                    continue;
                }

                results.Add(CreateNativeDependencyInfo(TensorRtNativeDependencySource.LoadedProcessModule, modulePath, moduleName));
            }
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            AddDependencyProbeDiagnostic(diagnostics, "Loaded process module enumeration failed", exception);
        }

        return results;
    }

    private static IReadOnlyList<TensorRtNativeDependencyInfo> EnumerateSearchPathDependencyCandidates(TensorRtApiLine line, List<string> diagnostics)
    {
        List<TensorRtNativeDependencyInfo> results = new List<TensorRtNativeDependencyInfo>();
        HashSet<string> seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        IReadOnlyList<string> patterns = GetDependencySearchPatterns(line);

        foreach (string directory in EnumerateDependencySearchDirectories(diagnostics))
        {
            foreach (string pattern in patterns)
            {
                try
                {
                    foreach (string candidate in Directory.EnumerateFiles(directory, pattern, SearchOption.TopDirectoryOnly))
                    {
                        string path = NormalizeProbePath(candidate);
                        if (string.IsNullOrWhiteSpace(path) || !seen.Add(path))
                        {
                            continue;
                        }

                        results.Add(CreateNativeDependencyInfo(TensorRtNativeDependencySource.SearchPathCandidate, path, string.Empty));
                    }
                }
                catch (Exception exception) when (IsProbeException(exception) || exception is IOException || exception is UnauthorizedAccessException)
                {
                    AddDependencyProbeDiagnostic(diagnostics, $"Dependency search failed in '{directory}' for '{pattern}'", exception);
                }
            }
        }

        return results;
    }

    private static IReadOnlyList<string> EnumerateDependencySearchDirectories(List<string> diagnostics)
    {
        List<string> results = new List<string>();
        HashSet<string> seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        void AddDirectory(string? directory)
        {
            string path = NormalizeProbePath(directory ?? string.Empty);
            if (string.IsNullOrWhiteSpace(path) || !seen.Add(path))
            {
                return;
            }

            try
            {
                if (Directory.Exists(path))
                {
                    results.Add(path);
                }
            }
            catch (Exception exception) when (IsProbeException(exception) || exception is IOException || exception is UnauthorizedAccessException)
            {
                AddDependencyProbeDiagnostic(diagnostics, $"Dependency directory check failed for '{path}'", exception);
            }
        }

        string currentPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        foreach (string entry in currentPath.Split(new[] { Path.PathSeparator }, StringSplitOptions.RemoveEmptyEntries))
        {
            AddDirectory(entry);
        }

        try
        {
            foreach (string entry in NativeBridgePathResolver.EnumerateDependencyDirectories(typeof(NativeMethodsCommon).Assembly))
            {
                AddDirectory(entry);
            }
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            AddDependencyProbeDiagnostic(diagnostics, "Dependency directory resolver enumeration failed", exception);
        }

        return results;
    }

    private static IReadOnlyList<string> GetDependencySearchPatterns(TensorRtApiLine line)
    {
        List<string> patterns = new List<string>();
        if (IsWindowsDependencyProbe())
        {
            switch (line)
            {
                case TensorRtApiLine.TensorRt10:
                    patterns.Add("nvinfer_10.dll");
                    patterns.Add("nvinfer_plugin_10.dll");
                    patterns.Add("nvonnxparser_10.dll");
                    break;
                case TensorRtApiLine.TensorRt11:
                    patterns.Add("nvinfer_11.dll");
                    patterns.Add("nvinfer_plugin_11.dll");
                    patterns.Add("nvonnxparser_11.dll");
                    break;
                default:
                    patterns.Add("nvinfer.dll");
                    patterns.Add("nvinfer_plugin.dll");
                    patterns.Add("nvonnxparser.dll");
                    break;
            }

            patterns.Add("cudart64_*.dll");
            patterns.Add("cudnn*.dll");
            patterns.Add("nvcuda.dll");
            patterns.Add(NativeBridgePathResolver.GetBridgeFileName());
            return patterns;
        }

        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                patterns.Add("libnvinfer.so.10*");
                patterns.Add("libnvinfer_plugin.so.10*");
                patterns.Add("libnvonnxparser.so.10*");
                break;
            case TensorRtApiLine.TensorRt11:
                patterns.Add("libnvinfer.so.11*");
                patterns.Add("libnvinfer_plugin.so.11*");
                patterns.Add("libnvonnxparser.so.11*");
                break;
            default:
                patterns.Add("libnvinfer.so*");
                patterns.Add("libnvinfer_plugin.so*");
                patterns.Add("libnvonnxparser.so*");
                break;
        }

        patterns.Add("libcudart.so*");
        patterns.Add("libcudnn.so*");
        patterns.Add("libcuda.so*");
        patterns.Add(NativeBridgePathResolver.GetBridgeFileName());
        return patterns;
    }

    private static bool IsWindowsDependencyProbe()
    {
#if JYPPX_NETFRAMEWORK
        PlatformID platform = Environment.OSVersion.Platform;
        return platform == PlatformID.Win32NT
            || platform == PlatformID.Win32S
            || platform == PlatformID.Win32Windows
            || platform == PlatformID.WinCE;
#else
        return RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
#endif
    }

    private static TensorRtNativeDependencyInfo CreateNativeDependencyInfo(TensorRtNativeDependencySource source, string path, string name)
    {
        string moduleName = !string.IsNullOrWhiteSpace(name) ? name : Path.GetFileName(path);
        bool exists = false;
        string fileVersion = string.Empty;
        string productVersion = string.Empty;
        string diagnostic = string.Empty;

        try
        {
            exists = !string.IsNullOrWhiteSpace(path) && File.Exists(path);
        }
        catch (Exception exception) when (IsProbeException(exception) || exception is IOException || exception is UnauthorizedAccessException)
        {
            diagnostic = $"Existence check failed: {exception.Message}";
        }

        if (exists)
        {
            try
            {
                FileVersionInfo versionInfo = FileVersionInfo.GetVersionInfo(path);
                fileVersion = versionInfo.FileVersion ?? string.Empty;
                productVersion = versionInfo.ProductVersion ?? string.Empty;
            }
            catch (Exception exception) when (IsProbeException(exception) || exception is IOException || exception is UnauthorizedAccessException)
            {
                diagnostic = $"Version metadata read failed: {exception.Message}";
            }
        }

        return new TensorRtNativeDependencyInfo(source, moduleName, path, exists, fileVersion, productVersion, diagnostic);
    }

    private static bool IsInterestingDependencyName(string moduleName)
    {
        if (string.IsNullOrWhiteSpace(moduleName))
        {
            return false;
        }

        string lower = moduleName.ToLowerInvariant();
        return lower.Contains("jyppxtrtbridge") ||
               lower.Contains("nvinfer") ||
               lower.Contains("nvonnxparser") ||
               lower.Contains("cudart") ||
               lower.Contains("cudnn") ||
               lower.Contains("nvcuda") ||
               lower.Contains("cuda");
    }

    private static string NormalizeProbePath(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        try
        {
            return Path.GetFullPath(value.Trim().Trim('"'));
        }
        catch (Exception exception) when (IsProbeException(exception) || exception is IOException || exception is UnauthorizedAccessException)
        {
            return value.Trim().Trim('"');
        }
    }

    private static string SafeGetModuleName(ProcessModule module, List<string> diagnostics)
    {
        try
        {
            return module.ModuleName ?? string.Empty;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            AddDependencyProbeDiagnostic(diagnostics, "Process module name read failed", exception);
            return string.Empty;
        }
    }

    private static string SafeGetModulePath(ProcessModule module, List<string> diagnostics)
    {
        try
        {
            return module.FileName ?? string.Empty;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            AddDependencyProbeDiagnostic(diagnostics, "Process module path read failed", exception);
            return string.Empty;
        }
    }

    private static void AddDependencyProbeDiagnostic(List<string> diagnostics, string stage, Exception exception)
    {
        if (diagnostics.Count >= 64)
        {
            return;
        }

        diagnostics.Add($"{stage}: {exception.Message}");
    }

    private static TensorRtAdapterInfo GetAdapterInfoOrFallback(TensorRtApiLine line, BridgeBuildInfo buildInfo)
    {
        try
        {
            return BridgeInfoMapper.ToManaged(NativeBridgeApi.GetAdapterInfo(line));
        }
        catch (BridgeProbeException exception) when (exception.StatusCode == BridgeStatusCode.NotSupported)
        {
            return new TensorRtAdapterInfo(
                line,
                buildInfo.HasTensorRt,
                runtimeCreationSupported: false,
                builderCreationSupported: false,
                networkCreationSupported: false,
                engineDeserializationSupported: false,
                buildInfo.TensorRtVersion,
                exception.Message);
        }
    }

    /// <summary>
    /// Tries to create a TensorRT logger for one API line.
    /// 尝试为一个 API line 创建 TensorRT logger。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when logger creation succeeds. logger 创建成功时返回 <see langword="true"/>。</returns>
    public static bool TryCreateLogger(TensorRtApiLine line, out string message)
    {
        NativeBridgeLoader.EnsureInitialized();

        try
        {
            using SafeTensorRtObjectHandle logger = NativeBridgeApi.CreateLogger(line);
            if (!logger.IsInvalid)
            {
                message = "Logger handle created successfully.";
                return true;
            }

            message = NativeBridgeApi.GetLastErrorMessageOrFallback("Logger creation failed.");
            return false;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            message = FormatProbeException("Logger creation", exception);
            return false;
        }
    }

    /// <summary>
    /// Tries to create a TensorRT runtime for one API line.
    /// 尝试为一个 API line 创建 TensorRT runtime。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when runtime creation succeeds. runtime 创建成功时返回 <see langword="true"/>。</returns>
    public static bool TryCreateRuntime(TensorRtApiLine line, out string message)
    {
        NativeBridgeLoader.EnsureInitialized();

        try
        {
            using SafeTensorRtObjectHandle logger = NativeBridgeApi.CreateLogger(line);
            return NativeBridgeApi.TryCreateRuntime(line, logger, out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
        catch (SEHException exception)
        {
            message = $"Runtime creation raised SEHException: {exception.Message}";
            return false;
        }
        catch (AccessViolationException exception)
        {
            message = $"Runtime creation raised AccessViolationException: {exception.Message}";
            return false;
        }
    }

    /// <summary>
    /// Gets a copied no-throw diagnostic snapshot for TensorRT runtime creation.
    /// 获取 TensorRT runtime 创建的复制型 no-throw 诊断快照。
    /// </summary>
    /// <param name="line">The TensorRT API line to diagnose. 要诊断的 TensorRT API line。</param>
    /// <returns>A pointer-free runtime creation diagnostic snapshot. 无指针 runtime 创建诊断快照。</returns>
    /// <remarks>
    /// The native diagnostic entry is currently implemented for TensorRT 11. Other lines return a managed not-supported snapshot.
    /// native 诊断入口目前仅实现 TensorRT 11；其他版本线返回托管 not-supported 快照。
    /// </remarks>
    public static TensorRtRuntimeCreateDiagnosticSnapshot GetRuntimeCreateDiagnostic(TensorRtApiLine line)
    {
        NativeBridgeLoader.EnsureInitialized();

        try
        {
            using SafeTensorRtObjectHandle logger = NativeBridgeApi.CreateLogger(line);
            return NativeBridgeApi.GetRuntimeCreateDiagnostic(line, logger);
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            return new TensorRtRuntimeCreateDiagnosticSnapshot(
                line,
                diagnosticAvailable: false,
                attempted: false,
                loggerHandlePresent: false,
                loggerPayloadPresent: false,
                createInferRuntimeReturnedNonNull: false,
                createInferRuntimeReturnedNull: false,
                lastStatus: exception is BridgeProbeException bridgeException ? bridgeException.StatusCode : BridgeStatusCode.RuntimeError,
                tensorRtAvailable: false,
                expectedMajor: line == TensorRtApiLine.TensorRt11 ? 11 : 0,
                bridgeBuiltMajor: 0,
                detectedVersion: string.Empty,
                loggerCallbackAvailable: false,
                loggerMessageCount: 0,
                lastLoggerSeverity: 0,
                lastLoggerMessage: string.Empty,
                createRuntimePhase: "managed-probe-exception",
                nativeDetail: "Managed probe caught an exception before a native diagnostic snapshot was available.",
                diagnostic: FormatProbeException("Runtime create diagnostic", exception));
        }
    }

    /// <summary>
    /// Tries to create a TensorRT builder for one API line.
    /// 尝试为一个 API line 创建 TensorRT builder。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when builder creation succeeds. builder 创建成功时返回 <see langword="true"/>。</returns>
    public static bool TryCreateBuilder(TensorRtApiLine line, out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryCreateBuilder(line, out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
        catch (SEHException exception)
        {
            message = $"Builder creation raised SEHException: {exception.Message}";
            return false;
        }
        catch (AccessViolationException exception)
        {
            message = $"Builder creation raised AccessViolationException: {exception.Message}";
            return false;
        }
    }

    /// <summary>
    /// Tries to run the minimal TensorRT 10 build chain.
    /// 尝试运行最小 TensorRT 10 构建链路。
    /// </summary>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when the minimal build chain succeeds. 最小构建链路成功时返回 <see langword="true"/>。</returns>
    public static bool TryRunTensorRt10MinimalBuildChain(out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryRunTrt10MinimalBuildChain(out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to build a TensorRT 10 serialized network without running inference.
    /// 尝试构建一个不执行推理的 TensorRT 10 serialized network。
    /// </summary>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when serialized-network build succeeds. serialized network 构建成功时返回 <see langword="true"/>。</returns>
    public static bool TryBuildTensorRt10SerializedNetworkOnly(out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryBuildTrt10SerializedNetworkOnly(out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to run the minimal TensorRT 8 build chain.
    /// 尝试运行最小 TensorRT 8 构建链路。
    /// </summary>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when the minimal build chain succeeds. 最小构建链路成功时返回 <see langword="true"/>。</returns>
    public static bool TryRunTensorRt8MinimalBuildChain(out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryRunTrt8MinimalBuildChain(out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to build a TensorRT 8 serialized network without running inference.
    /// 尝试构建一个不执行推理的 TensorRT 8 serialized network。
    /// </summary>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when serialized-network build succeeds. serialized network 构建成功时返回 <see langword="true"/>。</returns>
    public static bool TryBuildTensorRt8SerializedNetworkOnly(out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryBuildTrt8SerializedNetworkOnly(out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to run the minimal TensorRT 11 build chain.
    /// 尝试运行最小 TensorRT 11 构建链路。
    /// </summary>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when the minimal build chain succeeds. 最小构建链路成功时返回 <see langword="true"/>。</returns>
    public static bool TryRunTensorRt11MinimalBuildChain(out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryRunTrt11MinimalBuildChain(out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to build a TensorRT 11 serialized network without running inference.
    /// 尝试构建一个不执行推理的 TensorRT 11 serialized network。
    /// </summary>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when serialized-network build succeeds. serialized network 构建成功时返回 <see langword="true"/>。</returns>
    public static bool TryBuildTensorRt11SerializedNetworkOnly(out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryBuildTrt11SerializedNetworkOnly(out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
    }

    private static bool IsProbeException(Exception exception)
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

    private static string FormatProbeException(string stageName, Exception exception)
    {
        if (exception is SEHException)
        {
            return $"{stageName} raised SEHException: {exception.Message}";
        }

        if (exception is AccessViolationException)
        {
            return $"{stageName} raised AccessViolationException: {exception.Message}";
        }

        return exception.Message;
    }

    private static bool TryAddProbeStage<T>(List<TensorRtRuntimeProbeStage> stages, string stageName, Func<T> action, Func<T, string> formatMessage, out T value)
    {
        try
        {
            value = action();
            stages.Add(new TensorRtRuntimeProbeStage(stageName, succeeded: true, formatMessage(value)));
            return true;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            value = default!;
            stages.Add(new TensorRtRuntimeProbeStage(stageName, succeeded: false, FormatProbeException(stageName, exception)));
            return false;
        }
    }
}
