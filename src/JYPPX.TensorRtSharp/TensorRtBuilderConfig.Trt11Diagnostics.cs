using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtBuilderConfig
{
    /// <summary>
    /// Resets this TensorRT 11 builder configuration back to TensorRT defaults.
    /// 将当前 TensorRT 11 构建配置重置为 TensorRT 默认状态。
    /// </summary>
    /// <remarks>
    /// Use this only before building or after you intentionally discard all current options. Existing optimization
    /// profiles, flags, timing cache references, and deployment options may be cleared by TensorRT.
    /// 仅应在构建前或明确要丢弃当前配置时调用；TensorRT 可能会清除已有 optimization profile、flag、timing cache 引用和部署选项。
    /// </remarks>
    public void Reset()
    {
        NativeBridgeApi.ResetBuilderConfig(Line, _handle);
    }

    /// <summary>
    /// Gets whether this TensorRT 11 builder configuration currently references a timing cache.
    /// 获取当前 TensorRT 11 构建配置是否已经绑定 timing cache。
    /// </summary>
    public bool HasTimingCache => NativeBridgeApi.HasBuilderConfigTimingCache(Line, _handle);

    /// <summary>
    /// Checks whether TensorRT reports that a layer can run on DLA for this configuration.
    /// 查询 TensorRT 是否认为指定 layer 可在当前配置下运行于 DLA。
    /// </summary>
    /// <param name="layer">The layer to query. / 要查询的 layer。</param>
    /// <returns><see langword="true"/> when TensorRT reports DLA support. / TensorRT 报告支持 DLA 时返回 <see langword="true"/>。</returns>
    public bool CanRunOnDla(TensorRtLayer layer)
    {
        ValidateLayer(layer);
        return NativeBridgeApi.CanBuilderConfigRunLayerOnDla(Line, _handle, layer.Handle);
    }

    /// <summary>
    /// Clears the plugin-library path list serialized into TensorRT 10/11 version-compatible engines.
    /// 清空会被序列化进 TensorRT 10/11 version-compatible engine 的插件库路径列表。
    /// </summary>
    public void ClearPluginsToSerialize()
    {
        NativeBridgeApi.ClearBuilderConfigPluginsToSerialize(Line, _handle);
    }

    /// <summary>
    /// Gets the number of plugin-library paths that TensorRT reports for version-compatible serialization.
    /// 获取 TensorRT 报告的 version-compatible 序列化插件库路径数量。
    /// </summary>
    /// <remarks>
    /// TensorRT 8 is supported for count-only compatibility. Path copying remains supported only for TensorRT 10/11.
    /// TensorRT 8 仅支持数量查询；路径复制仍仅支持 TensorRT 10/11。
    /// </remarks>
    public int PluginToSerializeCount => NativeBridgeApi.GetBuilderConfigPluginToSerializeCount(Line, _handle);

    /// <summary>
    /// Gets the count-only serialized plugin inventory across TensorRT 8, 10, and 11.
    /// 获取跨 TensorRT 8、10、11 的只读 serialized plugin 数量。
    /// </summary>
    public int SerializedPluginPathCountCompatibility => PluginToSerializeCount;

    /// <summary>
    /// Gets a plugin-library path from TensorRT's version-compatible serialization list.
    /// 从 TensorRT version-compatible 序列化列表中获取一个插件库路径。
    /// </summary>
    /// <param name="index">Zero-based plugin path index. / 从零开始的插件路径索引。</param>
    /// <returns>The plugin library path copied from TensorRT. / 从 TensorRT 复制出来的插件库路径。</returns>
    public string GetPluginToSerialize(int index)
    {
        return NativeBridgeApi.GetBuilderConfigPluginToSerialize(Line, _handle, index);
    }

    /// <summary>
    /// Gets every plugin-library path that TensorRT will serialize with a version-compatible engine.
    /// 获取 TensorRT 将随 version-compatible engine 序列化的全部插件库路径。
    /// </summary>
    /// <returns>
    /// A snapshot of plugin-library paths copied from TensorRT.
    /// 从 TensorRT 复制出来的插件库路径快照。
    /// </returns>
    /// <remarks>
    /// This is a read-only inventory helper; it does not load, create, own, or redistribute plugin binaries.
    /// 这是只读 inventory 辅助方法；它不会加载、创建、持有或再分发插件二进制。
    /// </remarks>
    public IReadOnlyList<string> GetPluginsToSerialize()
    {
        int count = PluginToSerializeCount;
        List<string> pluginLibraryPaths = new List<string>(count);
        for (int index = 0; index < count; index++)
        {
            pluginLibraryPaths.Add(GetPluginToSerialize(index));
        }

        return pluginLibraryPaths;
    }

    /// <summary>
    /// Tries to read the plugin-library path inventory without throwing for unsupported TensorRT lines.
    /// 尝试读取插件库路径 inventory；当当前 TensorRT 版本线不支持时不会抛出异常。
    /// </summary>
    /// <param name="pluginLibraryPaths">The copied plugin-library paths when the query succeeds. 查询成功时复制出的插件库路径。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when the inventory was read successfully. 成功读取 inventory 时返回 <see langword="true"/>。</returns>
    public bool TryGetPluginsToSerialize(out IReadOnlyList<string> pluginLibraryPaths, out string diagnostic)
    {
        try
        {
            pluginLibraryPaths = GetPluginsToSerialize();
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            pluginLibraryPaths = Array.Empty<string>();
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Gets a copied read-only snapshot of TensorRT serialized plugin path state.
    /// 获取 TensorRT serialized plugin path 状态的复制型只读快照。
    /// </summary>
    /// <returns>A copied snapshot containing count, optional copied paths, and diagnostics. 包含数量、可选路径副本和诊断信息的复制型快照。</returns>
    /// <remarks>
    /// TensorRT 8 supports count-only compatibility through this bridge. TensorRT 10/11 additionally support copied path inventory.
    /// This method does not load plugin libraries, create plugins, deserialize plugins, or expose TensorRT-owned pointers.
    /// TensorRT 8 通过该桥接层支持仅数量兼容查询；TensorRT 10/11 额外支持复制 path inventory。
    /// 该方法不会加载插件库、创建插件、反序列化插件或暴露 TensorRT 拥有的指针。
    /// </remarks>
    public TensorRtBuilderConfigSerializedPluginSnapshot GetSerializedPluginSnapshot()
    {
        int count = SerializedPluginPathCountCompatibility;
        if (TryGetPluginsToSerialize(out IReadOnlyList<string> pluginLibraryPaths, out string diagnostic))
        {
            return new TensorRtBuilderConfigSerializedPluginSnapshot(Line, count, pluginLibraryPaths, true, diagnostic);
        }

        return new TensorRtBuilderConfigSerializedPluginSnapshot(Line, count, Array.Empty<string>(), false, diagnostic);
    }

    /// <summary>
    /// Tries to get a copied read-only snapshot of TensorRT serialized plugin path state.
    /// 尝试获取 TensorRT serialized plugin path 状态的复制型只读快照。
    /// </summary>
    /// <param name="snapshot">The copied snapshot. 复制出的快照。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for path inventory unavailability. 描述成功或 path inventory 不可用原因的简短诊断。</param>
    /// <returns><see langword="true"/> when copied path inventory is available; count-only snapshots return <see langword="false"/> with a valid snapshot. 当复制 path inventory 可用时返回 <see langword="true"/>；仅数量快照会返回 <see langword="false"/>，但仍提供有效快照。</returns>
    public bool TryGetSerializedPluginSnapshot(out TensorRtBuilderConfigSerializedPluginSnapshot snapshot, out string diagnostic)
    {
        snapshot = GetSerializedPluginSnapshot();
        diagnostic = snapshot.Diagnostic;
        return snapshot.HasPathInventory;
    }

    /// <summary>
    /// Gets whether a TensorRT progress monitor is attached to this builder configuration.
    /// 获取当前 builder config 是否绑定了 TensorRT progress monitor；不会暴露 monitor 指针或接管其生命周期。
    /// </summary>
    public bool HasProgressMonitor => NativeBridgeApi.HasBuilderConfigProgressMonitor(Line, _handle);

    /// <summary>
    /// Attaches a managed TensorRT progress monitor to this builder configuration.
    /// 将托管 TensorRT progress monitor 绑定到当前 builder config。
    /// </summary>
    /// <param name="monitor">The managed progress monitor to borrow. 要借用的托管 progress monitor。</param>
    /// <remarks>
    /// TensorRT borrows the native monitor pointer and does not take ownership. This builder config keeps the managed
    /// monitor alive until <see cref="ClearProgressMonitor"/> or <see cref="Dispose"/> detaches it. Dispose the builder config
    /// or clear the monitor before disposing the monitor when possible; if the monitor is disposed first, native release is
    /// deferred until this config detaches it.
    /// TensorRT 只借用 native monitor 指针，不接管所有权。当前 builder config 会保持托管 monitor 存活，直到
    /// <see cref="ClearProgressMonitor"/> 或 <see cref="Dispose"/> 解除绑定。建议先释放 builder config 或清除 monitor 再释放 monitor；
    /// 如果先释放 monitor，native 释放会延迟到 config 解除绑定之后。
    /// </remarks>
    public void SetProgressMonitor(TensorRtProgressMonitor monitor)
    {
        if (monitor == null)
        {
            throw new ArgumentNullException(nameof(monitor));
        }

        ThrowIfDisposed();
        monitor.ThrowIfDisposed();
        monitor.AttachBorrower(Line);
        try
        {
            NativeBridgeApi.SetBuilderConfigProgressMonitor(Line, _handle, monitor.Handle);
            TensorRtProgressMonitor? previous = _progressMonitorKeepAlive;
            _progressMonitorKeepAlive = monitor;
            previous?.DetachBorrower();
        }
        catch
        {
            monitor.DetachBorrower();
            throw;
        }
    }

    /// <summary>
    /// Clears any TensorRT progress monitor attached to this builder configuration.
    /// 清除当前 builder config 上绑定的 TensorRT progress monitor；不会调用用户回调。
    /// </summary>
    public void ClearProgressMonitor()
    {
        ThrowIfDisposed();
        NativeBridgeApi.ClearBuilderConfigProgressMonitor(Line, _handle);
        DetachProgressMonitor();
    }

}
