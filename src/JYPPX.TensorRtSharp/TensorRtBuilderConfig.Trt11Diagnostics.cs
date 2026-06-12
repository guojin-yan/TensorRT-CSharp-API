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
    /// Clears the plugin-library path list serialized into TensorRT 11 version-compatible engines.
    /// 清空会被序列化进 TensorRT 11 version-compatible engine 的插件库路径列表。
    /// </summary>
    public void ClearPluginsToSerialize()
    {
        NativeBridgeApi.ClearBuilderConfigPluginsToSerialize(Line, _handle);
    }

    /// <summary>
    /// Gets the number of plugin-library paths that TensorRT will serialize with compatible engines.
    /// 获取 TensorRT 将随兼容 engine 序列化的插件库路径数量。
    /// </summary>
    public int PluginToSerializeCount => NativeBridgeApi.GetBuilderConfigPluginToSerializeCount(Line, _handle);

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
    /// Gets whether a TensorRT progress monitor is attached to this builder configuration.
    /// 获取当前 builder config 是否绑定了 TensorRT progress monitor。
    /// </summary>
    public bool HasProgressMonitor => NativeBridgeApi.HasBuilderConfigProgressMonitor(Line, _handle);

    /// <summary>
    /// Clears any TensorRT progress monitor attached to this builder configuration.
    /// 清除当前 builder config 上绑定的 TensorRT progress monitor。
    /// </summary>
    public void ClearProgressMonitor()
    {
        NativeBridgeApi.ClearBuilderConfigProgressMonitor(Line, _handle);
    }

}
