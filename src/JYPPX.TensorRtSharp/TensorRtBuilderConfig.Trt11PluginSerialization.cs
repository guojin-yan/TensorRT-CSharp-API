using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtBuilderConfig
{
    /// <summary>
    /// Sets the plugin library paths that TensorRT 11 should serialize with the generated engine plan.
    /// 设置 TensorRT 11 在生成 engine plan 时需要一并序列化的插件库路径。
    /// </summary>
    /// <param name="pluginLibraryPaths">
    /// Plugin library paths visible to TensorRT during build and deployment.
    /// 构建和部署阶段 TensorRT 可访问的插件库路径。
    /// </param>
    /// <returns>
    /// <see langword="true"/> when TensorRT accepted the plugin path list.
    /// 当 TensorRT 接受插件路径列表时返回 <see langword="true"/>。
    /// </returns>
    /// <remarks>
    /// Passing an empty collection clears the list. The bridge only passes path strings; it does not load,
    /// own, or redistribute plugin binaries.
    /// 传入空集合会清空列表。桥接层只传递路径字符串，不负责加载、持有或再分发插件二进制文件。
    /// </remarks>
    public bool SetPluginsToSerialize(IReadOnlyList<string> pluginLibraryPaths)
    {
        return NativeBridgeApi.SetBuilderConfigPluginsToSerialize(Line, _handle, pluginLibraryPaths);
    }

    /// <summary>
    /// Sets the plugin library paths that TensorRT 11 should serialize with the generated engine plan.
    /// 设置 TensorRT 11 在生成 engine plan 时需要一并序列化的插件库路径。
    /// </summary>
    /// <param name="pluginLibraryPaths">
    /// Plugin library paths visible to TensorRT during build and deployment.
    /// 构建和部署阶段 TensorRT 可访问的插件库路径。
    /// </param>
    /// <returns>
    /// <see langword="true"/> when TensorRT accepted the plugin path list.
    /// 当 TensorRT 接受插件路径列表时返回 <see langword="true"/>。
    /// </returns>
    public bool SetPluginsToSerialize(params string[] pluginLibraryPaths)
    {
        if (pluginLibraryPaths == null)
        {
            throw new ArgumentNullException(nameof(pluginLibraryPaths));
        }

        return SetPluginsToSerialize((IReadOnlyList<string>)pluginLibraryPaths);
    }
}
