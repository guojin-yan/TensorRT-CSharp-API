namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes a native bridge, TensorRT, CUDA, cuDNN, or parser DLL discovered by a dependency probe.
/// 描述依赖探针发现的 native bridge、TensorRT、CUDA、cuDNN 或 parser DLL。
/// </summary>
public sealed class TensorRtNativeDependencyInfo
{
    internal TensorRtNativeDependencyInfo(
        TensorRtNativeDependencySource source,
        string name,
        string path,
        bool exists,
        string fileVersion,
        string productVersion,
        string diagnostic)
    {
        Source = source;
        Name = name ?? string.Empty;
        Path = path ?? string.Empty;
        Exists = exists;
        FileVersion = fileVersion ?? string.Empty;
        ProductVersion = productVersion ?? string.Empty;
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>
    /// Gets how this dependency path was discovered.
    /// 获取该依赖路径的发现方式。
    /// </summary>
    public TensorRtNativeDependencySource Source { get; }

    /// <summary>
    /// Gets the module or file name.
    /// 获取模块名或文件名。
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the full path when it is available.
    /// 获取可用时的完整路径。
    /// </summary>
    public string Path { get; }

    /// <summary>
    /// Gets whether the path exists at probe time.
    /// 获取探测时该路径是否存在。
    /// </summary>
    public bool Exists { get; }

    /// <summary>
    /// Gets the file version reported by Windows file-version metadata when available.
    /// 获取可用时 Windows 文件版本元数据报告的文件版本。
    /// </summary>
    public string FileVersion { get; }

    /// <summary>
    /// Gets the product version reported by Windows file-version metadata when available.
    /// 获取可用时 Windows 文件版本元数据报告的产品版本。
    /// </summary>
    public string ProductVersion { get; }

    /// <summary>
    /// Gets a non-throwing diagnostic collected while reading this entry.
    /// 获取读取该条目时收集的非抛异常诊断信息。
    /// </summary>
    public string Diagnostic { get; }

    /// <summary>
    /// Formats the dependency entry for diagnostics.
    /// 将依赖条目格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString()
    {
        string version = !string.IsNullOrWhiteSpace(FileVersion) ? FileVersion : ProductVersion;
        return $"{Source}:{Name}:{(Exists ? "exists" : "missing")}:{(string.IsNullOrWhiteSpace(version) ? "version=n/a" : $"version={version}")}";
    }
}
