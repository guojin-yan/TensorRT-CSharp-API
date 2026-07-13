namespace JYPPX.TensorRtSharp;

/// <summary>
/// TensorRT versioned-interface implementation language.
/// TensorRT versioned-interface 的实现语言。
/// </summary>
public enum TensorRtApiLanguage
{
    /// <summary>
    /// Unknown or unavailable language metadata. 未知或不可用的语言元数据。
    /// </summary>
    Unknown = -1,

    /// <summary>
    /// C++ implementation. C++ 实现。
    /// </summary>
    Cpp = 0,

    /// <summary>
    /// Python implementation. Python 实现。
    /// </summary>
    Python = 1
}

/// <summary>
/// Copied TensorRT versioned-interface metadata.
/// 从 TensorRT 复制出的 versioned-interface 元数据。
/// </summary>
public readonly struct TensorRtInterfaceInfo
{
    internal TensorRtInterfaceInfo(string kind, int major, int minor)
    {
        Kind = kind ?? string.Empty;
        Major = major;
        Minor = minor;
    }

    /// <summary>
    /// Gets the TensorRT interface kind string.
    /// 获取 TensorRT interface kind 字符串。
    /// </summary>
    public string Kind { get; }

    /// <summary>
    /// Gets the TensorRT interface major version.
    /// 获取 TensorRT interface major 版本。
    /// </summary>
    public int Major { get; }

    /// <summary>
    /// Gets the TensorRT interface minor version.
    /// 获取 TensorRT interface minor 版本。
    /// </summary>
    public int Minor { get; }

    /// <summary>
    /// Returns a compact display string for the interface metadata.
    /// 返回该 interface 元数据的简短显示字符串。
    /// </summary>
    /// <returns>A display string containing the kind and version. 包含 kind 和版本的显示字符串。</returns>
    public override string ToString() => $"{Kind}:{Major}.{Minor}";
}
