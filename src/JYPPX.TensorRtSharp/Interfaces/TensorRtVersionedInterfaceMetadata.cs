using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents copied metadata from a TensorRT versioned interface without retaining its native pointer.
/// 表示从 TensorRT versioned interface 复制出的元数据，不持有其原生指针。
/// </summary>
public sealed class TensorRtVersionedInterfaceMetadata
{
    internal TensorRtVersionedInterfaceMetadata(
        TensorRtApiLine line,
        TensorRtInterfaceInfo interfaceInfo,
        TensorRtApiLanguage apiLanguage)
    {
        Line = line;
        InterfaceInfo = interfaceInfo;
        ApiLanguage = apiLanguage;
    }

    /// <summary>Gets the TensorRT API line used for the query. 获取执行查询的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the copied interface kind and version. 获取复制出的 interface 类型与版本。</summary>
    public TensorRtInterfaceInfo InterfaceInfo { get; }

    /// <summary>Gets the implementation language reported by TensorRT. 获取 TensorRT 报告的实现语言。</summary>
    public TensorRtApiLanguage ApiLanguage { get; }

    /// <summary>Gets whether the metadata is copied and pointer-free. 获取该元数据是否为复制型且无指针。</summary>
    public bool PointerFreeCopiedMetadata => true;

    /// <summary>Gets whether this object owns or retains a native interface. 获取该对象是否拥有或持有原生 interface。</summary>
    public bool RetainsNativeInterface => false;

    /// <summary>Formats the copied metadata for diagnostics. 将复制型元数据格式化为诊断文本。</summary>
    public override string ToString()
    {
        return $"{Line}:{InterfaceInfo}:language={ApiLanguage}:pointerFree={PointerFreeCopiedMetadata}";
    }
}
