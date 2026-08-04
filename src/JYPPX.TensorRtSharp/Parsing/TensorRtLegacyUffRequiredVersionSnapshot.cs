using System;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Contains the UFF version required by the TensorRT 8 legacy parser.
/// 包含 TensorRT 8 legacy parser 所需的 UFF 版本。
/// </summary>
public sealed class TensorRtLegacyUffRequiredVersionSnapshot
{
    internal TensorRtLegacyUffRequiredVersionSnapshot(TensorRtApiLine line, int major, int minor, int patch)
    {
        Line = line;
        Major = major;
        Minor = minor;
        Patch = patch;
    }

    /// <summary>Gets the TensorRT adapter line. 获取 TensorRT 适配版本线。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the required UFF major version. 获取所需 UFF 主版本。</summary>
    public int Major { get; }

    /// <summary>Gets the required UFF minor version. 获取所需 UFF 次版本。</summary>
    public int Minor { get; }

    /// <summary>Gets the required UFF patch version. 获取所需 UFF 修订版本。</summary>
    public int Patch { get; }

    /// <summary>Gets a managed version value. 获取托管版本值。</summary>
    public Version Version => new Version(Major, Minor, Patch);

    /// <summary>Gets whether the native parser is retained. 获取是否保留 native parser。</summary>
    public bool RetainsNativeParser => false;

    /// <summary>Gets whether process-global protobuf shutdown is called. 获取是否调用进程级 protobuf shutdown。</summary>
    public bool CallsProcessGlobalProtobufShutdown => false;

    /// <summary>Gets whether the snapshot is pointer-free. 获取快照是否不含指针。</summary>
    public bool PointerFreeCopiedMetadata => true;

    /// <summary>Formats the required version. 格式化所需版本。</summary>
    public override string ToString() => Version.ToString();
}
