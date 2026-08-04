using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Read-only TensorRT global runtime version details collected without creating a runtime.
/// 无需创建 runtime 即可采集的 TensorRT 全局 runtime 版本信息。
/// </summary>
public sealed class TensorRtGlobalRuntimeVersion
{
    internal TensorRtGlobalRuntimeVersion(
        TensorRtApiLine line,
        int inferLibVersion,
        int major,
        int minor,
        int patch,
        int build,
        int onnxParserVersion,
        bool hasGlobalLogger)
    {
        Line = line;
        InferLibVersion = inferLibVersion;
        Major = major;
        Minor = minor;
        Patch = patch;
        Build = build;
        OnnxParserVersion = onnxParserVersion;
        HasGlobalLogger = hasGlobalLogger;
    }

    /// <summary>
    /// Gets the TensorRT API line used for the query.
    /// 获取查询使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the packed TensorRT infer library version reported by TensorRT.
    /// 获取 TensorRT 报告的 packed infer library version。
    /// </summary>
    public int InferLibVersion { get; }

    /// <summary>
    /// Gets the TensorRT infer library major version.
    /// 获取 TensorRT infer library major version。
    /// </summary>
    public int Major { get; }

    /// <summary>
    /// Gets the TensorRT infer library minor version.
    /// 获取 TensorRT infer library minor version。
    /// </summary>
    public int Minor { get; }

    /// <summary>
    /// Gets the TensorRT infer library patch version.
    /// 获取 TensorRT infer library patch version。
    /// </summary>
    public int Patch { get; }

    /// <summary>
    /// Gets the TensorRT infer library build version.
    /// 获取 TensorRT infer library build version。
    /// </summary>
    public int Build { get; }

    /// <summary>
    /// Gets the ONNX parser version reported by TensorRT's parser library.
    /// 获取 TensorRT ONNX parser library 报告的版本。
    /// </summary>
    public int OnnxParserVersion { get; }

    /// <summary>
    /// Gets whether TensorRT exposes a non-null global logger pointer.
    /// 获取 TensorRT 是否暴露非空全局 logger 指针。
    /// </summary>
    public bool HasGlobalLogger { get; }

    /// <summary>
    /// Formats the global runtime version snapshot for diagnostics.
    /// 将全局 runtime 版本快照格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString() => $"{Line}:{Major}.{Minor}.{Patch}.{Build}:packed={InferLibVersion}:onnx={OnnxParserVersion}:globalLogger={HasGlobalLogger}";
}
