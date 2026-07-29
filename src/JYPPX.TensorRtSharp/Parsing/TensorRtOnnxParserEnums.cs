using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a bitmask of ONNX parser flags.
/// 表示 ONNX parser 标志位掩码。
/// </summary>
[Flags]
public enum TensorRtOnnxParserFlags : uint
{
    /// <summary>
    /// No parser flags are enabled.
    /// 不启用任何 parser 标志。
    /// </summary>
    None = 0,

    /// <summary>
    /// Prefer TensorRT native instance-normalization handling when supported.
    /// 在支持时优先使用 TensorRT 原生 instance normalization 处理。
    /// </summary>
    NativeInstanceNormalization = 1u << (int)TensorRtOnnxParserFlag.NativeInstanceNormalization,

    /// <summary>
    /// Enables TensorRT 10 parser handling for UInt8 and asymmetric quantization on DLA.
    /// 启用 TensorRT 10 parser 对 DLA UInt8 和非对称量化的处理。
    /// </summary>
    EnableUInt8AndAsymmetricQuantizationDla = 1u << (int)TensorRtOnnxParserFlag.EnableUInt8AndAsymmetricQuantizationDla,

    /// <summary>
    /// Reports TensorRT 11 parser capability for DLA deployment.
    /// 报告 TensorRT 11 parser 的 DLA 部署能力。
    /// </summary>
    ReportCapabilityDla = 1u << (int)TensorRtOnnxParserFlag.ReportCapabilityDla,

    /// <summary>
    /// Enables TensorRT 11 plugin override handling during ONNX parsing.
    /// 启用 TensorRT 11 ONNX 解析期间的 plugin override 处理。
    /// </summary>
    EnablePluginOverride = 1u << (int)TensorRtOnnxParserFlag.EnablePluginOverride,

    /// <summary>
    /// Adjusts the TensorRT 11 parsed network for DLA deployment.
    /// 针对 DLA 部署调整 TensorRT 11 解析后的 network。
    /// </summary>
    AdjustForDla = 1u << (int)TensorRtOnnxParserFlag.AdjustForDla
}
/// <summary>
/// Identifies an individual ONNX parser flag.
/// 标识单个 ONNX parser 标志。
/// </summary>
public enum TensorRtOnnxParserFlag
{
    /// <summary>
    /// Native instance-normalization parser flag.
    /// 原生 instance normalization parser 标志。
    /// </summary>
    NativeInstanceNormalization = 0,

    /// <summary>
    /// TensorRT 10 UInt8 and asymmetric quantization DLA parser flag.
    /// TensorRT 10 的 UInt8 与 DLA 非对称量化 parser 标志。
    /// </summary>
    EnableUInt8AndAsymmetricQuantizationDla = 1,

    /// <summary>
    /// TensorRT 11 DLA capability-report parser flag.
    /// TensorRT 11 DLA 能力报告 parser 标志。
    /// </summary>
    ReportCapabilityDla = 2,

    /// <summary>
    /// TensorRT 11 plugin-override parser flag.
    /// TensorRT 11 plugin override parser 标志。
    /// </summary>
    EnablePluginOverride = 3,

    /// <summary>
    /// TensorRT 11 DLA-adjustment parser flag.
    /// TensorRT 11 DLA 调整 parser 标志。
    /// </summary>
    AdjustForDla = 4
}
