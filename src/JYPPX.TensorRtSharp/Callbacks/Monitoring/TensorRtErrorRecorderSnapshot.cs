using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a copied read-only snapshot of a TensorRT error recorder.
/// 表示 TensorRT error recorder 的只读托管快照。
/// </summary>
public sealed class TensorRtErrorRecorderSnapshot
{
    internal TensorRtErrorRecorderSnapshot(
        TensorRtApiLine line,
        bool hasRecorder,
        int errorCount,
        bool hasOverflowed,
        IReadOnlyList<TensorRtErrorRecord> records)
        : this(
            line,
            hasRecorder,
            errorCount,
            hasOverflowed,
            interfaceInfoAvailable: false,
            interfaceInfo: new TensorRtInterfaceInfo(string.Empty, 0, 0),
            records)
    {
    }

    internal TensorRtErrorRecorderSnapshot(
        TensorRtApiLine line,
        bool hasRecorder,
        int errorCount,
        bool hasOverflowed,
        bool interfaceInfoAvailable,
        TensorRtInterfaceInfo interfaceInfo,
        IReadOnlyList<TensorRtErrorRecord> records)
    {
        Line = line;
        HasRecorder = hasRecorder;
        ErrorCount = errorCount < 0 ? 0 : errorCount;
        HasOverflowed = hasOverflowed;
        InterfaceInfoAvailable = interfaceInfoAvailable;
        InterfaceInfo = interfaceInfo;
        Records = records ?? Array.Empty<TensorRtErrorRecord>();
    }

    /// <summary>
    /// Gets the TensorRT API line used to collect this snapshot.
    /// 获取采集该快照的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets whether the source object had an error recorder attached during the snapshot.
    /// 获取采集快照时来源对象是否附加了 error recorder。
    /// </summary>
    public bool HasRecorder { get; }

    /// <summary>
    /// Gets the number of errors reported by the recorder at snapshot time.
    /// 获取采集快照时 recorder 报告的错误数量。
    /// </summary>
    public int ErrorCount { get; }

    /// <summary>
    /// Gets whether the recorder reported that errors had overflowed.
    /// 获取 recorder 是否报告曾发生错误溢出。
    /// </summary>
    public bool HasOverflowed { get; }

    /// <summary>
    /// Gets whether TensorRT reported copied versioned-interface metadata for the attached recorder.
    /// 获取 TensorRT 是否为已附加 recorder 报告复制型 versioned-interface 元数据。
    /// </summary>
    public bool InterfaceInfoAvailable { get; }

    /// <summary>
    /// Gets copied TensorRT interface metadata for the recorder when available.
    /// 获取已复制的 recorder TensorRT interface 元数据；可用性由 <see cref="InterfaceInfoAvailable"/> 指示。
    /// </summary>
    public TensorRtInterfaceInfo InterfaceInfo { get; }

    /// <summary>
    /// Gets copied error records. No native recorder pointer is exposed or retained.
    /// 获取已复制的错误记录；不会暴露或持有原生 recorder 指针。
    /// </summary>
    public IReadOnlyList<TensorRtErrorRecord> Records { get; }

    /// <summary>
    /// Converts the copied snapshot into a compact pointer-free summary.
    /// 将已复制 snapshot 转换为紧凑的无指针摘要。
    /// </summary>
    /// <returns>A copied summary suitable for logs, smoke output, and package-consumer probes. 适用于日志、smoke 输出和包消费验证的复制型摘要。</returns>
    public TensorRtErrorRecorderSummary ToSummary()
    {
        TensorRtErrorRecord? firstRecord = Records.Count > 0 ? Records[0] : null;
        string firstDescription = firstRecord?.Description ?? string.Empty;
        return new TensorRtErrorRecorderSummary(
            Line,
            HasRecorder,
            ErrorCount,
            Records.Count,
            HasOverflowed,
            InterfaceInfoAvailable,
            InterfaceInfo.Kind,
            InterfaceInfo.Major,
            InterfaceInfo.Minor,
            firstRecord?.Code,
            string.IsNullOrWhiteSpace(firstDescription) ? 0 : firstDescription.Length);
    }

    /// <summary>
    /// Converts the snapshot to a compact diagnostic string.
    /// 转换为简短诊断字符串。
    /// </summary>
    /// <returns>A compact diagnostic string. 简短诊断字符串。</returns>
    public override string ToString()
    {
        string interfaceInfo = InterfaceInfoAvailable ? InterfaceInfo.ToString() : "n/a";
        return $"{Line}:hasRecorder={HasRecorder}:errors={ErrorCount}:overflow={HasOverflowed}:interface={interfaceInfo}:copied={Records.Count}";
    }
}
