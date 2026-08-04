using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Captures copied read-only diagnostics for a TensorRT runtime instance.
/// 捕获 TensorRT runtime 实例的复制型只读诊断信息。
/// </summary>
public sealed class TensorRtRuntimeDiagnosticSnapshot
{
    internal TensorRtRuntimeDiagnosticSnapshot(
        TensorRtApiLine line,
        int dlaCore,
        int dlaCoreCount,
        int maxThreads,
        bool engineHostCodeAllowed,
        TensorRtTempfileControlFlags tempfileControlFlags,
        string temporaryDirectory,
        bool hasLogger,
        bool hasErrorRecorder,
        TensorRtErrorRecorderSnapshot errorRecorder,
        IReadOnlyList<string> diagnostics)
    {
        Line = line;
        DlaCore = dlaCore;
        DlaCoreCount = dlaCoreCount;
        MaxThreads = maxThreads;
        EngineHostCodeAllowed = engineHostCodeAllowed;
        TempfileControlFlags = tempfileControlFlags;
        TemporaryDirectory = temporaryDirectory ?? string.Empty;
        HasLogger = hasLogger;
        HasErrorRecorder = hasErrorRecorder;
        ErrorRecorder = errorRecorder ?? new TensorRtErrorRecorderSnapshot(line, false, 0, false, Array.Empty<TensorRtErrorRecord>());
        Diagnostics = diagnostics ?? Array.Empty<string>();
    }

    /// <summary>
    /// Gets the TensorRT API line used to collect this snapshot.
    /// 获取采集该快照的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the selected DLA core copied from the runtime.
    /// 获取从 runtime 复制出的已选 DLA core。
    /// </summary>
    public int DlaCore { get; }

    /// <summary>
    /// Gets the available DLA core count copied from the runtime.
    /// 获取从 runtime 复制出的可用 DLA core 数量。
    /// </summary>
    public int DlaCoreCount { get; }

    /// <summary>
    /// Gets the maximum thread count copied from the runtime.
    /// 获取从 runtime 复制出的最大线程数。
    /// </summary>
    public int MaxThreads { get; }

    /// <summary>
    /// Gets whether engine host code was allowed by the runtime.
    /// 获取 runtime 是否允许 engine host code。
    /// </summary>
    public bool EngineHostCodeAllowed { get; }

    /// <summary>
    /// Gets the temporary-file control flags copied from the runtime.
    /// 获取从 runtime 复制出的 temporary-file control flags。
    /// </summary>
    public TensorRtTempfileControlFlags TempfileControlFlags { get; }

    /// <summary>
    /// Gets the temporary directory copied from the runtime.
    /// 获取从 runtime 复制出的临时目录。
    /// </summary>
    public string TemporaryDirectory { get; }

    /// <summary>
    /// Gets whether the runtime reported an associated logger.
    /// 获取 runtime 是否报告关联的 logger。
    /// </summary>
    public bool HasLogger { get; }

    /// <summary>
    /// Gets whether the runtime reported an associated error recorder.
    /// 获取 runtime 是否报告关联的 error recorder。
    /// </summary>
    public bool HasErrorRecorder { get; }

    /// <summary>
    /// Gets the copied error-recorder snapshot.
    /// 获取复制出的 error recorder 快照。
    /// </summary>
    public TensorRtErrorRecorderSnapshot ErrorRecorder { get; }

    /// <summary>
    /// Gets diagnostics collected while creating the snapshot.
    /// 获取创建快照时收集的诊断信息。
    /// </summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>
    /// Creates a compact pointer-free summary from this copied runtime diagnostic snapshot.
    /// 从当前已复制的 runtime 诊断快照创建简短无指针摘要。
    /// </summary>
    /// <remarks>
    /// This method only reads managed snapshot values. It does not call TensorRT, expose native pointers, or promote runtime proof.
    /// 该方法只读取托管快照值；不会调用 TensorRT、暴露 native 指针，也不会提升 runtime proof。
    /// </remarks>
    /// <returns>A pointer-free runtime diagnostic summary. 无指针 runtime 诊断摘要。</returns>
    public TensorRtRuntimeDiagnosticSummary ToSummary()
    {
        return new TensorRtRuntimeDiagnosticSummary(
            Line,
            DlaCore,
            DlaCoreCount,
            MaxThreads,
            EngineHostCodeAllowed,
            TempfileControlFlags,
            !string.IsNullOrWhiteSpace(TemporaryDirectory),
            HasLogger,
            HasErrorRecorder,
            ErrorRecorder.ErrorCount,
            ErrorRecorder.Records.Count,
            ErrorRecorder.HasOverflowed,
            Diagnostics.Count);
    }

    /// <summary>
    /// Converts the snapshot to a compact diagnostic string.
    /// 将快照转换为简短诊断字符串。
    /// </summary>
    /// <returns>A compact diagnostic string. 简短诊断字符串。</returns>
    public override string ToString()
    {
        return $"Line={(int)Line} DlaCore={DlaCore}/{DlaCoreCount} MaxThreads={MaxThreads} HostCode={EngineHostCodeAllowed} TempFlags={TempfileControlFlags} Logger={HasLogger} ErrorRecorder={HasErrorRecorder}/{ErrorRecorder.ErrorCount} diagnostics={Diagnostics.Count}";
    }
}
