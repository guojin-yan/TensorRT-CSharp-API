using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

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

    public TensorRtApiLine Line { get; }

    public int DlaCore { get; }

    public int DlaCoreCount { get; }

    public int MaxThreads { get; }

    public bool EngineHostCodeAllowed { get; }

    public TensorRtTempfileControlFlags TempfileControlFlags { get; }

    public string TemporaryDirectory { get; }

    public bool HasLogger { get; }

    public bool HasErrorRecorder { get; }

    public TensorRtErrorRecorderSnapshot ErrorRecorder { get; }

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

    public override string ToString()
    {
        return $"Line={(int)Line} DlaCore={DlaCore}/{DlaCoreCount} MaxThreads={MaxThreads} HostCode={EngineHostCodeAllowed} TempFlags={TempfileControlFlags} Logger={HasLogger} ErrorRecorder={HasErrorRecorder}/{ErrorRecorder.ErrorCount} diagnostics={Diagnostics.Count}";
    }
}

/// <summary>
/// Compact pointer-free summary for a TensorRT runtime diagnostic snapshot.
/// TensorRT runtime 诊断快照的简短无指针摘要。
/// </summary>
public sealed class TensorRtRuntimeDiagnosticSummary
{
    internal TensorRtRuntimeDiagnosticSummary(
        TensorRtApiLine line,
        int dlaCore,
        int dlaCoreCount,
        int maxThreads,
        bool engineHostCodeAllowed,
        TensorRtTempfileControlFlags tempfileControlFlags,
        bool hasTemporaryDirectory,
        bool hasLogger,
        bool hasErrorRecorder,
        int errorCount,
        int copiedErrorRecordCount,
        bool hasErrorOverflowed,
        int diagnosticCount)
    {
        Line = line;
        DlaCore = dlaCore;
        DlaCoreCount = dlaCoreCount;
        MaxThreads = maxThreads;
        EngineHostCodeAllowed = engineHostCodeAllowed;
        TempfileControlFlags = tempfileControlFlags;
        HasTemporaryDirectory = hasTemporaryDirectory;
        HasLogger = hasLogger;
        HasErrorRecorder = hasErrorRecorder;
        ErrorCount = errorCount;
        CopiedErrorRecordCount = copiedErrorRecordCount;
        HasErrorOverflowed = hasErrorOverflowed;
        DiagnosticCount = diagnosticCount;
    }

    /// <summary>
    /// Gets the TensorRT API line used to collect the source snapshot.
    /// 获取采集源快照的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the selected DLA core value copied from the runtime snapshot.
    /// 获取从 runtime 快照复制出的 DLA core 值。
    /// </summary>
    public int DlaCore { get; }

    /// <summary>
    /// Gets the DLA core count copied from the runtime snapshot.
    /// 获取从 runtime 快照复制出的 DLA core 数量。
    /// </summary>
    public int DlaCoreCount { get; }

    /// <summary>
    /// Gets the max thread count copied from the runtime snapshot.
    /// 获取从 runtime 快照复制出的最大线程数。
    /// </summary>
    public int MaxThreads { get; }

    /// <summary>
    /// Gets whether engine host code was allowed in the runtime snapshot.
    /// 获取 runtime 快照中是否允许 engine host code。
    /// </summary>
    public bool EngineHostCodeAllowed { get; }

    /// <summary>
    /// Gets the copied temporary-file control flags.
    /// 获取复制出的 temporary-file control flags。
    /// </summary>
    public TensorRtTempfileControlFlags TempfileControlFlags { get; }

    /// <summary>
    /// Gets whether the copied snapshot reported a non-empty temporary directory.
    /// 获取复制快照是否报告非空临时目录。
    /// </summary>
    public bool HasTemporaryDirectory { get; }

    /// <summary>
    /// Gets whether the runtime snapshot reported a logger.
    /// 获取 runtime 快照是否报告 logger。
    /// </summary>
    public bool HasLogger { get; }

    /// <summary>
    /// Gets whether the runtime snapshot reported an error recorder.
    /// 获取 runtime 快照是否报告 error recorder。
    /// </summary>
    public bool HasErrorRecorder { get; }

    /// <summary>
    /// Gets the copied error count.
    /// 获取复制出的错误数量。
    /// </summary>
    public int ErrorCount { get; }

    /// <summary>
    /// Gets the number of copied error records.
    /// 获取复制出的 error record 数量。
    /// </summary>
    public int CopiedErrorRecordCount { get; }

    /// <summary>
    /// Gets whether the copied error recorder snapshot overflowed.
    /// 获取复制出的 error recorder 快照是否发生溢出。
    /// </summary>
    public bool HasErrorOverflowed { get; }

    /// <summary>
    /// Gets the number of diagnostics collected while creating the snapshot.
    /// 获取创建快照时收集到的诊断数量。
    /// </summary>
    public int DiagnosticCount { get; }

    /// <summary>
    /// Gets the runtime evidence kind represented by this copied summary.
    /// 获取该 copied summary 表示的 runtime evidence 类型。
    /// </summary>
    public string RuntimeEvidenceKind => "copied-readonly-summary";

    /// <summary>
    /// Gets whether this summary is runtime execution evidence.
    /// 获取该摘要是否为 runtime execution evidence。
    /// </summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>
    /// Gets whether this summary is runtime execution proof.
    /// 获取该摘要是否为 runtime execution proof。
    /// </summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>
    /// Gets whether this summary is copied and pointer-free.
    /// 获取该摘要是否为复制型且不暴露指针。
    /// </summary>
    public bool PointerFreeCopiedSummary => true;

    /// <summary>
    /// Gets whether this summary can be promoted as runtime proof.
    /// 获取该摘要是否可晋级为 runtime proof。
    /// </summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>
    /// Gets whether this summary can promote public release proof.
    /// 获取该摘要是否可晋级为 public release proof。
    /// </summary>
    public bool CanPromoteReleaseProof => false;

    /// <summary>
    /// Gets whether this summary allows deleting deferred records.
    /// 获取该摘要是否允许删除 deferred 记录。
    /// </summary>
    public bool CanDeleteDeferredRecord => false;

    /// <summary>
    /// Converts the summary to a compact diagnostic string.
    /// 转换为简短诊断字符串。
    /// </summary>
    /// <returns>A compact diagnostic string. 简短诊断字符串。</returns>
    public override string ToString()
    {
        return $"Line={(int)Line} Dla={DlaCore}/{DlaCoreCount} MaxThreads={MaxThreads} HostCode={EngineHostCodeAllowed} TempFlags={TempfileControlFlags} TempDir={HasTemporaryDirectory} Logger={HasLogger} ErrorRecorder={HasErrorRecorder}/{ErrorCount}/{CopiedErrorRecordCount}/Overflow={HasErrorOverflowed} Diagnostics={DiagnosticCount} RuntimeProof={CanPromoteRuntimeProof}";
    }
}
