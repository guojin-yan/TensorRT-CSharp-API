using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Compact pointer-free summary for a TensorRT refitter diagnostic snapshot.
/// TensorRT refitter 诊断快照的简短无指针摘要。
/// </summary>
public sealed class TensorRtRefitterDiagnosticSummary
{
    internal TensorRtRefitterDiagnosticSummary(
        TensorRtApiLine line,
        int maxThreads,
        bool weightsValidation,
        bool hasLogger,
        bool hasErrorRecorder,
        int errorCount,
        int copiedErrorRecordCount,
        bool hasErrorOverflowed,
        int dynamicRangeTensorCount,
        int copiedDynamicRangeTensorNameCount,
        int missingNamedWeightCount,
        int copiedMissingNamedWeightCount,
        int allNamedWeightCount,
        int copiedAllNamedWeightCount,
        int diagnosticCount)
    {
        Line = line;
        MaxThreads = maxThreads;
        WeightsValidation = weightsValidation;
        HasLogger = hasLogger;
        HasErrorRecorder = hasErrorRecorder;
        ErrorCount = errorCount;
        CopiedErrorRecordCount = copiedErrorRecordCount;
        HasErrorOverflowed = hasErrorOverflowed;
        DynamicRangeTensorCount = dynamicRangeTensorCount;
        CopiedDynamicRangeTensorNameCount = copiedDynamicRangeTensorNameCount;
        MissingNamedWeightCount = missingNamedWeightCount;
        CopiedMissingNamedWeightCount = copiedMissingNamedWeightCount;
        AllNamedWeightCount = allNamedWeightCount;
        CopiedAllNamedWeightCount = copiedAllNamedWeightCount;
        DiagnosticCount = diagnosticCount;
    }

    /// <summary>
    /// Gets the TensorRT API line used to collect the source snapshot.
    /// 获取采集源快照的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the max thread count copied from the refitter snapshot.
    /// 获取从 refitter 快照复制出的最大线程数。
    /// </summary>
    public int MaxThreads { get; }

    /// <summary>
    /// Gets whether weights validation was enabled in the refitter snapshot.
    /// 获取 refitter 快照中是否启用 weights validation。
    /// </summary>
    public bool WeightsValidation { get; }

    /// <summary>
    /// Gets whether the refitter snapshot reported a logger.
    /// 获取 refitter 快照是否报告 logger。
    /// </summary>
    public bool HasLogger { get; }

    /// <summary>
    /// Gets whether the refitter snapshot reported an error recorder.
    /// 获取 refitter 快照是否报告 error recorder。
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
    /// Gets the copied dynamic range tensor count.
    /// 获取复制出的 dynamic range tensor 数量。
    /// </summary>
    public int DynamicRangeTensorCount { get; }

    /// <summary>
    /// Gets the number of copied dynamic range tensor names.
    /// 获取复制出的 dynamic range tensor 名称数量。
    /// </summary>
    public int CopiedDynamicRangeTensorNameCount { get; }

    /// <summary>
    /// Gets the copied missing named weight count.
    /// 获取复制出的 missing named weight 数量。
    /// </summary>
    public int MissingNamedWeightCount { get; }

    /// <summary>
    /// Gets the number of copied missing named weight names.
    /// 获取复制出的 missing named weight 名称数量。
    /// </summary>
    public int CopiedMissingNamedWeightCount { get; }

    /// <summary>
    /// Gets the copied all named weight count.
    /// 获取复制出的 all named weight 数量。
    /// </summary>
    public int AllNamedWeightCount { get; }

    /// <summary>
    /// Gets the number of copied all named weight names.
    /// 获取复制出的 all named weight 名称数量。
    /// </summary>
    public int CopiedAllNamedWeightCount { get; }

    /// <summary>
    /// Gets the number of diagnostics collected while creating the snapshot.
    /// 获取创建快照时收集到的诊断数量。
    /// </summary>
    public int DiagnosticCount { get; }

    /// <summary>
    /// Converts the summary to a compact diagnostic string.
    /// 转换为简短诊断字符串。
    /// </summary>
    /// <returns>A compact diagnostic string. 简短诊断字符串。</returns>
    public override string ToString()
    {
        return $"Line={(int)Line} MaxThreads={MaxThreads} WeightsValidation={WeightsValidation} Logger={HasLogger} ErrorRecorder={HasErrorRecorder}/{ErrorCount}/{CopiedErrorRecordCount}/Overflow={HasErrorOverflowed} MissingWeights={MissingNamedWeightCount}/{CopiedMissingNamedWeightCount} AllWeights={AllNamedWeightCount}/{CopiedAllNamedWeightCount} DynamicRanges={DynamicRangeTensorCount}/{CopiedDynamicRangeTensorNameCount} Diagnostics={DiagnosticCount}";
    }
}
