using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Captures copied read-only diagnostics for a TensorRT refitter instance.
/// 捕获 TensorRT refitter 实例的复制型只读诊断信息。
/// </summary>
public sealed class TensorRtRefitterDiagnosticSnapshot
{
    internal TensorRtRefitterDiagnosticSnapshot(
        TensorRtApiLine line,
        int maxThreads,
        bool weightsValidation,
        bool hasLogger,
        bool hasErrorRecorder,
        TensorRtErrorRecorderSnapshot errorRecorder,
        int dynamicRangeTensorCount,
        int missingNamedWeightCount,
        int allNamedWeightCount,
        IReadOnlyList<string> dynamicRangeTensorNames,
        IReadOnlyList<string> missingNamedWeights,
        IReadOnlyList<string> allNamedWeights,
        IReadOnlyList<string> diagnostics)
    {
        Line = line;
        MaxThreads = maxThreads;
        WeightsValidation = weightsValidation;
        HasLogger = hasLogger;
        HasErrorRecorder = hasErrorRecorder;
        ErrorRecorder = errorRecorder ?? new TensorRtErrorRecorderSnapshot(line, false, 0, false, Array.Empty<TensorRtErrorRecord>());
        DynamicRangeTensorCount = dynamicRangeTensorCount;
        MissingNamedWeightCount = missingNamedWeightCount;
        AllNamedWeightCount = allNamedWeightCount;
        DynamicRangeTensorNames = dynamicRangeTensorNames ?? Array.Empty<string>();
        MissingNamedWeights = missingNamedWeights ?? Array.Empty<string>();
        AllNamedWeights = allNamedWeights ?? Array.Empty<string>();
        Diagnostics = diagnostics ?? Array.Empty<string>();
    }

    /// <summary>
    /// Gets the TensorRT API line used to collect this snapshot.
    /// 获取采集该快照的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the maximum thread count copied from the refitter.
    /// 获取从 refitter 复制出的最大线程数。
    /// </summary>
    public int MaxThreads { get; }

    /// <summary>
    /// Gets whether weight validation was enabled on the refitter.
    /// 获取 refitter 是否启用了权重验证。
    /// </summary>
    public bool WeightsValidation { get; }

    /// <summary>
    /// Gets whether the refitter reported an associated logger.
    /// 获取 refitter 是否报告关联的 logger。
    /// </summary>
    public bool HasLogger { get; }

    /// <summary>
    /// Gets whether the refitter reported an associated error recorder.
    /// 获取 refitter 是否报告关联的 error recorder。
    /// </summary>
    public bool HasErrorRecorder { get; }

    /// <summary>
    /// Gets the copied error-recorder snapshot.
    /// 获取复制出的 error recorder 快照。
    /// </summary>
    public TensorRtErrorRecorderSnapshot ErrorRecorder { get; }

    /// <summary>
    /// Gets the dynamic-range tensor count reported by the refitter.
    /// 获取 refitter 报告的 dynamic range tensor 数量。
    /// </summary>
    public int DynamicRangeTensorCount { get; }

    /// <summary>
    /// Gets the missing named-weight count reported by the refitter.
    /// 获取 refitter 报告的 missing named weight 数量。
    /// </summary>
    public int MissingNamedWeightCount { get; }

    /// <summary>
    /// Gets the total named-weight count reported by the refitter.
    /// 获取 refitter 报告的全部 named weight 数量。
    /// </summary>
    public int AllNamedWeightCount { get; }

    /// <summary>
    /// Gets the copied dynamic-range tensor names.
    /// 获取复制出的 dynamic range tensor 名称。
    /// </summary>
    public IReadOnlyList<string> DynamicRangeTensorNames { get; }

    /// <summary>
    /// Gets the copied missing named-weight names.
    /// 获取复制出的 missing named weight 名称。
    /// </summary>
    public IReadOnlyList<string> MissingNamedWeights { get; }

    /// <summary>
    /// Gets all copied named-weight names.
    /// 获取复制出的全部 named weight 名称。
    /// </summary>
    public IReadOnlyList<string> AllNamedWeights { get; }

    /// <summary>
    /// Gets diagnostics collected while creating the snapshot.
    /// 获取创建快照时收集的诊断信息。
    /// </summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>
    /// Creates a compact pointer-free summary from this copied refitter diagnostic snapshot.
    /// 从当前已复制的 refitter 诊断快照创建简短无指针摘要。
    /// </summary>
    /// <remarks>
    /// This method only reads managed snapshot values. It does not call TensorRT, expose native pointers, or promote runtime proof.
    /// 该方法只读取托管快照值；不会调用 TensorRT、暴露 native 指针，也不会提升 runtime proof。
    /// </remarks>
    /// <returns>A pointer-free refitter diagnostic summary. 无指针 refitter 诊断摘要。</returns>
    public TensorRtRefitterDiagnosticSummary ToSummary()
    {
        return new TensorRtRefitterDiagnosticSummary(
            Line,
            MaxThreads,
            WeightsValidation,
            HasLogger,
            HasErrorRecorder,
            ErrorRecorder.ErrorCount,
            ErrorRecorder.Records.Count,
            ErrorRecorder.HasOverflowed,
            DynamicRangeTensorCount,
            DynamicRangeTensorNames.Count,
            MissingNamedWeightCount,
            MissingNamedWeights.Count,
            AllNamedWeightCount,
            AllNamedWeights.Count,
            Diagnostics.Count);
    }

    /// <summary>
    /// Converts the snapshot to a compact diagnostic string.
    /// 将快照转换为简短诊断字符串。
    /// </summary>
    /// <returns>A compact diagnostic string. 简短诊断字符串。</returns>
    public override string ToString()
    {
        return $"Line={(int)Line} MaxThreads={MaxThreads} WeightsValidation={WeightsValidation} Logger={HasLogger} ErrorRecorder={HasErrorRecorder}/{ErrorRecorder.ErrorCount} MissingWeights={MissingNamedWeightCount}/{MissingNamedWeights.Count} AllWeights={AllNamedWeightCount}/{AllNamedWeights.Count} DynamicRanges={DynamicRangeTensorCount}/{DynamicRangeTensorNames.Count} diagnostics={Diagnostics.Count}";
    }
}

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
