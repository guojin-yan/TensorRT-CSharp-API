namespace JYPPX.TensorRtSharp;

/// <summary>
/// A copied TensorRT progress monitor event.
/// 从 TensorRT 复制出的 progress monitor 事件。
/// </summary>
public readonly struct TensorRtProgressMonitorEvent
{
    internal TensorRtProgressMonitorEvent(TensorRtProgressMonitorEventKind kind, string phaseName, string? parentPhase, int step, int stepCount)
    {
        Kind = kind;
        PhaseName = phaseName;
        ParentPhase = parentPhase;
        Step = step;
        StepCount = stepCount;
    }

    /// <summary>
    /// Gets the event kind. 获取事件类型。
    /// </summary>
    public TensorRtProgressMonitorEventKind Kind { get; }

    /// <summary>
    /// Gets the TensorRT phase name copied from native memory.
    /// 获取从 native 内存复制出的 TensorRT 阶段名称。
    /// </summary>
    public string PhaseName { get; }

    /// <summary>
    /// Gets the parent phase name, when TensorRT provided one.
    /// 获取父阶段名称；当 TensorRT 未提供父阶段时为 <see langword="null"/>。
    /// </summary>
    public string? ParentPhase { get; }

    /// <summary>
    /// Gets the completed step index for <see cref="TensorRtProgressMonitorEventKind.StepComplete"/> events.
    /// 获取 <see cref="TensorRtProgressMonitorEventKind.StepComplete"/> 事件的完成步骤索引。
    /// </summary>
    public int Step { get; }

    /// <summary>
    /// Gets the total step count reported for <see cref="TensorRtProgressMonitorEventKind.PhaseStart"/> events.
    /// 获取 <see cref="TensorRtProgressMonitorEventKind.PhaseStart"/> 事件中 TensorRT 报告的总步骤数。
    /// </summary>
    public int StepCount { get; }
}
