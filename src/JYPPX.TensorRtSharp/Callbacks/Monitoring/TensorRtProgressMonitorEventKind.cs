namespace JYPPX.TensorRtSharp;

/// <summary>
/// TensorRT progress monitor event kinds.
/// TensorRT progress monitor 事件类型。
/// </summary>
public enum TensorRtProgressMonitorEventKind
{
    /// <summary>
    /// Unknown event kind. 未知事件。
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// A build phase has started. 构建阶段开始。
    /// </summary>
    PhaseStart = 1,

    /// <summary>
    /// A build phase step has completed. 构建阶段中的一个步骤完成。
    /// </summary>
    StepComplete = 2,

    /// <summary>
    /// A build phase has finished. 构建阶段结束。
    /// </summary>
    PhaseFinish = 3
}
