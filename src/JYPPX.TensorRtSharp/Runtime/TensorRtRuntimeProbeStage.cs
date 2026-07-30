namespace JYPPX.TensorRtSharp;

/// <summary>
/// Result for one TensorRT runtime probe stage.
/// TensorRT runtime 分阶段探针的单阶段结果。
/// </summary>
public sealed class TensorRtRuntimeProbeStage
{
    internal TensorRtRuntimeProbeStage(string name, bool succeeded, string message)
    {
        Name = name ?? string.Empty;
        Succeeded = succeeded;
        Message = message ?? string.Empty;
    }

    /// <summary>
    /// Gets the stage name.
    /// 获取阶段名称。
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets whether the stage succeeded.
    /// 获取该阶段是否成功。
    /// </summary>
    public bool Succeeded { get; }

    /// <summary>
    /// Gets the diagnostic message for this stage.
    /// 获取该阶段的诊断消息。
    /// </summary>
    public string Message { get; }

    /// <summary>
    /// Formats the probe stage for diagnostics.
    /// 将探针阶段格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString() => $"{Name}={Succeeded}:{Message}";
}
