namespace JYPPX.TensorRtSharp;

/// <summary>
/// Summarizes one TensorRT inference enqueue attempt.
/// 汇总一次 TensorRT 推理 enqueue 操作。
/// </summary>
public sealed class TensorRtInferenceExecutionSummary
{
    internal TensorRtInferenceExecutionSummary(int profileIndex, int boundTensorCount, bool synchronized, TensorRtExecutionContextReadiness readiness)
    {
        ProfileIndex = profileIndex;
        BoundTensorCount = boundTensorCount;
        Synchronized = synchronized;
        Readiness = readiness;
    }

    /// <summary>
    /// Gets the optimization profile used by the binding set.
    /// 获取绑定集使用的 optimization profile。
    /// </summary>
    public int ProfileIndex { get; }

    /// <summary>
    /// Gets the number of CUDA buffers bound before enqueue.
    /// 获取 enqueue 前已绑定的 CUDA 缓冲区数量。
    /// </summary>
    public int BoundTensorCount { get; }

    /// <summary>
    /// Gets whether the CUDA stream was synchronized before returning.
    /// 获取返回前是否已同步 CUDA stream。
    /// </summary>
    public bool Synchronized { get; }

    /// <summary>
    /// Gets the readiness snapshot captured before enqueue.
    /// 获取 enqueue 前采集的就绪状态快照。
    /// </summary>
    public TensorRtExecutionContextReadiness Readiness { get; }

    /// <summary>
    /// Gets a short human-readable summary for logs.
    /// 获取用于日志的简短可读摘要。
    /// </summary>
    /// <returns>A summary line. 摘要文本。</returns>
    public override string ToString()
    {
        return $"profile={ProfileIndex} bound={BoundTensorCount} synchronized={Synchronized} ready={Readiness.IsReadyForEnqueue}";
    }
}
