namespace JYPPX.TensorRtSharp;

/// <summary>
/// Result returned by a native progress monitor diagnostic emission.
/// native progress monitor 诊断触发返回的结果。
/// </summary>
public readonly struct TensorRtProgressMonitorDiagnosticResult
{
    internal TensorRtProgressMonitorDiagnosticResult(bool shouldContinue, bool callbackAccepted)
    {
        ShouldContinue = shouldContinue;
        CallbackAccepted = callbackAccepted;
    }

    /// <summary>
    /// Gets whether TensorRT should continue after a step-complete callback.
    /// 获取 step-complete 回调后 TensorRT 是否应继续。
    /// </summary>
    public bool ShouldContinue { get; }

    /// <summary>
    /// Gets whether the callback completed without a managed exception or non-OK status.
    /// 获取回调是否未发生托管异常或非 OK 状态。
    /// </summary>
    public bool CallbackAccepted { get; }
}
