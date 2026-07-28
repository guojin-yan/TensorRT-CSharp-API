namespace JYPPX.CudaSharp;

/// <summary>
/// Captures CUDA stream graph-capture metadata.
/// 表示 CUDA stream 的 graph capture 元数据。
/// </summary>
public readonly struct CudaStreamCaptureInfo
{
    /// <summary>
    /// Creates stream capture metadata.
    /// 创建 stream 捕获元数据。
    /// </summary>
    /// <param name="status">The capture status. 捕获状态。</param>
    /// <param name="captureId">The CUDA capture id, or zero when not capturing. CUDA 捕获 ID；未捕获时通常为零。</param>
    public CudaStreamCaptureInfo(CudaStreamCaptureStatus status, ulong captureId)
        : this(status, captureId, false, 0, false)
    {
    }

    internal CudaStreamCaptureInfo(
        CudaStreamCaptureStatus status,
        ulong captureId,
        bool hasCapturedGraph,
        ulong dependencyCount,
        bool hasDependencyEdgeData)
    {
        Status = status;
        CaptureId = captureId;
        HasCapturedGraph = hasCapturedGraph;
        DependencyCount = dependencyCount;
        HasDependencyEdgeData = hasDependencyEdgeData;
    }

    /// <summary>
    /// Gets the stream capture status.
    /// 获取 stream 捕获状态。
    /// </summary>
    public CudaStreamCaptureStatus Status { get; }

    /// <summary>
    /// Gets the CUDA capture id.
    /// 获取 CUDA 捕获 ID。
    /// </summary>
    public ulong CaptureId { get; }

    /// <summary>Gets whether CUDA reported a borrowed capture graph during the native call. 获取 CUDA 是否在 native 调用期间报告 borrowed capture graph。</summary>
    public bool HasCapturedGraph { get; }

    /// <summary>Gets the copied number of current capture dependencies. 获取复制出的当前 capture dependency 数量。</summary>
    public ulong DependencyCount { get; }

    /// <summary>Gets whether CUDA reported edge-data metadata for the dependency list. 获取 CUDA 是否为 dependency list 报告 edge-data metadata。</summary>
    public bool HasDependencyEdgeData { get; }

    /// <summary>
    /// Returns a compact diagnostic string.
    /// 返回简短诊断字符串。
    /// </summary>
    /// <returns>A readable stream capture string. 可读的 stream 捕获字符串。</returns>
    public override string ToString()
    {
        return $"Status={Status} CaptureId={CaptureId} HasGraph={HasCapturedGraph} Dependencies={DependencyCount} HasEdgeData={HasDependencyEdgeData}";
    }
}
