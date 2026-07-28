namespace JYPPX.CudaSharp;

/// <summary>
/// Captures the scalar metadata returned by CUDA's per-thread stream capture query.
/// 表示 CUDA per-thread stream capture 查询返回的标量元数据。
/// </summary>
public readonly struct CudaStreamCaptureScalarInfo
{
    /// <summary>
    /// Creates per-thread stream capture metadata.
    /// 创建 per-thread stream capture 元数据。
    /// </summary>
    /// <param name="status">The capture status. 捕获状态。</param>
    /// <param name="captureId">The CUDA capture id. CUDA 捕获 ID。</param>
    public CudaStreamCaptureScalarInfo(CudaStreamCaptureStatus status, ulong captureId)
    {
        Status = status;
        CaptureId = captureId;
    }

    /// <summary>
    /// Gets the stream capture status.
    /// 获取 stream capture 状态。
    /// </summary>
    public CudaStreamCaptureStatus Status { get; }

    /// <summary>
    /// Gets the CUDA capture id.
    /// 获取 CUDA capture ID。
    /// </summary>
    public ulong CaptureId { get; }

    /// <summary>
    /// Returns a compact diagnostic string.
    /// 返回简短诊断字符串。
    /// </summary>
    public override string ToString() => $"Status={Status} CaptureId={CaptureId}";
}
