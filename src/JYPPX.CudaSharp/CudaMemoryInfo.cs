namespace JYPPX.CudaSharp;

/// <summary>
/// Reports CUDA device memory availability for the current device.
/// 报告当前 CUDA 设备的显存可用性。
/// </summary>
public sealed class CudaMemoryInfo
{
    /// <summary>
    /// Creates a CUDA memory information snapshot.
    /// 创建 CUDA 显存信息快照。
    /// </summary>
    /// <param name="freeBytes">Free device memory in bytes. 空闲显存字节数。</param>
    /// <param name="totalBytes">Total device memory in bytes. 总显存字节数。</param>
    public CudaMemoryInfo(ulong freeBytes, ulong totalBytes)
    {
        FreeBytes = freeBytes;
        TotalBytes = totalBytes;
    }

    /// <summary>
    /// Gets free device memory in bytes.
    /// 获取空闲显存字节数。
    /// </summary>
    public ulong FreeBytes { get; }

    /// <summary>
    /// Gets total device memory in bytes.
    /// 获取总显存字节数。
    /// </summary>
    public ulong TotalBytes { get; }

    /// <summary>
    /// Gets used device memory in bytes.
    /// 获取已使用显存字节数。
    /// </summary>
    public ulong UsedBytes => TotalBytes >= FreeBytes ? TotalBytes - FreeBytes : 0;

    /// <summary>
    /// Gets free device memory as a ratio in the range 0 to 1.
    /// 获取空闲显存比例，范围为 0 到 1。
    /// </summary>
    public double FreeRatio => TotalBytes == 0 ? 0 : (double)FreeBytes / TotalBytes;

    /// <summary>
    /// Gets used device memory as a ratio in the range 0 to 1.
    /// 获取已使用显存比例，范围为 0 到 1。
    /// </summary>
    public double UsedRatio => TotalBytes == 0 ? 0 : (double)UsedBytes / TotalBytes;
}
