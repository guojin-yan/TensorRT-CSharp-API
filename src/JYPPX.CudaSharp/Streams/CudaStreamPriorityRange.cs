namespace JYPPX.CudaSharp;

/// <summary>
/// CUDA stream priority range for the current device.
/// 当前设备的 CUDA stream 优先级范围。
/// </summary>
public readonly struct CudaStreamPriorityRange
{
    /// <summary>
    /// Initializes a CUDA stream-priority range.
    /// 初始化 CUDA stream 优先级范围。
    /// </summary>
    /// <param name="leastPriority">The least-urgent priority value. 最低优先级值。</param>
    /// <param name="greatestPriority">The most-urgent priority value. 最高优先级值。</param>
    public CudaStreamPriorityRange(int leastPriority, int greatestPriority)
    {
        LeastPriority = leastPriority;
        GreatestPriority = greatestPriority;
    }

    /// <summary>
    /// Priority value for the lowest-priority stream.
    /// 最低优先级 stream 的优先级值。
    /// </summary>
    public int LeastPriority { get; }

    /// <summary>
    /// Priority value for the highest-priority stream.
    /// 最高优先级 stream 的优先级值。
    /// </summary>
    public int GreatestPriority { get; }

    /// <summary>
    /// Formats the priority range for diagnostics.
    /// 将优先级范围格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString()
    {
        return $"Least={LeastPriority}, Greatest={GreatestPriority}";
    }
}
