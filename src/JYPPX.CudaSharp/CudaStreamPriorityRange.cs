namespace JYPPX.CudaSharp;

/// <summary>
/// CUDA stream priority range for the current device.
/// </summary>
public readonly struct CudaStreamPriorityRange
{
    public CudaStreamPriorityRange(int leastPriority, int greatestPriority)
    {
        LeastPriority = leastPriority;
        GreatestPriority = greatestPriority;
    }

    /// <summary>
    /// Priority value for the lowest-priority stream.
    /// </summary>
    public int LeastPriority { get; }

    /// <summary>
    /// Priority value for the highest-priority stream.
    /// </summary>
    public int GreatestPriority { get; }

    public override string ToString()
    {
        return $"Least={LeastPriority}, Greatest={GreatestPriority}";
    }
}
