namespace JYPPX.TensorRtSharp;

/// <summary>Identifies the origin used by the most recent native stream seek. 标识最近一次原生流定位使用的基准位置。</summary>
public enum TensorRtStreamSeekPosition
{
    /// <summary>No seek has been observed. 尚未观察到定位操作。</summary>
    Unknown = -1,

    /// <summary>The seek offset is relative to the beginning. 定位偏移量相对于流的开头。</summary>
    Begin = 0,

    /// <summary>The seek offset is relative to the current position. 定位偏移量相对于当前位置。</summary>
    Current = 1,

    /// <summary>The seek offset is relative to the end. 定位偏移量相对于流的末尾。</summary>
    End = 2
}
