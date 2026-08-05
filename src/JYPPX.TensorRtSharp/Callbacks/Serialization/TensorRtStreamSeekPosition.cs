namespace JYPPX.TensorRtSharp;

/// <summary>Identifies the origin used by the most recent native stream seek.</summary>
public enum TensorRtStreamSeekPosition
{
    /// <summary>No seek has been observed.</summary>
    Unknown = -1,

    /// <summary>The seek offset is relative to the beginning.</summary>
    Begin = 0,

    /// <summary>The seek offset is relative to the current position.</summary>
    Current = 1,

    /// <summary>The seek offset is relative to the end.</summary>
    End = 2
}
