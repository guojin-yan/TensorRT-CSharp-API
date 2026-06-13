namespace JYPPX.Shared.Interop;

/// <summary>
/// TensorRT major lines exposed by the bridge.
/// bridge 对外暴露的 TensorRT 主版本线。
/// </summary>
public enum TensorRtApiLine
{
    /// <summary>
    /// The TensorRT line is unknown or not resolved.
    /// TensorRT 主版本线未知或尚未解析。
    /// </summary>
    Unknown = 0,
    /// <summary>
    /// TensorRT 8 API line.
    /// TensorRT 8 API 主线。
    /// </summary>
    TensorRt8 = 8,
    /// <summary>
    /// TensorRT 10 API line.
    /// TensorRT 10 API 主线。
    /// </summary>
    TensorRt10 = 10,
    /// <summary>
    /// TensorRT 11 API line.
    /// TensorRT 11 API 主线。
    /// </summary>
    TensorRt11 = 11
}
