using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Flags that control CUDA stream creation behavior.
/// 控制 CUDA stream 创建行为的标志。
/// </summary>
[Flags]
public enum CudaStreamCreationFlags : uint
{
    /// <summary>
    /// Uses CUDA default stream-creation behavior.
    /// 使用 CUDA 默认的 stream 创建行为。
    /// </summary>
    Default = 0,
    /// <summary>
    /// Creates a non-blocking CUDA stream.
    /// 创建一个 non-blocking CUDA stream。
    /// </summary>
    NonBlocking = 1
}
/// <summary>
/// Modes that control CUDA stream capture validation.
/// 控制 CUDA stream capture 校验行为的模式。
/// </summary>
public enum CudaStreamCaptureMode
{
    /// <summary>
    /// Uses global capture validation.
    /// 使用全局 capture 校验。
    /// </summary>
    Global = 0,
    /// <summary>
    /// Uses thread-local capture validation.
    /// 使用线程本地 capture 校验。
    /// </summary>
    ThreadLocal = 1,
    /// <summary>
    /// Uses relaxed capture validation.
    /// 使用宽松 capture 校验。
    /// </summary>
    Relaxed = 2
}

/// <summary>
/// Controls whether stream-capture dependencies are added or replaced.
/// 控制 stream capture dependencies 是追加还是替换。
/// </summary>
public enum CudaStreamCaptureDependencyMode
{
    /// <summary>Adds nodes to the current dependency set. 向当前 dependency set 追加节点。</summary>
    Add = 0,
    /// <summary>Replaces the current dependency set. 替换当前 dependency set。</summary>
    Replace = 1
}

/// <summary>
/// Describes the CUDA stream capture state used by CUDA Graph capture.
/// 描述 CUDA Graph 捕获流程中的 CUDA stream 捕获状态。
/// </summary>
public enum CudaStreamCaptureStatus
{
    /// <summary>
    /// The stream is not currently capturing.
    /// 当前 stream 没有处于捕获状态。
    /// </summary>
    None = 0,

    /// <summary>
    /// The stream is actively capturing commands.
    /// 当前 stream 正在捕获命令。
    /// </summary>
    Active = 1,

    /// <summary>
    /// The stream capture was invalidated by an error.
    /// stream 捕获流程已因错误失效。
    /// </summary>
    Invalidated = 2
}
