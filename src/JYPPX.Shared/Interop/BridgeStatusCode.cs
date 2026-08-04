namespace JYPPX.TensorRtSharp.Shared.Interop;

/// <summary>
/// Status codes returned by the native bridge.
/// 原生 bridge 返回的状态码。
/// </summary>
public enum BridgeStatusCode
{
    /// <summary>
    /// The operation completed successfully. 操作成功完成。
    /// </summary>
    Ok = 0,
    /// <summary>
    /// One or more arguments were invalid. 一个或多个参数无效。
    /// </summary>
    InvalidArgument = 1,
    /// <summary>
    /// The provided buffer was too small. 提供的缓冲区过小。
    /// </summary>
    BufferTooSmall = 2,
    /// <summary>
    /// The requested item was not found. 未找到请求的对象。
    /// </summary>
    NotFound = 3,
    /// <summary>
    /// The requested operation is not supported. 请求的操作不受支持。
    /// </summary>
    NotSupported = 4,
    /// <summary>
    /// A required dependency is missing. 缺少必需依赖。
    /// </summary>
    DependencyMissing = 5,
    /// <summary>
    /// The operation is not ready yet. 操作尚未就绪。
    /// </summary>
    NotReady = 6,
    /// <summary>
    /// The native runtime reported an execution error. 原生运行时报告了执行错误。
    /// </summary>
    RuntimeError = 7,
    /// <summary>
    /// The operation could not proceed because the object state was invalid. 由于对象状态无效，操作无法继续。
    /// </summary>
    InvalidState = 8,
    /// <summary>
    /// The operation failed because memory could not be allocated. 由于无法分配内存，操作失败。
    /// </summary>
    OutOfMemory = 9,
    /// <summary>
    /// The bridge intentionally leaves this operation unimplemented. bridge 有意保留该操作为未实现状态。
    /// </summary>
    NotImplemented = 10
}
