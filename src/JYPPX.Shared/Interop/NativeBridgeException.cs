using System;

namespace JYPPX.Shared.Interop;

/// <summary>
/// Base exception for managed wrappers around the native bridge.
/// 原生 bridge 托管封装的基础异常类型。
/// </summary>
public class NativeBridgeException : InvalidOperationException
{
    /// <summary>
    /// Creates a native bridge exception with bridge status metadata.
    /// 使用 bridge 状态元数据创建原生 bridge 异常。
    /// </summary>
    public NativeBridgeException(BridgeStatusCode statusCode, BridgeErrorCategory errorCategory, string message)
        : base(message)
    {
        StatusCode = statusCode;
        ErrorCategory = errorCategory;
    }

    /// <summary>
    /// Gets the bridge status code associated with the exception.
    /// 获取与该异常关联的 bridge 状态码。
    /// </summary>
    public BridgeStatusCode StatusCode { get; }
    /// <summary>
    /// Gets the bridge error category associated with the exception.
    /// 获取与该异常关联的 bridge 错误类别。
    /// </summary>
    public BridgeErrorCategory ErrorCategory { get; }
}
