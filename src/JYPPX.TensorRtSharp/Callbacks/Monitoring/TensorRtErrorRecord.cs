using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents one copied TensorRT error-recorder entry.
/// 表示一条已复制到托管内存的 TensorRT error-recorder 记录。
/// </summary>
public sealed class TensorRtErrorRecord
{
    internal TensorRtErrorRecord(int index, int code, string description)
    {
        Index = index;
        Code = code;
        Description = description ?? string.Empty;
    }

    /// <summary>
    /// Gets the zero-based error index reported by TensorRT.
    /// 获取 TensorRT 报告的从零开始的错误索引。
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Gets the raw TensorRT error code value.
    /// 获取 TensorRT error code 的原始数值。
    /// </summary>
    public int Code { get; }

    /// <summary>
    /// Gets the copied error description.
    /// 获取已复制的错误描述。
    /// </summary>
    public string Description { get; }

    /// <summary>
    /// Converts the error record to a compact diagnostic string.
    /// 转换为简短诊断字符串。
    /// </summary>
    /// <returns>A compact diagnostic string. 简短诊断字符串。</returns>
    public override string ToString()
    {
        return string.IsNullOrWhiteSpace(Description)
            ? $"TensorRT error #{Index} code={Code}"
            : $"TensorRT error #{Index} code={Code}: {Description}";
    }
}
