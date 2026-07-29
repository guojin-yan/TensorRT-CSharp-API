using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Parses ONNX models into a TensorRT network definition.
/// 将 ONNX 模型解析到 TensorRT network definition。
/// </summary>
public sealed partial class TensorRtOnnxParser
{
    /// <summary>
    /// Gets a single ONNX parser flag.
    /// 获取单个 ONNX parser 标志。
    /// </summary>
    /// <param name="flag">The parser flag to query. 要查询的 parser 标志。</param>
    /// <returns><c>true</c> when the flag is enabled. 标志启用时返回 <c>true</c>。</returns>
    public bool GetFlag(TensorRtOnnxParserFlag flag)
    {
        ValidateParserFlag(flag);
        return NativeBridgeApi.GetOnnxParserFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Enables a single ONNX parser flag.
    /// 启用单个 ONNX parser 标志。
    /// </summary>
    /// <param name="flag">The parser flag to enable. 要启用的 parser 标志。</param>
    public void SetFlag(TensorRtOnnxParserFlag flag)
    {
        ValidateParserFlag(flag);
        NativeBridgeApi.SetOnnxParserFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Clears a single ONNX parser flag.
    /// 清除单个 ONNX parser 标志。
    /// </summary>
    /// <param name="flag">The parser flag to clear. 要清除的 parser 标志。</param>
    public void ClearFlag(TensorRtOnnxParserFlag flag)
    {
        ValidateParserFlag(flag);
        NativeBridgeApi.ClearOnnxParserFlag(Line, _handle, flag);
    }

}
