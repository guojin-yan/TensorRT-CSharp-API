using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT engine.
/// TensorRT engine 的托管封装。
/// </summary>
public sealed partial class TensorRtEngine
{
    /// <summary>
    /// Creates a TensorRT engine inspector for this engine.
    /// 为当前 engine 创建一个 TensorRT engine inspector。
    /// </summary>
    /// <returns>A managed engine-inspector wrapper. 托管 engine inspector 封装。</returns>
    public TensorRtEngineInspector CreateInspector()
    {
        return new TensorRtEngineInspector(Line, NativeBridgeApi.CreateEngineInspector(Line, _handle));
    }

}
