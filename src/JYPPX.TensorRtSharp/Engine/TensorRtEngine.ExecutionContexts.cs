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
    /// Creates a TensorRT execution context with engine-managed device memory.
    /// 创建一个由 engine 管理设备内存的 TensorRT execution context。
    /// </summary>
    /// <returns>A managed execution-context wrapper. 托管 execution context 封装。</returns>
    public TensorRtExecutionContext CreateExecutionContext()
    {
        return new TensorRtExecutionContext(Line, NativeBridgeApi.CreateExecutionContext(Line, _handle));
    }

    /// <summary>
    /// Creates an execution context without internally allocated device memory.
    /// 创建一个不由 TensorRT 内部分配 device memory 的 execution context。
    /// </summary>
    /// <returns>A managed execution context that requires device memory to be set before enqueue. 需要在 enqueue 前设置 device memory 的托管 execution context。</returns>
    public TensorRtExecutionContext CreateExecutionContextWithoutDeviceMemory()
    {
        return new TensorRtExecutionContext(Line, NativeBridgeApi.CreateExecutionContextWithoutDeviceMemory(Line, _handle));
    }

}
