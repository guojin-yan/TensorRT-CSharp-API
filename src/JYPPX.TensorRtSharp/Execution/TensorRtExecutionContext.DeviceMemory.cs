using System;
using System.Collections.Generic;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Execution Context wrapper.
/// 表示托管 TensorRT Tensor Rt Execution Context 包装器。
/// </summary>
public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Sets the Device Memory value.
    /// 设置 Device Memory 值。
    /// </summary>
    public void SetDeviceMemory(CudaMemory memory)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(nameof(memory));
        }

        NativeBridgeApi.SetExecutionContextDeviceMemory(Line, _handle, memory.Handle);
    }

    /// <summary>
    /// Updates the Device Memory Size For Shapes value.
    /// 更新 Device Memory Size For Shapes 值。
    /// </summary>
    public ulong UpdateDeviceMemorySizeForShapes()
    {
        return NativeBridgeApi.UpdateExecutionContextDeviceMemorySizeForShapes(Line, _handle);
    }

}
