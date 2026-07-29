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
    /// Sets the Input Consumed Event value.
    /// 设置 Input Consumed Event 值。
    /// </summary>
    public void SetInputConsumedEvent(CudaEvent cudaEvent)
    {
        if (cudaEvent == null)
        {
            throw new ArgumentNullException(nameof(cudaEvent));
        }

        NativeBridgeApi.SetExecutionContextInputConsumedEvent(Line, _handle, cudaEvent.Handle);
    }

}
