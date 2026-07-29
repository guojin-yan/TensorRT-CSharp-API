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
    /// Sets the Tensor Address value.
    /// 设置 Tensor Address 值。
    /// </summary>
    public void SetTensorAddress(string tensorName, CudaMemory memory)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(nameof(memory));
        }

        NativeBridgeApi.SetTensorAddress(Line, _handle, tensorName, memory.Handle);
    }

    /// <summary>
    /// Binds a CUDA allocation to a named input tensor.
    /// 将 CUDA 设备内存分配绑定到指定输入 tensor。
    /// </summary>
    /// <param name="tensorName">The input tensor name. 输入 tensor 名称。</param>
    /// <param name="memory">The CUDA allocation used as the input buffer. 作为输入缓冲区的 CUDA 设备内存。</param>
    public void SetInputTensorAddress(string tensorName, CudaMemory memory)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(nameof(memory));
        }

        NativeBridgeApi.SetInputTensorAddress(Line, _handle, tensorName, memory.Handle);
    }

    /// <summary>
    /// Binds a CUDA allocation to a named output tensor.
    /// 将 CUDA 设备内存分配绑定到指定输出 tensor。
    /// </summary>
    /// <param name="tensorName">The output tensor name. 输出 tensor 名称。</param>
    /// <param name="memory">The CUDA allocation used as the output buffer. 作为输出缓冲区的 CUDA 设备内存。</param>
    public void SetOutputTensorAddress(string tensorName, CudaMemory memory)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(nameof(memory));
        }

        NativeBridgeApi.SetOutputTensorAddress(Line, _handle, tensorName, memory.Handle);
    }

    /// <summary>
    /// Checks whether Tensor Address Bound is true.
    /// 检查 Tensor Address Bound 是否为 true。
    /// </summary>
    public bool IsTensorAddressBound(string tensorName)
    {
        return NativeBridgeApi.IsExecutionContextTensorAddressBound(Line, _handle, tensorName);
    }

}
