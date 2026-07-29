using System;
using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtInferenceBindings
{
    /// <summary>
    /// Binds one tensor buffer to the execution context.
    /// 将一个 tensor 缓冲区绑定到 execution context。
    /// </summary>
    /// <param name="tensorName">The TensorRT tensor name. TensorRT tensor 名称。</param>
    /// <returns>The current binding set for chaining. 当前绑定集，便于链式调用。</returns>
    public TensorRtInferenceBindings BindTensor(string tensorName)
    {
        ThrowIfDisposed();
        if (!_buffers.TryGetValue(tensorName, out TensorRtInferenceBuffer? buffer))
        {
            throw new InvalidOperationException($"Tensor '{tensorName}' does not have an attached CUDA buffer.");
        }

        if (buffer.Tensor.Location != TensorRtTensorLocation.Device)
        {
            throw new NotSupportedException("High-level inference bindings currently support device tensors only.");
        }

        if (buffer.Tensor.IOMode == TensorRtIOMode.Input)
        {
            _context.SetInputTensorAddress(buffer.Tensor.Name, buffer.Memory);
        }
        else if (buffer.Tensor.IOMode == TensorRtIOMode.Output)
        {
            _context.SetOutputTensorAddress(buffer.Tensor.Name, buffer.Memory);
        }
        else
        {
            _context.SetTensorAddress(buffer.Tensor.Name, buffer.Memory);
        }

        buffer.IsBound = true;
        return this;
    }

    /// <summary>
    /// Binds every attached CUDA buffer to the execution context.
    /// 将所有已附加 CUDA 缓冲区绑定到 execution context。
    /// </summary>
    /// <returns>The current binding set for chaining. 当前绑定集，便于链式调用。</returns>
    public TensorRtInferenceBindings BindAll()
    {
        ThrowIfDisposed();
        foreach (string tensorName in new List<string>(_buffers.Keys))
        {
            BindTensor(tensorName);
        }

        RefreshReport(runShapeInference: false);
        return this;
    }
}
