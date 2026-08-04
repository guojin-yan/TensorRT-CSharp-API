using System;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtInferenceBindings
{
    /// <summary>
    /// Copies single-precision input data into a named input buffer, allocating the buffer when needed.
    /// 将单精度输入数据复制到指定输入缓冲区；必要时自动分配缓冲区。
    /// </summary>
    /// <param name="tensorName">The input tensor name. 输入 tensor 名称。</param>
    /// <param name="values">The input values. 输入数据。</param>
    /// <param name="runtimeShape">Optional runtime shape used for allocation. 用于分配的可选运行时 shape。</param>
    /// <returns>The current binding set for chaining. 当前绑定集，便于链式调用。</returns>
    public TensorRtInferenceBindings CopyInputFromHost(string tensorName, float[] values, TensorRtDims? runtimeShape = null)
    {
        ThrowIfDisposed();
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        TensorRtEngineTensorBinding tensor = GetTensor(tensorName);
        EnsureSinglePrecisionTensor(tensor, "copied from a float array");

        TensorRtInferenceBuffer buffer = EnsureBuffer(tensorName, runtimeShape, checked(values.Length * sizeof(float)));
        if (buffer.Tensor.IOMode != TensorRtIOMode.Input)
        {
            throw new ArgumentException("Only input tensors can be copied from host.", nameof(tensorName));
        }

        buffer.Memory.CopyFrom(values);
        return this;
    }

    /// <summary>
    /// Copies byte input data into a named input buffer, allocating the buffer when needed.
    /// 将字节输入数据复制到指定输入缓冲区；必要时自动分配缓冲区。
    /// </summary>
    /// <param name="tensorName">The input tensor name. 输入 tensor 名称。</param>
    /// <param name="bytes">The input bytes. 输入字节。</param>
    /// <param name="runtimeShape">Optional runtime shape used for allocation. 用于分配的可选运行时 shape。</param>
    /// <returns>The current binding set for chaining. 当前绑定集，便于链式调用。</returns>
    public TensorRtInferenceBindings CopyInputFromHost(string tensorName, byte[] bytes, TensorRtDims? runtimeShape = null)
    {
        ThrowIfDisposed();
        if (bytes == null)
        {
            throw new ArgumentNullException(nameof(bytes));
        }

        TensorRtInferenceBuffer buffer = EnsureBuffer(tensorName, runtimeShape, bytes.Length);
        if (buffer.Tensor.IOMode != TensorRtIOMode.Input)
        {
            throw new ArgumentException("Only input tensors can be copied from host.", nameof(tensorName));
        }

        buffer.Memory.CopyFrom(bytes);
        return this;
    }

    /// <summary>
    /// Copies a named output tensor to a single-precision managed array.
    /// 将指定输出 tensor 复制到单精度托管数组。
    /// </summary>
    /// <param name="tensorName">The output tensor name. 输出 tensor 名称。</param>
    /// <param name="elementCount">The number of float elements to read. 要读取的 float 元素数量。</param>
    /// <returns>The copied output values. 复制出的输出数据。</returns>
    public float[] ReadOutputSingles(string tensorName, int elementCount)
    {
        ThrowIfDisposed();
        TensorRtInferenceBuffer buffer = GetOutputBuffer(tensorName);
        EnsureSinglePrecisionTensor(buffer.Tensor, "read as a float array");
        return buffer.Memory.ToSingleArray(elementCount);
    }

    /// <summary>
    /// Copies the complete named output tensor buffer to a managed byte array.
    /// 将指定输出 tensor 的完整缓冲区复制到托管字节数组。
    /// </summary>
    /// <param name="tensorName">The output tensor name. 输出 tensor 名称。</param>
    /// <returns>
    /// The raw output bytes in the TensorRT binding layout. TensorRT 绑定布局中的原始输出字节。
    /// </returns>
    /// <remarks>
    /// Use this method for Half, BFloat16, integer, Boolean, packed, or vectorized output tensors,
    /// then decode the bytes according to <see cref="TensorRtInferenceBuffer.Tensor"/> metadata.
    /// 对 Half、BFloat16、整数、布尔、打包或向量化输出 tensor 使用此方法，
    /// 并根据 <see cref="TensorRtInferenceBuffer.Tensor"/> 元数据解码字节。
    /// </remarks>
    public byte[] ReadOutputBytes(string tensorName)
    {
        ThrowIfDisposed();
        TensorRtInferenceBuffer buffer = GetOutputBuffer(tensorName);
        return buffer.Memory.ToArray(buffer.SizeInBytes);
    }

    private TensorRtInferenceBuffer GetOutputBuffer(string tensorName)
    {
        if (!_buffers.TryGetValue(tensorName, out TensorRtInferenceBuffer? buffer))
        {
            throw new InvalidOperationException($"Tensor '{tensorName}' does not have an attached CUDA buffer.");
        }

        if (buffer.Tensor.IOMode != TensorRtIOMode.Output)
        {
            throw new ArgumentException("Only output tensors can be read as outputs.", nameof(tensorName));
        }

        return buffer;
    }

    private static void EnsureSinglePrecisionTensor(TensorRtEngineTensorBinding tensor, string operation)
    {
        if (tensor.DataType != TensorRtDataType.Float)
        {
            throw new NotSupportedException(
                $"Tensor '{tensor.Name}' is {tensor.DataType} and cannot be {operation}. " +
                "Use the byte-array transfer API and decode the binding according to its tensor metadata.");
        }
    }
}
