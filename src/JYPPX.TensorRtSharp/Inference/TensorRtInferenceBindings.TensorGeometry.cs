using System;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtInferenceBindings
{
    /// <summary>
    /// Sets a dynamic input shape on the execution context and refreshes the binding report.
    /// 在 execution context 上设置动态输入 shape，并刷新绑定报告。
    /// </summary>
    /// <param name="tensorName">The input tensor name. 输入 tensor 名称。</param>
    /// <param name="shape">The runtime input shape. 运行时输入 shape。</param>
    /// <returns>The current binding set for chaining. 当前绑定集，便于链式调用。</returns>
    public TensorRtInferenceBindings SetInputShape(string tensorName, TensorRtDims shape)
    {
        ThrowIfDisposed();
        if (shape == null)
        {
            throw new ArgumentNullException(nameof(shape));
        }

        TensorRtEngineTensorBinding tensor = GetTensor(tensorName);
        if (tensor.IOMode != TensorRtIOMode.Input)
        {
            throw new ArgumentException("Only input tensors can receive an input shape.", nameof(tensorName));
        }

        _context.SetInputShape(tensorName, shape);
        if (_buffers.TryGetValue(tensorName, out TensorRtInferenceBuffer? buffer))
        {
            buffer.RuntimeShape = shape;
            buffer.SizeInBytes = EstimateTensorByteSize(tensor, shape);
        }

        RefreshReport(runShapeInference: false);
        return this;
    }

    private TensorRtDims ResolveRuntimeShape(TensorRtEngineTensorBinding tensor)
    {
        try
        {
            TensorRtDims shape = _context.GetTensorShape(tensor.Name);
            if (CanEstimate(shape))
            {
                return shape;
            }
        }
        catch (Exception)
        {
        }

        if (tensor.EngineShape != null && CanEstimate(tensor.EngineShape))
        {
            return tensor.EngineShape;
        }

        if (tensor.ProfileOptShape != null && CanEstimate(tensor.ProfileOptShape))
        {
            return tensor.ProfileOptShape;
        }

        throw new InvalidOperationException($"Tensor '{tensor.Name}' does not have a concrete runtime shape. Set input shapes or provide a runtime shape explicitly.");
    }

    private static bool CanEstimate(TensorRtDims shape)
    {
        if (shape == null || shape.Values.Length == 0)
        {
            return false;
        }

        foreach (int value in shape.Values)
        {
            if (value <= 0)
            {
                return false;
            }
        }

        return true;
    }

    private static int EstimateTensorByteSize(TensorRtEngineTensorBinding tensor, TensorRtDims shape)
    {
        if (!CanEstimate(shape))
        {
            throw new InvalidOperationException($"Tensor '{tensor.Name}' shape is not concrete enough for byte-size estimation.");
        }

        long elementCount = 1;
        foreach (int value in shape.Values)
        {
            elementCount = checked(elementCount * value);
        }

        int bytesPerComponent = tensor.EffectiveBytesPerComponent;
        int componentsPerElement = tensor.EffectiveComponentsPerElement;
        if (bytesPerComponent <= 0)
        {
            throw new NotSupportedException(
                $"Tensor '{tensor.Name}' data type {tensor.DataType} does not have an integral byte-size fallback.");
        }

        long bytes = checked(elementCount * bytesPerComponent * componentsPerElement);
        if (bytes <= 0 || bytes > int.MaxValue)
        {
            throw new InvalidOperationException($"Tensor '{tensor.Name}' estimated byte size is outside the managed allocation range.");
        }

        return (int)bytes;
    }
}
