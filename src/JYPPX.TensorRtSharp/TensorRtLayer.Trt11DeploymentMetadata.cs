using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets the TensorRT gather mode.
    /// 获取 TensorRT gather 层模式。
    /// </summary>
    public TensorRtGatherMode GetGatherMode()
    {
        return NativeBridgeApi.GetGatherMode(Line, _handle);
    }

    /// <summary>
    /// Sets the TensorRT gather mode.
    /// 设置 TensorRT gather 层模式。
    /// </summary>
    public void SetGatherMode(TensorRtGatherMode mode)
    {
        NativeBridgeApi.SetGatherMode(Line, _handle, mode);
    }

    /// <summary>
    /// Gets the number of elementwise dimensions used by a gather layer.
    /// 获取 gather 层使用的 elementwise 维度数量。
    /// </summary>
    public int GetGatherElementWiseDimensions()
    {
        return NativeBridgeApi.GetGatherElementWiseDimensions(Line, _handle);
    }

    /// <summary>
    /// Sets the number of elementwise dimensions used by a gather layer.
    /// 设置 gather 层使用的 elementwise 维度数量。
    /// </summary>
    public void SetGatherElementWiseDimensions(int dimensions)
    {
        if (dimensions < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensions));
        }

        NativeBridgeApi.SetGatherElementWiseDimensions(Line, _handle, dimensions);
    }

    /// <summary>
    /// Gets the TensorRT scatter mode.
    /// 获取 TensorRT scatter 层模式。
    /// </summary>
    public TensorRtScatterMode GetScatterMode()
    {
        return NativeBridgeApi.GetScatterMode(Line, _handle);
    }

    /// <summary>
    /// Sets the TensorRT scatter mode.
    /// 设置 TensorRT scatter 层模式。
    /// </summary>
    public void SetScatterMode(TensorRtScatterMode mode)
    {
        NativeBridgeApi.SetScatterMode(Line, _handle, mode);
    }

    /// <summary>
    /// Gets the axis used by a scatter layer in element mode.
    /// 获取 scatter 层在 element 模式下使用的轴。
    /// </summary>
    public int GetScatterAxis()
    {
        return NativeBridgeApi.GetScatterAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the axis used by a scatter layer in element mode.
    /// 设置 scatter 层在 element 模式下使用的轴。
    /// </summary>
    public void SetScatterAxis(int axis)
    {
        NativeBridgeApi.SetScatterAxis(Line, _handle, axis);
    }

    /// <summary>
    /// Gets the axis used by a one-hot layer.
    /// 获取 one-hot 层使用的轴。
    /// </summary>
    public int GetOneHotAxis()
    {
        return NativeBridgeApi.GetOneHotAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the axis used by a one-hot layer.
    /// 设置 one-hot 层使用的轴。
    /// </summary>
    public void SetOneHotAxis(int axis)
    {
        NativeBridgeApi.SetOneHotAxis(Line, _handle, axis);
    }

    /// <summary>
    /// Gets the cumulative operation.
    /// 获取 cumulative 层的累计运算类型。
    /// </summary>
    public TensorRtCumulativeOperation GetCumulativeOperation()
    {
        return NativeBridgeApi.GetCumulativeOperation(Line, _handle);
    }

    /// <summary>
    /// Sets the cumulative operation.
    /// 设置 cumulative 层的累计运算类型。
    /// </summary>
    public void SetCumulativeOperation(TensorRtCumulativeOperation operation)
    {
        NativeBridgeApi.SetCumulativeOperation(Line, _handle, operation);
    }

    /// <summary>
    /// Gets whether the cumulative layer uses exclusive accumulation.
    /// 获取 cumulative 层是否使用 exclusive 累计。
    /// </summary>
    public bool GetCumulativeExclusive()
    {
        return NativeBridgeApi.GetCumulativeExclusive(Line, _handle);
    }

    /// <summary>
    /// Sets whether the cumulative layer uses exclusive accumulation.
    /// 设置 cumulative 层是否使用 exclusive 累计。
    /// </summary>
    public void SetCumulativeExclusive(bool exclusive)
    {
        NativeBridgeApi.SetCumulativeExclusive(Line, _handle, exclusive);
    }

    /// <summary>
    /// Gets whether the cumulative layer accumulates in reverse order.
    /// 获取 cumulative 层是否按反向顺序累计。
    /// </summary>
    public bool GetCumulativeReverse()
    {
        return NativeBridgeApi.GetCumulativeReverse(Line, _handle);
    }

    /// <summary>
    /// Sets whether the cumulative layer accumulates in reverse order.
    /// 设置 cumulative 层是否按反向顺序累计。
    /// </summary>
    public void SetCumulativeReverse(bool reverse)
    {
        NativeBridgeApi.SetCumulativeReverse(Line, _handle, reverse);
    }

    /// <summary>
    /// Gets the assertion message.
    /// 获取 assertion 层的断言消息。
    /// </summary>
    public string GetAssertionMessage()
    {
        return NativeBridgeApi.GetAssertionMessage(Line, _handle);
    }

    /// <summary>
    /// Sets the assertion message.
    /// 设置 assertion 层的断言消息。
    /// </summary>
    public void SetAssertionMessage(string message)
    {
        NativeBridgeApi.SetAssertionMessage(Line, _handle, message);
    }

    /// <summary>
    /// Gets the interpolation mode of a grid-sample layer.
    /// 获取 grid-sample 层的插值模式。
    /// </summary>
    public TensorRtInterpolationMode GetGridSampleInterpolationMode()
    {
        return NativeBridgeApi.GetGridSampleInterpolationMode(Line, _handle);
    }

    /// <summary>
    /// Sets the interpolation mode of a grid-sample layer.
    /// 设置 grid-sample 层的插值模式。
    /// </summary>
    public void SetGridSampleInterpolationMode(TensorRtInterpolationMode mode)
    {
        NativeBridgeApi.SetGridSampleInterpolationMode(Line, _handle, mode);
    }

    /// <summary>
    /// Gets whether a grid-sample layer aligns corners.
    /// 获取 grid-sample 层是否对齐角点。
    /// </summary>
    public bool GetGridSampleAlignCorners()
    {
        return NativeBridgeApi.GetGridSampleAlignCorners(Line, _handle);
    }

    /// <summary>
    /// Sets whether a grid-sample layer aligns corners.
    /// 设置 grid-sample 层是否对齐角点。
    /// </summary>
    public void SetGridSampleAlignCorners(bool alignCorners)
    {
        NativeBridgeApi.SetGridSampleAlignCorners(Line, _handle, alignCorners);
    }

    /// <summary>
    /// Gets the out-of-bounds sample mode of a grid-sample layer.
    /// 获取 grid-sample 层的越界采样模式。
    /// </summary>
    public TensorRtSampleMode GetGridSampleMode()
    {
        return NativeBridgeApi.GetGridSampleMode(Line, _handle);
    }

    /// <summary>
    /// Sets the out-of-bounds sample mode of a grid-sample layer.
    /// 设置 grid-sample 层的越界采样模式。
    /// </summary>
    public void SetGridSampleMode(TensorRtSampleMode mode)
    {
        NativeBridgeApi.SetGridSampleMode(Line, _handle, mode);
    }

    /// <summary>
    /// Gets the epsilon value of a normalization layer.
    /// 获取 normalization 层的 epsilon 参数。
    /// </summary>
    public double GetNormalizationEpsilon()
    {
        return NativeBridgeApi.GetNormalizationEpsilon(Line, _handle);
    }

    /// <summary>
    /// Sets the epsilon value of a normalization layer.
    /// 设置 normalization 层的 epsilon 参数。
    /// </summary>
    public void SetNormalizationEpsilon(double epsilon)
    {
        NativeBridgeApi.SetNormalizationEpsilon(Line, _handle, epsilon);
    }

    /// <summary>
    /// Gets the axes bitmask of a normalization layer.
    /// 获取 normalization 层的 axes 位掩码。
    /// </summary>
    public uint GetNormalizationAxes()
    {
        return NativeBridgeApi.GetNormalizationAxes(Line, _handle);
    }

    /// <summary>
    /// Sets the axes bitmask of a normalization layer.
    /// 设置 normalization 层的 axes 位掩码。
    /// </summary>
    public void SetNormalizationAxes(uint axes)
    {
        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "Normalization axes bitmask must not be zero.");
        }

        NativeBridgeApi.SetNormalizationAxes(Line, _handle, axes);
    }

    /// <summary>
    /// Gets the compute precision used by a TensorRT 8 or TensorRT 10 normalization layer.
    /// 获取 TensorRT 8 或 TensorRT 10 normalization 层使用的计算精度。
    /// </summary>
    /// <returns>The normalization compute precision data type. normalization 计算精度的数据类型。</returns>
    public TensorRtDataType GetNormalizationComputePrecision()
    {
        return NativeBridgeApi.GetNormalizationComputePrecision(Line, _handle);
    }

    /// <summary>
    /// Sets the compute precision used by a TensorRT 8 or TensorRT 10 normalization layer.
    /// 设置 TensorRT 8 或 TensorRT 10 normalization 层使用的计算精度。
    /// </summary>
    /// <param name="dataType">The compute precision data type. 计算精度的数据类型。</param>
    public void SetNormalizationComputePrecision(TensorRtDataType dataType)
    {
        NativeBridgeApi.SetNormalizationComputePrecision(Line, _handle, dataType);
    }

    /// <summary>
    /// Gets the group count of a normalization layer.
    /// 获取 normalization 层的分组数量。
    /// </summary>
    public long GetNormalizationGroupCount()
    {
        return NativeBridgeApi.GetNormalizationGroupCount(Line, _handle);
    }

    /// <summary>
    /// Sets the group count of a normalization layer.
    /// 设置 normalization 层的分组数量。
    /// </summary>
    public void SetNormalizationGroupCount(long groupCount)
    {
        if (groupCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(groupCount));
        }

        NativeBridgeApi.SetNormalizationGroupCount(Line, _handle, groupCount);
    }

    /// <summary>
    /// Gets whether a normalization layer is a TensorRT V2 normalization layer.
    /// 获取 normalization 层是否为 TensorRT V2 normalization 层。
    /// </summary>
    public bool IsNormalizationV2()
    {
        return NativeBridgeApi.IsNormalizationV2(Line, _handle);
    }

    /// <summary>
    /// Gets the output data type of a dynamic-quantize layer.
    /// 获取 dynamic-quantize 层的输出数据类型。
    /// </summary>
    public TensorRtDataType GetDynamicQuantizeToType()
    {
        return NativeBridgeApi.GetDynamicQuantizeToType(Line, _handle);
    }

    /// <summary>
    /// Sets the output data type of a dynamic-quantize layer.
    /// 设置 dynamic-quantize 层的输出数据类型。
    /// </summary>
    public void SetDynamicQuantizeToType(TensorRtDataType dataType)
    {
        NativeBridgeApi.SetDynamicQuantizeToType(Line, _handle, dataType);
    }

    /// <summary>
    /// Gets the scale tensor data type of a dynamic-quantize layer.
    /// 获取 dynamic-quantize 层的 scale 张量数据类型。
    /// </summary>
    public TensorRtDataType GetDynamicQuantizeScaleType()
    {
        return NativeBridgeApi.GetDynamicQuantizeScaleType(Line, _handle);
    }

    /// <summary>
    /// Sets the scale tensor data type of a dynamic-quantize layer.
    /// 设置 dynamic-quantize 层的 scale 张量数据类型。
    /// </summary>
    public void SetDynamicQuantizeScaleType(TensorRtDataType dataType)
    {
        NativeBridgeApi.SetDynamicQuantizeScaleType(Line, _handle, dataType);
    }

    /// <summary>
    /// Gets the compatibility quantization axis of a TensorRT 10/11 dynamic-quantize layer.
    /// 获取 TensorRT 10 dynamic-quantize 层的量化 axis。
    /// </summary>
    public int GetDynamicQuantizeAxis()
    {
        return NativeBridgeApi.GetDynamicQuantizeAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the compatibility quantization axis of a TensorRT 10/11 dynamic-quantize layer.
    /// 设置 TensorRT 10 dynamic-quantize 层的量化 axis。
    /// </summary>
    public void SetDynamicQuantizeAxis(int axis)
    {
        NativeBridgeApi.SetDynamicQuantizeAxis(Line, _handle, axis);
    }

    /// <summary>
    /// Gets the compatibility block size of a TensorRT 10/11 dynamic-quantize layer.
    /// 获取 TensorRT 10 dynamic-quantize 层的 block size。
    /// </summary>
    public int GetDynamicQuantizeBlockSize()
    {
        return NativeBridgeApi.GetDynamicQuantizeBlockSize(Line, _handle);
    }

    /// <summary>
    /// Sets the compatibility block size of a TensorRT 10/11 dynamic-quantize layer.
    /// 设置 TensorRT 10 dynamic-quantize 层的 block size。
    /// </summary>
    public void SetDynamicQuantizeBlockSize(int blockSize)
    {
        if (blockSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(blockSize), "Dynamic quantize block size must be positive.");
        }

        NativeBridgeApi.SetDynamicQuantizeBlockSize(Line, _handle, blockSize);
    }

    /// <summary>
    /// Gets the quantization block shape of a dynamic-quantize layer.
    /// 获取 dynamic-quantize 层的量化 block shape。
    /// </summary>
    public TensorRtDims GetDynamicQuantizeBlockShape()
    {
        return NativeBridgeApi.GetDynamicQuantizeBlockShape(Line, _handle);
    }

    /// <summary>
    /// Sets the quantization block shape of a dynamic-quantize layer.
    /// 设置 dynamic-quantize 层的量化 block shape。
    /// </summary>
    public void SetDynamicQuantizeBlockShape(TensorRtDims blockShape)
    {
        if (blockShape == null)
        {
            throw new ArgumentNullException(nameof(blockShape));
        }

        NativeBridgeApi.SetDynamicQuantizeBlockShape(Line, _handle, blockShape);
    }

    /// <summary>
    /// Gets the dimensions stored on a TensorRT constant layer.
    /// 获取 TensorRT constant 层中保存的常量维度。
    /// </summary>
    public TensorRtDims GetConstantLayerDimensions()
    {
        return NativeBridgeApi.GetConstantLayerDimensions(Line, _handle);
    }

    /// <summary>
    /// Sets the dimensions stored on a TensorRT constant layer.
    /// 设置 TensorRT constant 层中保存的常量维度。
    /// </summary>
    /// <param name="dimensions">The constant dimensions. 常量维度。</param>
    public void SetConstantLayerDimensions(TensorRtDims dimensions)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeBridgeApi.SetConstantLayerDimensions(Line, _handle, dimensions);
    }

    /// <summary>
    /// Gets the equation string from a TensorRT einsum layer.
    /// 获取 TensorRT einsum 层的方程字符串。
    /// </summary>
    public string GetEinsumEquation()
    {
        return NativeBridgeApi.GetEinsumEquation(Line, _handle);
    }

    /// <summary>
    /// Sets the equation string on a TensorRT einsum layer.
    /// 设置 TensorRT einsum 层的方程字符串。
    /// </summary>
    /// <param name="equation">The einsum equation, for example <c>"ij,jk-&gt;ik"</c>. Einsum 方程，例如 <c>"ij,jk-&gt;ik"</c>。</param>
    /// <returns><see langword="true"/> when TensorRT accepts the new equation. 当 TensorRT 接受新方程时返回 <see langword="true"/>。</returns>
    public bool SetEinsumEquation(string equation)
    {
        if (string.IsNullOrWhiteSpace(equation))
        {
            throw new ArgumentException("Einsum equation must not be null, empty, or whitespace.", nameof(equation));
        }

        return NativeBridgeApi.SetEinsumEquation(Line, _handle, equation);
    }
}
