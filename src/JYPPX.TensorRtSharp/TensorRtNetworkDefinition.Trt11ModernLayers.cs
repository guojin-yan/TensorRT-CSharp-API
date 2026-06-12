using System;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>
    /// Adds a TensorRT scatter layer.
    /// 添加 TensorRT scatter 层。
    /// </summary>
    /// <remarks>
    /// This API is currently exposed through the TensorRT 11 adapter. The returned layer is owned by the network.
    /// 当前此 API 通过 TensorRT 11 适配器暴露。返回的层由 network 持有生命周期。
    /// </remarks>
    public TensorRtLayer AddScatter(TensorRtTensor data, TensorRtTensor indices, TensorRtTensor updates, TensorRtScatterMode mode)
    {
        ValidateInputTensor(data, nameof(data));
        ValidateInputTensor(indices, nameof(indices));
        ValidateInputTensor(updates, nameof(updates));
        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddScatterLayer(Line, _handle, data.Handle, indices.Handle, updates.Handle, mode));
    }

    /// <summary>
    /// Adds a TensorRT one-hot layer.
    /// 添加 TensorRT one-hot 层。
    /// </summary>
    /// <remarks>
    /// This API is currently exposed through the TensorRT 11 adapter.
    /// 当前此 API 通过 TensorRT 11 适配器暴露。
    /// </remarks>
    public TensorRtLayer AddOneHot(TensorRtTensor indices, TensorRtTensor values, TensorRtTensor depth, int axis)
    {
        ValidateInputTensor(indices, nameof(indices));
        ValidateInputTensor(values, nameof(values));
        ValidateInputTensor(depth, nameof(depth));
        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddOneHotLayer(Line, _handle, indices.Handle, values.Handle, depth.Handle, axis));
    }

    /// <summary>
    /// Adds a TensorRT cumulative operation layer.
    /// 添加 TensorRT cumulative 累计运算层。
    /// </summary>
    /// <remarks>
    /// The axis tensor must be a TensorRT build-time constant 0D shape tensor; a 1D constant with one element is rejected by TensorRT 11.
    /// axis 张量必须是 TensorRT 构建期常量 0D shape tensor；只有一个元素的一维常量会被 TensorRT 11 拒绝。
    /// </remarks>
    public TensorRtLayer AddCumulative(TensorRtTensor input, TensorRtTensor axis, TensorRtCumulativeOperation operation, bool exclusive = false, bool reverse = false)
    {
        ValidateInputTensor(input, nameof(input));
        ValidateInputTensor(axis, nameof(axis));
        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddCumulativeLayer(Line, _handle, input.Handle, axis.Handle, operation, exclusive, reverse));
    }

    /// <summary>
    /// Adds a TensorRT assertion layer.
    /// 添加 TensorRT assertion 断言层。
    /// </summary>
    /// <remarks>
    /// The condition tensor must evaluate to a TensorRT boolean condition.
    /// condition 张量必须表示 TensorRT 布尔断言条件。
    /// </remarks>
    public TensorRtLayer AddAssertion(TensorRtTensor condition, string message)
    {
        ValidateInputTensor(condition, nameof(condition));
        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddAssertionLayer(Line, _handle, condition.Handle, message));
    }

    /// <summary>
    /// Adds a TensorRT grid-sample layer.
    /// 添加 TensorRT grid-sample 层。
    /// </summary>
    /// <remarks>
    /// Configure interpolation and sample modes on the returned layer.
    /// 可在返回的层对象上继续配置插值模式和越界采样模式。
    /// </remarks>
    public TensorRtLayer AddGridSample(TensorRtTensor input, TensorRtTensor grid)
    {
        ValidateInputTensor(input, nameof(input));
        ValidateInputTensor(grid, nameof(grid));
        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddGridSampleLayer(Line, _handle, input.Handle, grid.Handle));
    }

    /// <summary>
    /// Adds a TensorRT normalization V2 layer.
    /// 添加 TensorRT normalization V2 层。
    /// </summary>
    /// <remarks>
    /// The scale and bias tensors are passed directly to TensorRT; their shape rules are validated by TensorRT.
    /// scale 和 bias 张量直接传入 TensorRT；其 shape 规则由 TensorRT 原生侧校验。
    /// </remarks>
    public TensorRtLayer AddNormalizationV2(TensorRtTensor input, TensorRtTensor scale, TensorRtTensor bias, uint axes)
    {
        ValidateInputTensor(input, nameof(input));
        ValidateInputTensor(scale, nameof(scale));
        ValidateInputTensor(bias, nameof(bias));
        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "Normalization axes bitmask must not be zero.");
        }

        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddNormalizationV2Layer(Line, _handle, input.Handle, scale.Handle, bias.Handle, axes));
    }

    /// <summary>
    /// Adds a TensorRT squeeze layer.
    /// 添加 TensorRT squeeze 层。
    /// </summary>
    /// <remarks>
    /// The axes tensor controls which dimensions are removed.
    /// axes 张量控制需要移除的维度。
    /// </remarks>
    public TensorRtLayer AddSqueeze(TensorRtTensor input, TensorRtTensor axes)
    {
        ValidateInputTensor(input, nameof(input));
        ValidateInputTensor(axes, nameof(axes));
        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddSqueezeLayer(Line, _handle, input.Handle, axes.Handle));
    }

    /// <summary>
    /// Adds a TensorRT unsqueeze layer.
    /// 添加 TensorRT unsqueeze 层。
    /// </summary>
    /// <remarks>
    /// The axes tensor controls where new dimensions are inserted.
    /// axes 张量控制新增维度插入的位置。
    /// </remarks>
    public TensorRtLayer AddUnsqueeze(TensorRtTensor input, TensorRtTensor axes)
    {
        ValidateInputTensor(input, nameof(input));
        ValidateInputTensor(axes, nameof(axes));
        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddUnsqueezeLayer(Line, _handle, input.Handle, axes.Handle));
    }

    /// <summary>
    /// Adds a TensorRT dynamic-quantize V2 layer.
    /// 添加 TensorRT dynamic-quantize V2 动态量化层。
    /// </summary>
    /// <remarks>
    /// The block shape and data types are forwarded to TensorRT 11. TensorRT validates whether the combination is legal.
    /// block shape 和数据类型会传递给 TensorRT 11；组合是否合法由 TensorRT 原生侧校验。
    /// </remarks>
    public TensorRtLayer AddDynamicQuantizeV2(TensorRtTensor input, TensorRtDims blockShape, TensorRtDataType outputType, TensorRtDataType scaleType)
    {
        ValidateInputTensor(input, nameof(input));
        if (blockShape == null)
        {
            throw new ArgumentNullException(nameof(blockShape));
        }

        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddDynamicQuantizeV2Layer(Line, _handle, input.Handle, blockShape, outputType, scaleType));
    }

    /// <summary>
    /// Adds a TensorRT 10 dynamic-quantize layer using axis and block-size metadata.
    /// 使用 axis 和 block-size 元数据添加 TensorRT 10 dynamic-quantize 层。
    /// </summary>
    /// <remarks>
    /// TensorRT 10 uses <paramref name="axis"/> and <paramref name="blockSize"/>. TensorRT 11 uses <see cref="AddDynamicQuantizeV2"/> with a block shape.
    /// TensorRT 10 使用 <paramref name="axis"/> 和 <paramref name="blockSize"/>；TensorRT 11 请使用带 block shape 的 <see cref="AddDynamicQuantizeV2"/>。
    /// </remarks>
    public TensorRtLayer AddDynamicQuantize(TensorRtTensor input, int axis, int blockSize, TensorRtDataType outputType, TensorRtDataType scaleType)
    {
        ValidateInputTensor(input, nameof(input));
        if (blockSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(blockSize), "Dynamic quantize block size must be positive.");
        }

        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddDynamicQuantizeLayer(Line, _handle, input.Handle, axis, blockSize, outputType, scaleType));
    }
}
