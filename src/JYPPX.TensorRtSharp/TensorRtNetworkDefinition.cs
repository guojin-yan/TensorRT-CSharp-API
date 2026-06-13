using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Network Definition wrapper.
/// 表示托管 TensorRT Tensor Rt Network Definition 包装器。
/// </summary>
public sealed partial class TensorRtNetworkDefinition : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtNetworkDefinition(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets or sets the Line value.
    /// 获取或设置 Line 值。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets or sets the Name value.
    /// 获取或设置 Name 值。
    /// </summary>
    public string Name
    {
        get => NativeBridgeApi.GetNetworkName(Line, _handle);
        set => NativeBridgeApi.SetNetworkName(Line, _handle, value);
    }

    /// <summary>
    /// Gets the Input Count value.
    /// 获取 Input Count 值。
    /// </summary>
    public int InputCount => NativeBridgeApi.GetNetworkInputCount(Line, _handle);

    /// <summary>
    /// Gets the Output Count value.
    /// 获取 Output Count 值。
    /// </summary>
    public int OutputCount => NativeBridgeApi.GetNetworkOutputCount(Line, _handle);

    /// <summary>
    /// Gets the Layer Count value.
    /// 获取 Layer Count 值。
    /// </summary>
    public int LayerCount => NativeBridgeApi.GetNetworkLayerCount(Line, _handle);

    /// <summary>
    /// Gets the Flags value.
    /// 获取 Flags 值。
    /// </summary>
    public TensorRtNetworkDefinitionCreationFlags Flags => NativeBridgeApi.GetNetworkFlags(Line, _handle);

    /// <summary>
    /// Gets whether this network uses TensorRT implicit batch dimensions.
    /// 获取当前网络是否使用 TensorRT 隐式 batch 维度。
    /// </summary>
    public bool HasImplicitBatchDimension => NativeBridgeApi.HasImplicitBatchDimension(Line, _handle);

    /// <summary>
    /// Gets the Flag value.
    /// 获取 Flag 值。
    /// </summary>
    public bool GetFlag(TensorRtNetworkDefinitionCreationFlags flag)
    {
        return NativeBridgeApi.GetNetworkFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Adds a Input layer or object.
    /// 添加 Input 层或对象。
    /// </summary>
    public TensorRtTensor AddInput(string name, TensorRtDataType dataType, TensorRtDims shape)
    {
        return new TensorRtTensor(Line, NativeBridgeApi.AddNetworkInput(Line, _handle, name, dataType, shape));
    }

    /// <summary>
    /// Gets the Input value.
    /// 获取 Input 值。
    /// </summary>
    public TensorRtTensor GetInput(int index)
    {
        return new TensorRtTensor(Line, NativeBridgeApi.GetNetworkInput(Line, _handle, index));
    }

    /// <summary>
    /// Gets a TensorRT 11 network input tensor shape with 64-bit dimension extents.
    /// 获取 TensorRT 11 网络输入张量形状，并保留 64 位维度 extent。
    /// </summary>
    /// <param name="index">The zero-based input index. 从零开始的输入索引。</param>
    /// <returns>The input tensor shape reported by TensorRT. TensorRT 报告的输入张量形状。</returns>
    public TensorRtDims64 GetInputShape64(int index)
    {
        return NativeBridgeApi.GetNetworkInputTensorShape64(Line, _handle, index);
    }

    /// <summary>
    /// Gets one TensorRT 11 network input tensor dimension extent as a 64-bit value.
    /// 以 64 位整数获取 TensorRT 11 网络输入张量的单个维度 extent。
    /// </summary>
    /// <param name="index">The zero-based input index. 从零开始的输入索引。</param>
    /// <param name="dimensionIndex">The zero-based dimension index. 从零开始的维度索引。</param>
    /// <returns>The input dimension extent reported by TensorRT. TensorRT 报告的输入维度 extent。</returns>
    public long GetInputDimensionExtent64(int index, int dimensionIndex)
    {
        return NativeBridgeApi.GetNetworkInputTensorDimensionExtent64(Line, _handle, index, dimensionIndex);
    }

    /// <summary>
    /// Gets the Output value.
    /// 获取 Output 值。
    /// </summary>
    public TensorRtTensor GetOutput(int index)
    {
        return new TensorRtTensor(Line, NativeBridgeApi.GetNetworkOutput(Line, _handle, index));
    }

    /// <summary>
    /// Gets a TensorRT 11 network output tensor shape with 64-bit dimension extents.
    /// 获取 TensorRT 11 网络输出张量形状，并保留 64 位维度 extent。
    /// </summary>
    /// <param name="index">The zero-based output index. 从零开始的输出索引。</param>
    /// <returns>The output tensor shape reported by TensorRT. TensorRT 报告的输出张量形状。</returns>
    public TensorRtDims64 GetOutputShape64(int index)
    {
        return NativeBridgeApi.GetNetworkOutputTensorShape64(Line, _handle, index);
    }

    /// <summary>
    /// Gets one TensorRT 11 network output tensor dimension extent as a 64-bit value.
    /// 以 64 位整数获取 TensorRT 11 网络输出张量的单个维度 extent。
    /// </summary>
    /// <param name="index">The zero-based output index. 从零开始的输出索引。</param>
    /// <param name="dimensionIndex">The zero-based dimension index. 从零开始的维度索引。</param>
    /// <returns>The output dimension extent reported by TensorRT. TensorRT 报告的输出维度 extent。</returns>
    public long GetOutputDimensionExtent64(int index, int dimensionIndex)
    {
        return NativeBridgeApi.GetNetworkOutputTensorDimensionExtent64(Line, _handle, index, dimensionIndex);
    }

    /// <summary>
    /// Gets the Layer value.
    /// 获取 Layer 值。
    /// </summary>
    public TensorRtLayer GetLayer(int index)
    {
        return new TensorRtLayer(Line, NativeBridgeApi.GetNetworkLayer(Line, _handle, index));
    }

    /// <summary>
    /// Adds a Identity layer or object.
    /// 添加 Identity 层或对象。
    /// </summary>
    public TensorRtLayer AddIdentity(TensorRtTensor input)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddIdentityLayer(Line, _handle, input.Handle));
    }

    /// <summary>
    /// Adds a Constant layer or object.
    /// 添加 Constant 层或对象。
    /// </summary>
    public TensorRtLayer AddConstant(TensorRtDims shape, TensorRtWeights weights)
    {
        if (shape == null)
        {
            throw new ArgumentNullException(nameof(shape));
        }

        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddConstantLayer(Line, _handle, shape, weights));
    }

    /// <summary>
    /// Adds a Convolution layer or object.
    /// 添加 Convolution 层或对象。
    /// </summary>
    public TensorRtLayer AddConvolution(TensorRtTensor input, int outputMaps, TensorRtDims kernelSize, TensorRtWeights kernelWeights, TensorRtWeights? biasWeights = null)
    {
        ValidateInputTensor(input, nameof(input));
        if (kernelSize == null)
        {
            throw new ArgumentNullException(nameof(kernelSize));
        }

        if (kernelWeights == null)
        {
            throw new ArgumentNullException(nameof(kernelWeights));
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddConvolutionLayer(Line, _handle, input.Handle, outputMaps, kernelSize, kernelWeights, biasWeights));
    }

    /// <summary>
    /// Adds a Scale layer or object.
    /// 添加 Scale 层或对象。
    /// </summary>
    public TensorRtLayer AddScale(
        TensorRtTensor input,
        TensorRtScaleMode mode,
        TensorRtWeights? shift = null,
        TensorRtWeights? scale = null,
        TensorRtWeights? power = null,
        int channelAxis = 0)
    {
        ValidateInputTensor(input, nameof(input));
        return new TensorRtLayer(Line, NativeBridgeApi.AddScaleLayer(Line, _handle, input.Handle, mode, shift, scale, power, channelAxis));
    }

    /// <summary>
    /// Adds a Padding layer or object.
    /// 添加 Padding 层或对象。
    /// </summary>
    public TensorRtLayer AddPadding(TensorRtTensor input, TensorRtDims prePadding, TensorRtDims postPadding)
    {
        ValidateInputTensor(input, nameof(input));
        if (prePadding == null)
        {
            throw new ArgumentNullException(nameof(prePadding));
        }

        if (postPadding == null)
        {
            throw new ArgumentNullException(nameof(postPadding));
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddPaddingLayer(Line, _handle, input.Handle, prePadding, postPadding));
    }

    /// <summary>
    /// Adds a Element Wise layer or object.
    /// 添加 Element Wise 层或对象。
    /// </summary>
    public TensorRtLayer AddElementWise(TensorRtTensor left, TensorRtTensor right, TensorRtElementWiseOperation operation)
    {
        if (left == null)
        {
            throw new ArgumentNullException(nameof(left));
        }

        if (right == null)
        {
            throw new ArgumentNullException(nameof(right));
        }

        if (left.Line != Line || right.Line != Line)
        {
            throw new ArgumentException("Input tensors must belong to the same TensorRT API line as the network.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddElementWiseLayer(Line, _handle, left.Handle, right.Handle, operation));
    }

    /// <summary>
    /// Adds a Matrix Multiply layer or object.
    /// 添加 Matrix Multiply 层或对象。
    /// </summary>
    public TensorRtLayer AddMatrixMultiply(
        TensorRtTensor left,
        TensorRtMatrixOperation leftOperation,
        TensorRtTensor right,
        TensorRtMatrixOperation rightOperation)
    {
        if (left == null)
        {
            throw new ArgumentNullException(nameof(left));
        }

        if (right == null)
        {
            throw new ArgumentNullException(nameof(right));
        }

        if (left.Line != Line || right.Line != Line)
        {
            throw new ArgumentException("Input tensors must belong to the same TensorRT API line as the network.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddMatrixMultiplyLayer(Line, _handle, left.Handle, leftOperation, right.Handle, rightOperation));
    }

    /// <summary>
    /// Adds a Shuffle layer or object.
    /// 添加 Shuffle 层或对象。
    /// </summary>
    public TensorRtLayer AddShuffle(TensorRtTensor input, TensorRtDims reshapeDimensions)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (reshapeDimensions == null)
        {
            throw new ArgumentNullException(nameof(reshapeDimensions));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        TensorRtLayer layer = new TensorRtLayer(Line, NativeBridgeApi.AddShuffleLayer(Line, _handle, input.Handle));
        try
        {
            layer.SetShuffleReshapeDimensions(reshapeDimensions);
            return layer;
        }
        catch
        {
            layer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Adds a Reduce layer or object.
    /// 添加 Reduce 层或对象。
    /// </summary>
    public TensorRtLayer AddReduce(TensorRtTensor input, TensorRtReduceOperation operation, uint axes, bool keepDimensions)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "Reduce axes bitmask must not be zero.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddReduceLayer(Line, _handle, input.Handle, operation, axes, keepDimensions));
    }

    /// <summary>
    /// Adds a Soft Max layer or object.
    /// 添加 Soft Max 层或对象。
    /// </summary>
    public TensorRtLayer AddSoftMax(TensorRtTensor input, uint axes)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "SoftMax axes bitmask must not be zero.");
        }

        TensorRtLayer layer = new TensorRtLayer(Line, NativeBridgeApi.AddSoftMaxLayer(Line, _handle, input.Handle));
        try
        {
            layer.SetSoftMaxAxes(axes);
            return layer;
        }
        catch
        {
            layer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Adds a Unary layer or object.
    /// 添加 Unary 层或对象。
    /// </summary>
    public TensorRtLayer AddUnary(TensorRtTensor input, TensorRtUnaryOperation operation)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddUnaryLayer(Line, _handle, input.Handle, operation));
    }

    /// <summary>
    /// Adds a Top K layer or object.
    /// 添加 Top K 层或对象。
    /// </summary>
    public TensorRtLayer AddTopK(TensorRtTensor input, TensorRtTopKOperation operation, int k, uint axes)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "TopK k must be greater than zero.");
        }

        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "TopK axes bitmask must not be zero.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddTopKLayer(Line, _handle, input.Handle, operation, k, axes));
    }

    /// <summary>
    /// Adds a Gather layer or object.
    /// 添加 Gather 层或对象。
    /// </summary>
    public TensorRtLayer AddGather(TensorRtTensor data, TensorRtTensor indices, int axis)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (indices == null)
        {
            throw new ArgumentNullException(nameof(indices));
        }

        if (data.Line != Line || indices.Line != Line)
        {
            throw new ArgumentException("Input tensors must belong to the same TensorRT API line as the network.");
        }

        if (axis < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axis), "Gather axis must be greater than or equal to zero.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddGatherLayer(Line, _handle, data.Handle, indices.Handle, axis));
    }

    /// <summary>
    /// Adds a Activation layer or object.
    /// 添加 Activation 层或对象。
    /// </summary>
    public TensorRtLayer AddActivation(TensorRtTensor input, TensorRtActivationType activationType)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddActivationLayer(Line, _handle, input.Handle, activationType));
    }

    /// <summary>
    /// Adds a Pooling layer or object.
    /// 添加 Pooling 层或对象。
    /// </summary>
    public TensorRtLayer AddPooling(TensorRtTensor input, TensorRtPoolingType poolingType, TensorRtDims windowSize)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (windowSize == null)
        {
            throw new ArgumentNullException(nameof(windowSize));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddPoolingLayer(Line, _handle, input.Handle, poolingType, windowSize));
    }

    /// <summary>
    /// Adds a Resize layer or object.
    /// 添加 Resize 层或对象。
    /// </summary>
    public TensorRtLayer AddResize(TensorRtTensor input, TensorRtDims outputDimensions, TensorRtResizeMode resizeMode = TensorRtResizeMode.Nearest)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (outputDimensions == null)
        {
            throw new ArgumentNullException(nameof(outputDimensions));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        TensorRtLayer layer = new TensorRtLayer(Line, NativeBridgeApi.AddResizeLayer(Line, _handle, input.Handle));
        try
        {
            layer.SetResizeMode(resizeMode);
            layer.SetResizeOutputDimensions(outputDimensions);
            return layer;
        }
        catch
        {
            layer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Adds a Concatenation layer or object.
    /// 添加 Concatenation 层或对象。
    /// </summary>
    public TensorRtLayer AddConcatenation(params TensorRtTensor[] inputs)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (inputs.Length < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(inputs), "Concatenation requires at least two input tensors.");
        }

        SafeTensorRtObjectHandle[] inputHandles = new SafeTensorRtObjectHandle[inputs.Length];
        for (int index = 0; index < inputs.Length; index++)
        {
            if (inputs[index] == null)
            {
                throw new ArgumentNullException(nameof(inputs), "Input tensors must not contain null entries.");
            }

            if (inputs[index].Line != Line)
            {
                throw new ArgumentException("Input tensors must belong to the same TensorRT API line as the network.");
            }

            inputHandles[index] = inputs[index].Handle;
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddConcatenationLayer(Line, _handle, inputHandles));
    }

    /// <summary>
    /// Adds a Slice layer or object.
    /// 添加 Slice 层或对象。
    /// </summary>
    public TensorRtLayer AddSlice(TensorRtTensor input, TensorRtDims start, TensorRtDims size, TensorRtDims stride)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (start == null)
        {
            throw new ArgumentNullException(nameof(start));
        }

        if (size == null)
        {
            throw new ArgumentNullException(nameof(size));
        }

        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddSliceLayer(Line, _handle, input.Handle, start, size, stride));
    }

    /// <summary>
    /// Adds a Shape layer or object.
    /// 添加 Shape 层或对象。
    /// </summary>
    public TensorRtLayer AddShape(TensorRtTensor input)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddShapeLayer(Line, _handle, input.Handle));
    }

    /// <summary>
    /// Adds a Select layer or object.
    /// 添加 Select 层或对象。
    /// </summary>
    public TensorRtLayer AddSelect(TensorRtTensor condition, TensorRtTensor thenInput, TensorRtTensor elseInput)
    {
        if (condition == null)
        {
            throw new ArgumentNullException(nameof(condition));
        }

        if (thenInput == null)
        {
            throw new ArgumentNullException(nameof(thenInput));
        }

        if (elseInput == null)
        {
            throw new ArgumentNullException(nameof(elseInput));
        }

        if (condition.Line != Line || thenInput.Line != Line || elseInput.Line != Line)
        {
            throw new ArgumentException("Input tensors must belong to the same TensorRT API line as the network.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddSelectLayer(Line, _handle, condition.Handle, thenInput.Handle, elseInput.Handle));
    }

    /// <summary>
    /// Adds a Fill layer or object.
    /// 添加 Fill 层或对象。
    /// </summary>
    public TensorRtLayer AddFill(TensorRtDims dimensions, TensorRtFillOperation operation)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddFillLayer(Line, _handle, dimensions, operation));
    }

    /// <summary>
    /// Marks the Output value.
    /// 标记 Output 值。
    /// </summary>
    public void MarkOutput(TensorRtTensor tensor)
    {
        if (tensor == null)
        {
            throw new ArgumentNullException(nameof(tensor));
        }

        if (tensor.Line != Line)
        {
            throw new ArgumentException("Output tensor must belong to the same TensorRT API line as the network.");
        }

        NativeBridgeApi.MarkNetworkOutput(Line, _handle, tensor.Handle);
    }

    /// <summary>
    /// Unmarks the Output value.
    /// 取消标记 Output 值。
    /// </summary>
    public void UnmarkOutput(TensorRtTensor tensor)
    {
        if (tensor == null)
        {
            throw new ArgumentNullException(nameof(tensor));
        }

        if (tensor.Line != Line)
        {
            throw new ArgumentException("Output tensor must belong to the same TensorRT API line as the network.");
        }

        NativeBridgeApi.UnmarkNetworkOutput(Line, _handle, tensor.Handle);
    }

    /// <summary>
    /// Releases the native TensorRT resources held by this object.
    /// 释放此对象持有的 native TensorRT 资源。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ValidateInputTensor(TensorRtTensor tensor, string argumentName)
    {
        if (tensor == null)
        {
            throw new ArgumentNullException(argumentName);
        }

        if (tensor.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.", argumentName);
        }
    }
}
