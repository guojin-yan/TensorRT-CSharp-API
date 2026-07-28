using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Layer wrapper.
/// 表示托管 TensorRT Tensor Rt Layer 包装器。
/// </summary>
public sealed partial class TensorRtLayer : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;
    private readonly SafeTensorRtObjectHandleLease? _ownerLease;

    internal TensorRtLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle handle,
        SafeTensorRtObjectHandleLease? ownerLease = null)
    {
        Line = line;
        _handle = handle;
        _ownerLease = ownerLease;
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
        get => NativeBridgeApi.GetLayerName(Line, _handle);
        set => NativeBridgeApi.SetLayerName(Line, _handle, value);
    }

    /// <summary>
    /// Gets the Type value.
    /// 获取 Type 值。
    /// </summary>
    public TensorRtLayerType Type => NativeBridgeApi.GetLayerType(Line, _handle);

    /// <summary>
    /// Gets or sets the Input Count value.
    /// 获取或设置 Input Count 值。
    /// </summary>
    public int InputCount => NativeBridgeApi.GetLayerInputCount(Line, _handle);

    /// <summary>
    /// Gets or sets the Output Count value.
    /// 获取或设置 Output Count 值。
    /// </summary>
    public int OutputCount => NativeBridgeApi.GetLayerOutputCount(Line, _handle);

    /// <summary>
    /// Gets or sets the Precision value.
    /// 获取或设置 Precision 值。
    /// </summary>
    public TensorRtDataType Precision
    {
        get => NativeBridgeApi.GetLayerPrecision(Line, _handle);
        set => NativeBridgeApi.SetLayerPrecision(Line, _handle, value);
    }

    /// <summary>
    /// Gets the Is Precision Set value.
    /// 获取 Is Precision Set 值。
    /// </summary>
    public bool IsPrecisionSet => NativeBridgeApi.IsLayerPrecisionSet(Line, _handle);

    /// <summary>
    /// Resets the Precision setting.
    /// 重置 Precision 设置。
    /// </summary>
    public void ResetPrecision()
    {
        NativeBridgeApi.ResetLayerPrecision(Line, _handle);
    }

    /// <summary>
    /// Gets the Input value.
    /// 获取 Input 值。
    /// </summary>
    public TensorRtTensor GetInput(int index)
    {
        return new TensorRtTensor(
            Line,
            NativeBridgeApi.GetLayerInput(Line, _handle, index),
            _ownerLease?.Clone());
    }

    /// <summary>
    /// Gets the Output value.
    /// 获取 Output 值。
    /// </summary>
    public TensorRtTensor GetOutput(int index)
    {
        return new TensorRtTensor(
            Line,
            NativeBridgeApi.GetLayerOutput(Line, _handle, index),
            _ownerLease?.Clone());
    }

    /// <summary>
    /// Sets the Output Type value.
    /// 设置 Output Type 值。
    /// </summary>
    public void SetOutputType(int index, TensorRtDataType dataType)
    {
        ValidateOutputIndex(index);
        NativeBridgeApi.SetLayerOutputType(Line, _handle, index, dataType);
    }

    /// <summary>
    /// Gets the Output Type value.
    /// 获取 Output Type 值。
    /// </summary>
    public TensorRtDataType GetOutputType(int index)
    {
        ValidateOutputIndex(index);
        return NativeBridgeApi.GetLayerOutputType(Line, _handle, index);
    }

    /// <summary>
    /// Checks whether Output Type Set is true.
    /// 检查 Output Type Set 是否为 true。
    /// </summary>
    public bool IsOutputTypeSet(int index)
    {
        ValidateOutputIndex(index);
        return NativeBridgeApi.IsLayerOutputTypeSet(Line, _handle, index);
    }

    /// <summary>
    /// Resets the Output Type setting.
    /// 重置 Output Type 设置。
    /// </summary>
    public void ResetOutputType(int index)
    {
        ValidateOutputIndex(index);
        NativeBridgeApi.ResetLayerOutputType(Line, _handle, index);
    }

    /// <summary>
    /// Sets the Shuffle Reshape Dimensions value.
    /// 设置 Shuffle Reshape Dimensions 值。
    /// </summary>
    public void SetShuffleReshapeDimensions(TensorRtDims dims)
    {
        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        NativeBridgeApi.SetShuffleReshapeDimensions(Line, _handle, dims);
    }

    /// <summary>
    /// Gets the Shuffle Reshape Dimensions value.
    /// 获取 Shuffle Reshape Dimensions 值。
    /// </summary>
    public TensorRtDims GetShuffleReshapeDimensions()
    {
        return NativeBridgeApi.GetShuffleReshapeDimensions(Line, _handle);
    }

    /// <summary>
    /// Sets the Shuffle layer permutation applied before reshaping.
    /// 设置 Shuffle 层在 reshape 之前应用的转置排列。
    /// </summary>
    /// <param name="permutation">Permutation dimensions. 转置排列维度。</param>
    public void SetShuffleFirstTranspose(TensorRtDims permutation)
    {
        if (permutation == null)
        {
            throw new ArgumentNullException(nameof(permutation));
        }

        NativeBridgeApi.SetShuffleFirstTranspose(Line, _handle, permutation);
    }

    /// <summary>
    /// Gets the Shuffle layer permutation applied before reshaping.
    /// 获取 Shuffle 层在 reshape 之前应用的转置排列。
    /// </summary>
    /// <returns>The first transpose permutation. 第一段转置排列。</returns>
    public TensorRtDims GetShuffleFirstTranspose()
    {
        return NativeBridgeApi.GetShuffleFirstTranspose(Line, _handle);
    }

    /// <summary>
    /// Sets the Shuffle layer permutation applied after reshaping.
    /// 设置 Shuffle 层在 reshape 之后应用的转置排列。
    /// </summary>
    /// <param name="permutation">Permutation dimensions. 转置排列维度。</param>
    public void SetShuffleSecondTranspose(TensorRtDims permutation)
    {
        if (permutation == null)
        {
            throw new ArgumentNullException(nameof(permutation));
        }

        NativeBridgeApi.SetShuffleSecondTranspose(Line, _handle, permutation);
    }

    /// <summary>
    /// Gets the Shuffle layer permutation applied after reshaping.
    /// 获取 Shuffle 层在 reshape 之后应用的转置排列。
    /// </summary>
    /// <returns>The second transpose permutation. 第二段转置排列。</returns>
    public TensorRtDims GetShuffleSecondTranspose()
    {
        return NativeBridgeApi.GetShuffleSecondTranspose(Line, _handle);
    }

    /// <summary>
    /// Sets the Shuffle Zero Is Placeholder value.
    /// 设置 Shuffle Zero Is Placeholder 值。
    /// </summary>
    public void SetShuffleZeroIsPlaceholder(bool zeroIsPlaceholder)
    {
        NativeBridgeApi.SetShuffleZeroIsPlaceholder(Line, _handle, zeroIsPlaceholder);
    }

    /// <summary>
    /// Gets the Shuffle Zero Is Placeholder value.
    /// 获取 Shuffle Zero Is Placeholder 值。
    /// </summary>
    public bool GetShuffleZeroIsPlaceholder()
    {
        return NativeBridgeApi.GetShuffleZeroIsPlaceholder(Line, _handle);
    }

    /// <summary>
    /// Sets the Matrix Multiply Operation value.
    /// 设置 Matrix Multiply Operation 值。
    /// </summary>
    public void SetMatrixMultiplyOperation(int inputIndex, TensorRtMatrixOperation operation)
    {
        if (inputIndex < 0 || inputIndex > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(inputIndex), "MatrixMultiply input index must be 0 or 1.");
        }

        NativeBridgeApi.SetMatrixMultiplyOperation(Line, _handle, inputIndex, operation);
    }

    /// <summary>
    /// Gets the Matrix Multiply Operation value.
    /// 获取 Matrix Multiply Operation 值。
    /// </summary>
    public TensorRtMatrixOperation GetMatrixMultiplyOperation(int inputIndex)
    {
        if (inputIndex < 0 || inputIndex > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(inputIndex), "MatrixMultiply input index must be 0 or 1.");
        }

        return NativeBridgeApi.GetMatrixMultiplyOperation(Line, _handle, inputIndex);
    }

    /// <summary>
    /// Gets the Reduce Operation value.
    /// 获取 Reduce Operation 值。
    /// </summary>
    public TensorRtReduceOperation GetReduceOperation()
    {
        return NativeBridgeApi.GetReduceOperation(Line, _handle);
    }

    /// <summary>
    /// Sets the Reduce Operation value.
    /// 设置 Reduce Operation 值。
    /// </summary>
    public void SetReduceOperation(TensorRtReduceOperation operation)
    {
        NativeBridgeApi.SetReduceOperation(Line, _handle, operation);
    }

    /// <summary>
    /// Gets the Reduce Axes value.
    /// 获取 Reduce Axes 值。
    /// </summary>
    public uint GetReduceAxes()
    {
        return NativeBridgeApi.GetReduceAxes(Line, _handle);
    }

    /// <summary>
    /// Sets the Reduce Axes value.
    /// 设置 Reduce Axes 值。
    /// </summary>
    public void SetReduceAxes(uint axes)
    {
        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "Reduce axes bitmask must not be zero.");
        }

        NativeBridgeApi.SetReduceAxes(Line, _handle, axes);
    }

    /// <summary>
    /// Gets the Reduce Keep Dimensions value.
    /// 获取 Reduce Keep Dimensions 值。
    /// </summary>
    public bool GetReduceKeepDimensions()
    {
        return NativeBridgeApi.GetReduceKeepDimensions(Line, _handle);
    }

    /// <summary>
    /// Sets the Reduce Keep Dimensions value.
    /// 设置 Reduce Keep Dimensions 值。
    /// </summary>
    public void SetReduceKeepDimensions(bool keepDimensions)
    {
        NativeBridgeApi.SetReduceKeepDimensions(Line, _handle, keepDimensions);
    }

    /// <summary>
    /// Sets the Soft Max Axes value.
    /// 设置 Soft Max Axes 值。
    /// </summary>
    public void SetSoftMaxAxes(uint axes)
    {
        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "SoftMax axes bitmask must not be zero.");
        }

        NativeBridgeApi.SetSoftMaxAxes(Line, _handle, axes);
    }

    /// <summary>
    /// Gets the Soft Max Axes value.
    /// 获取 Soft Max Axes 值。
    /// </summary>
    public uint GetSoftMaxAxes()
    {
        return NativeBridgeApi.GetSoftMaxAxes(Line, _handle);
    }

    /// <summary>
    /// Gets the Unary Operation value.
    /// 获取 Unary Operation 值。
    /// </summary>
    public TensorRtUnaryOperation GetUnaryOperation()
    {
        return NativeBridgeApi.GetUnaryOperation(Line, _handle);
    }

    /// <summary>
    /// Sets the Unary Operation value.
    /// 设置 Unary Operation 值。
    /// </summary>
    public void SetUnaryOperation(TensorRtUnaryOperation operation)
    {
        NativeBridgeApi.SetUnaryOperation(Line, _handle, operation);
    }

    /// <summary>
    /// Gets the Top K Operation value.
    /// 获取 Top K Operation 值。
    /// </summary>
    public TensorRtTopKOperation GetTopKOperation()
    {
        return NativeBridgeApi.GetTopKOperation(Line, _handle);
    }

    /// <summary>
    /// Sets the Top K Operation value.
    /// 设置 Top K Operation 值。
    /// </summary>
    public void SetTopKOperation(TensorRtTopKOperation operation)
    {
        NativeBridgeApi.SetTopKOperation(Line, _handle, operation);
    }

    /// <summary>
    /// Gets the Top K Value value.
    /// 获取 Top K Value 值。
    /// </summary>
    public int GetTopKValue()
    {
        return NativeBridgeApi.GetTopKValue(Line, _handle);
    }

    /// <summary>
    /// Sets the Top K Value value.
    /// 设置 Top K Value 值。
    /// </summary>
    public void SetTopKValue(int k)
    {
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k));
        }

        NativeBridgeApi.SetTopKValue(Line, _handle, k);
    }

    /// <summary>
    /// Gets the Top K Axes value.
    /// 获取 Top K Axes 值。
    /// </summary>
    public uint GetTopKAxes()
    {
        return NativeBridgeApi.GetTopKAxes(Line, _handle);
    }

    /// <summary>
    /// Sets the Top K Axes value.
    /// 设置 Top K Axes 值。
    /// </summary>
    public void SetTopKAxes(uint axes)
    {
        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "TopK axes bitmask must not be zero.");
        }

        NativeBridgeApi.SetTopKAxes(Line, _handle, axes);
    }

    /// <summary>
    /// Gets the index output data type configured on a TensorRT 11 TopK layer.
    /// 获取 TensorRT 11 TopK 层索引输出的数据类型。
    /// </summary>
    /// <returns>The configured TopK indices data type. 已配置的 TopK indices 数据类型。</returns>
    public TensorRtDataType GetTopKIndicesType()
    {
        return NativeBridgeApi.GetTopKIndicesType(Line, _handle);
    }

    /// <summary>
    /// Sets the index output data type on a TensorRT 11 TopK layer.
    /// 设置 TensorRT 11 TopK 层索引输出的数据类型。
    /// </summary>
    /// <param name="dataType">The requested indices data type. 请求设置的 indices 数据类型。</param>
    /// <returns><c>true</c> when TensorRT accepts the value; otherwise <c>false</c>. TensorRT 接受该值时返回 <c>true</c>，否则返回 <c>false</c>。</returns>
    public bool SetTopKIndicesType(TensorRtDataType dataType)
    {
        return NativeBridgeApi.SetTopKIndicesType(Line, _handle, dataType);
    }

    /// <summary>
    /// Gets the Gather Axis value.
    /// 获取 Gather Axis 值。
    /// </summary>
    public int GetGatherAxis()
    {
        return NativeBridgeApi.GetGatherAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the Gather Axis value.
    /// 设置 Gather Axis 值。
    /// </summary>
    public void SetGatherAxis(int axis)
    {
        NativeBridgeApi.SetGatherAxis(Line, _handle, axis);
    }

    /// <summary>
    /// Gets the Element Wise Operation value.
    /// 获取 Element Wise Operation 值。
    /// </summary>
    public TensorRtElementWiseOperation GetElementWiseOperation()
    {
        return NativeBridgeApi.GetElementWiseOperation(Line, _handle);
    }

    /// <summary>
    /// Sets the Element Wise Operation value.
    /// 设置 Element Wise Operation 值。
    /// </summary>
    public void SetElementWiseOperation(TensorRtElementWiseOperation operation)
    {
        NativeBridgeApi.SetElementWiseOperation(Line, _handle, operation);
    }

    /// <summary>
    /// Gets the Activation Type value.
    /// 获取 Activation Type 值。
    /// </summary>
    public TensorRtActivationType GetActivationType()
    {
        return NativeBridgeApi.GetActivationType(Line, _handle);
    }

    /// <summary>
    /// Sets the Activation Type value.
    /// 设置 Activation Type 值。
    /// </summary>
    public void SetActivationType(TensorRtActivationType activationType)
    {
        NativeBridgeApi.SetActivationType(Line, _handle, activationType);
    }

    /// <summary>
    /// Gets the Activation Alpha value.
    /// 获取 Activation Alpha 值。
    /// </summary>
    public double GetActivationAlpha()
    {
        return NativeBridgeApi.GetActivationAlpha(Line, _handle);
    }

    /// <summary>
    /// Sets the Activation Alpha value.
    /// 设置 Activation Alpha 值。
    /// </summary>
    public void SetActivationAlpha(double alpha)
    {
        NativeBridgeApi.SetActivationAlpha(Line, _handle, alpha);
    }

    /// <summary>
    /// Gets the Activation Beta value.
    /// 获取 Activation Beta 值。
    /// </summary>
    public double GetActivationBeta()
    {
        return NativeBridgeApi.GetActivationBeta(Line, _handle);
    }

    /// <summary>
    /// Sets the Activation Beta value.
    /// 设置 Activation Beta 值。
    /// </summary>
    public void SetActivationBeta(double beta)
    {
        NativeBridgeApi.SetActivationBeta(Line, _handle, beta);
    }

    /// <summary>
    /// Gets the Pooling Type value.
    /// 获取 Pooling Type 值。
    /// </summary>
    public TensorRtPoolingType GetPoolingType()
    {
        return NativeBridgeApi.GetPoolingType(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Type value.
    /// 设置 Pooling Type 值。
    /// </summary>
    public void SetPoolingType(TensorRtPoolingType poolingType)
    {
        NativeBridgeApi.SetPoolingType(Line, _handle, poolingType);
    }

    /// <summary>
    /// Sets the Pooling Window Size value.
    /// 设置 Pooling Window Size 值。
    /// </summary>
    public void SetPoolingWindowSize(TensorRtDims windowSize)
    {
        if (windowSize == null)
        {
            throw new ArgumentNullException(nameof(windowSize));
        }

        NativeBridgeApi.SetPoolingWindowSize(Line, _handle, windowSize);
    }

    /// <summary>
    /// Gets the Pooling Window Size value.
    /// 获取 Pooling Window Size 值。
    /// </summary>
    public TensorRtDims GetPoolingWindowSize()
    {
        return NativeBridgeApi.GetPoolingWindowSize(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Stride value.
    /// 设置 Pooling Stride 值。
    /// </summary>
    public void SetPoolingStride(TensorRtDims stride)
    {
        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        NativeBridgeApi.SetPoolingStride(Line, _handle, stride);
    }

    /// <summary>
    /// Gets the Pooling Stride value.
    /// 获取 Pooling Stride 值。
    /// </summary>
    public TensorRtDims GetPoolingStride()
    {
        return NativeBridgeApi.GetPoolingStride(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Padding value.
    /// 设置 Pooling Padding 值。
    /// </summary>
    public void SetPoolingPadding(TensorRtDims padding)
    {
        if (padding == null)
        {
            throw new ArgumentNullException(nameof(padding));
        }

        NativeBridgeApi.SetPoolingPadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Pooling Padding value.
    /// 获取 Pooling Padding 值。
    /// </summary>
    public TensorRtDims GetPoolingPadding()
    {
        return NativeBridgeApi.GetPoolingPadding(Line, _handle);
    }

    /// <summary>
    /// Gets the Pooling Blend Factor value.
    /// 获取 Pooling Blend Factor 值。
    /// </summary>
    public double GetPoolingBlendFactor()
    {
        return NativeBridgeApi.GetPoolingBlendFactor(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Blend Factor value.
    /// 设置 Pooling Blend Factor 值。
    /// </summary>
    public void SetPoolingBlendFactor(double blendFactor)
    {
        NativeBridgeApi.SetPoolingBlendFactor(Line, _handle, blendFactor);
    }

    /// <summary>
    /// Gets the Pooling Average Count Excludes Padding value.
    /// 获取 Pooling Average Count Excludes Padding 值。
    /// </summary>
    public bool GetPoolingAverageCountExcludesPadding()
    {
        return NativeBridgeApi.GetPoolingAverageCountExcludesPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Average Count Excludes Padding value.
    /// 设置 Pooling Average Count Excludes Padding 值。
    /// </summary>
    public void SetPoolingAverageCountExcludesPadding(bool excludesPadding)
    {
        NativeBridgeApi.SetPoolingAverageCountExcludesPadding(Line, _handle, excludesPadding);
    }

    /// <summary>
    /// Gets the Pooling Pre Padding value.
    /// 获取 Pooling Pre Padding 值。
    /// </summary>
    public TensorRtDims GetPoolingPrePadding()
    {
        return NativeBridgeApi.GetPoolingPrePadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Pre Padding value.
    /// 设置 Pooling Pre Padding 值。
    /// </summary>
    public void SetPoolingPrePadding(TensorRtDims padding)
    {
        if (padding == null)
        {
            throw new ArgumentNullException(nameof(padding));
        }

        NativeBridgeApi.SetPoolingPrePadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Pooling Post Padding value.
    /// 获取 Pooling Post Padding 值。
    /// </summary>
    public TensorRtDims GetPoolingPostPadding()
    {
        return NativeBridgeApi.GetPoolingPostPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Post Padding value.
    /// 设置 Pooling Post Padding 值。
    /// </summary>
    public void SetPoolingPostPadding(TensorRtDims padding)
    {
        if (padding == null)
        {
            throw new ArgumentNullException(nameof(padding));
        }

        NativeBridgeApi.SetPoolingPostPadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Pooling Padding Mode value.
    /// 获取 Pooling Padding Mode 值。
    /// </summary>
    public TensorRtPaddingMode GetPoolingPaddingMode()
    {
        return NativeBridgeApi.GetPoolingPaddingMode(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Padding Mode value.
    /// 设置 Pooling Padding Mode 值。
    /// </summary>
    public void SetPoolingPaddingMode(TensorRtPaddingMode paddingMode)
    {
        NativeBridgeApi.SetPoolingPaddingMode(Line, _handle, paddingMode);
    }

    /// <summary>
    /// Gets the Convolution Output Maps value.
    /// 获取 Convolution Output Maps 值。
    /// </summary>
    public int GetConvolutionOutputMaps()
    {
        return NativeBridgeApi.GetConvolutionOutputMaps(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Output Maps value.
    /// 设置 Convolution Output Maps 值。
    /// </summary>
    public void SetConvolutionOutputMaps(int outputMaps)
    {
        NativeBridgeApi.SetConvolutionOutputMaps(Line, _handle, outputMaps);
    }

    /// <summary>
    /// Gets the Convolution Groups value.
    /// 获取 Convolution Groups 值。
    /// </summary>
    public int GetConvolutionGroups()
    {
        return NativeBridgeApi.GetConvolutionGroups(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Groups value.
    /// 设置 Convolution Groups 值。
    /// </summary>
    public void SetConvolutionGroups(int groups)
    {
        NativeBridgeApi.SetConvolutionGroups(Line, _handle, groups);
    }

    /// <summary>
    /// Gets the Convolution Stride value.
    /// 获取 Convolution Stride 值。
    /// </summary>
    public TensorRtDims GetConvolutionStride()
    {
        return NativeBridgeApi.GetConvolutionStride(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Stride value.
    /// 设置 Convolution Stride 值。
    /// </summary>
    public void SetConvolutionStride(TensorRtDims stride)
    {
        ValidateDims(stride, nameof(stride));
        NativeBridgeApi.SetConvolutionStride(Line, _handle, stride);
    }

    /// <summary>
    /// Gets the explicit N-D padding of a convolution layer.
    /// 获取 convolution 层的显式 N-D padding。
    /// </summary>
    /// <returns>The current convolution padding dimensions. 当前 convolution padding 维度。</returns>
    public TensorRtDims GetConvolutionPadding()
    {
        return NativeBridgeApi.GetConvolutionPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the explicit N-D padding of a convolution layer.
    /// 设置 convolution 层的显式 N-D padding。
    /// </summary>
    /// <param name="padding">The padding dimensions to apply. 要应用的 padding 维度。</param>
    public void SetConvolutionPadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetConvolutionPadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Convolution Pre Padding value.
    /// 获取 Convolution Pre Padding 值。
    /// </summary>
    public TensorRtDims GetConvolutionPrePadding()
    {
        return NativeBridgeApi.GetConvolutionPrePadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Pre Padding value.
    /// 设置 Convolution Pre Padding 值。
    /// </summary>
    public void SetConvolutionPrePadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetConvolutionPrePadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Convolution Post Padding value.
    /// 获取 Convolution Post Padding 值。
    /// </summary>
    public TensorRtDims GetConvolutionPostPadding()
    {
        return NativeBridgeApi.GetConvolutionPostPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Post Padding value.
    /// 设置 Convolution Post Padding 值。
    /// </summary>
    public void SetConvolutionPostPadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetConvolutionPostPadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Convolution Dilation value.
    /// 获取 Convolution Dilation 值。
    /// </summary>
    public TensorRtDims GetConvolutionDilation()
    {
        return NativeBridgeApi.GetConvolutionDilation(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Dilation value.
    /// 设置 Convolution Dilation 值。
    /// </summary>
    public void SetConvolutionDilation(TensorRtDims dilation)
    {
        ValidateDims(dilation, nameof(dilation));
        NativeBridgeApi.SetConvolutionDilation(Line, _handle, dilation);
    }

    /// <summary>
    /// Gets the Convolution Padding Mode value.
    /// 获取 Convolution Padding Mode 值。
    /// </summary>
    public TensorRtPaddingMode GetConvolutionPaddingMode()
    {
        return NativeBridgeApi.GetConvolutionPaddingMode(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Padding Mode value.
    /// 设置 Convolution Padding Mode 值。
    /// </summary>
    public void SetConvolutionPaddingMode(TensorRtPaddingMode paddingMode)
    {
        NativeBridgeApi.SetConvolutionPaddingMode(Line, _handle, paddingMode);
    }

    /// <summary>
    /// Gets the Scale Mode value.
    /// 获取 Scale Mode 值。
    /// </summary>
    public TensorRtScaleMode GetScaleMode()
    {
        return NativeBridgeApi.GetScaleMode(Line, _handle);
    }

    /// <summary>
    /// Sets the Scale Mode value.
    /// 设置 Scale Mode 值。
    /// </summary>
    public void SetScaleMode(TensorRtScaleMode mode)
    {
        NativeBridgeApi.SetScaleMode(Line, _handle, mode);
    }

    /// <summary>
    /// Gets the Scale Channel Axis value.
    /// 获取 Scale Channel Axis 值。
    /// </summary>
    public int GetScaleChannelAxis()
    {
        return NativeBridgeApi.GetScaleChannelAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the Scale Channel Axis value.
    /// 设置 Scale Channel Axis 值。
    /// </summary>
    public void SetScaleChannelAxis(int channelAxis)
    {
        NativeBridgeApi.SetScaleChannelAxis(Line, _handle, channelAxis);
    }

    /// <summary>
    /// Gets the Padding Pre Padding value.
    /// 获取 Padding Pre Padding 值。
    /// </summary>
    public TensorRtDims GetPaddingPrePadding()
    {
        return NativeBridgeApi.GetPaddingPrePadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Padding Pre Padding value.
    /// 设置 Padding Pre Padding 值。
    /// </summary>
    public void SetPaddingPrePadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetPaddingPrePadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Padding Post Padding value.
    /// 获取 Padding Post Padding 值。
    /// </summary>
    public TensorRtDims GetPaddingPostPadding()
    {
        return NativeBridgeApi.GetPaddingPostPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Padding Post Padding value.
    /// 设置 Padding Post Padding 值。
    /// </summary>
    public void SetPaddingPostPadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetPaddingPostPadding(Line, _handle, padding);
    }

    /// <summary>
    /// Sets the Resize Output Dimensions value.
    /// 设置 Resize Output Dimensions 值。
    /// </summary>
    public void SetResizeOutputDimensions(TensorRtDims dimensions)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeBridgeApi.SetResizeOutputDimensions(Line, _handle, dimensions);
    }

    /// <summary>
    /// Gets the Resize Output Dimensions value.
    /// 获取 Resize Output Dimensions 值。
    /// </summary>
    public TensorRtDims GetResizeOutputDimensions()
    {
        return NativeBridgeApi.GetResizeOutputDimensions(Line, _handle);
    }

    /// <summary>
    /// Sets the Resize Mode value.
    /// 设置 Resize Mode 值。
    /// </summary>
    public void SetResizeMode(TensorRtResizeMode resizeMode)
    {
        NativeBridgeApi.SetResizeMode(Line, _handle, resizeMode);
    }

    /// <summary>
    /// Gets the Resize Mode value.
    /// 获取 Resize Mode 值。
    /// </summary>
    public TensorRtResizeMode GetResizeMode()
    {
        return NativeBridgeApi.GetResizeMode(Line, _handle);
    }

    /// <summary>
    /// Gets the TensorRT 8 resize align-corners flag.
    /// 获取 TensorRT 8 resize 层的 align-corners 标志。
    /// </summary>
    /// <returns><c>true</c> when corner alignment is enabled. 启用 corner alignment 时返回 <c>true</c>。</returns>
    public bool GetResizeAlignCorners()
    {
        return NativeBridgeApi.GetResizeAlignCorners(Line, _handle);
    }

    /// <summary>
    /// Sets the TensorRT 8 resize align-corners flag.
    /// 设置 TensorRT 8 resize 层的 align-corners 标志。
    /// </summary>
    /// <param name="alignCorners">Whether resize should align the corner pixels. resize 是否应对齐角点像素。</param>
    public void SetResizeAlignCorners(bool alignCorners)
    {
        NativeBridgeApi.SetResizeAlignCorners(Line, _handle, alignCorners);
    }

    /// <summary>
    /// Gets the Resize Coordinate Transformation value.
    /// 获取 Resize Coordinate Transformation 值。
    /// </summary>
    public TensorRtResizeCoordinateTransformation GetResizeCoordinateTransformation()
    {
        return NativeBridgeApi.GetResizeCoordinateTransformation(Line, _handle);
    }

    /// <summary>
    /// Sets the Resize Coordinate Transformation value.
    /// 设置 Resize Coordinate Transformation 值。
    /// </summary>
    public void SetResizeCoordinateTransformation(TensorRtResizeCoordinateTransformation transformation)
    {
        NativeBridgeApi.SetResizeCoordinateTransformation(Line, _handle, transformation);
    }

    /// <summary>
    /// Gets the Resize Selector For Single Pixel value.
    /// 获取 Resize Selector For Single Pixel 值。
    /// </summary>
    public TensorRtResizeSelector GetResizeSelectorForSinglePixel()
    {
        return NativeBridgeApi.GetResizeSelectorForSinglePixel(Line, _handle);
    }

    /// <summary>
    /// Sets the Resize Selector For Single Pixel value.
    /// 设置 Resize Selector For Single Pixel 值。
    /// </summary>
    public void SetResizeSelectorForSinglePixel(TensorRtResizeSelector selector)
    {
        NativeBridgeApi.SetResizeSelectorForSinglePixel(Line, _handle, selector);
    }

    /// <summary>
    /// Gets the Resize Nearest Rounding value.
    /// 获取 Resize Nearest Rounding 值。
    /// </summary>
    public TensorRtResizeRoundMode GetResizeNearestRounding()
    {
        return NativeBridgeApi.GetResizeNearestRounding(Line, _handle);
    }

    /// <summary>
    /// Sets the Resize Nearest Rounding value.
    /// 设置 Resize Nearest Rounding 值。
    /// </summary>
    public void SetResizeNearestRounding(TensorRtResizeRoundMode rounding)
    {
        NativeBridgeApi.SetResizeNearestRounding(Line, _handle, rounding);
    }

    /// <summary>
    /// Gets the Resize Cubic Coefficient value.
    /// 获取 Resize Cubic Coefficient 值。
    /// </summary>
    public double GetResizeCubicCoefficient()
    {
        return NativeBridgeApi.GetResizeCubicCoefficient(Line, _handle);
    }

    /// <summary>
    /// Sets the Resize Cubic Coefficient value.
    /// 设置 Resize Cubic Coefficient 值。
    /// </summary>
    public void SetResizeCubicCoefficient(double value)
    {
        NativeBridgeApi.SetResizeCubicCoefficient(Line, _handle, value);
    }

    /// <summary>
    /// Gets the Resize Exclude Outside value.
    /// 获取 Resize Exclude Outside 值。
    /// </summary>
    public bool GetResizeExcludeOutside()
    {
        return NativeBridgeApi.GetResizeExcludeOutside(Line, _handle);
    }

    /// <summary>
    /// Sets the Resize Exclude Outside value.
    /// 设置 Resize Exclude Outside 值。
    /// </summary>
    public void SetResizeExcludeOutside(bool value)
    {
        NativeBridgeApi.SetResizeExcludeOutside(Line, _handle, value);
    }

    /// <summary>
    /// Sets the Resize Scales value.
    /// 设置 Resize Scales 值。
    /// </summary>
    public void SetResizeScales(float[] scales)
    {
        NativeBridgeApi.SetResizeScales(Line, _handle, scales);
    }

    /// <summary>
    /// Gets the Resize Scales value.
    /// 获取 Resize Scales 值。
    /// </summary>
    public float[] GetResizeScales()
    {
        return NativeBridgeApi.GetResizeScales(Line, _handle);
    }

    /// <summary>
    /// Sets the Concatenation Axis value.
    /// 设置 Concatenation Axis 值。
    /// </summary>
    public void SetConcatenationAxis(int axis)
    {
        NativeBridgeApi.SetConcatenationAxis(Line, _handle, axis);
    }

    /// <summary>
    /// Gets the Concatenation Axis value.
    /// 获取 Concatenation Axis 值。
    /// </summary>
    public int GetConcatenationAxis()
    {
        return NativeBridgeApi.GetConcatenationAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the Slice Start value.
    /// 设置 Slice Start 值。
    /// </summary>
    public void SetSliceStart(TensorRtDims start)
    {
        if (start == null)
        {
            throw new ArgumentNullException(nameof(start));
        }

        NativeBridgeApi.SetSliceStart(Line, _handle, start);
    }

    /// <summary>
    /// Gets the Slice Start value.
    /// 获取 Slice Start 值。
    /// </summary>
    public TensorRtDims GetSliceStart()
    {
        return NativeBridgeApi.GetSliceStart(Line, _handle);
    }

    /// <summary>
    /// Sets the Slice Size value.
    /// 设置 Slice Size 值。
    /// </summary>
    public void SetSliceSize(TensorRtDims size)
    {
        if (size == null)
        {
            throw new ArgumentNullException(nameof(size));
        }

        NativeBridgeApi.SetSliceSize(Line, _handle, size);
    }

    /// <summary>
    /// Gets the Slice Size value.
    /// 获取 Slice Size 值。
    /// </summary>
    public TensorRtDims GetSliceSize()
    {
        return NativeBridgeApi.GetSliceSize(Line, _handle);
    }

    /// <summary>
    /// Sets the Slice Stride value.
    /// 设置 Slice Stride 值。
    /// </summary>
    public void SetSliceStride(TensorRtDims stride)
    {
        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        NativeBridgeApi.SetSliceStride(Line, _handle, stride);
    }

    /// <summary>
    /// Gets the Slice Stride value.
    /// 获取 Slice Stride 值。
    /// </summary>
    public TensorRtDims GetSliceStride()
    {
        return NativeBridgeApi.GetSliceStride(Line, _handle);
    }

    /// <summary>
    /// Sets the axes vector used by a TensorRT 10 or TensorRT 11 slice layer.
    /// 设置 TensorRT 10 或 TensorRT 11 slice 层使用的 axes 向量。
    /// </summary>
    /// <param name="axes">The axes dimensions to apply. 要应用的 axes 维度。</param>
    public void SetSliceAxes(TensorRtDims axes)
    {
        if (axes == null)
        {
            throw new ArgumentNullException(nameof(axes));
        }

        NativeBridgeApi.SetSliceAxes(Line, _handle, axes);
    }

    /// <summary>
    /// Gets the axes vector used by a TensorRT 10 or TensorRT 11 slice layer.
    /// 获取 TensorRT 10 或 TensorRT 11 slice 层使用的 axes 向量。
    /// </summary>
    /// <returns>The current slice axes dimensions. 当前 slice axes 维度。</returns>
    public TensorRtDims GetSliceAxes()
    {
        return NativeBridgeApi.GetSliceAxes(Line, _handle);
    }

    /// <summary>
    /// Sets the Slice Mode value.
    /// 设置 Slice Mode 值。
    /// </summary>
    public void SetSliceMode(TensorRtSliceMode mode)
    {
        NativeBridgeApi.SetSliceMode(Line, _handle, mode);
    }

    /// <summary>
    /// Gets the Slice Mode value.
    /// 获取 Slice Mode 值。
    /// </summary>
    public TensorRtSliceMode GetSliceMode()
    {
        return NativeBridgeApi.GetSliceMode(Line, _handle);
    }

    /// <summary>
    /// Sets the Fill Dimensions value.
    /// 设置 Fill Dimensions 值。
    /// </summary>
    public void SetFillDimensions(TensorRtDims dimensions)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeBridgeApi.SetFillDimensions(Line, _handle, dimensions);
    }

    /// <summary>
    /// Gets the Fill Dimensions value.
    /// 获取 Fill Dimensions 值。
    /// </summary>
    public TensorRtDims GetFillDimensions()
    {
        return NativeBridgeApi.GetFillDimensions(Line, _handle);
    }

    /// <summary>
    /// Sets the Fill Operation value.
    /// 设置 Fill Operation 值。
    /// </summary>
    public void SetFillOperation(TensorRtFillOperation operation)
    {
        NativeBridgeApi.SetFillOperation(Line, _handle, operation);
    }

    /// <summary>
    /// Gets the Fill Operation value.
    /// 获取 Fill Operation 值。
    /// </summary>
    public TensorRtFillOperation GetFillOperation()
    {
        return NativeBridgeApi.GetFillOperation(Line, _handle);
    }

    /// <summary>
    /// Sets the Fill Alpha value.
    /// 设置 Fill Alpha 值。
    /// </summary>
    public void SetFillAlpha(double alpha)
    {
        NativeBridgeApi.SetFillAlpha(Line, _handle, alpha);
    }

    /// <summary>
    /// Gets the Fill Alpha value.
    /// 获取 Fill Alpha 值。
    /// </summary>
    public double GetFillAlpha()
    {
        return NativeBridgeApi.GetFillAlpha(Line, _handle);
    }

    /// <summary>
    /// Sets the Fill Beta value.
    /// 设置 Fill Beta 值。
    /// </summary>
    public void SetFillBeta(double beta)
    {
        NativeBridgeApi.SetFillBeta(Line, _handle, beta);
    }

    /// <summary>
    /// Gets the Fill Beta value.
    /// 获取 Fill Beta 值。
    /// </summary>
    public double GetFillBeta()
    {
        return NativeBridgeApi.GetFillBeta(Line, _handle);
    }

    /// <summary>
    /// Releases the native TensorRT resources held by this object.
    /// 释放此对象持有的 native TensorRT 资源。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        _ownerLease?.Dispose();
        GC.SuppressFinalize(this);
    }

    internal SafeTensorRtObjectHandleLease CloneRequiredOwnerLease()
    {
        if (_ownerLease == null)
        {
            throw new InvalidOperationException(
                "This layer is not bound to a network owner. Retrieve the TensorRT 8 RNNv2 layer through TensorRtNetworkDefinition.GetLayer before querying borrowed state tensors.");
        }

        return _ownerLease.Clone();
    }

    private void ValidateOutputIndex(int index)
    {
        if (index < 0 || index >= OutputCount)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }
    }

    private static void ValidateDims(TensorRtDims dims, string argumentName)
    {
        if (dims == null)
        {
            throw new ArgumentNullException(argumentName);
        }
    }
}
