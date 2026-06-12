using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLayer : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtLayer(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    public TensorRtApiLine Line { get; }

    public string Name
    {
        get => NativeBridgeApi.GetLayerName(Line, _handle);
        set => NativeBridgeApi.SetLayerName(Line, _handle, value);
    }

    public TensorRtLayerType Type => NativeBridgeApi.GetLayerType(Line, _handle);

    public int InputCount => NativeBridgeApi.GetLayerInputCount(Line, _handle);

    public int OutputCount => NativeBridgeApi.GetLayerOutputCount(Line, _handle);

    public TensorRtDataType Precision
    {
        get => NativeBridgeApi.GetLayerPrecision(Line, _handle);
        set => NativeBridgeApi.SetLayerPrecision(Line, _handle, value);
    }

    public bool IsPrecisionSet => NativeBridgeApi.IsLayerPrecisionSet(Line, _handle);

    public void ResetPrecision()
    {
        NativeBridgeApi.ResetLayerPrecision(Line, _handle);
    }

    public TensorRtTensor GetInput(int index)
    {
        return new TensorRtTensor(Line, NativeBridgeApi.GetLayerInput(Line, _handle, index));
    }

    public TensorRtTensor GetOutput(int index)
    {
        return new TensorRtTensor(Line, NativeBridgeApi.GetLayerOutput(Line, _handle, index));
    }

    public void SetOutputType(int index, TensorRtDataType dataType)
    {
        ValidateOutputIndex(index);
        NativeBridgeApi.SetLayerOutputType(Line, _handle, index, dataType);
    }

    public TensorRtDataType GetOutputType(int index)
    {
        ValidateOutputIndex(index);
        return NativeBridgeApi.GetLayerOutputType(Line, _handle, index);
    }

    public bool IsOutputTypeSet(int index)
    {
        ValidateOutputIndex(index);
        return NativeBridgeApi.IsLayerOutputTypeSet(Line, _handle, index);
    }

    public void ResetOutputType(int index)
    {
        ValidateOutputIndex(index);
        NativeBridgeApi.ResetLayerOutputType(Line, _handle, index);
    }

    public void SetShuffleReshapeDimensions(TensorRtDims dims)
    {
        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        NativeBridgeApi.SetShuffleReshapeDimensions(Line, _handle, dims);
    }

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

    public void SetShuffleZeroIsPlaceholder(bool zeroIsPlaceholder)
    {
        NativeBridgeApi.SetShuffleZeroIsPlaceholder(Line, _handle, zeroIsPlaceholder);
    }

    public bool GetShuffleZeroIsPlaceholder()
    {
        return NativeBridgeApi.GetShuffleZeroIsPlaceholder(Line, _handle);
    }

    public void SetMatrixMultiplyOperation(int inputIndex, TensorRtMatrixOperation operation)
    {
        if (inputIndex < 0 || inputIndex > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(inputIndex), "MatrixMultiply input index must be 0 or 1.");
        }

        NativeBridgeApi.SetMatrixMultiplyOperation(Line, _handle, inputIndex, operation);
    }

    public TensorRtMatrixOperation GetMatrixMultiplyOperation(int inputIndex)
    {
        if (inputIndex < 0 || inputIndex > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(inputIndex), "MatrixMultiply input index must be 0 or 1.");
        }

        return NativeBridgeApi.GetMatrixMultiplyOperation(Line, _handle, inputIndex);
    }

    public TensorRtReduceOperation GetReduceOperation()
    {
        return NativeBridgeApi.GetReduceOperation(Line, _handle);
    }

    public void SetReduceOperation(TensorRtReduceOperation operation)
    {
        NativeBridgeApi.SetReduceOperation(Line, _handle, operation);
    }

    public uint GetReduceAxes()
    {
        return NativeBridgeApi.GetReduceAxes(Line, _handle);
    }

    public void SetReduceAxes(uint axes)
    {
        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "Reduce axes bitmask must not be zero.");
        }

        NativeBridgeApi.SetReduceAxes(Line, _handle, axes);
    }

    public bool GetReduceKeepDimensions()
    {
        return NativeBridgeApi.GetReduceKeepDimensions(Line, _handle);
    }

    public void SetReduceKeepDimensions(bool keepDimensions)
    {
        NativeBridgeApi.SetReduceKeepDimensions(Line, _handle, keepDimensions);
    }

    public void SetSoftMaxAxes(uint axes)
    {
        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "SoftMax axes bitmask must not be zero.");
        }

        NativeBridgeApi.SetSoftMaxAxes(Line, _handle, axes);
    }

    public uint GetSoftMaxAxes()
    {
        return NativeBridgeApi.GetSoftMaxAxes(Line, _handle);
    }

    public TensorRtUnaryOperation GetUnaryOperation()
    {
        return NativeBridgeApi.GetUnaryOperation(Line, _handle);
    }

    public void SetUnaryOperation(TensorRtUnaryOperation operation)
    {
        NativeBridgeApi.SetUnaryOperation(Line, _handle, operation);
    }

    public TensorRtTopKOperation GetTopKOperation()
    {
        return NativeBridgeApi.GetTopKOperation(Line, _handle);
    }

    public void SetTopKOperation(TensorRtTopKOperation operation)
    {
        NativeBridgeApi.SetTopKOperation(Line, _handle, operation);
    }

    public int GetTopKValue()
    {
        return NativeBridgeApi.GetTopKValue(Line, _handle);
    }

    public void SetTopKValue(int k)
    {
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k));
        }

        NativeBridgeApi.SetTopKValue(Line, _handle, k);
    }

    public uint GetTopKAxes()
    {
        return NativeBridgeApi.GetTopKAxes(Line, _handle);
    }

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
    /// <returns>The configured TopK indices data type.</returns>
    public TensorRtDataType GetTopKIndicesType()
    {
        return NativeBridgeApi.GetTopKIndicesType(Line, _handle);
    }

    /// <summary>
    /// Sets the index output data type on a TensorRT 11 TopK layer.
    /// 设置 TensorRT 11 TopK 层索引输出的数据类型。
    /// </summary>
    /// <param name="dataType">The requested indices data type.</param>
    /// <returns><c>true</c> when TensorRT accepts the value; otherwise <c>false</c>.</returns>
    public bool SetTopKIndicesType(TensorRtDataType dataType)
    {
        return NativeBridgeApi.SetTopKIndicesType(Line, _handle, dataType);
    }

    public int GetGatherAxis()
    {
        return NativeBridgeApi.GetGatherAxis(Line, _handle);
    }

    public void SetGatherAxis(int axis)
    {
        NativeBridgeApi.SetGatherAxis(Line, _handle, axis);
    }

    public TensorRtElementWiseOperation GetElementWiseOperation()
    {
        return NativeBridgeApi.GetElementWiseOperation(Line, _handle);
    }

    public void SetElementWiseOperation(TensorRtElementWiseOperation operation)
    {
        NativeBridgeApi.SetElementWiseOperation(Line, _handle, operation);
    }

    public TensorRtActivationType GetActivationType()
    {
        return NativeBridgeApi.GetActivationType(Line, _handle);
    }

    public void SetActivationType(TensorRtActivationType activationType)
    {
        NativeBridgeApi.SetActivationType(Line, _handle, activationType);
    }

    public double GetActivationAlpha()
    {
        return NativeBridgeApi.GetActivationAlpha(Line, _handle);
    }

    public void SetActivationAlpha(double alpha)
    {
        NativeBridgeApi.SetActivationAlpha(Line, _handle, alpha);
    }

    public double GetActivationBeta()
    {
        return NativeBridgeApi.GetActivationBeta(Line, _handle);
    }

    public void SetActivationBeta(double beta)
    {
        NativeBridgeApi.SetActivationBeta(Line, _handle, beta);
    }

    public TensorRtPoolingType GetPoolingType()
    {
        return NativeBridgeApi.GetPoolingType(Line, _handle);
    }

    public void SetPoolingType(TensorRtPoolingType poolingType)
    {
        NativeBridgeApi.SetPoolingType(Line, _handle, poolingType);
    }

    public void SetPoolingWindowSize(TensorRtDims windowSize)
    {
        if (windowSize == null)
        {
            throw new ArgumentNullException(nameof(windowSize));
        }

        NativeBridgeApi.SetPoolingWindowSize(Line, _handle, windowSize);
    }

    public TensorRtDims GetPoolingWindowSize()
    {
        return NativeBridgeApi.GetPoolingWindowSize(Line, _handle);
    }

    public void SetPoolingStride(TensorRtDims stride)
    {
        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        NativeBridgeApi.SetPoolingStride(Line, _handle, stride);
    }

    public TensorRtDims GetPoolingStride()
    {
        return NativeBridgeApi.GetPoolingStride(Line, _handle);
    }

    public void SetPoolingPadding(TensorRtDims padding)
    {
        if (padding == null)
        {
            throw new ArgumentNullException(nameof(padding));
        }

        NativeBridgeApi.SetPoolingPadding(Line, _handle, padding);
    }

    public TensorRtDims GetPoolingPadding()
    {
        return NativeBridgeApi.GetPoolingPadding(Line, _handle);
    }

    public double GetPoolingBlendFactor()
    {
        return NativeBridgeApi.GetPoolingBlendFactor(Line, _handle);
    }

    public void SetPoolingBlendFactor(double blendFactor)
    {
        NativeBridgeApi.SetPoolingBlendFactor(Line, _handle, blendFactor);
    }

    public bool GetPoolingAverageCountExcludesPadding()
    {
        return NativeBridgeApi.GetPoolingAverageCountExcludesPadding(Line, _handle);
    }

    public void SetPoolingAverageCountExcludesPadding(bool excludesPadding)
    {
        NativeBridgeApi.SetPoolingAverageCountExcludesPadding(Line, _handle, excludesPadding);
    }

    public TensorRtDims GetPoolingPrePadding()
    {
        return NativeBridgeApi.GetPoolingPrePadding(Line, _handle);
    }

    public void SetPoolingPrePadding(TensorRtDims padding)
    {
        if (padding == null)
        {
            throw new ArgumentNullException(nameof(padding));
        }

        NativeBridgeApi.SetPoolingPrePadding(Line, _handle, padding);
    }

    public TensorRtDims GetPoolingPostPadding()
    {
        return NativeBridgeApi.GetPoolingPostPadding(Line, _handle);
    }

    public void SetPoolingPostPadding(TensorRtDims padding)
    {
        if (padding == null)
        {
            throw new ArgumentNullException(nameof(padding));
        }

        NativeBridgeApi.SetPoolingPostPadding(Line, _handle, padding);
    }

    public TensorRtPaddingMode GetPoolingPaddingMode()
    {
        return NativeBridgeApi.GetPoolingPaddingMode(Line, _handle);
    }

    public void SetPoolingPaddingMode(TensorRtPaddingMode paddingMode)
    {
        NativeBridgeApi.SetPoolingPaddingMode(Line, _handle, paddingMode);
    }

    public int GetConvolutionOutputMaps()
    {
        return NativeBridgeApi.GetConvolutionOutputMaps(Line, _handle);
    }

    public void SetConvolutionOutputMaps(int outputMaps)
    {
        NativeBridgeApi.SetConvolutionOutputMaps(Line, _handle, outputMaps);
    }

    public int GetConvolutionGroups()
    {
        return NativeBridgeApi.GetConvolutionGroups(Line, _handle);
    }

    public void SetConvolutionGroups(int groups)
    {
        NativeBridgeApi.SetConvolutionGroups(Line, _handle, groups);
    }

    public TensorRtDims GetConvolutionStride()
    {
        return NativeBridgeApi.GetConvolutionStride(Line, _handle);
    }

    public void SetConvolutionStride(TensorRtDims stride)
    {
        ValidateDims(stride, nameof(stride));
        NativeBridgeApi.SetConvolutionStride(Line, _handle, stride);
    }

    /// <summary>
    /// Gets the explicit N-D padding of a convolution layer.
    /// 获取 convolution 层的显式 N-D padding。
    /// </summary>
    /// <returns>The current convolution padding dimensions.</returns>
    public TensorRtDims GetConvolutionPadding()
    {
        return NativeBridgeApi.GetConvolutionPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the explicit N-D padding of a convolution layer.
    /// 设置 convolution 层的显式 N-D padding。
    /// </summary>
    /// <param name="padding">The padding dimensions to apply.</param>
    public void SetConvolutionPadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetConvolutionPadding(Line, _handle, padding);
    }

    public TensorRtDims GetConvolutionPrePadding()
    {
        return NativeBridgeApi.GetConvolutionPrePadding(Line, _handle);
    }

    public void SetConvolutionPrePadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetConvolutionPrePadding(Line, _handle, padding);
    }

    public TensorRtDims GetConvolutionPostPadding()
    {
        return NativeBridgeApi.GetConvolutionPostPadding(Line, _handle);
    }

    public void SetConvolutionPostPadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetConvolutionPostPadding(Line, _handle, padding);
    }

    public TensorRtDims GetConvolutionDilation()
    {
        return NativeBridgeApi.GetConvolutionDilation(Line, _handle);
    }

    public void SetConvolutionDilation(TensorRtDims dilation)
    {
        ValidateDims(dilation, nameof(dilation));
        NativeBridgeApi.SetConvolutionDilation(Line, _handle, dilation);
    }

    public TensorRtPaddingMode GetConvolutionPaddingMode()
    {
        return NativeBridgeApi.GetConvolutionPaddingMode(Line, _handle);
    }

    public void SetConvolutionPaddingMode(TensorRtPaddingMode paddingMode)
    {
        NativeBridgeApi.SetConvolutionPaddingMode(Line, _handle, paddingMode);
    }

    public TensorRtScaleMode GetScaleMode()
    {
        return NativeBridgeApi.GetScaleMode(Line, _handle);
    }

    public void SetScaleMode(TensorRtScaleMode mode)
    {
        NativeBridgeApi.SetScaleMode(Line, _handle, mode);
    }

    public int GetScaleChannelAxis()
    {
        return NativeBridgeApi.GetScaleChannelAxis(Line, _handle);
    }

    public void SetScaleChannelAxis(int channelAxis)
    {
        NativeBridgeApi.SetScaleChannelAxis(Line, _handle, channelAxis);
    }

    public TensorRtDims GetPaddingPrePadding()
    {
        return NativeBridgeApi.GetPaddingPrePadding(Line, _handle);
    }

    public void SetPaddingPrePadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetPaddingPrePadding(Line, _handle, padding);
    }

    public TensorRtDims GetPaddingPostPadding()
    {
        return NativeBridgeApi.GetPaddingPostPadding(Line, _handle);
    }

    public void SetPaddingPostPadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetPaddingPostPadding(Line, _handle, padding);
    }

    public void SetResizeOutputDimensions(TensorRtDims dimensions)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeBridgeApi.SetResizeOutputDimensions(Line, _handle, dimensions);
    }

    public TensorRtDims GetResizeOutputDimensions()
    {
        return NativeBridgeApi.GetResizeOutputDimensions(Line, _handle);
    }

    public void SetResizeMode(TensorRtResizeMode resizeMode)
    {
        NativeBridgeApi.SetResizeMode(Line, _handle, resizeMode);
    }

    public TensorRtResizeMode GetResizeMode()
    {
        return NativeBridgeApi.GetResizeMode(Line, _handle);
    }

    /// <summary>
    /// Gets the TensorRT 8 resize align-corners flag.
    /// 获取 TensorRT 8 resize 层的 align-corners 标志。
    /// </summary>
    /// <returns><c>true</c> when corner alignment is enabled.</returns>
    public bool GetResizeAlignCorners()
    {
        return NativeBridgeApi.GetResizeAlignCorners(Line, _handle);
    }

    /// <summary>
    /// Sets the TensorRT 8 resize align-corners flag.
    /// 设置 TensorRT 8 resize 层的 align-corners 标志。
    /// </summary>
    /// <param name="alignCorners">Whether resize should align the corner pixels.</param>
    public void SetResizeAlignCorners(bool alignCorners)
    {
        NativeBridgeApi.SetResizeAlignCorners(Line, _handle, alignCorners);
    }

    public TensorRtResizeCoordinateTransformation GetResizeCoordinateTransformation()
    {
        return NativeBridgeApi.GetResizeCoordinateTransformation(Line, _handle);
    }

    public void SetResizeCoordinateTransformation(TensorRtResizeCoordinateTransformation transformation)
    {
        NativeBridgeApi.SetResizeCoordinateTransformation(Line, _handle, transformation);
    }

    public TensorRtResizeSelector GetResizeSelectorForSinglePixel()
    {
        return NativeBridgeApi.GetResizeSelectorForSinglePixel(Line, _handle);
    }

    public void SetResizeSelectorForSinglePixel(TensorRtResizeSelector selector)
    {
        NativeBridgeApi.SetResizeSelectorForSinglePixel(Line, _handle, selector);
    }

    public TensorRtResizeRoundMode GetResizeNearestRounding()
    {
        return NativeBridgeApi.GetResizeNearestRounding(Line, _handle);
    }

    public void SetResizeNearestRounding(TensorRtResizeRoundMode rounding)
    {
        NativeBridgeApi.SetResizeNearestRounding(Line, _handle, rounding);
    }

    public double GetResizeCubicCoefficient()
    {
        return NativeBridgeApi.GetResizeCubicCoefficient(Line, _handle);
    }

    public void SetResizeCubicCoefficient(double value)
    {
        NativeBridgeApi.SetResizeCubicCoefficient(Line, _handle, value);
    }

    public bool GetResizeExcludeOutside()
    {
        return NativeBridgeApi.GetResizeExcludeOutside(Line, _handle);
    }

    public void SetResizeExcludeOutside(bool value)
    {
        NativeBridgeApi.SetResizeExcludeOutside(Line, _handle, value);
    }

    public void SetResizeScales(float[] scales)
    {
        NativeBridgeApi.SetResizeScales(Line, _handle, scales);
    }

    public float[] GetResizeScales()
    {
        return NativeBridgeApi.GetResizeScales(Line, _handle);
    }

    public void SetConcatenationAxis(int axis)
    {
        NativeBridgeApi.SetConcatenationAxis(Line, _handle, axis);
    }

    public int GetConcatenationAxis()
    {
        return NativeBridgeApi.GetConcatenationAxis(Line, _handle);
    }

    public void SetSliceStart(TensorRtDims start)
    {
        if (start == null)
        {
            throw new ArgumentNullException(nameof(start));
        }

        NativeBridgeApi.SetSliceStart(Line, _handle, start);
    }

    public TensorRtDims GetSliceStart()
    {
        return NativeBridgeApi.GetSliceStart(Line, _handle);
    }

    public void SetSliceSize(TensorRtDims size)
    {
        if (size == null)
        {
            throw new ArgumentNullException(nameof(size));
        }

        NativeBridgeApi.SetSliceSize(Line, _handle, size);
    }

    public TensorRtDims GetSliceSize()
    {
        return NativeBridgeApi.GetSliceSize(Line, _handle);
    }

    public void SetSliceStride(TensorRtDims stride)
    {
        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        NativeBridgeApi.SetSliceStride(Line, _handle, stride);
    }

    public TensorRtDims GetSliceStride()
    {
        return NativeBridgeApi.GetSliceStride(Line, _handle);
    }

    /// <summary>
    /// Sets the axes vector used by a TensorRT 10 or TensorRT 11 slice layer.
    /// 设置 TensorRT 10 或 TensorRT 11 slice 层使用的 axes 向量。
    /// </summary>
    /// <param name="axes">The axes dimensions to apply.</param>
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
    /// <returns>The current slice axes dimensions.</returns>
    public TensorRtDims GetSliceAxes()
    {
        return NativeBridgeApi.GetSliceAxes(Line, _handle);
    }

    public void SetSliceMode(TensorRtSliceMode mode)
    {
        NativeBridgeApi.SetSliceMode(Line, _handle, mode);
    }

    public TensorRtSliceMode GetSliceMode()
    {
        return NativeBridgeApi.GetSliceMode(Line, _handle);
    }

    public void SetFillDimensions(TensorRtDims dimensions)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeBridgeApi.SetFillDimensions(Line, _handle, dimensions);
    }

    public TensorRtDims GetFillDimensions()
    {
        return NativeBridgeApi.GetFillDimensions(Line, _handle);
    }

    public void SetFillOperation(TensorRtFillOperation operation)
    {
        NativeBridgeApi.SetFillOperation(Line, _handle, operation);
    }

    public TensorRtFillOperation GetFillOperation()
    {
        return NativeBridgeApi.GetFillOperation(Line, _handle);
    }

    public void SetFillAlpha(double alpha)
    {
        NativeBridgeApi.SetFillAlpha(Line, _handle, alpha);
    }

    public double GetFillAlpha()
    {
        return NativeBridgeApi.GetFillAlpha(Line, _handle);
    }

    public void SetFillBeta(double beta)
    {
        NativeBridgeApi.SetFillBeta(Line, _handle, beta);
    }

    public double GetFillBeta()
    {
        return NativeBridgeApi.GetFillBeta(Line, _handle);
    }

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
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
