using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets whether the specified TensorRT layer input slot contains a tensor.
    /// 获取指定 TensorRT layer 输入槽位是否包含 tensor。
    /// </summary>
    /// <param name="index">The zero-based input slot index. 从零开始的输入槽位索引。</param>
    /// <returns><c>true</c> when TensorRT reports a tensor at that slot. 当 TensorRT 在该槽位报告 tensor 时返回 <c>true</c>。</returns>
    public bool HasInputTensor(int index)
    {
        ValidateInputIndex(index);
        return NativeBridgeApi.HasLayerInputTensor(Line, _handle, index);
    }

    /// <summary>
    /// Gets whether the specified TensorRT layer output slot contains a tensor.
    /// 获取指定 TensorRT layer 输出槽位是否包含 tensor。
    /// </summary>
    /// <param name="index">The zero-based output slot index. 从零开始的输出槽位索引。</param>
    /// <returns><c>true</c> when TensorRT reports a tensor at that slot. 当 TensorRT 在该槽位报告 tensor 时返回 <c>true</c>。</returns>
    public bool HasOutputTensor(int index)
    {
        ValidateOutputIndex(index);
        return NativeBridgeApi.HasLayerOutputTensor(Line, _handle, index);
    }

    /// <summary>
    /// Gets a deployment diagnostic snapshot for one TensorRT layer input tensor slot.
    /// 获取 TensorRT layer 指定输入 tensor 槽位的部署诊断快照。
    /// </summary>
    /// <param name="index">The zero-based input slot index. 从零开始的输入槽位索引。</param>
    /// <returns>A managed metadata snapshot. 托管元数据快照。</returns>
    /// <remarks>
    /// The returned object is a snapshot and does not expose native <c>ITensor*</c> pointers.
    /// 返回对象是快照，不暴露原生 <c>ITensor*</c> 指针。
    /// </remarks>
    public TensorRtLayerTensorMetadata GetInputTensorMetadata(int index)
    {
        ValidateInputIndex(index);
        return GetLayerTensorMetadata(index, isOutputSlot: false);
    }

    /// <summary>
    /// Gets a deployment diagnostic snapshot for one TensorRT layer output tensor slot.
    /// 获取 TensorRT layer 指定输出 tensor 槽位的部署诊断快照。
    /// </summary>
    /// <param name="index">The zero-based output slot index. 从零开始的输出槽位索引。</param>
    /// <returns>A managed metadata snapshot. 托管元数据快照。</returns>
    /// <remarks>
    /// Use this method when validating direct network-construction flows and checking whether layer outputs are network outputs.
    /// 在验证直接构图流程以及检查 layer 输出是否为 network output 时使用此方法。
    /// </remarks>
    public TensorRtLayerTensorMetadata GetOutputTensorMetadata(int index)
    {
        ValidateOutputIndex(index);
        return GetLayerTensorMetadata(index, isOutputSlot: true);
    }

    /// <summary>
    /// Gets the compact native diagnostic summary for one input tensor slot.
    /// 获取指定输入 tensor 槽位的原生紧凑诊断摘要。
    /// </summary>
    /// <param name="index">The zero-based input slot index. 从零开始的输入槽位索引。</param>
    /// <returns>A compact native summary string. 原生紧凑摘要字符串。</returns>
    public string GetInputTensorSummary(int index)
    {
        ValidateInputIndex(index);
        return NativeBridgeApi.GetLayerInputTensorSummary(Line, _handle, index);
    }

    /// <summary>
    /// Gets the compact native diagnostic summary for one output tensor slot.
    /// 获取指定输出 tensor 槽位的原生紧凑诊断摘要。
    /// </summary>
    /// <param name="index">The zero-based output slot index. 从零开始的输出槽位索引。</param>
    /// <returns>A compact native summary string. 原生紧凑摘要字符串。</returns>
    public string GetOutputTensorSummary(int index)
    {
        ValidateOutputIndex(index);
        return NativeBridgeApi.GetLayerOutputTensorSummary(Line, _handle, index);
    }

    private TensorRtLayerTensorMetadata GetLayerTensorMetadata(int index, bool isOutputSlot)
    {
        bool hasTensor = isOutputSlot
            ? NativeBridgeApi.HasLayerOutputTensor(Line, _handle, index)
            : NativeBridgeApi.HasLayerInputTensor(Line, _handle, index);

        if (!hasTensor)
        {
            return new TensorRtLayerTensorMetadata(
                isOutputSlot,
                index,
                hasTensor: false,
                name: string.Empty,
                dataType: null,
                shape: null,
                location: null,
                allowedFormats: null,
                isShapeTensor: false,
                isExecutionTensor: false,
                isNetworkInput: false,
                isNetworkOutput: false,
                hasDynamicDimension: false,
                dimensionNames: Array.Empty<string>(),
                dimensionExtents: Array.Empty<int>(),
                summary: isOutputSlot ? $"output[{index}]=null" : $"input[{index}]=null",
                shape64: null,
                dimensionExtents64: Array.Empty<long>());
        }

        string name = isOutputSlot
            ? NativeBridgeApi.GetLayerOutputTensorName(Line, _handle, index)
            : NativeBridgeApi.GetLayerInputTensorName(Line, _handle, index);
        TensorRtDataType dataType = isOutputSlot
            ? NativeBridgeApi.GetLayerOutputTensorDataType(Line, _handle, index)
            : NativeBridgeApi.GetLayerInputTensorDataType(Line, _handle, index);
        TensorRtDims? shape = TryGetLayerTensorShape(index, isOutputSlot);
        TensorRtDims64? shape64 = TryGetLayerTensorShape64(index, isOutputSlot);
        int rawDimensionCount = isOutputSlot
            ? NativeBridgeApi.GetLayerOutputTensorDimensionCount(Line, _handle, index)
            : NativeBridgeApi.GetLayerInputTensorDimensionCount(Line, _handle, index);
        int dimensionCount = Math.Max(rawDimensionCount, 0);
        TensorRtTensorLocation location = isOutputSlot
            ? NativeBridgeApi.GetLayerOutputTensorLocation(Line, _handle, index)
            : NativeBridgeApi.GetLayerInputTensorLocation(Line, _handle, index);
        TensorRtTensorFormats formats = isOutputSlot
            ? NativeBridgeApi.GetLayerOutputTensorAllowedFormats(Line, _handle, index)
            : NativeBridgeApi.GetLayerInputTensorAllowedFormats(Line, _handle, index);
        bool isShapeTensor = isOutputSlot
            ? NativeBridgeApi.IsLayerOutputShapeTensor(Line, _handle, index)
            : NativeBridgeApi.IsLayerInputShapeTensor(Line, _handle, index);
        bool isExecutionTensor = isOutputSlot
            ? NativeBridgeApi.IsLayerOutputExecutionTensor(Line, _handle, index)
            : NativeBridgeApi.IsLayerInputExecutionTensor(Line, _handle, index);
        bool isNetworkInput = isOutputSlot
            ? NativeBridgeApi.IsLayerOutputNetworkInput(Line, _handle, index)
            : NativeBridgeApi.IsLayerInputNetworkInput(Line, _handle, index);
        bool isNetworkOutput = isOutputSlot
            ? NativeBridgeApi.IsLayerOutputNetworkOutput(Line, _handle, index)
            : NativeBridgeApi.IsLayerInputNetworkOutput(Line, _handle, index);
        bool hasDynamicDimension = isOutputSlot
            ? NativeBridgeApi.LayerOutputTensorHasDynamicDimension(Line, _handle, index)
            : NativeBridgeApi.LayerInputTensorHasDynamicDimension(Line, _handle, index);
        string summary = isOutputSlot
            ? NativeBridgeApi.GetLayerOutputTensorSummary(Line, _handle, index)
            : NativeBridgeApi.GetLayerInputTensorSummary(Line, _handle, index);
        if (shape is null)
        {
            summary = $"{summary} shape=unknown-rank";
        }

        string[] dimensionNames = new string[Math.Max(dimensionCount, 0)];
        int[] dimensionExtents = new int[Math.Max(dimensionCount, 0)];
        long[] dimensionExtents64 = new long[Math.Max(dimensionCount, 0)];
        for (int dimensionIndex = 0; dimensionIndex < dimensionCount; ++dimensionIndex)
        {
            dimensionExtents[dimensionIndex] = isOutputSlot
                ? NativeBridgeApi.GetLayerOutputTensorDimensionExtent(Line, _handle, index, dimensionIndex)
                : NativeBridgeApi.GetLayerInputTensorDimensionExtent(Line, _handle, index, dimensionIndex);
            dimensionExtents64[dimensionIndex] = isOutputSlot
                ? NativeBridgeApi.GetLayerOutputTensorDimensionExtent64(Line, _handle, index, dimensionIndex)
                : NativeBridgeApi.GetLayerInputTensorDimensionExtent64(Line, _handle, index, dimensionIndex);

            bool hasDimensionName = isOutputSlot
                ? NativeBridgeApi.LayerOutputTensorDimensionHasName(Line, _handle, index, dimensionIndex)
                : NativeBridgeApi.LayerInputTensorDimensionHasName(Line, _handle, index, dimensionIndex);
            dimensionNames[dimensionIndex] = hasDimensionName
                ? (isOutputSlot
                    ? NativeBridgeApi.GetLayerOutputTensorDimensionName(Line, _handle, index, dimensionIndex)
                    : NativeBridgeApi.GetLayerInputTensorDimensionName(Line, _handle, index, dimensionIndex))
                : string.Empty;
        }

        return new TensorRtLayerTensorMetadata(
            isOutputSlot,
            index,
            hasTensor,
            name,
            dataType,
            shape,
            location,
            formats,
            isShapeTensor,
            isExecutionTensor,
            isNetworkInput,
            isNetworkOutput,
            hasDynamicDimension,
            dimensionNames,
            dimensionExtents,
            summary,
            shape64,
            dimensionExtents64);
    }

    private TensorRtDims? TryGetLayerTensorShape(int index, bool isOutputSlot)
    {
        try
        {
            return isOutputSlot
                ? NativeBridgeApi.GetLayerOutputTensorShape(Line, _handle, index)
                : NativeBridgeApi.GetLayerInputTensorShape(Line, _handle, index);
        }
        catch (ArgumentOutOfRangeException)
        {
            return null;
        }
    }

    private TensorRtDims64? TryGetLayerTensorShape64(int index, bool isOutputSlot)
    {
        try
        {
            return isOutputSlot
                ? NativeBridgeApi.GetLayerOutputTensorShape64(Line, _handle, index)
                : NativeBridgeApi.GetLayerInputTensorShape64(Line, _handle, index);
        }
        catch (ArgumentOutOfRangeException)
        {
            return null;
        }
    }

    private void ValidateInputIndex(int index)
    {
        if (index < 0 || index >= InputCount)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }
    }
}
