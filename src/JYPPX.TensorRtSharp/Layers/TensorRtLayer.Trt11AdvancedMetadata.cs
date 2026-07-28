using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets TensorRT layer metadata text.
    /// 获取 TensorRT 层的 metadata 文本。
    /// </summary>
    /// <returns>The metadata string, or an empty string when unset. / 返回 metadata 字符串；未设置时返回空字符串。</returns>
    public string GetMetadata()
    {
        return NativeBridgeApi.GetLayerMetadata(Line, _handle);
    }

    /// <summary>
    /// Sets TensorRT layer metadata text.
    /// 设置 TensorRT 层的 metadata 文本。
    /// </summary>
    /// <param name="metadata">Metadata text. Use an empty string to clear. / metadata 文本；传入空字符串可清空。</param>
    public void SetMetadata(string metadata)
    {
        NativeBridgeApi.SetLayerMetadata(Line, _handle, metadata);
    }

    /// <summary>
    /// Replaces or appends an input tensor on TensorRT 11 layers that support dynamic input slots.
    /// 在支持动态输入槽位的 TensorRT 11 层上替换或追加输入张量。
    /// </summary>
    /// <param name="index">Input index to replace, or the current input count to append. / 要替换的输入索引，或当前输入数量表示追加。</param>
    /// <param name="tensor">Tensor to assign. / 要设置的张量。</param>
    public void SetInput(int index, TensorRtTensor tensor)
    {
        ValidateLayerTensor(tensor, nameof(tensor));
        NativeBridgeApi.SetLayerInput(Line, _handle, index, tensor.Handle);
    }

    /// <summary>
    /// Gets the multi-rank execution hint count configured on the layer.
    /// 获取层上配置的多 rank 执行提示数量。
    /// </summary>
    public int GetRankCount()
    {
        return NativeBridgeApi.GetLayerRankCount(Line, _handle);
    }

    /// <summary>
    /// Sets the multi-rank execution hint count when supported by TensorRT for this layer type.
    /// 当 TensorRT 对该层类型支持时，设置多 rank 执行提示数量。
    /// </summary>
    /// <param name="rankCount">Rank count. / rank 数量。</param>
    /// <returns>True when TensorRT accepted the value. / 当 TensorRT 接受该值时返回 true。</returns>
    public bool SetRankCount(int rankCount)
    {
        return NativeBridgeApi.SetLayerRankCount(Line, _handle, rankCount);
    }

    /// <summary>
    /// Gets the output data type configured for a Cast layer.
    /// 获取 Cast 层配置的输出数据类型。
    /// </summary>
    public TensorRtDataType GetCastToType()
    {
        return NativeBridgeApi.GetCastToType(Line, _handle);
    }

    /// <summary>
    /// Sets the output data type for a Cast layer.
    /// 设置 Cast 层的输出数据类型。
    /// </summary>
    /// <param name="dataType">Destination data type. / 目标数据类型。</param>
    public void SetCastToType(TensorRtDataType dataType)
    {
        NativeBridgeApi.SetCastToType(Line, _handle, dataType);
    }

    /// <summary>
    /// Gets the indices output type for a NonZero layer.
    /// 获取 NonZero 层索引输出的数据类型。
    /// </summary>
    public TensorRtDataType GetNonZeroIndicesType()
    {
        return NativeBridgeApi.GetNonZeroIndicesType(Line, _handle);
    }

    /// <summary>
    /// Sets the indices output type for a NonZero layer.
    /// 设置 NonZero 层索引输出的数据类型。
    /// </summary>
    /// <param name="dataType">Indices type, usually Int32 or Int64. / 索引类型，通常为 Int32 或 Int64。</param>
    /// <returns>True when TensorRT accepted the value. / 当 TensorRT 接受该值时返回 true。</returns>
    public bool SetNonZeroIndicesType(TensorRtDataType dataType)
    {
        return NativeBridgeApi.SetNonZeroIndicesType(Line, _handle, dataType);
    }

    /// <summary>
    /// Gets the bounding-box format used by an NMS layer.
    /// 获取 NMS 层使用的边界框格式。
    /// </summary>
    public TensorRtBoundingBoxFormat GetNmsBoundingBoxFormat()
    {
        return NativeBridgeApi.GetNmsBoundingBoxFormat(Line, _handle);
    }

    /// <summary>
    /// Sets the bounding-box format used by an NMS layer.
    /// 设置 NMS 层使用的边界框格式。
    /// </summary>
    /// <param name="format">Bounding-box coordinate format. / 边界框坐标格式。</param>
    public void SetNmsBoundingBoxFormat(TensorRtBoundingBoxFormat format)
    {
        NativeBridgeApi.SetNmsBoundingBoxFormat(Line, _handle, format);
    }

    /// <summary>
    /// Gets the NMS TopK box limit.
    /// 获取 NMS TopK 候选框数量限制。
    /// </summary>
    public int GetNmsTopKBoxLimit()
    {
        return NativeBridgeApi.GetNmsTopKBoxLimit(Line, _handle);
    }

    /// <summary>
    /// Sets the NMS TopK box limit.
    /// 设置 NMS TopK 候选框数量限制。
    /// </summary>
    /// <param name="limit">Positive TopK limit. / 正数 TopK 限制。</param>
    public void SetNmsTopKBoxLimit(int limit)
    {
        NativeBridgeApi.SetNmsTopKBoxLimit(Line, _handle, limit);
    }

    /// <summary>
    /// Sets the optional IoU threshold tensor input for an NMS layer.
    /// 设置 NMS 层可选的 IoU 阈值张量输入。
    /// </summary>
    /// <param name="tensor">Scalar IoU threshold tensor. / IoU 阈值标量张量。</param>
    public void SetNmsIouThresholdTensor(TensorRtTensor tensor)
    {
        ValidateLayerTensor(tensor, nameof(tensor));
        NativeBridgeApi.SetNmsIouThresholdTensor(Line, _handle, tensor.Handle);
    }

    /// <summary>
    /// Sets the optional score threshold tensor input for an NMS layer.
    /// 设置 NMS 层可选的分数阈值张量输入。
    /// </summary>
    /// <param name="tensor">Scalar score threshold tensor. / 分数阈值标量张量。</param>
    public void SetNmsScoreThresholdTensor(TensorRtTensor tensor)
    {
        ValidateLayerTensor(tensor, nameof(tensor));
        NativeBridgeApi.SetNmsScoreThresholdTensor(Line, _handle, tensor.Handle);
    }

    /// <summary>
    /// Gets the indices output type for an NMS layer.
    /// 获取 NMS 层索引输出的数据类型。
    /// </summary>
    public TensorRtDataType GetNmsIndicesType()
    {
        return NativeBridgeApi.GetNmsIndicesType(Line, _handle);
    }

    /// <summary>
    /// Sets the indices output type for an NMS layer.
    /// 设置 NMS 层索引输出的数据类型。
    /// </summary>
    /// <param name="dataType">Indices type, usually Int32 or Int64. / 索引类型，通常为 Int32 或 Int64。</param>
    /// <returns>True when TensorRT accepted the value. / 当 TensorRT 接受该值时返回 true。</returns>
    public bool SetNmsIndicesType(TensorRtDataType dataType)
    {
        return NativeBridgeApi.SetNmsIndicesType(Line, _handle, dataType);
    }

    /// <summary>
    /// Gets the batch axis configured on a ReverseSequence layer.
    /// 获取 ReverseSequence 层配置的 batch 轴。
    /// </summary>
    public int GetReverseSequenceBatchAxis()
    {
        return NativeBridgeApi.GetReverseSequenceBatchAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the batch axis configured on a ReverseSequence layer.
    /// 设置 ReverseSequence 层配置的 batch 轴。
    /// </summary>
    /// <param name="axis">Batch axis. / batch 轴。</param>
    public void SetReverseSequenceBatchAxis(int axis)
    {
        NativeBridgeApi.SetReverseSequenceBatchAxis(Line, _handle, axis);
    }

    /// <summary>
    /// Gets the sequence axis configured on a ReverseSequence layer.
    /// 获取 ReverseSequence 层配置的 sequence 轴。
    /// </summary>
    public int GetReverseSequenceSequenceAxis()
    {
        return NativeBridgeApi.GetReverseSequenceSequenceAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the sequence axis configured on a ReverseSequence layer.
    /// 设置 ReverseSequence 层配置的 sequence 轴。
    /// </summary>
    /// <param name="axis">Sequence axis. / sequence 轴。</param>
    public void SetReverseSequenceSequenceAxis(int axis)
    {
        NativeBridgeApi.SetReverseSequenceSequenceAxis(Line, _handle, axis);
    }

    private void ValidateLayerTensor(TensorRtTensor tensor, string argumentName)
    {
        if (tensor == null)
        {
            throw new ArgumentNullException(argumentName);
        }

        if (tensor.Line != Line)
        {
            throw new ArgumentException("Tensor must belong to the same TensorRT API line as the layer.", argumentName);
        }
    }
}
