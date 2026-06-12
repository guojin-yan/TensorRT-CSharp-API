using System;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>
    /// Adds a TensorRT 11 cast layer that converts a tensor to another data type.
    /// 添加 TensorRT 11 Cast 层，用于把张量转换为另一个数据类型。
    /// </summary>
    /// <param name="input">Input tensor to cast. / 需要转换的输入张量。</param>
    /// <param name="toType">Destination TensorRT data type. / 目标 TensorRT 数据类型。</param>
    /// <returns>The created network-owned layer. / 返回由网络持有生命周期的层对象。</returns>
    public TensorRtLayer AddCast(TensorRtTensor input, TensorRtDataType toType)
    {
        ValidateInputTensor(input, nameof(input));
        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddCastLayer(Line, _handle, input.Handle, toType));
    }

    /// <summary>
    /// Adds a TensorRT 10 or TensorRT 11 NonZero layer.
    /// 添加 TensorRT 10 或 TensorRT 11 NonZero 层。
    /// </summary>
    /// <param name="input">Input tensor to scan for non-zero elements. / 用于查找非零元素的输入张量。</param>
    /// <param name="indicesType">Indices output type. TensorRT 10 only supports Int32 here. / 索引输出类型；TensorRT 10 仅支持 Int32。</param>
    /// <returns>The created network-owned layer. / 返回由网络持有生命周期的层对象。</returns>
    public TensorRtLayer AddNonZero(TensorRtTensor input, TensorRtDataType indicesType = TensorRtDataType.Int32)
    {
        ValidateInputTensor(input, nameof(input));
        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddNonZeroLayer(Line, _handle, input.Handle, indicesType));
    }

    /// <summary>
    /// Adds a TensorRT ragged softmax layer for variable-length sequences.
    /// 添加 TensorRT RaggedSoftMax 层，用于可变长度序列的 softmax。
    /// </summary>
    /// <param name="input">Input tensor. / 输入张量。</param>
    /// <param name="bounds">Bounds tensor that describes valid sequence lengths. / 描述有效序列长度的边界张量。</param>
    /// <returns>The created network-owned layer. / 返回由网络持有生命周期的层对象。</returns>
    public TensorRtLayer AddRaggedSoftMax(TensorRtTensor input, TensorRtTensor bounds)
    {
        ValidateInputTensor(input, nameof(input));
        ValidateInputTensor(bounds, nameof(bounds));
        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddRaggedSoftMaxLayer(Line, _handle, input.Handle, bounds.Handle));
    }

    /// <summary>
    /// Adds a TensorRT NMS layer for deployment-side post-processing.
    /// 添加 TensorRT NMS 层，用于部署端目标检测后处理。
    /// </summary>
    /// <param name="boxes">Boxes tensor. / 边界框张量。</param>
    /// <param name="scores">Scores tensor. / 置信度分数张量。</param>
    /// <param name="maxOutputBoxesPerClass">Scalar tensor for max boxes per class. / 每个类别最大输出框数量的标量张量。</param>
    /// <param name="indicesType">Indices output type, usually Int32 or Int64. / 索引输出类型，通常为 Int32 或 Int64。</param>
    /// <returns>The created network-owned layer. / 返回由网络持有生命周期的层对象。</returns>
    public TensorRtLayer AddNms(TensorRtTensor boxes, TensorRtTensor scores, TensorRtTensor maxOutputBoxesPerClass, TensorRtDataType indicesType = TensorRtDataType.Int32)
    {
        ValidateInputTensor(boxes, nameof(boxes));
        ValidateInputTensor(scores, nameof(scores));
        ValidateInputTensor(maxOutputBoxesPerClass, nameof(maxOutputBoxesPerClass));
        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddNmsLayer(Line, _handle, boxes.Handle, scores.Handle, maxOutputBoxesPerClass.Handle, indicesType));
    }

    /// <summary>
    /// Adds a TensorRT ReverseSequence layer on TensorRT 8, 10, or 11 adapters.
    /// 在 TensorRT 8、10 或 11 适配线上添加 ReverseSequence 层。
    /// </summary>
    /// <param name="input">Input tensor with rank greater than or equal to 2. / rank 大于等于 2 的输入张量。</param>
    /// <param name="sequenceLengths">1D tensor containing lengths to reverse. / 包含反转长度的一维张量。</param>
    /// <returns>The created network-owned layer. / 返回由网络持有生命周期的层对象。</returns>
    public TensorRtLayer AddReverseSequence(TensorRtTensor input, TensorRtTensor sequenceLengths)
    {
        ValidateInputTensor(input, nameof(input));
        ValidateInputTensor(sequenceLengths, nameof(sequenceLengths));
        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddReverseSequenceLayer(Line, _handle, input.Handle, sequenceLengths.Handle));
    }

    /// <summary>
    /// Adds a TensorRT Einsum layer from an equation and one or more input tensors.
    /// 根据公式和一个或多个输入张量添加 TensorRT Einsum 层。
    /// </summary>
    /// <param name="equation">Einsum equation accepted by TensorRT. / TensorRT 可接受的 Einsum 公式。</param>
    /// <param name="inputs">Input tensors. / 输入张量列表。</param>
    /// <returns>The created network-owned layer. / 返回由网络持有生命周期的层对象。</returns>
    public TensorRtLayer AddEinsum(string equation, params TensorRtTensor[] inputs)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (inputs.Length == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputs), "Einsum requires at least one input tensor.");
        }

        SafeHandleArrayBuilder builder = new SafeHandleArrayBuilder(inputs.Length);
        for (int i = 0; i < inputs.Length; i++)
        {
            ValidateInputTensor(inputs[i], nameof(inputs));
            builder.Add(inputs[i].Handle);
        }

        return new TensorRtLayer(Line, Internal.Interop.NativeBridgeApi.AddEinsumLayer(Line, _handle, builder.ToArray(), equation));
    }

    private sealed class SafeHandleArrayBuilder
    {
        private readonly Internal.Handles.SafeTensorRtObjectHandle[] _handles;
        private int _count;

        public SafeHandleArrayBuilder(int capacity)
        {
            _handles = new Internal.Handles.SafeTensorRtObjectHandle[capacity];
        }

        public void Add(Internal.Handles.SafeTensorRtObjectHandle handle)
        {
            _handles[_count++] = handle;
        }

        public Internal.Handles.SafeTensorRtObjectHandle[] ToArray()
        {
            return _handles;
        }
    }
}
