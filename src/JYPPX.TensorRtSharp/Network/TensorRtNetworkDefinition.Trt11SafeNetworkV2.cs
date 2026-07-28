using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>
    /// Adds a TensorRT 10 or TensorRT 11 GatherV2 layer with the requested gather mode.
    /// 添加 TensorRT 10 或 TensorRT 11 GatherV2 层，并指定 gather 模式。
    /// </summary>
    /// <param name="data">Tensor to gather from. / 被 gather 的数据张量。</param>
    /// <param name="indices">Tensor containing gather indices. / 包含 gather 索引的张量。</param>
    /// <param name="mode">Gather semantic mode. / Gather 语义模式。</param>
    /// <returns>The created network-owned layer. / 返回由网络持有生命周期的层对象。</returns>
    public TensorRtLayer AddGatherV2(TensorRtTensor data, TensorRtTensor indices, TensorRtGatherMode mode)
    {
        ValidateInputTensor(data, nameof(data));
        ValidateInputTensor(indices, nameof(indices));
        return new TensorRtLayer(Line, NativeBridgeApi.AddGatherV2Layer(Line, _handle, data.Handle, indices.Handle, mode));
    }

    /// <summary>
    /// Adds a TensorRT 10 or TensorRT 11 FillV2 layer and selects the output tensor type at creation time.
    /// 添加 TensorRT 10 或 TensorRT 11 FillV2 层，并在创建时指定输出张量类型。
    /// </summary>
    /// <param name="dimensions">Static output dimensions when input 0 is absent. / 当第 0 输入不存在时使用的静态输出维度。</param>
    /// <param name="operation">Fill operation. / Fill 操作。</param>
    /// <param name="outputType">Requested output tensor data type. / 请求的输出张量数据类型。</param>
    /// <returns>The created network-owned layer. / 返回由网络持有生命周期的层对象。</returns>
    public TensorRtLayer AddFillV2(TensorRtDims dimensions, TensorRtFillOperation operation, TensorRtDataType outputType)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddFillV2Layer(Line, _handle, dimensions, operation, outputType));
    }

    /// <summary>
    /// Adds a TensorRT parametric ReLU layer.
    /// 添加 TensorRT ParametricReLU 层。
    /// </summary>
    /// <param name="input">Input tensor. / 输入张量。</param>
    /// <param name="slope">Slope tensor broadcastable to the input tensor. / 可广播到输入张量的 slope 张量。</param>
    /// <returns>The created network-owned layer. / 返回由网络持有生命周期的层对象。</returns>
    public TensorRtLayer AddParametricReLU(TensorRtTensor input, TensorRtTensor slope)
    {
        ValidateInputTensor(input, nameof(input));
        ValidateInputTensor(slope, nameof(slope));
        return new TensorRtLayer(Line, NativeBridgeApi.AddParametricReluLayer(Line, _handle, input.Handle, slope.Handle));
    }

    /// <summary>
    /// Adds a TensorRT 11 DistCollective layer. This layer may require a TensorRT multi-device capability set at build or runtime.
    /// 添加 TensorRT 11 DistCollective 层；该层在构建或运行时可能需要 TensorRT 多设备能力集。
    /// </summary>
    /// <param name="input">Input tensor. / 输入张量。</param>
    /// <param name="collectiveOperation">Collective operation to perform. / 要执行的集合通信操作。</param>
    /// <param name="reduceOperation">Reduction operation, or None when unused. / 归约操作；不使用归约时传 None。</param>
    /// <param name="root">Root rank for root-based collectives, or -1 when unused. / root 型集合通信的根 rank，不使用时传 -1。</param>
    /// <param name="groups">Optional ordered participating rank IDs. / 可选的参与 rank ID 有序列表。</param>
    /// <returns>The created network-owned layer. / 返回由网络持有生命周期的层对象。</returns>
    public TensorRtLayer AddDistCollective(
        TensorRtTensor input,
        TensorRtCollectiveOperation collectiveOperation,
        TensorRtDistributedReduceOperation reduceOperation = TensorRtDistributedReduceOperation.None,
        long root = -1,
        IReadOnlyList<long>? groups = null)
    {
        ValidateInputTensor(input, nameof(input));
        return new TensorRtLayer(
            Line,
            NativeBridgeApi.AddDistCollectiveLayer(Line, _handle, input.Handle, collectiveOperation, reduceOperation, root, groups));
    }
}
