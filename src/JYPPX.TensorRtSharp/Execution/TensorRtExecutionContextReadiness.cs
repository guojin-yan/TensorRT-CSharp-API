using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Summarizes whether a TensorRT execution context has enough shape and address state for enqueue.
/// 汇总 TensorRT execution context 是否已经具备 enqueue 所需的 shape 与地址绑定状态。
/// </summary>
public sealed class TensorRtExecutionContextReadiness
{
    /// <summary>
    /// Creates an execution-context readiness snapshot.
    /// 创建 execution context 就绪状态快照。
    /// </summary>
    public TensorRtExecutionContextReadiness(
        bool allInputDimensionsSpecified,
        bool allInputShapesSpecified,
        int activeOptimizationProfile,
        int? shapeInferenceMissingTensorCount,
        IReadOnlyList<TensorRtTensorBindingState> tensors)
    {
        AllInputDimensionsSpecified = allInputDimensionsSpecified;
        AllInputShapesSpecified = allInputShapesSpecified;
        ActiveOptimizationProfile = activeOptimizationProfile;
        ShapeInferenceMissingTensorCount = shapeInferenceMissingTensorCount;
        Tensors = tensors;
    }

    /// <summary>
    /// Gets whether TensorRT reports all input dimensions as specified.
    /// 获取 TensorRT 是否报告所有输入维度都已指定。
    /// </summary>
    public bool AllInputDimensionsSpecified { get; }

    /// <summary>
    /// Gets whether TensorRT reports all input shape tensors as specified.
    /// 获取 TensorRT 是否报告所有输入 shape tensor 都已指定。
    /// </summary>
    public bool AllInputShapesSpecified { get; }

    /// <summary>
    /// Gets the currently active optimization profile index.
    /// 获取当前激活的 optimization profile 索引。
    /// </summary>
    public int ActiveOptimizationProfile { get; }

    /// <summary>
    /// Gets the missing tensor count returned by inferShapes when explicitly requested.
    /// 获取显式请求 inferShapes 时返回的缺失 tensor 数量。
    /// </summary>
    public int? ShapeInferenceMissingTensorCount { get; }

    /// <summary>
    /// Gets the per-tensor binding state snapshots.
    /// 获取逐 tensor 的绑定状态快照。
    /// </summary>
    public IReadOnlyList<TensorRtTensorBindingState> Tensors { get; }

    /// <summary>
    /// Gets whether every input/output tensor has a bound device address.
    /// 获取所有输入/输出 tensor 是否都已经绑定设备地址。
    /// </summary>
    public bool AllTensorAddressesBound
    {
        get
        {
            foreach (TensorRtTensorBindingState tensor in Tensors)
            {
                if (tensor.RequiresAddress && !tensor.IsAddressBound)
                {
                    return false;
                }
            }

            return true;
        }
    }

    /// <summary>
    /// Gets whether the snapshot indicates the context is ready for enqueue.
    /// 获取该快照是否表示 context 已经可以执行 enqueue。
    /// </summary>
    public bool IsReadyForEnqueue =>
        AllInputDimensionsSpecified &&
        AllInputShapesSpecified &&
        ShapeInferenceMissingTensorCount.GetValueOrDefault(0) == 0 &&
        AllTensorAddressesBound;
}
