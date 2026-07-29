using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Selects the output semantics of a TensorRT loop output layer.
/// 选择 TensorRT loop output 层的输出语义。
/// </summary>
public enum TensorRtLoopOutputKind
{
    /// <summary>
    /// Output the tensor value from the last loop iteration.
    /// 输出最后一次循环迭代的张量值。
    /// </summary>
    LastValue = 0,

    /// <summary>
    /// Concatenate values from all iterations in forward order.
    /// 按正向顺序拼接每次迭代的值。
    /// </summary>
    Concatenate = 1,

    /// <summary>
    /// Concatenate values from all iterations in reverse order.
    /// 按反向顺序拼接每次迭代的值。
    /// </summary>
    Reverse = 2
}
/// <summary>
/// Selects how TensorRT limits loop iteration count.
/// 选择 TensorRT 如何限制循环迭代次数。
/// </summary>
public enum TensorRtTripLimitKind
{
    /// <summary>
    /// A scalar Int32/Int64 tensor provides the maximum trip count.
    /// 使用 Int32/Int64 标量张量提供最大迭代次数。
    /// </summary>
    Count = 0,

    /// <summary>
    /// A scalar Bool tensor controls whether the loop should continue.
    /// 使用 Bool 标量张量控制循环是否继续。
    /// </summary>
    While = 1
}
